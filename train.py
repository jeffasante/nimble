# train.py
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.training import train_state
from flax.training import checkpoints
import optax
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Dict, Optional, Any
from tqdm import tqdm
import logging
from functools import partial
from jax.tree_util import tree_map

from dataset import get_dataloaders
from encoder import BitmapEncoder
from decoder import TransformerDecoder
from diffusion import DiffusionProcess
from losses import NimbleLoss

def get_device_name():
    """Get the best available device"""
    if jax.default_backend() == "gpu":
        return "cuda"
    elif jax.default_backend() == "cpu":
        # Check for MPS (Apple Silicon)
        try:
            if tf.config.list_physical_devices("GPU"):
                return "mps"
        except:
            pass
    return "cpu"

class TrainState(train_state.TrainState):
    batch_stats: Dict[str, Any]

def create_train_state(
    rng: jnp.ndarray,
    learning_rate: float,
    batch_size: int = 1,
    optimizer: Optional[optax.GradientTransformation] = None
) -> TrainState:
    """Initialize training state with optimized configuration"""
    # Split PRNG key
    rng, encoder_key, decoder_key = random.split(rng, 3)
    
    # Initialize models
    encoder = BitmapEncoder(embed_dim=256)
    decoder = TransformerDecoder(d_model=256)
    
    # Initialize parameters with optimized settings
    encoder_variables = encoder.init(
        {"params": encoder_key, "batch_stats": encoder_key},
        jnp.ones((batch_size, 28, 28)),
        training=True
    )
    
    decoder_variables = decoder.init(
        {"params": decoder_key},
        jnp.ones((batch_size, 100, 2)),
        jnp.zeros((batch_size,), dtype=jnp.int32),
        jnp.ones((batch_size, 256)),
        deterministic=False
    )
    
    # Combine parameters and batch stats
    params = {
        "encoder": encoder_variables["params"],
        "decoder": decoder_variables["params"]
    }
    batch_stats = {
        "encoder": encoder_variables.get("batch_stats", {})
    }
    
    # Use provided optimizer or create default
    if optimizer is None:
        optimizer = optax.adam(learning_rate)
    
    return TrainState.create(
        apply_fn=None,
        params=params,
        tx=optimizer,
        batch_stats=batch_stats
    )



class NimbleTrainer:
    def __init__(
        self,
        learning_rate: float = 1e-5,  # Further reduced learning rate
        checkpoint_dir: str = "checkpoints",
        use_wandb: bool = False,
        batch_size: int = 32
    ):
        """Initialize trainer with further optimized configurations"""
        # Previous initialization code remains the same
        self.device = get_device_name()
        print(f"Using device: {self.device}")
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.use_wandb = use_wandb
        
        # Initialize models
        self.encoder = BitmapEncoder(embed_dim=256)
        self.decoder = TransformerDecoder(d_model=256)
        self.diffusion = DiffusionProcess()
        self.criterion = NimbleLoss(
            coord_weight=0.5,    # Reduced coordinate weight
            raster_weight=0.05   # Further reduced raster weight
        )
        
        # Initialize PRNG key
        self.rng = random.PRNGKey(0)
        
        # Create training state with optimized schedule
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=200,     # Increased warmup steps
            decay_steps=15000,    # Increased decay steps
            end_value=learning_rate / 20.0  # More aggressive decay
        )
        
        # Enhanced optimizer chain
        optimizer = optax.chain(
            optax.clip_by_global_norm(0.5),    # More aggressive gradient clipping
            optax.scale_by_adam(b1=0.9, b2=0.99),  # Adjusted beta parameters
            optax.scale_by_schedule(schedule)
        )
        
        self.state = create_train_state(
            self.rng, 
            learning_rate, 
            batch_size,
            optimizer=optimizer
        )
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Enhanced loss normalization
        self.coord_scale = 2000.0    # Increased coordinate scaling
        self.raster_scale = 0.5      # Reduced raster scaling


    def _forward(
        self,
        params: Dict,
        batch_stats: Dict,
        bitmap: jnp.ndarray,
        coords: jnp.ndarray,
        timesteps: jnp.ndarray,
        training: bool = True,
        rngs: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Forward pass through the models"""
        # Get variables
        encoder_variables = {
            "params": params["encoder"],
            "batch_stats": batch_stats["encoder"],
        }
        decoder_variables = {
            "params": params["decoder"]
        }
        
        # Default RNGs if none provided
        if rngs is None:
            rngs = {}
        
        # Run encoder
        encoder_output = self.encoder.apply(
            encoder_variables,
            bitmap,
            training=training,
            mutable=["batch_stats"] if training else False,
            rngs=rngs
        )
        if training:
            image_embeddings, new_encoder_stats = encoder_output
            new_batch_stats = {"encoder": new_encoder_stats["batch_stats"]}
        else:
            image_embeddings = encoder_output
            new_batch_stats = {}

        # Run decoder
        pred_coords = self.decoder.apply(
            decoder_variables,
            coords,
            timesteps,
            image_embeddings,
            deterministic=not training,
            rngs=rngs
        )

        return pred_coords, new_batch_stats

    @partial(jax.jit, static_argnames=("self", "training"))
    def _forward_jit(
        self,
        params: Dict,
        batch_stats: Dict,
        bitmap: jnp.ndarray,
        coords: jnp.ndarray,
        timesteps: jnp.ndarray,
        training: bool = True,
        rngs: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        return self._forward(
            params, batch_stats, bitmap, coords, timesteps, training, rngs
        )

    def compute_loss(
        self,
        pred_coords: jnp.ndarray,
        target_coords: jnp.ndarray,
        target_bitmap: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """Compute losses with enhanced normalization"""
        # Apply pre-normalization to coordinates
        pred_coords = pred_coords / self.coord_scale
        target_coords = target_coords / self.coord_scale
        
        losses = self.criterion(pred_coords, target_coords, target_bitmap)
        
        # Post-process losses
        losses = {
            'coord_loss': losses['coord_loss'] * self.coord_scale,
            'raster_loss': losses['raster_loss'] * self.raster_scale,
            'total_loss': (losses['coord_loss'] * self.coord_scale * self.criterion.coord_weight +
                         losses['raster_loss'] * self.raster_scale * self.criterion.raster_weight)
        }
        
        return losses

    @partial(jax.jit, static_argnames=("self",))
    def compute_loss_jit(
        self,
        pred_coords: jnp.ndarray,
        target_coords: jnp.ndarray,
        target_bitmap: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        return self.compute_loss(pred_coords, target_coords, target_bitmap)

    def _train_step_impl(
        self,
        state: TrainState,
        bitmap: jnp.ndarray,
        coords: jnp.ndarray,
        timesteps: jnp.ndarray,
        noise_key: jnp.ndarray,
        dropout_key: jnp.ndarray,
    ) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        """Implementation of single training step"""
        # Add noise to target coordinates
        noised_coords, noise = self.diffusion.add_noise(noise_key, coords, timesteps)
        
        # Prepare RNG dict for dropout
        rngs = {"dropout": dropout_key}
        
        # Forward pass and loss computation
        pred_coords, new_batch_stats = self._forward_jit(
            state.params,
            state.batch_stats,
            bitmap,
            noised_coords,
            timesteps,
            True,  # training
            rngs=rngs
        )
        
        losses = self.compute_loss_jit(pred_coords, coords, bitmap)
        
        # Compute gradients
        grad_fn = jax.value_and_grad(lambda p: losses["total_loss"])
        _, grads = grad_fn(state.params)
        
        # Update parameters and batch stats
        state = state.apply_gradients(grads=grads, batch_stats=new_batch_stats)
        
        return state, losses

    @partial(jax.jit, static_argnames=("self",))
    def train_step_jit(
        self,
        state: TrainState,
        bitmap: jnp.ndarray,
        coords: jnp.ndarray,
        timesteps: jnp.ndarray,
        noise_key: jnp.ndarray,
        dropout_key: jnp.ndarray,
    ) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        return self._train_step_impl(
            state, bitmap, coords, timesteps, noise_key, dropout_key
        )

    def train_step(
        self,
        state: TrainState,
        batch: Tuple[jnp.ndarray, jnp.ndarray],
        rng: jnp.ndarray
    ) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        """Single training step with proper handling of dynamic shapes"""
        bitmap, coords = batch
        
        # Split PRNG keys for different random operations
        rng, timestep_key, noise_key, dropout_key = random.split(rng, 4)
        
        # Sample timesteps
        timesteps = self.diffusion.sample_timesteps(timestep_key, bitmap.shape[0])
        
        return self.train_step_jit(
            state, bitmap, coords, timesteps, noise_key, dropout_key
        )

    def eval_step(
        self,
        state: TrainState,
        batch: Tuple[jnp.ndarray, jnp.ndarray],
        rng: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """Single evaluation step"""
        bitmap, target_coords = batch
        
        # Split PRNG keys
        rng, timestep_key, noise_key = random.split(rng, 3)
        
        # Sample timesteps
        timesteps = self.diffusion.sample_timesteps(timestep_key, bitmap.shape[0])
        
        # Add noise to target coordinates
        noised_coords, noise = self.diffusion.add_noise(
            noise_key, target_coords, timesteps
        )
        
        # Forward pass (no dropout in eval)
        pred_coords, _ = self._forward_jit(
            state.params,
            state.batch_stats,
            bitmap,
            noised_coords,
            timesteps,
            False,  # not training
        )
        
        # Compute losses
        return self.compute_loss_jit(pred_coords, target_coords, bitmap)

    @partial(jax.jit, static_argnames=("self",))
    def eval_step_jit(self, *args, **kwargs):
        return self.eval_step(*args, **kwargs)


    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
        num_epochs: int = 100,
        validate_every: int = 1,
    ):
        """Train the model"""
        best_val_loss = float("inf")
        
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_losses = []
            pbar = tqdm(train_dataset, desc="Training")
            for batch in pbar:
                # Convert TF tensors to JAX arrays
                bitmap = jnp.array(batch[0].numpy())
                coords = jnp.array(batch[1].numpy())
                
                # Split PRNG key
                self.rng, step_key = random.split(self.rng)
                
                # Training step
                self.state, losses = self.train_step(
                    self.state,
                    (bitmap, coords),
                    step_key
                )
                train_losses.append(losses)
                
                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": float(losses["total_loss"]),
                        "coord_loss": float(losses["coord_loss"]),
                        "raster_loss": float(losses["raster_loss"]),
                    }
                )
            
            # Calculate average training losses using updated JAX function
            from jax.tree_util import tree_map
            train_metrics = tree_map(
                lambda *x: jnp.mean(jnp.stack(x)),
                *train_losses
            )
            
            # Save regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    epoch,
                    train_metrics,
                    is_best=False,
                    checkpoint_name=f"checkpoint_epoch_{epoch}"
                )
            
            # Validation
            if val_dataset is not None and epoch % validate_every == 0:
                val_losses = []
                for batch in tqdm(val_dataset, desc="Validation"):
                    bitmap = jnp.array(batch[0].numpy())
                    coords = jnp.array(batch[1].numpy())
                    
                    self.rng, step_key = random.split(self.rng)
                    losses = self.eval_step(
                        self.state,
                        (bitmap, coords),
                        step_key
                    )
                    val_losses.append(losses)
                
                # Calculate average validation losses
                val_metrics = tree_map(
                    lambda *x: jnp.mean(jnp.stack(x)),
                    *val_losses
                )
                
                # Log metrics
                self.logger.info(
                    f"Train Loss: {float(train_metrics['total_loss']):.4f}, "
                    f"Val Loss: {float(val_metrics['total_loss']):.4f}"
                )
                
                if self.use_wandb:
                    metrics = {
                        **{f"train_{k}": float(v) for k, v in train_metrics.items()},
                        **{f"val_{k}": float(v) for k, v in val_metrics.items()},
                    }
                    wandb.log(metrics, step=epoch)
                
                # Check and save best model
                current_val_loss = float(val_metrics["total_loss"])
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    self.logger.info(f"New best validation loss: {best_val_loss:.4f}")
                    # Save best model in a separate directory
                    best_checkpoint_path = f"best_model_epoch_{epoch}"
                    self.save_checkpoint(
                        epoch,
                        val_metrics,
                        is_best=True,
                        checkpoint_name=best_checkpoint_path
                    )


    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, jnp.ndarray],
        is_best: bool = False,
        checkpoint_name: str = None
    ):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "state": self.state,
            "metrics": metrics,
            "best_val_loss": getattr(self, "best_val_loss", None)
        }

        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_epoch_{epoch}"
        
        # Convert to absolute path
        checkpoint_path = self.checkpoint_dir.absolute() / checkpoint_name
        
        # Set keep parameter based on checkpoint type
        if is_best:
            keep = 1  # Only keep the best checkpoint
        else:
            keep = 2  # Keep last 2 regular checkpoints
        
        checkpoints.save_checkpoint(
            str(checkpoint_path),  # Convert Path to string
            checkpoint,
            epoch,
            keep=keep  # Always specify a number for keep
        )
        
        self.logger.info(
            f"Saved {'best ' if is_best else ''}checkpoint to {checkpoint_path}"
        )

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        # Convert to absolute path if necessary
        checkpoint_path = Path(checkpoint_path).absolute()
        checkpoint = checkpoints.restore_checkpoint(str(checkpoint_path), target=None)
        self.state = checkpoint["state"]
        if "best_val_loss" in checkpoint:
            self.best_val_loss = checkpoint["best_val_loss"]
        return checkpoint["epoch"], checkpoint["metrics"]

def main():
    """Main training function with improved configurations"""
    # Get datasets
    train_ds, val_ds = get_dataloaders(
        data_path='../data/full-simplified-bird.ndjson',
        batch_size=32,
        num_points=100
    )
    
    # Create trainer with improved hyperparameters
    trainer = NimbleTrainer(
        learning_rate=3e-5,  # Lower learning rate
        use_wandb=False,
        batch_size=32
    )
    
    # Start training
    trainer.train(
        train_ds,
        val_ds,
        num_epochs=1,
        # num_epochs=100,
        validate_every=1
    )


if __name__ == "__main__":
    main()