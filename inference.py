# inference.py

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.training import checkpoints
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple, List
from pathlib import Path
from functools import partial

from encoder import BitmapEncoder
from decoder import TransformerDecoder
from diffusion import DiffusionProcess

class NimbleInference:
    def __init__(
        self,
        checkpoint_path: str,
        image_size: Tuple[int, int] = (28, 28),
        num_points: int = 100
    ):
        """Initialize inference model"""
        self.image_size = image_size
        self.num_points = num_points
        self.rng = random.PRNGKey(0)
        
        self.device = self._get_device()
        print(f"Using device: {self.device}")
        
        try:
            self.encoder = BitmapEncoder(embed_dim=256)
            self.decoder = TransformerDecoder(d_model=256)
            self.diffusion = DiffusionProcess()
            
            self._forward_encoder_impl = jax.jit(self._forward_encoder_impl)
            self._forward_decoder_impl = jax.jit(self._forward_decoder_impl)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize models: {str(e)}")
        
        self._load_checkpoint_with_verification(checkpoint_path)

    def _process_input(
        self, 
        input_data: Union[str, Image.Image, jnp.ndarray, np.ndarray]
    ) -> jnp.ndarray:
        """Process input into bitmap array"""
        if isinstance(input_data, str):
            input_data = Image.open(input_data).convert('L')
        
        if isinstance(input_data, Image.Image):
            input_data = input_data.resize(self.image_size)
            input_data = np.array(input_data)
        
        if isinstance(input_data, np.ndarray):
            if input_data.ndim == 3:
                input_data = np.mean(input_data, axis=2)
            if input_data.shape != self.image_size:
                temp_img = Image.fromarray((input_data * 255).astype('uint8'))
                temp_img = temp_img.resize(self.image_size)
                input_data = np.array(temp_img)
        
        bitmap = jnp.array(input_data, dtype=jnp.float32) / 255.0
        if bitmap.ndim == 2:
            bitmap = bitmap[None, ..., None]
        elif bitmap.ndim == 3 and bitmap.shape[-1] > 1:
            bitmap = jnp.mean(bitmap, axis=-1)[None, ..., None]
        
        return bitmap

    def _forward_encoder_impl(
        self,
        encoder_params: dict,
        batch_stats: dict,
        bitmap: jnp.ndarray
    ) -> jnp.ndarray:
        """Implementation of encoder forward pass"""
        encoder_variables = {
            'params': encoder_params,
            'batch_stats': batch_stats
        }
        return self.encoder.apply(
            encoder_variables,
            bitmap,
            training=False,
            mutable=False
        )

    def _forward_decoder_impl(
        self,
        decoder_params: dict,
        coords: jnp.ndarray,
        t: jnp.ndarray,
        image_embedding: jnp.ndarray
    ) -> jnp.ndarray:
        """Implementation of decoder forward pass"""
        decoder_variables = {'params': decoder_params}
        return self.decoder.apply(
            decoder_variables,
            coords,
            t,
            image_embedding,
            deterministic=True
        )

    def _forward_encoder(
        self,
        bitmap: jnp.ndarray
    ) -> jnp.ndarray:
        """Forward pass through encoder using stored parameters"""
        return self._forward_encoder_impl(
            self.encoder_params,
            self.batch_stats,
            bitmap
        )

    def _forward_decoder(
        self,
        coords: jnp.ndarray,
        t: jnp.ndarray,
        image_embedding: jnp.ndarray
    ) -> jnp.ndarray:
        """Forward pass through decoder using stored parameters"""
        return self._forward_decoder_impl(
            self.decoder_params,
            coords,
            t,
            image_embedding
        )

    def _get_device(self) -> str:
        """Get the best available device with error handling"""
        try:
            if jax.default_backend() == "gpu":
                return "cuda"
            elif jax.default_backend() == "cpu":
                try:
                    if any(dev.device_type == "gpu" for dev in jax.local_devices()):
                        return "mps"
                except:
                    pass
            return "cpu"
        except Exception as e:
            print(f"Warning: Error detecting device, defaulting to CPU. Error: {str(e)}")
            return "cpu"
    
    def _load_checkpoint_with_verification(self, checkpoint_path: str):
        """Load and verify checkpoint with proper parameter restructuring"""
        try:
            checkpoint_path = Path(checkpoint_path).absolute()
            if not checkpoint_path.exists():
                checkpoint_with_suffix = checkpoint_path.parent / "checkpoint_0"
                if checkpoint_with_suffix.exists():
                    checkpoint_path = checkpoint_with_suffix
                else:
                    raise FileNotFoundError(
                        f"Checkpoint path does not exist: {checkpoint_path}\n"
                        f"Attempted alternate path: {checkpoint_with_suffix}"
                    )
            
            checkpoint = checkpoints.restore_checkpoint(str(checkpoint_path), target=None)
            if checkpoint is None:
                raise ValueError(f"Failed to load checkpoint from {checkpoint_path}")
            
            print("Checkpoint keys:", checkpoint.keys())
            
            if 'state' not in checkpoint:
                raise ValueError("Checkpoint missing 'state' key")
            
            state = checkpoint['state']
            print("State keys:", state.keys())
            
            if 'params' not in state or 'batch_stats' not in state:
                raise ValueError("State missing required keys: params and/or batch_stats")
            
            # Extract encoder and decoder parameters separately
            self.encoder_params = state['params']['encoder']
            self.decoder_params = state['params']['decoder']
            self.batch_stats = state['batch_stats']['encoder']
            
            # Debug print to verify parameter structure
            print("Encoder params keys:", list(self.encoder_params.keys()))
            print("Decoder params keys:", list(self.decoder_params.keys()))
            print("Batch stats keys:", list(self.batch_stats.keys()))
            
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
            
        except Exception as e:
            error_msg = (
                f"Error loading checkpoint: {str(e)}\n\n"
                "Potential solutions:\n"
                "1. Ensure you have run the training process first\n"
                "2. Check the checkpoint path is correct\n"
                "3. Verify the checkpoint file isn't corrupted\n"
                "4. Make sure you're using the same model version that created the checkpoint"
            )
            raise RuntimeError(error_msg) from e

    def generate(
        self,
        input_data: Union[str, Image.Image, jnp.ndarray, np.ndarray],
        temperature: float = 1.0,
        return_process: bool = False
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, List[jnp.ndarray]]]:
        """Generate sketch coordinates from input image"""
        bitmap = self._process_input(input_data)
        image_embedding = self._forward_encoder(bitmap)
        
        self.rng, init_key = random.split(self.rng)
        coords = random.normal(
            init_key,
            shape=(1, self.num_points, 2)
        ) * temperature
        
        process = [] if return_process else None
        
        timesteps = jnp.arange(self.diffusion.num_timesteps - 1, -1, -1)
        
        for t in timesteps:
            if process is not None:
                process.append(coords)
            
            t_batch = jnp.array([t])
            pred_coords = self._forward_decoder(coords, t_batch, image_embedding)
            
            if t > 0:
                coords = self.diffusion.remove_noise(coords, pred_coords, t_batch)
                self.rng, noise_key = random.split(self.rng)
                noise = random.normal(noise_key, coords.shape) * temperature
                coords = coords + noise * (t / self.diffusion.num_timesteps)
        
        if return_process:
            return coords, process
        return coords

    def visualize_generation(
        self,
        input_data: Union[str, Image.Image, jnp.ndarray, np.ndarray],
        temperature: float = 1.0,
        num_steps: int = 5
    ):
        """Visualize the generation process"""
        coords, process = self.generate(input_data, temperature, return_process=True)
        bitmap = self._process_input(input_data)
        bitmap_display = np.array(bitmap[0, ..., 0])
        
        step_indices = np.linspace(0, len(process) - 1, num_steps, dtype=int)
        
        fig, axes = plt.subplots(1, num_steps + 2, figsize=(3 * (num_steps + 2), 3))
        
        axes[0].imshow(bitmap_display, cmap='gray')
        axes[0].set_title("Input Image")
        axes[0].axis("off")
        
        for i, idx in enumerate(step_indices):
            step_coords = np.array(process[idx][0])
            axes[i + 1].scatter(step_coords[:, 0], step_coords[:, 1], s=1, c="black")
            axes[i + 1].set_xlim(0, 1)
            axes[i + 1].set_ylim(0, 1)
            axes[i + 1].invert_yaxis()
            axes[i + 1].set_aspect("equal")
            axes[i + 1].axis("off")
            axes[i + 1].set_title(f"Step {idx}")
        
        final_coords = np.array(coords[0])
        axes[-1].scatter(final_coords[:, 0], final_coords[:, 1], s=1, c="black")
        axes[-1].set_xlim(0, 1)
        axes[-1].set_ylim(0, 1)
        axes[-1].invert_yaxis()
        axes[-1].set_aspect("equal")
        axes[-1].axis("off")
        axes[-1].set_title("Final Sketch")
        
        plt.tight_layout()
        plt.show()

def test_inference():
    """Test the inference and visualization with enhanced error handling"""
    try:
        print("Initializing inference model...")
        inferencer = NimbleInference(
            checkpoint_path="checkpoints/best_model/checkpoint",
            image_size=(28, 28),
            num_points=100
        )
        
        test_cases = [
            ("Test with image file", "../test_images/bird_0.jpg"),
            ("Test with numpy array", np.random.rand(100, 100, 3)),
            ("Test with PIL Image", Image.fromarray(
                (np.random.rand(100, 100, 3) * 255).astype('uint8')
            ))
        ]
        
        for test_name, test_input in test_cases:
            try:
                print(f"\n{test_name}...")
                inferencer.visualize_generation(test_input, temperature=0.8)
                print(f"✓ {test_name} completed successfully")
            except Exception as e:
                print(f"✗ {test_name} failed: {str(e)}")
        
        return True
        
    except Exception as e:
        print("\nTest failed with error:")
        print(f"{str(e)}")
        print("\nPlease ensure:")
        print("1. The model has been trained and checkpoints are saved")
        print("2. All required dependencies are installed")
        print("3. Test images are available in the correct location")
        return False

if __name__ == "__main__":
    success = test_inference()
    if success:
        print("\nAll tests completed.")
    else:
        print("\nTests failed. Please check the error messages above.")