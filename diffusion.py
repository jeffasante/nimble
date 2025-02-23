# diffusion.py

import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Optional
from functools import partial

class DiffusionProcess:
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        dtype: jnp.dtype = jnp.float32
    ):
        self.num_timesteps = num_timesteps
        self.dtype = dtype
        
        # Define noise schedule
        self.betas = jnp.linspace(beta_start, beta_end, num_timesteps, dtype=dtype)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)
        
        # Pre-compute values for sampling
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
        
        # Pre-compute values for inference
        self.sqrt_recip_alphas = jnp.sqrt(1.0 / self.alphas)
        alphas_cumprod_prev = jnp.append(jnp.array([1.0]), self.alphas_cumprod[:-1])
        self.posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def _sample_timesteps_impl(self, key: jnp.ndarray, batch_size: int) -> jnp.ndarray:
        """Implementation of timestep sampling, separated for proper JIT compilation"""
        return random.randint(
            key,
            shape=(batch_size,),
            minval=0,
            maxval=self.num_timesteps,
            dtype=jnp.int32
        )

    def sample_timesteps(self, key: jnp.ndarray, batch_size: int) -> jnp.ndarray:
        """
        Sample random timesteps for a batch.
        
        Args:
            key: JAX PRNG key
            batch_size: Size of batch to generate timesteps for
            
        Returns:
            timesteps: Random timesteps array
        """
        # Create a JIT-compiled version of the implementation with batch_size as static
        sample_fn = jax.jit(
            self._sample_timesteps_impl,
            static_argnums=(1,)
        )
        return sample_fn(key, batch_size)

    @partial(jax.jit, static_argnums=(0,))
    def add_noise(
        self,
        key: jnp.ndarray,
        sketch_coords: jnp.ndarray,
        timesteps: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Add noise to sketch coordinates based on timestep.
        """
        noise = random.normal(
            key,
            sketch_coords.shape,
            dtype=self.dtype
        )
        
        sqrt_alphas = jnp.take(self.sqrt_alphas_cumprod, timesteps)
        sqrt_one_minus_alphas = jnp.take(self.sqrt_one_minus_alphas_cumprod, timesteps)
        
        sqrt_alphas = sqrt_alphas[:, None, None]
        sqrt_one_minus_alphas = sqrt_one_minus_alphas[:, None, None]
        
        noised_coords = sqrt_alphas * sketch_coords + sqrt_one_minus_alphas * noise
        return noised_coords, noise

    @partial(jax.jit, static_argnums=(0,))
    def remove_noise(
        self,
        noised_coords: jnp.ndarray,
        pred_noise: jnp.ndarray,
        timesteps: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Remove predicted noise from noised coordinates.
        """
        sqrt_alphas = jnp.take(self.sqrt_alphas_cumprod, timesteps)
        sqrt_one_minus_alphas = jnp.take(self.sqrt_one_minus_alphas_cumprod, timesteps)
        
        sqrt_alphas = sqrt_alphas[:, None, None]
        sqrt_one_minus_alphas = sqrt_one_minus_alphas[:, None, None]
        
        return (noised_coords - sqrt_one_minus_alphas * pred_noise) / sqrt_alphas

    @partial(jax.jit, static_argnums=(0,))
    def q_sample(
        self,
        key: jnp.ndarray,
        sketch_coords: jnp.ndarray,
        timesteps: jnp.ndarray
    ) -> jnp.ndarray:
        """Sample from the forward diffusion process."""
        noised_coords, _ = self.add_noise(key, sketch_coords, timesteps)
        return noised_coords

    @partial(jax.jit, static_argnums=(0,))
    def p_sample(
        self,
        key: jnp.ndarray,
        model_fn,
        noised_coords: jnp.ndarray,
        timesteps: jnp.ndarray,
        clip_denoised: bool = True
    ) -> jnp.ndarray:
        """Sample from the reverse diffusion process (single step)."""
        pred_noise = model_fn(noised_coords, timesteps)
        pred_coords = self.remove_noise(noised_coords, pred_noise, timesteps)
        
        if clip_denoised:
            pred_coords = jnp.clip(pred_coords, -1.0, 1.0)
        return pred_coords

def test_diffusion():
    """Test the diffusion process implementation"""
    import matplotlib.pyplot as plt
    
    # Initialize
    key = random.PRNGKey(0)
    diffusion = DiffusionProcess()
    
    # Create dummy data
    batch_size = 4
    num_points = 100
    key, subkey = random.split(key)
    sketch_coords = random.normal(
        subkey,
        shape=(batch_size, num_points, 2)
    )
    
    # Sample timesteps
    key, subkey = random.split(key)
    timesteps = diffusion.sample_timesteps(subkey, batch_size)
    
    # Add noise
    key, subkey = random.split(key)
    noised_coords, noise = diffusion.add_noise(subkey, sketch_coords, timesteps)
    
    # Test shapes
    print(f"Original coords shape: {sketch_coords.shape}")
    print(f"Noised coords shape: {noised_coords.shape}")
    print(f"Timesteps shape: {timesteps.shape}")
    
    # Visualize noising process for first example
    plt.figure(figsize=(15, 5))
    
    # Original
    plt.subplot(131)
    plt.scatter(sketch_coords[0, :, 0], sketch_coords[0, :, 1], s=1)
    plt.title("Original")
    plt.axis('equal')
    
    # Noised
    plt.subplot(132)
    plt.scatter(noised_coords[0, :, 0], noised_coords[0, :, 1], s=1)
    plt.title(f"Noised (t={timesteps[0]})")
    plt.axis('equal')
    
    # Noise
    plt.subplot(133)
    plt.scatter(noise[0, :, 0], noise[0, :, 1], s=1)
    plt.title("Added Noise")
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_diffusion()