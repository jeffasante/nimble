#losses.py
import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
from typing import Tuple, Dict, Optional
from functools import partial

class NimbleLoss(nn.Module):
    """Loss module for sketch generation"""
    coord_weight: float = 1.0
    raster_weight: float = 0.5
    canvas_size: Tuple[int, int] = (28, 28)  # Bitmap size
    
    def coordinate_loss(
        self,
        pred_coords: jnp.ndarray,
        target_coords: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute L2 loss between predicted and target coordinates"""
        return jnp.mean((pred_coords - target_coords) ** 2)

    def _rasterize_line(
        self,
        x0: jnp.ndarray,
        y0: jnp.ndarray,
        x1: jnp.ndarray,
        y1: jnp.ndarray,
        canvas: jnp.ndarray
    ) -> jnp.ndarray:
        """Rasterize a single line using JAX control flow"""
        dx = jnp.abs(x1 - x0)
        dy = jnp.abs(y1 - y0)
        
        def swap_points(x0, y0, x1, y1):
            return y0, x0, y1, x1

        # Handle steep lines
        steep = dy > dx
        x0, y0, x1, y1 = lax.cond(
            steep,
            lambda args: swap_points(*args),
            lambda args: args,
            (x0, y0, x1, y1)
        )
        
        # Ensure x0 <= x1
        swap = x0 > x1
        x0, x1 = lax.cond(
            swap,
            lambda x: (x[1], x[0]),
            lambda x: x,
            (x0, x1)
        )
        y0, y1 = lax.cond(
            swap,
            lambda y: (y[1], y[0]),
            lambda y: y,
            (y0, y1)
        )
        
        dx = x1 - x0
        dy = jnp.abs(y1 - y0)
        ystep = jnp.where(y0 < y1, 1, -1)
        
        def plot_point(args):
            x, y, canvas = args
            # Handle steep vs non-steep lines
            plot_x = lax.cond(steep, lambda: y, lambda: x)
            plot_y = lax.cond(steep, lambda: x, lambda: y)
            
            # Check bounds
            valid = (plot_x >= 0) & (plot_x < canvas.shape[1]) & \
                   (plot_y >= 0) & (plot_y < canvas.shape[0])
            
            # Update canvas
            return lax.cond(
                valid,
                lambda: canvas.at[plot_y, plot_x].set(1.0),
                lambda: canvas
            )
        
        def body_fn(i, state):
            x = x0 + i
            y = y0 + (ystep * dy * i / dx).astype(jnp.int32)
            canvas = plot_point((x, y, state))
            return canvas
        
        # Draw the line using fori_loop
        canvas = lax.fori_loop(
            0,
            dx.astype(jnp.int32) + 1,
            body_fn,
            canvas
        )
        
        return canvas

    @partial(jax.jit, static_argnums=(0,))
    def rasterize_strokes(
        self,
        coords: jnp.ndarray
    ) -> jnp.ndarray:
        """Rasterize vector coordinates to image"""
        batch_size, num_points, _ = coords.shape
        H, W = self.canvas_size
        
        # Scale coordinates to pixel space
        coords = coords * jnp.array([W - 1, H - 1])
        coords = coords.astype(jnp.int32)
        
        def rasterize_single(coords_single):
            # Initialize empty canvas
            canvas = jnp.zeros(self.canvas_size)
            
            def draw_segment(i, canvas):
                x0, y0 = coords_single[i]
                x1, y1 = coords_single[i + 1]
                return self._rasterize_line(x0, y0, x1, y1, canvas)
            
            return lax.fori_loop(0, num_points - 1, draw_segment, canvas)
        
        # Vectorize over batch
        return jax.vmap(rasterize_single)(coords)
    
    def rasterization_loss(
        self,
        pred_coords: jnp.ndarray,
        target_coords: jnp.ndarray,
        target_bitmap: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Compute rasterization loss"""
        pred_raster = self.rasterize_strokes(pred_coords)
        
        if target_bitmap is not None:
            # Binary cross entropy with target bitmap
            epsilon = 1e-7
            pred_raster = jnp.clip(pred_raster, epsilon, 1.0 - epsilon)
            bce = -(target_bitmap * jnp.log(pred_raster) + 
                   (1 - target_bitmap) * jnp.log(1 - pred_raster))
            return jnp.mean(bce)
        else:
            # Compare with rasterized target coordinates
            target_raster = self.rasterize_strokes(target_coords)
            pred_raster = jnp.clip(pred_raster, 1e-7, 1.0 - 1e-7)
            bce = -(target_raster * jnp.log(pred_raster) + 
                   (1 - target_raster) * jnp.log(1 - pred_raster))
            return jnp.mean(bce)
    
    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        pred_coords: jnp.ndarray,
        target_coords: jnp.ndarray,
        target_bitmap: Optional[jnp.ndarray] = None
    ) -> Dict[str, jnp.ndarray]:
        """Compute combined loss"""
        coord_loss = self.coordinate_loss(pred_coords, target_coords)
        raster_loss = self.rasterization_loss(pred_coords, target_coords, target_bitmap)
        
        total_loss = self.coord_weight * coord_loss + self.raster_weight * raster_loss
        
        return {
            'coord_loss': coord_loss,
            'raster_loss': raster_loss,
            'total_loss': total_loss
        }

def test_losses():
    """Test the loss implementation"""
    import jax.random as random
    import matplotlib.pyplot as plt
    
    # Create random key
    key = random.PRNGKey(0)
    
    # Create dummy data
    batch_size = 4
    num_points = 100
    H, W = 28, 28
    
    # Generate random coordinates
    key, subkey1, subkey2 = random.split(key, 3)
    pred_coords = random.uniform(subkey1, (batch_size, num_points, 2))
    target_coords = random.uniform(subkey2, (batch_size, num_points, 2))
    
    # Create random target bitmap
    key, subkey = random.split(key)
    target_bitmap = random.bernoulli(subkey, shape=(batch_size, H, W)).astype(jnp.float32)
    
    # Initialize loss
    criterion = NimbleLoss()
    
    # Test without target bitmap
    losses = criterion(pred_coords, target_coords)
    print("\nLosses without target bitmap:")
    for k, v in losses.items():
        print(f"{k}: {v}")
        
    # Test with target bitmap
    losses = criterion(pred_coords, target_coords, target_bitmap)
    print("\nLosses with target bitmap:")
    for k, v in losses.items():
        print(f"{k}: {v}")
    
    # Visualize rasterization
    rasterized = criterion.rasterize_strokes(pred_coords)
    
    # Plot first example
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.scatter(pred_coords[0, :, 0], pred_coords[0, :, 1], s=1)
    plt.title("Vector Coordinates")
    plt.axis('equal')
    
    plt.subplot(132)
    plt.imshow(rasterized[0], cmap='gray')
    plt.title("Rasterized Output")
    
    plt.subplot(133)
    plt.imshow(target_bitmap[0], cmap='gray')
    plt.title("Target Bitmap")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_losses()