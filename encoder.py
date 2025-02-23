#encoder.py
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional, Dict, Any
from functools import partial

class ConvBNAct(nn.Module):
    """Convolution-BatchNorm-Activation block"""
    features: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    padding: str = 'SAME'
    use_bias: bool = False
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias
        )(x)
        x = nn.BatchNorm(
            use_running_average=not training,
            momentum=0.9,
            epsilon=1e-5
        )(x)
        return nn.relu(x)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    features: int
    reduction_ratio: int = 4
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        input_channels = x.shape[-1]
        
        # Global Average Pooling
        pooled = jnp.mean(x, axis=(1, 2), keepdims=True)
        
        # Reduction
        reduced = nn.Conv(
            features=max(1, input_channels // self.reduction_ratio),
            kernel_size=(1, 1)
        )(pooled)
        reduced = nn.relu(reduced)
        
        # Expansion
        expanded = nn.Conv(
            features=input_channels,
            kernel_size=(1, 1)
        )(reduced)
        weights = nn.sigmoid(expanded)
        
        return x * weights

class InvertedResidual(nn.Module):
    """MobileNetV3 Inverted Residual block"""
    features: int
    stride: int
    expand_ratio: int = 6
    use_se: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        input_channels = x.shape[-1]
        expanded_features = input_channels * self.expand_ratio
        
        # Expansion
        if self.expand_ratio != 1:
            residual = ConvBNAct(
                features=expanded_features,
                kernel_size=(1, 1)
            )(x, training=training)
        else:
            residual = x
        
        # Depthwise
        residual = nn.Conv(
            features=expanded_features,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding='SAME',
            feature_group_count=expanded_features
        )(residual)
        residual = nn.BatchNorm(use_running_average=not training)(residual)
        residual = nn.relu(residual)
        
        # SE block
        if self.use_se:
            residual = SEBlock(features=expanded_features)(residual)
        
        # Projection
        residual = ConvBNAct(
            features=self.features,
            kernel_size=(1, 1)
        )(residual, training=training)
        
        # Skip connection
        if self.stride == 1 and input_channels == self.features:
            return x + residual
        return residual

class BitmapEncoder(nn.Module):
    """Encodes bitmap images into fixed-dimensional embeddings"""
    embed_dim: int = 256
    
    @nn.compact
    def __call__(self, bitmap: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Ensure input has correct shape
        if bitmap.ndim == 3:
            bitmap = jnp.expand_dims(bitmap, axis=-1)
        
        # Initial conv to get to 3 channels
        x = nn.Conv(features=3, kernel_size=(1, 1))(bitmap)
        
        # Early layers adapted for 28x28 input
        x = ConvBNAct(
            features=16,
            kernel_size=(3, 3),
            strides=(2, 2)
        )(x, training=training)
        
        x = InvertedResidual(
            features=24,
            stride=2,
            expand_ratio=4,
            use_se=True
        )(x, training=training)
        
        x = InvertedResidual(
            features=32,
            stride=1,
            expand_ratio=4,
            use_se=True
        )(x, training=training)
        
        # Feature processor
        x = ConvBNAct(
            features=64,
            kernel_size=(1, 1)
        )(x, training=training)
        
        x = ConvBNAct(
            features=self.embed_dim,
            kernel_size=(1, 1)
        )(x, training=training)
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))
        
        return x

def create_encoder_params(
    key: jnp.ndarray,
    batch_size: int = 1,
    image_size: Tuple[int, int] = (28, 28),
    embed_dim: int = 256
) -> Dict[str, Any]:
    """Initialize encoder parameters"""
    encoder = BitmapEncoder(embed_dim=embed_dim)
    dummy_bitmap = jnp.ones((batch_size, *image_size))
    
    variables = encoder.init(
        {'params': key, 'batch_stats': key},
        dummy_bitmap,
        training=False
    )
    
    return variables

def test_encoder():
    """Test the bitmap encoder implementation"""
    import jax.random as random
    
    # Create random key
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    
    # Create dummy batch
    batch_size = 4
    image_size = (28, 28)
    embed_dim = 256
    
    # Initialize model
    encoder = BitmapEncoder(embed_dim=embed_dim)
    variables = create_encoder_params(subkey, batch_size, image_size, embed_dim)
    
    # Create dummy batch
    dummy_batch = jnp.ones((batch_size, *image_size))
    
    # Forward pass
    output = encoder.apply(
        variables,
        dummy_batch,
        training=False,
        mutable=['batch_stats']
    )
    
    # When training=True, output is a tuple (output, mutated_variables)
    # When training=False, output is just the tensor
    if isinstance(output, tuple):
        output, updated_variables = output
    
    print(f"Input shape: {dummy_batch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output embedding dimension: {output.shape[-1]}")
    print("Test successful!")

if __name__ == "__main__":
    test_encoder()