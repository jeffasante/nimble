# decoder.py
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple
import math
from functools import partial

class PositionalEncoding(nn.Module):
    """Positional encoding module using sinusoidal functions"""
    d_model: int
    max_len: int = 5000
    
    def setup(self):
        position = jnp.arange(self.max_len)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model)
        )
        
        pe = jnp.zeros((self.max_len, self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        
        self.pe = pe
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x + self.pe[:x.shape[1]]
        return x

class TimestepEmbedding(nn.Module):
    """Timestep embedding module"""
    embed_dim: int
    
    @nn.compact
    def __call__(self, timesteps: jnp.ndarray) -> jnp.ndarray:
        timesteps = timesteps[:, None].astype(jnp.float32)
        x = nn.Dense(self.embed_dim)(timesteps)
        x = nn.silu(x)
        x = nn.Dense(self.embed_dim)(x)
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head attention module with proper masking support"""
    num_heads: int
    d_model: int
    dropout_rate: float = 0.1
    deterministic: bool = True
    
    def setup(self):
        head_dim = self.d_model // self.num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Dense(3 * self.d_model, use_bias=False)
        self.proj = nn.Dense(self.d_model)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
    
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: Optional[bool] = None
    ) -> jnp.ndarray:
        deterministic = self.deterministic if deterministic is None else deterministic
        batch_size, seq_len = x.shape[:2]
        
        # Get query, key, value projections
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, -1)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        
        if mask is not None:
            attn = jnp.where(mask == 0, float('-inf'), attn)
        
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn, deterministic=deterministic)
        
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        x = self.proj(x)
        x = self.dropout(x, deterministic=deterministic)
        
        return x

class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer"""
    d_model: int
    nhead: int
    dim_feedforward: int = 2048
    dropout: float = 0.1
    deterministic: bool = True
    
    @nn.compact
    def __call__(
        self,
        tgt: jnp.ndarray,
        memory: jnp.ndarray,
        tgt_mask: Optional[jnp.ndarray] = None,
        memory_mask: Optional[jnp.ndarray] = None,
        deterministic: Optional[bool] = None
    ) -> jnp.ndarray:
        deterministic = self.deterministic if deterministic is None else deterministic
        
        # Self attention
        x = nn.LayerNorm()(tgt)
        x = MultiHeadAttention(
            num_heads=self.nhead,
            d_model=self.d_model,
            dropout_rate=self.dropout,
            deterministic=deterministic
        )(x, mask=tgt_mask)
        tgt = tgt + x
        
        # Cross attention
        x = nn.LayerNorm()(tgt)
        x = MultiHeadAttention(
            num_heads=self.nhead,
            d_model=self.d_model,
            dropout_rate=self.dropout,
            deterministic=deterministic
        )(x, mask=memory_mask)
        tgt = tgt + x
        
        # Feedforward
        x = nn.LayerNorm()(tgt)
        x = nn.Dense(self.dim_feedforward)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = nn.Dense(self.d_model)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        tgt = tgt + x
        
        return tgt

class TransformerDecoder(nn.Module):
    """Complete transformer decoder"""
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 1000
    deterministic: bool = True
    
    def setup(self):
        # Create embedding modules
        self.coord_embed = nn.Dense(self.d_model)
        self.timestep_embed = TimestepEmbedding(self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, self.max_seq_length)
        
        # Create decoder layers
        self.layers = [
            TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                deterministic=self.deterministic
            )
            for _ in range(self.num_layers)
        ]
        
        # Output projection
        self.output_proj = nn.Dense(2)
    
    def __call__(
        self,
        noised_coords: jnp.ndarray,
        timesteps: jnp.ndarray,
        image_embedding: jnp.ndarray,
        key_padding_mask: Optional[jnp.ndarray] = None,
        deterministic: Optional[bool] = None
    ) -> jnp.ndarray:
        deterministic = self.deterministic if deterministic is None else deterministic
        
        # Embed coordinates
        x = self.coord_embed(noised_coords)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Get timestep embeddings and expand to sequence length
        t_emb = self.timestep_embed(timesteps)
        t_emb = jnp.expand_dims(t_emb, axis=1)
        t_emb = jnp.tile(t_emb, (1, x.shape[1], 1))
        
        # Combine coordinate embeddings with timestep embeddings
        x = x + t_emb
        
        # Prepare image embedding as memory for cross-attention
        memory = jnp.expand_dims(image_embedding, axis=1)
        
        # Apply transformer decoder layers
        for layer in self.layers:
            x = layer(
                x,
                memory,
                tgt_mask=None,
                memory_mask=None,
                deterministic=deterministic
            )
        
        # Project back to coordinates
        pred_coords = self.output_proj(x)
        
        return pred_coords

def create_decoder_params(
    key: jnp.ndarray,
    batch_size: int = 1,
    seq_length: int = 100,
    d_model: int = 512
) -> dict:
    """Initialize decoder parameters"""
    decoder = TransformerDecoder(d_model=d_model)
    
    # Create dummy inputs for initialization
    dummy_coords = jnp.ones((batch_size, seq_length, 2))
    dummy_timesteps = jnp.zeros((batch_size,), dtype=jnp.int32)
    dummy_image_embedding = jnp.ones((batch_size, d_model))
    
    # Initialize parameters
    return decoder.init(
        key,
        dummy_coords,
        dummy_timesteps,
        dummy_image_embedding,
        deterministic=True
    )

def test_decoder():
    """Test the transformer decoder implementation"""
    import jax.random as random
    
    # Create random key
    key = random.PRNGKey(0)
    
    # Create dummy inputs
    batch_size = 4
    seq_length = 100
    d_model = 256
    
    noised_coords = jnp.ones((batch_size, seq_length, 2))
    timesteps = jnp.zeros((batch_size,), dtype=jnp.int32)
    image_embedding = jnp.ones((batch_size, d_model))
    
    # Initialize model
    decoder = TransformerDecoder(d_model=d_model)
    params = create_decoder_params(key, batch_size, seq_length, d_model)
    
    # Forward pass
    output = decoder.apply(
        params,
        noised_coords,
        timesteps,
        image_embedding,
        deterministic=True
    )
    
    print(f"Input coords shape: {noised_coords.shape}")
    print(f"Output coords shape: {output.shape}")
    print("Test successful!")

if __name__ == "__main__":
    test_decoder()