import jax
import jax.numpy as jnp
import numpy as np
import json
from typing import Tuple, Dict, Any
from flax.training import train_state
import tensorflow as tf

class PhotoSketchDataset:
    def __init__(self, data_path: str, num_points: int = 100, image_size: int = 28):
        self.data_path = data_path
        self.num_points = num_points
        self.image_size = image_size
        self.drawings = []
        
        # Load the NDJSON file
        with open(data_path, 'r') as f:
            for line in f:
                drawing_data = json.loads(line)
                if drawing_data["recognized"]:  # Only use recognized drawings
                    self.drawings.append(drawing_data)
        
        # Load bitmap data
        bitmap_path = self.data_path.replace('-simplified-bird.ndjson', '-numpy_bitmap-bird.npy')
        self.bitmaps = np.load(bitmap_path)
        
        # Convert to numpy arrays for better performance
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess all data at once for better performance"""
        processed_points = []
        processed_bitmaps = []
        
        for idx in range(len(self.drawings)):
            points, bitmap = self._process_single_item(idx)
            processed_points.append(points)
            processed_bitmaps.append(bitmap)
        
        self.processed_points = np.stack(processed_points)
        self.processed_bitmaps = np.stack(processed_bitmaps)
    
    def _process_single_item(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single drawing item"""
        drawing_data = self.drawings[idx]
        strokes = drawing_data["drawing"]
        
        # Convert strokes to absolute coordinates
        all_points = []
        for stroke in strokes:
            x_coords = np.array(stroke[0])
            y_coords = np.array(stroke[1])
            points = np.stack([x_coords, y_coords], axis=1)
            all_points.append(points)
        
        # Concatenate all strokes
        all_points = np.concatenate(all_points, axis=0)
        
        # Normalize coordinates to [0, 1]
        all_points = all_points.astype(np.float32)
        all_points[:, 0] = all_points[:, 0] / 255.0  # X coordinates
        all_points[:, 1] = all_points[:, 1] / 255.0  # Y coordinates
        
        # Sample or pad to desired number of points
        if len(all_points) > self.num_points:
            indices = np.random.choice(len(all_points), self.num_points, replace=False)
            all_points = all_points[indices]
        else:
            padding = np.pad(
                all_points,
                ((0, self.num_points - len(all_points)), (0, 0)),
                mode='wrap'
            )
            all_points = padding
        
        # Get and process bitmap
        bitmap = self.bitmaps[idx]
        bitmap = bitmap.reshape(self.image_size, self.image_size)
        bitmap = bitmap.astype(np.float32) / 255.0
        
        return all_points, bitmap

    def get_dataset(self, batch_size: int = 32, shuffle: bool = True) -> tf.data.Dataset:
        """Create a tf.data.Dataset for efficient batching and prefetching"""
        # Convert numpy arrays to tensors
        points_tensor = tf.convert_to_tensor(self.processed_points, dtype=tf.float32)
        bitmaps_tensor = tf.convert_to_tensor(self.processed_bitmaps, dtype=tf.float32)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((bitmaps_tensor, points_tensor))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.drawings))
        
        # Batch and prefetch
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset

def get_dataloaders(data_path: str, batch_size: int = 32, num_points: int = 100) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Create train and test datasets"""
    dataset = PhotoSketchDataset(data_path, num_points=num_points)
    
    # Calculate split sizes
    total_size = len(dataset.drawings)
    train_size = int(0.8 * total_size)
    
    # Create train/test splits
    indices = np.random.permutation(total_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Split the preprocessed data
    train_points = dataset.processed_points[train_indices]
    train_bitmaps = dataset.processed_bitmaps[train_indices]
    test_points = dataset.processed_points[test_indices]
    test_bitmaps = dataset.processed_bitmaps[test_indices]
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_bitmaps, train_points))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_bitmaps, test_points))
    
    # Configure datasets
    train_dataset = train_dataset.shuffle(buffer_size=len(train_indices))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, test_dataset

# Utility functions for visualization
def show_samples(dataset: tf.data.Dataset, num_samples: int = 5):
    """
    Show samples from the dataset
    Args:
        dataset: tf.data.Dataset instance
        num_samples: Number of samples to show
    """
    import matplotlib.pyplot as plt
    
    # Get samples
    samples = next(iter(dataset.take(1)))
    bitmaps, coords = samples
    
    # Convert to numpy for matplotlib
    bitmaps = bitmaps.numpy()
    coords = coords.numpy()
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(min(num_samples, bitmaps.shape[0])):
        # Plot bitmap
        axes[0, i].imshow(bitmaps[i], cmap='gray')
        axes[0, i].set_title(f'Bitmap {i+1}')
        axes[0, i].axis('off')
        
        # Plot vector coordinates
        axes[1, i].scatter(coords[i, :, 0], coords[i, :, 1], s=1, c='black', alpha=0.5)
        axes[1, i].set_xlim(0, 1)
        axes[1, i].set_ylim(0, 1)
        axes[1, i].invert_yaxis()  # Invert Y axis to match image coordinates
        axes[1, i].set_aspect('equal')
        axes[1, i].set_title(f'Vector {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Test the dataset implementation
def test_dataset():
    # Create a small test dataset
    dataset = PhotoSketchDataset('../data/full-simplified-bird.ndjson', num_points=100)
    train_ds, test_ds = get_dataloaders(
        '../data/full-simplified-bird.ndjson',
        batch_size=32,
        num_points=100
    )
    
    # Print dataset info
    print(f"Dataset size: {len(dataset.drawings)}")
    
    # Test batch iteration
    for bitmaps, coords in train_ds.take(1):
        print(f"Batch shapes - Bitmaps: {bitmaps.shape}, Coords: {coords.shape}")
        break

if __name__ == "__main__":
    test_dataset()