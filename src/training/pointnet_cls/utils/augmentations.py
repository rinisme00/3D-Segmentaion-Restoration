import numpy as np


def shuffle_data(data, labels, rng=None):
    rng = rng or np.random.default_rng()
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    return data[indices], labels[indices]


def rotate_point_cloud(batch_data, rng=None):
    rng = rng or np.random.default_rng()
    rotated = np.zeros(batch_data.shape, dtype=np.float32)
    for idx in range(batch_data.shape[0]):
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        cos_val, sin_val = np.cos(angle), np.sin(angle)
        rotation = np.array(
            [[cos_val, 0.0, sin_val], [0.0, 1.0, 0.0], [-sin_val, 0.0, cos_val]],
            dtype=np.float32,
        )
        rotated[idx] = batch_data[idx] @ rotation
    return rotated


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05, rng=None):
    rng = rng or np.random.default_rng()
    jitter = np.clip(sigma * rng.standard_normal(batch_data.shape), -clip, clip)
    return batch_data + jitter.astype(np.float32)


def random_scale_point_cloud(batch_data, scale_low=0.9, scale_high=1.1, rng=None):
    rng = rng or np.random.default_rng()
    scaled = np.empty_like(batch_data, dtype=np.float32)
    for idx in range(batch_data.shape[0]):
        scale = float(rng.uniform(scale_low, scale_high))
        scaled[idx] = batch_data[idx] * scale
    return scaled


def shift_point_cloud(batch_data, shift_range=0.1, rng=None):
    rng = rng or np.random.default_rng()
    shifts = rng.uniform(-shift_range, shift_range, size=(batch_data.shape[0], 1, 3))
    return batch_data + shifts.astype(np.float32)


def random_point_dropout(batch_data, max_dropout_ratio=0.1, rng=None):
    rng = rng or np.random.default_rng()
    dropped = batch_data.copy()
    for batch_idx in range(batch_data.shape[0]):
        dropout_ratio = float(rng.uniform(0.0, max_dropout_ratio))
        drop_indices = rng.random(batch_data.shape[1]) < dropout_ratio
        if np.any(drop_indices):
            dropped[batch_idx, drop_indices, :] = dropped[batch_idx, 0, :]
    return dropped
