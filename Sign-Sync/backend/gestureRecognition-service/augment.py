import numpy as np
import random

def rotate_keypoints(kps, axis='z', angle_deg=5):
    """Rotate 3D keypoints around the specified axis, centered on their centroid."""
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    if axis == 'x':
        R = np.array([[1, 0, 0],
                      [0, cos_a, -sin_a],
                      [0, sin_a, cos_a]])
    elif axis == 'y':
        R = np.array([[cos_a, 0, sin_a],
                      [0, 1, 0],
                      [-sin_a, 0, cos_a]])
    else:  # z-axis
        R = np.array([[cos_a, -sin_a, 0],
                      [sin_a, cos_a, 0],
                      [0, 0, 1]])

    center = kps.mean(axis=0, keepdims=True)
    return np.dot(kps - center, R.T) + center

def augment_keypoints_3d(seq, prob_drop=0.1, seed=None, global_dropout=False, intensity=1.0):
    """
    Augments a (T, D) sequence of 3D keypoints with jitter, shift, rotation, occlusion, and dropout.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    seq = seq.copy()
    T, D = seq.shape
    K = D // 3
    keypoints = seq.reshape(T, K, 3)


    transforms = random.sample(['jitter', 'shift', 'rotate', 'drop', 'occlude'], k=random.randint(1, 4))

    if 'jitter' in transforms:
        jitter_std = np.random.uniform(0.005, 0.02 * intensity, size=(1, 1, 3))
        keypoints += np.random.normal(0, jitter_std, keypoints.shape)

    if 'shift' in transforms:
        shift_vec = np.random.uniform(-0.03 * intensity, 0.03 * intensity, size=(1, 1, 3))
        keypoints += shift_vec

    if 'rotate' in transforms:
        axis = random.choice(['x', 'y', 'z'])
        angles = np.random.uniform(-8 * intensity, 8 * intensity, size=T)
        for t in range(T):
            keypoints[t] = rotate_keypoints(keypoints[t], axis=axis, angle_deg=angles[t])

    if 'occlude' in transforms:
        num_joints_to_zero = random.randint(1, max(1, int(5 * intensity)))
        joint_indices = np.random.choice(K, size=num_joints_to_zero, replace=False)
        keypoints[:, joint_indices] = 0.0

    if 'drop' in transforms:
        if global_dropout:
            num_to_drop = int(T * prob_drop)
            drop_indices = np.random.choice(T, num_to_drop, replace=False)
            keypoints[drop_indices] = 0.0
        else:
            for t in range(T):
                if random.random() < prob_drop:
                    keypoints[t] = 0.0

    return keypoints.reshape(T, D)
