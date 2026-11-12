import numpy as np
from ..search_spaces import SearchSpace


def sample_from_ball(center, search_space: SearchSpace, radius, n_samples):
    """
    Sample random points uniformly from a ball around a given center.
    
    Parameters:
        center (array-like): Coordinates of the center of the ball.
        search_space (SearchSpace): Search space of the current HPO task.
        radius (list or float): Radius of the ball per dimension.
        n_samples (int): Number of points to sample.

    Returns:
        np.ndarray: Array of shape (n_samples, dimension) with sampled points.
    """
    if isinstance(radius, list):
        radius = np.array(radius)
    points = []

    search_space_def = search_space.get_search_space_definition()

    # only apply ball sampling to real dimensions
    real_dims = [i for i, v in enumerate(search_space_def.values()) if v['dtype'] == 'float']
    other_dims = [(name, i) for i, (name, v) in enumerate(search_space_def.items()) if v['dtype'] != 'float']

    for i in range(n_samples):
        if len(real_dims) > 0:
            # Generate a random direction (unit vector) in `dimension` space
            direction = np.random.normal(size=len(real_dims))
            direction /= np.linalg.norm(direction)
            
            # Generate a random distance from the center, scaled to fit inside the radius
            distance = np.random.uniform(0, 1) ** (1 / len(real_dims)) * radius[real_dims]
            
            # Compute the point
            center[i, real_dims] = center[i, real_dims] + direction * distance
        for name, j in other_dims:
            coin = np.random.choice([0, 1], p=[1-radius[j], radius[j]])
            if coin == 1:
                choices = np.arange(0, len(search_space_def[name]['allowed']))
                rnd_cfg = np.random.choice(choices)
                center[i, j] = rnd_cfg
        points.append(center[i])
    return np.array(points)
