"""Bound scene layers spatially so first useful data need not be a whole scene."""

import numpy as np


def spatial_groups(means, maximum):
    """Median splits cover every original index once, without changing geometry."""
    if maximum <= 0 or means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("positive group size and N-by-3 means required")
    if not np.isfinite(means).all():
        raise ValueError("non-finite scene positions")
    pending = [np.arange(len(means))]
    groups = []
    while pending:
        indices = pending.pop()
        if len(indices) <= maximum:
            if len(indices):
                groups.append(indices)
            continue
        points = means[indices]
        axis = int(np.argmax(np.ptp(points, axis=0)))
        midpoint = len(indices) // 2
        order = np.argpartition(points[:, axis], midpoint)
        pending.extend((indices[order[midpoint:]], indices[order[:midpoint]]))
    return groups
