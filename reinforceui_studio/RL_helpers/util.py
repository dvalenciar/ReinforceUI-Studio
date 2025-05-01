import torch
import numpy as np
import random


def normalize_action(
    action: float, max_action_value: float, min_action_value: float
) -> float:
    """Normalize the given action value within the specified range.

    Args:
        action (float): The action value to be normalized.
        max_action_value (float): The maximum value of the action range.
        min_action_value (float): The minimum value of the action range.

    Returns:
        float: The normalized action value.

    """
    max_range_value: float = 1
    min_range_value: float = -1
    max_value_in: float = max_action_value
    min_value_in: float = min_action_value
    action_normalize: float = (action - min_value_in) * (
        max_range_value - min_range_value
    ) / (max_value_in - min_value_in) + min_range_value
    return action_normalize


def denormalize_action(
    action: float, max_action_value: float, min_action_value: float
) -> float:
    """Denormalize the given action value within the specified range.

    Args:
        action (float): The action value to be denormalized.
        max_action_value (float): The maximum value of the action range.
        min_action_value (float): The minimum value of the action range.

    Returns:
        float: The denormalized action value.

    """
    max_range_value: float = max_action_value
    min_range_value: float = min_action_value
    max_value_in: float = 1
    min_value_in: float = -1
    action_denormalize: float = (action - min_value_in) * (
        max_range_value - min_range_value
    ) / (max_value_in - min_value_in) + min_range_value
    return action_denormalize


def set_seed(seed: int) -> None:
    """Set seen for reproducibility

    Args:
        seed (int): seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
