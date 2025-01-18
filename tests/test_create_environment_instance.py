import pytest
from RL_algorithms.training_policy_loop import create_environment_instance


def test_create_environment_instance():
    config_data = {
        "selected_platform": "Gymnasium",
        "selected_environment": "Pendulum-v1",
        "Seed": 42,
    }
    environment = create_environment_instance(config_data)
    assert environment.env.spec.id == "Pendulum-v1"
