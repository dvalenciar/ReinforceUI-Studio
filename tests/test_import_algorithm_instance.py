# tests/test_import_algorithm_instance.py
from RL_loops.training_policy_loop import import_algorithm_instance


def test_import_algorithm_instance():
    config_data = {"Algorithm": "TD3"}
    algorithm_class = import_algorithm_instance(config_data)
    assert algorithm_class.__name__ == "TD3"
