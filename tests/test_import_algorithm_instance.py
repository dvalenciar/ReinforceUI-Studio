from RL_loops.training_policy_loop import import_algorithm_instance


def test_import_algorithm_instance() -> None:
    """Test the import of algorithm instances.

    This function tests the import of various algorithm instances
    to ensure that the correct class and name are returned.

    Raises:
        AssertionError: If the imported algorithm class or name does not match the expected values.
    """
    algorithms = ["CTD4", "DDPG", "DQN", "PPO", "SAC", "TD3", "TQC"]
    for algorithm_name in algorithms:
        algorithm_class = import_algorithm_instance(algorithm_name)
        assert algorithm_class.__name__ == algorithm_name
