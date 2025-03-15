from RL_loops.training_policy_loop import import_algorithm_instance


def test_import_algorithm_instance() -> None:
    """Test the import of algorithm instances.

    This function tests the import of various algorithm instances
    to ensure that the correct class and name are returned.

    Raises:
        AssertionError: If the imported algorithm class or name does not match the expected values.
    """
    algorithms = ["CTD4", "DDPG", "DQN", "PPO", "SAC", "TD3", "TQC"]
    for algorithm in algorithms:
        config_data = {"Algorithm": algorithm}
        algorithm_class, algorithm_name = import_algorithm_instance(
            config_data
        )
        assert algorithm_class.__name__ == algorithm
        assert algorithm_name == algorithm
