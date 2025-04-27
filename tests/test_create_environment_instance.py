import os
import yaml
from RL_loops.training_policy_loop import create_environment_instance


def test_create_environment_instance() -> None:
    """Test the creation of environment instances.

    This function tests the creation of environment instances for various
    platforms and environments specified in the configuration file.

    Raises:
        AssertionError: If the environment instance is not created successfully
                        or does not have the required methods.
    """
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "config_platform.yaml"
    )

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    for platform, envs in config["platforms"].items():
        all_envs = envs.get("environments", []) + envs.get("discrete_environments", [])
        for env in all_envs:
            config_data = {
                "selected_platform": platform,
                "selected_environment": env,
                "Seed": 42,
            }
            environment = create_environment_instance(config_data)

            assert (
                environment is not None
            ), f"Failed to create environment for {platform} - {env}"

            assert hasattr(
                environment, "reset"
            ), "Environment instance should have a reset method"
            assert hasattr(
                environment, "step"
            ), "Environment instance should have a step method"
            assert hasattr(
                environment, "sample_action"
            ), "Environment instance should have a sample_action method"
