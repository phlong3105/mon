import pytest
from click.exceptions import MissingParameter, NoArgsIsHelpError
from urllib3.exceptions import HTTPError

from hafnia import utils
from hafnia.dataset.dataset_recipe.dataset_recipe import DatasetRecipe
from hafnia.utils import is_hafnia_configured
from tests import helper_testing
from tests.helper_testing_datasets import DATASET_SPEC_MNIST


@pytest.mark.slow
def test_cli_integration_test():
    """
    Run actual CLI commands against Hafnia platform to ensure end-to-end functionality
    using simple smoke tests
    """
    from hafnia_cli.__main__ import main as hafnia_cli

    if not is_hafnia_configured():
        pytest.skip("Hafnia platform not configured. Skipping CLI integration test.")

    # Main help
    with pytest.raises(NoArgsIsHelpError):
        hafnia_cli(args=[], standalone_mode=False)
    hafnia_cli(args=["--version"], standalone_mode=False)

    # Configuration
    CMD_CONFIGURE = "configure"
    hafnia_cli(args=[CMD_CONFIGURE, "--help"], standalone_mode=False)

    # Profile
    CMD_PROFILE = "profile"
    hafnia_cli(args=[CMD_PROFILE, "--help"], standalone_mode=False)
    hafnia_cli(args=[CMD_PROFILE, "ls"], standalone_mode=False)
    hafnia_cli(args=[CMD_PROFILE, "active"], standalone_mode=False)
    with pytest.raises(MissingParameter):
        hafnia_cli(args=[CMD_PROFILE, "create"], standalone_mode=False)

    # Dataset commands
    CMD_DATASET = "dataset"
    with pytest.raises(NoArgsIsHelpError):
        hafnia_cli(args=[CMD_DATASET], standalone_mode=False)
    hafnia_cli(args=[CMD_DATASET, "--help"], standalone_mode=False)
    hafnia_cli(args=[CMD_DATASET, "ls"], standalone_mode=False)
    hafnia_cli(
        args=[CMD_DATASET, "download", "mnist", "-v", DATASET_SPEC_MNIST.version, "--force"], standalone_mode=False
    )
    hafnia_cli(args=[CMD_DATASET, "download", "mnist", "-v", DATASET_SPEC_MNIST.version], standalone_mode=False)

    # Dataset recipe commands
    CMD_DATASET_RECIPE = "recipe"
    with pytest.raises(NoArgsIsHelpError):
        hafnia_cli(args=[CMD_DATASET_RECIPE], standalone_mode=False)
    hafnia_cli(args=[CMD_DATASET_RECIPE, "--help"], standalone_mode=False)
    hafnia_cli(args=[CMD_DATASET_RECIPE, "ls"], standalone_mode=False)
    with pytest.raises(MissingParameter):
        hafnia_cli(args=[CMD_DATASET_RECIPE, "create"], standalone_mode=False)

    ## Create dataset recipe from local path
    dataset_recipe_name = "smoke_test_recipe"
    path_recipe = utils.PATH_DATASET_RECIPES / f"{dataset_recipe_name}.json"
    path_recipe_str = str(path_recipe.absolute())
    dataset_recipe = DatasetRecipe.from_name("mnist", version=DATASET_SPEC_MNIST.version).shuffle().select_samples(10)
    dataset_recipe.as_json_file(path_recipe)
    assert path_recipe.exists()

    hafnia_cli(args=[CMD_DATASET_RECIPE, "create", path_recipe_str, "-n", dataset_recipe_name], standalone_mode=False)
    hafnia_cli(args=[CMD_DATASET_RECIPE, "ls"], standalone_mode=False)
    hafnia_cli(args=[CMD_DATASET_RECIPE, "rm", "--help"], standalone_mode=False)
    with pytest.raises(MissingParameter):
        hafnia_cli(args=[CMD_DATASET_RECIPE, "rm"], standalone_mode=False)
    hafnia_cli(args=[CMD_DATASET_RECIPE, "rm", "-n", dataset_recipe_name], standalone_mode=False)

    # Trainer package commands
    CMD_TRAINER_PACKAGE = "trainer"
    with pytest.raises(NoArgsIsHelpError):
        hafnia_cli(args=[CMD_TRAINER_PACKAGE], standalone_mode=False)
    hafnia_cli(args=[CMD_TRAINER_PACKAGE, "--help"], standalone_mode=False)
    hafnia_cli(args=[CMD_TRAINER_PACKAGE, "ls"], standalone_mode=False)

    path_trainer = (helper_testing.get_path_workspace() / ".." / "trainer-classification").absolute()
    if path_trainer.exists():
        hafnia_cli(args=[CMD_TRAINER_PACKAGE, "create-zip", str(path_trainer)], standalone_mode=False)
        hafnia_cli(args=[CMD_TRAINER_PACKAGE, "view-zip", "--path", "trainer.zip"], standalone_mode=False)

    # Trainer update: surface checks (run regardless of whether a real trainer is available)
    hafnia_cli(args=[CMD_TRAINER_PACKAGE, "update", "--help"], standalone_mode=False)
    with pytest.raises(MissingParameter):
        hafnia_cli(args=[CMD_TRAINER_PACKAGE, "update"], standalone_mode=False)
    with pytest.raises(HTTPError, match="Not found"):
        hafnia_cli(args=[CMD_TRAINER_PACKAGE, "update", "non-existent-id", "--name", "Blah"], standalone_mode=False)

    # Experiment commands
    CMD_EXPERIMENT = "experiment"
    hafnia_cli(args=[CMD_EXPERIMENT, "ls"], standalone_mode=False)
    hafnia_cli(args=[CMD_EXPERIMENT, "environments"], standalone_mode=False)
    with pytest.raises(NoArgsIsHelpError):
        hafnia_cli(args=[CMD_EXPERIMENT], standalone_mode=False)
    with pytest.raises(MissingParameter):
        hafnia_cli(args=[CMD_EXPERIMENT, "create", "--dataset", "mnist"], standalone_mode=False)

    if path_trainer.exists():
        trainer_response = hafnia_cli(
            args=[
                CMD_TRAINER_PACKAGE,
                "create",
                str(path_trainer),
                "--name",
                f"IntegrationTestTrainer{utils.now_as_str()}",
                "--description",
                "Integration test trainer package created by CLI integration test.",
            ],
            standalone_mode=False,
        )
        assert trainer_response is not None
        assert isinstance(trainer_response, dict)
        assert "id" in trainer_response

        update_response = hafnia_cli(
            args=[
                CMD_TRAINER_PACKAGE,
                "update",
                trainer_response["id"],
                "--name",
                f"{trainer_response['name']} (updated)",
                "--description",
                f"Updated by integration test at {utils.now_as_str()}.",
                "--path",
                str(path_trainer),
            ],
            standalone_mode=False,
        )
        assert isinstance(update_response, dict)
        assert update_response.get("id") == trainer_response["id"]

        hafnia_cli(
            args=[
                CMD_EXPERIMENT,
                "create",
                "--dataset",
                "mnist",
                "--trainer-id",
                trainer_response["id"],
                "--name",
                f"integration_test_experiment_{utils.now_as_str()}",
            ],
            standalone_mode=False,
        )
