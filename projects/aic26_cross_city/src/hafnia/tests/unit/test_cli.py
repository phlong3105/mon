import subprocess
from pathlib import Path
from typing import Dict

import pytest
from click.testing import CliRunner

import hafnia_cli.__main__ as cli
import hafnia_cli.consts as consts
import hafnia_cli.runc_cmds as runc_cmds
from hafnia_cli.config import Config, ConfigFileSchema, ConfigSchema


@pytest.fixture
def cli_runner(tmp_path: Path) -> CliRunner:
    env = {"MDI_CONFIG_PATH": str(tmp_path / "config.json")}
    return CliRunner(env=env)


@pytest.fixture
def api_key() -> str:
    return "test-api-key-12345678"


@pytest.fixture
def test_config_path(tmp_path: Path) -> Path:
    """Return a temporary config file path for testing."""
    return tmp_path / "config.json"


@pytest.fixture()
def profile_data(api_key: str) -> Dict:
    """Base profile data that can be reused across different profiles."""
    return {"platform_url": consts.DEFAULT_API_URL, "api_key": api_key}


@pytest.fixture
def empty_config(test_config_path: Path) -> Config:
    return Config(config_path=test_config_path)


@pytest.fixture(scope="function")
def config_with_profiles(test_config_path: Path, profile_data: dict) -> Config:
    config = Config(config_path=test_config_path)
    config.add_profile("default", ConfigSchema(**profile_data), set_active=True)
    config.add_profile("staging", ConfigSchema(**profile_data))
    config.add_profile("production", ConfigSchema(**profile_data))
    return config


def test_configure(cli_runner: CliRunner, empty_config: Config, api_key: str) -> None:
    inputs = f"default\nApiKey some-fake-test-api-key\n{consts.DEFAULT_API_URL}\nN\n"
    result = cli_runner.invoke(cli.main, ["configure"], input="".join(inputs))
    assert result.exit_code == 0
    assert f"{consts.PROFILE_TABLE_HEADER} default" in result.output
    assert "ApiKey some" in result.output


def test_configure_api_key_autofix(cli_runner: CliRunner, empty_config: Config, api_key: str) -> None:
    """
    The submitted api key should always contain an "ApiKey " prefix.
    Namely the submitted api key should be in this form "ApiKey [HASH_VALUE]"
    Many users submit the api key without the prefix.
    This test ensures that the CLI will automatically add the prefix if missing.
    """
    inputs = f"default\nfake-api-key-with-out-prefix\n{consts.DEFAULT_API_URL}\nN\n"
    result = cli_runner.invoke(cli.main, ["configure"], input="".join(inputs))
    assert result.exit_code == 0
    assert f"{consts.PROFILE_TABLE_HEADER} default" in result.output
    assert "ApiKey fake" in result.output, (
        "'ApiKey ' was not added automatically. API key should be automatically prefixed with 'ApiKey ' when missing"
    )


def test_create_profile(cli_runner: CliRunner, empty_config: Config, api_key: str) -> None:
    fake_api_key = "SomeFakeApiKey123"
    args = [
        "profile",
        "create",
        fake_api_key,
        "--name",
        "test_profile",
        "--api-url",
        consts.DEFAULT_API_URL,
        "--activate",
    ]

    result = cli_runner.invoke(cli.main, args)
    assert result.exit_code == 0
    assert f"ApiKey {fake_api_key[:3]}" in result.output, (
        "'ApiKey ' was not added automatically. API key should be automatically prefixed with 'ApiKey ' when missing"
    )


class TestProfile:
    def test_list_profiles(
        self,
        cli_runner: CliRunner,
        empty_config: Config,
        config_with_profiles: Config,
    ) -> None:
        """Test list of profiles functionality."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("hafnia_cli.__main__.Config", lambda *args, **kwargs: empty_config)
            result = cli_runner.invoke(cli.main, ["profile", "ls"])
            assert result.exit_code != 0
            assert consts.ERROR_CONFIGURE in result.output

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("hafnia_cli.__main__.Config", lambda *args, **kwargs: config_with_profiles)
            result = cli_runner.invoke(cli.main, ["profile", "ls"])
            assert result.exit_code == 0
            assert "default" in result.output
            assert "staging" in result.output
            assert "production" in result.output
            assert "Active profile: default" in result.output

    def test_switch_profile(self, cli_runner: CliRunner, empty_config: Config, config_with_profiles: Config) -> None:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("hafnia_cli.__main__.Config", lambda *args, **kwargs: empty_config)
            result = cli_runner.invoke(cli.main, ["profile", "use", "default"])
            assert result.exit_code != 0
            assert f"Error: {consts.ERROR_CONFIGURE}" in result.output

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("hafnia_cli.__main__.Config", lambda *args, **kwargs: config_with_profiles)
            result = cli_runner.invoke(cli.main, ["profile", "active"])
            assert result.exit_code == 0
            assert f"{consts.PROFILE_TABLE_HEADER} default" in result.output

            result = cli_runner.invoke(cli.main, ["profile", "use", "staging"])
            assert result.exit_code == 0
            assert f"{consts.PROFILE_SWITCHED_SUCCESS} staging" in result.output

            result = cli_runner.invoke(cli.main, ["profile", "active"])
            assert result.exit_code == 0
            assert f"{consts.PROFILE_TABLE_HEADER} staging" in result.output

            result = cli_runner.invoke(cli.main, ["profile", "use", "nonexistent"])
            assert result.exit_code != 0
            assert consts.ERROR_PROFILE_NOT_EXIST in result.output

    def test_remove_profile(self, cli_runner: CliRunner, empty_config: Config, config_with_profiles: Config) -> None:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("hafnia_cli.__main__.Config", lambda *args, **kwargs: empty_config)
            result = cli_runner.invoke(cli.main, ["profile", "rm", "default"])
            assert result.exit_code != 0
            assert consts.ERROR_CONFIGURE in result.output

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("hafnia_cli.__main__.Config", lambda *args, **kwargs: config_with_profiles)
            result = cli_runner.invoke(cli.main, ["profile", "rm", "staging"])
            assert result.exit_code == 0
            assert f"{consts.PROFILE_REMOVED_SUCCESS} staging" in result.output

            result = cli_runner.invoke(cli.main, ["profile", "ls"])
            assert result.exit_code == 0
            assert "staging" not in result.output
            assert "production" in result.output
            assert "default" in result.output

            result = cli_runner.invoke(cli.main, ["profile", "rm", "nonexistent"])
            assert result.exit_code != 0
            assert consts.ERROR_PROFILE_NOT_EXIST in result.output

            result = cli_runner.invoke(cli.main, ["profile", "rm", "default"])
            assert result.exit_code != 0
            assert consts.ERROR_PROFILE_REMOVE_ACTIVE in result.output


def test_setting_environment_variables(tmp_path: Path):
    # 1) Environment variables when no config file is present
    path_fake_config_file = tmp_path / "env_settings" / "config.json"

    env = {
        "MDI_CONFIG_PATH": str(path_fake_config_file),
        "HAFNIA_API_KEY": "ApiKey fake",
        "HAFNIA_PLATFORM_URL": "https://fake.hafnia.ai",
    }

    another_fake_profile_schema = ConfigFileSchema(
        active_profile="another_fake_profile",
        profiles={
            "another_fake_profile": ConfigSchema(
                platform_url="https://another_fake.hafnia.ai", api_key="ApiKey another_fake_api_key"
            )
        },
    )

    # Test 1: Verify environment variables are used correctly when no config file is present
    with pytest.MonkeyPatch.context() as mp:
        for key, value in env.items():
            mp.setenv(key, value)
        config = Config()

        assert config.api_key == env["HAFNIA_API_KEY"]
        assert config.platform_url == env["HAFNIA_PLATFORM_URL"]

    # Create a fake config file with another profile
    path_fake_config_file.write_text(another_fake_profile_schema.model_dump_json())

    # Test 2: Verify that the "MDI_CONFIG_PATH" environment variable is used to load the config file
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MDI_CONFIG_PATH", str(path_fake_config_file))
        config = Config()
        # assert config.active_profile == another_fake_profile_schema.active_profile
        assert config.api_key == another_fake_profile_schema.profiles[config.active_profile].api_key
        assert config.platform_url == another_fake_profile_schema.profiles[config.active_profile].platform_url

    # Test 3: Verify environment variables override config file when both are present
    with pytest.MonkeyPatch.context() as mp:
        for key, value in env.items():
            mp.setenv(key, value)
        config = Config()
        assert config.api_key == env["HAFNIA_API_KEY"]
        assert config.platform_url == env["HAFNIA_PLATFORM_URL"]


class TestRuncLaunchLocal:
    def test_local_dataset_detected_by_path_existence(
        self, cli_runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A local filesystem path must be detected as local even if it contains no '/'."""
        path_dataset = tmp_path / "my_dataset"
        path_dataset.mkdir()

        monkeypatch.setattr(runc_cmds.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0))

        result = cli_runner.invoke(
            cli.main,
            [
                "runc",
                "launch-local",
                "--dataset",
                str(path_dataset),
                "--image_name",
                "test-image:latest",
                "python train.py",
            ],
        )
        assert "Using local dataset" in result.output, (
            f"Local path not detected as local dataset. Output: {result.output}"
        )
        assert "Using Hafnia dataset" not in result.output

    def test_docker_volume_mount_uses_forward_slashes(
        self, cli_runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Docker -v argument must use forward slashes so it is valid on Linux."""
        path_dataset = tmp_path / "my_dataset"
        path_dataset.mkdir()

        captured: list = []

        def mock_run(args, **_):
            captured.append(args)
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr(runc_cmds.subprocess, "run", mock_run)

        cli_runner.invoke(
            cli.main,
            [
                "runc",
                "launch-local",
                "--dataset",
                str(path_dataset),
                "--image_name",
                "test-image:latest",
                "python train.py",
            ],
        )

        assert captured, "subprocess.run was never called"
        docker_args = captured[0]
        v_idx = docker_args.index("-v")
        volume_mount = docker_args[v_idx + 1]
        assert chr(92) not in volume_mount, f"Backslash in Docker volume mount: {volume_mount!r}"
        assert ":/opt/ml/input/data/training" in volume_mount
