#!/usr/bin/env python
import click

import hafnia
from hafnia_cli import (
    consts,
    dataset_cmds,
    dataset_recipe_cmds,
    experiment_cmds,
    profile_cmds,
    runc_cmds,
    trainer_package_cmds,
)
from hafnia_cli.config import Config, ConfigSchema


@click.group()
@click.version_option(version=hafnia.__version__)
@click.pass_context
def main(ctx: click.Context) -> None:
    """Hafnia CLI."""
    ctx.obj = Config()
    ctx.max_content_width = 120


@main.command("configure")
@click.pass_obj
def configure(cfg: Config) -> None:
    """Configure Hafnia CLI settings."""

    profile_name = click.prompt("Alias", type=str, default=consts.DEFAULT_PROFILE_NAME)
    profile_name = profile_name.strip()

    cfg.check_profile_name(profile_name)

    api_key = click.prompt("Hafnia API Key", type=str, hide_input=True)

    platform_url = click.prompt("Hafnia Platform URL", type=str, default=consts.DEFAULT_API_URL)

    use_keychain = click.confirm("Store API key in system keychain?", default=False)

    cfg_profile = ConfigSchema(platform_url=platform_url, api_key=api_key, use_keychain=use_keychain)
    cfg.add_profile(profile_name, cfg_profile, set_active=True)
    profile_cmds.profile_show(cfg)


@main.command("clear")
@click.pass_obj
def clear(cfg: Config) -> None:
    """Remove stored configuration."""
    cfg.clear()
    click.echo("Successfully cleared Hafnia configuration.")


main.add_command(profile_cmds.profile)
main.add_command(dataset_cmds.dataset)
main.add_command(runc_cmds.runc)
main.add_command(experiment_cmds.experiment)
main.add_command(trainer_package_cmds.trainer_package)
main.add_command(dataset_recipe_cmds.dataset_recipe)

if __name__ == "__main__":
    main(max_content_width=120)
