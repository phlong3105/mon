import click
from rich.console import Console
from rich.table import Table

import hafnia_cli.consts as consts
from hafnia_cli.config import Config, ConfigSchema


@click.group(invoke_without_command=True)
@click.pass_context
def profile(ctx):
    """Manage profile."""
    if ctx.invoked_subcommand is None:
        # No subcommand provided, show active profile and help
        cfg = ctx.obj
        profile_show(cfg)
        click.echo("\n" + ctx.get_help())


@profile.command("ls")
@click.pass_obj
def cmd_profile_ls(cfg: Config) -> None:
    """List all available profiles."""
    profiles = cfg.available_profiles
    if not profiles:
        raise click.ClickException(consts.ERROR_CONFIGURE)
    active = cfg.active_profile

    for profile in profiles:
        status = "* " if profile == active else "  "
        print(f"{status}{profile}")

    print(f"\nActive profile: {active}")


@profile.command("use")
@click.argument("profile_name", required=True)
@click.pass_obj
def cmd_profile_use(cfg: Config, profile_name: str) -> None:
    """Switch to a different profile."""
    if len(cfg.available_profiles) == 0:
        raise click.ClickException(consts.ERROR_CONFIGURE)
    try:
        cfg.active_profile = profile_name
        cfg.save_config()
    except ValueError:
        raise click.ClickException(consts.ERROR_PROFILE_NOT_EXIST)
    click.echo(f"{consts.PROFILE_SWITCHED_SUCCESS} {profile_name}")


@profile.command("create")
@click.argument("api-key", required=True)
@click.option("--name", help="Specify profile name", default=consts.DEFAULT_PROFILE_NAME, show_default=True)
@click.option("--api-url", help="API URL", default=consts.DEFAULT_API_URL, show_default=True)
@click.option(
    "--activate/--no-activate", help="Activate the created profile after creation", default=True, show_default=True
)
@click.option(
    "--use-keychain", is_flag=True, help="Store API key in system keychain instead of config file", default=False
)
@click.pass_obj
def cmd_profile_create(cfg: Config, name: str, api_url: str, api_key: str, activate: bool, use_keychain: bool) -> None:
    """Create a new profile."""
    cfg_profile = ConfigSchema(platform_url=api_url, api_key=api_key, use_keychain=use_keychain)

    cfg.add_profile(profile_name=name, profile=cfg_profile, set_active=activate)
    profile_show(cfg)


@profile.command("rm")
@click.argument("profile_name", required=True)
@click.pass_obj
def cmd_profile_rm(cfg: Config, profile_name: str) -> None:
    """Remove a profile."""
    if len(cfg.available_profiles) == 0:
        raise click.ClickException(consts.ERROR_CONFIGURE)

    if profile_name == cfg.active_profile:
        raise click.ClickException(consts.ERROR_PROFILE_REMOVE_ACTIVE)

    try:
        cfg.remove_profile(profile_name)
        cfg.save_config()
    except ValueError:
        raise click.ClickException(consts.ERROR_PROFILE_NOT_EXIST)
    click.echo(f"{consts.PROFILE_REMOVED_SUCCESS} {profile_name}")


@profile.command("active")
@click.pass_obj
def cmd_profile_active(cfg: Config) -> None:
    """Show the currently active profile."""
    try:
        profile_show(cfg)
    except Exception as e:
        raise click.ClickException(str(e))


def profile_show(cfg: Config) -> None:
    masked_key = f"{cfg.api_key[:11]}...{cfg.api_key[-4:]}" if len(cfg.api_key) > 20 else "****"
    console = Console()
    table = Table(title=f"{consts.PROFILE_TABLE_HEADER} {cfg.active_profile}", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("API Key", masked_key)
    table.add_row("Platform URL", cfg.platform_url)
    table.add_row("Config File", cfg.config_path.as_posix())
    console.print(table)
