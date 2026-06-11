"""Keychain storage for API keys using the system keychain."""

from typing import Optional

from hafnia.log import sys_logger

# Keyring is optional - gracefully degrade if not available
try:
    import keyring

    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False
    sys_logger.debug("keyring library not available, keychain storage disabled")

KEYRING_SERVICE_NAME = "hafnia-cli"


def store_api_key(profile_name: str, api_key: str) -> bool:
    """
    Store an API key in the system keychain.

    Args:
        profile_name: The profile name to associate with the key
        api_key: The API key to store

    Returns:
        True if successfully stored, False otherwise
    """
    if not KEYRING_AVAILABLE:
        sys_logger.warning("Keyring library not available, cannot store API key in keychain")
        return False

    try:
        keyring.set_password(KEYRING_SERVICE_NAME, profile_name, api_key[:])
        sys_logger.debug(f"Stored API key for profile '{profile_name}' in keychain")
        return True
    except Exception as e:
        sys_logger.warning(f"Failed to store API key in keychain: {e}")
        return False


def get_api_key(profile_name: str) -> Optional[str]:
    """
    Retrieve an API key from the system keychain.

    Args:
        profile_name: The profile name to retrieve the key for

    Returns:
        The API key if found, None otherwise
    """
    if not KEYRING_AVAILABLE:
        return None

    try:
        api_key = keyring.get_password(KEYRING_SERVICE_NAME, profile_name)
        if api_key:
            sys_logger.debug(f"Retrieved API key for profile '{profile_name}' from keychain")
        return api_key
    except Exception as e:
        sys_logger.warning(f"Failed to retrieve API key from keychain: {e}")
        return None


def delete_api_key(profile_name: str) -> bool:
    """
    Delete an API key from the system keychain.

    Args:
        profile_name: The profile name to delete the key for

    Returns:
        True if successfully deleted or didn't exist, False on error
    """
    if not KEYRING_AVAILABLE:
        return False

    try:
        keyring.delete_password(KEYRING_SERVICE_NAME, profile_name)
        sys_logger.debug(f"Deleted API key for profile '{profile_name}' from keychain")
        return True
    except keyring.errors.PasswordDeleteError:
        # Key didn't exist, which is fine
        return True
    except Exception as e:
        sys_logger.warning(f"Failed to delete API key from keychain: {e}")
        return False
