DEFAULT_API_URL = "https://api.hafnia.milestonesys.com"
DEFAULT_PROFILE_NAME = "default"

ERROR_CONFIGURE: str = "Please configure the CLI with `hafnia configure`"
ERROR_PROFILE_NOT_EXIST: str = "No active profile configured. Please configure the CLI with `hafnia configure`"
ERROR_PROFILE_REMOVE_ACTIVE: str = "Cannot remove active profile. Please switch to another profile first."
ERROR_API_KEY_NOT_SET: str = "API key not set. Please configure the CLI with `hafnia configure`."
ERROR_CREATE_PROFILE: str = "Failed to create profile. Profile name must be unique and not empty."

ERROR_GET_RESOURCE: str = "Failed to get the data from platform. Verify url or api key."

ERROR_EXPERIMENT_DIR: str = "Source directory does not exist"
ERROR_TRAINER_PACKAGE_FILE_FORMAT: str = "Trainer package must be a '.zip' file"

PROFILE_SWITCHED_SUCCESS: str = "Switched to profile:"
PROFILE_REMOVED_SUCCESS: str = "Removed profile:"
PROFILE_TABLE_HEADER: str = "Hafnia Platform Profile:"
