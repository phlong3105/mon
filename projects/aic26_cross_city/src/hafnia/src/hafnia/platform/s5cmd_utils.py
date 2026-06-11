import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import UnauthorizedSSOTokenError
from pydantic import BaseModel, field_validator

from hafnia.log import sys_logger, user_logger
from hafnia.utils import progress_bar


def find_s5cmd() -> Optional[str]:
    """Locate the s5cmd executable across different installation methods.

    Searches for s5cmd in:
    1. System PATH (via shutil.which)
    2. Python bin directory (Unix-like systems)
    3. Python executable directory (direct installs)

    Returns:
        str: Absolute path to s5cmd executable if found, None otherwise.
    """
    result = shutil.which("s5cmd")
    if result:
        return result
    python_dir = Path(sys.executable).parent
    locations = (
        python_dir / "Scripts" / "s5cmd.exe",
        python_dir / "bin" / "s5cmd",
        python_dir / "s5cmd",
    )
    for loc in locations:
        if loc.exists():
            return str(loc)
    return None


def execute_command(args: List[str], append_envs: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    s5cmd_bin = find_s5cmd()
    cmds = [s5cmd_bin] + args
    envs = os.environ.copy()
    if append_envs:
        envs.update(append_envs)

    result = subprocess.run(
        cmds,  # type: ignore[arg-type]
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=envs,
    )
    return result


def execute_commands(
    commands: List[str],
    append_envs: Optional[Dict[str, str]] = None,
    description: str = "Executing s5cmd commands",
) -> List[str]:
    append_envs = append_envs or {}

    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_file_path = Path(temp_dir, f"{uuid.uuid4().hex}.txt")
        tmp_file_path.write_text("\n".join(commands))

        s5cmd_bin = find_s5cmd()
        if s5cmd_bin is None:
            raise ValueError("Can not find s5cmd executable.")
        run_cmds = [s5cmd_bin, "run", str(tmp_file_path)]
        sys_logger.debug(run_cmds)
        envs = os.environ.copy()
        envs.update(append_envs)

        process = subprocess.Popen(
            run_cmds,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=envs,
        )

        error_lines = []
        lines = []
        for line in progress_bar(process.stdout, total=len(commands), description=description):  # type: ignore[arg-type]
            if "ERROR" in line or "error" in line:
                error_lines.append(line.strip())
            lines.append(line.strip())

        if len(error_lines) > 0:
            show_n_lines = min(5, len(error_lines))
            str_error_lines = "\n".join(error_lines[:show_n_lines])
            user_logger.error(
                f"Detected {len(error_lines)} errors occurred while executing a total of {len(commands)} "
                f" commands with s5cmd. The first {show_n_lines} is printed below:\n{str_error_lines}"
            )
            raise RuntimeError("Errors occurred during s5cmd execution.")
    return lines


def delete_bucket_content(
    bucket_prefix: str,
    remove_bucket: bool = True,
    append_envs: Optional[Dict[str, str]] = None,
) -> None:
    # Remove all files in the bucket
    returns = execute_command(["rm", f"{bucket_prefix}/*"], append_envs=append_envs)

    if returns.returncode != 0:
        bucket_content_is_already_deleted = "no object found" in returns.stderr.strip()
        bucket_is_already_deleted = "NoSuchBucket" in returns.stderr.strip()
        if bucket_content_is_already_deleted:
            user_logger.info(f"No action was taken. S3 bucket '{bucket_prefix}' is already empty.")
        elif bucket_is_already_deleted:
            user_logger.info(f"No action was taken. S3 bucket '{bucket_prefix}' does not exist.")
            return
        else:
            user_logger.error("Error during s5cmd rm command:")
            user_logger.error(returns.stdout)
            user_logger.error(returns.stderr)
            raise RuntimeError(f"Failed to delete all files in S3 bucket '{bucket_prefix}'.")

    if remove_bucket:
        # Remove the bucket itself
        returns = execute_command(["rb", bucket_prefix], append_envs=append_envs)
        if returns.returncode != 0:
            user_logger.error("Error during s5cmd rb command:")
            user_logger.error(returns.stdout)
            user_logger.error(returns.stderr)
            raise RuntimeError(f"Failed to delete S3 bucket '{bucket_prefix}'.")
    user_logger.info(f"S3 bucket '{bucket_prefix}' has been deleted.")


def list_bucket(bucket_prefix: str, append_envs: Optional[Dict[str, str]] = None) -> List[str]:
    output = execute_command(["ls", f"{bucket_prefix}/*"], append_envs=append_envs)
    has_missing_folder = "no object found" in output.stderr.strip()
    if output.returncode != 0 and not has_missing_folder:
        user_logger.error("Error during s5cmd ls command:")
        user_logger.error(output.stderr)
        raise RuntimeError(f"Failed to list dataset in S3 bucket '{bucket_prefix}'.")

    files_in_s3 = [f"{bucket_prefix}/{line.split(' ')[-1]}" for line in output.stdout.splitlines()]
    return files_in_s3


def fast_copy_files(
    src_paths: List[str],
    dst_paths: List[str],
    append_envs: Optional[Dict[str, str]] = None,
    description: str = "Copying files",
) -> List[str]:
    if len(src_paths) != len(dst_paths):
        raise ValueError("Source and destination paths must have the same length.")
    cmds = [f'cp "{src}" "{dst}"' for src, dst in zip(src_paths, dst_paths)]
    lines = execute_commands(cmds, append_envs=append_envs, description=description)
    return lines


ARN_PREFIX = "arn:aws:s3:::"


class AwsCredentials(BaseModel):
    access_key: str
    secret_key: str
    session_token: str
    region: Optional[str]

    def aws_credentials(self) -> Dict[str, str]:
        """
        Returns the AWS credentials as a dictionary.
        """
        environment_vars = {
            "AWS_ACCESS_KEY_ID": self.access_key,
            "AWS_SECRET_ACCESS_KEY": self.secret_key,
            "AWS_SESSION_TOKEN": self.session_token,
        }
        if self.region:
            environment_vars["AWS_REGION"] = self.region

        return environment_vars

    @staticmethod
    def from_session(session: boto3.Session) -> "AwsCredentials":
        """
        Creates AwsCredentials from a Boto3 session.
        """
        try:
            frozen_credentials = session.get_credentials().get_frozen_credentials()
        except UnauthorizedSSOTokenError as e:
            raise RuntimeError(
                f"Failed to get AWS credentials from the session for profile '{session.profile_name}'.\n"
                f"Ensure the profile exists in your AWS config in '~/.aws/config' and that you are logged in via AWS SSO.\n"
                f"\tUse 'aws sso login --profile {session.profile_name}' to log in."
            ) from e
        return AwsCredentials(
            access_key=frozen_credentials.access_key,
            secret_key=frozen_credentials.secret_key,
            session_token=frozen_credentials.token,
            region=session.region_name,
        )

    def to_resource_credentials(self, bucket_name: str) -> "ResourceCredentials":
        """
        Converts AwsCredentials to ResourceCredentials by adding the S3 ARN.
        """
        payload = self.model_dump()
        payload["s3_arn"] = f"{ARN_PREFIX}{bucket_name}"
        return ResourceCredentials(**payload)


class ResourceCredentials(AwsCredentials):
    s3_arn: str

    @staticmethod
    def fix_naming(payload: Dict[str, str]) -> "ResourceCredentials":
        """
        The endpoint returns a payload with a key called 's3_path', but it
        is actually an ARN path (starts with arn:aws:s3::). This method renames it to 's3_arn' for consistency.
        """
        if "s3_path" in payload and payload["s3_path"].startswith(ARN_PREFIX):
            payload["s3_arn"] = payload.pop("s3_path")

        if "region" not in payload:
            payload["region"] = "eu-west-1"
        return ResourceCredentials(**payload)

    @field_validator("s3_arn")
    @classmethod
    def validate_s3_arn(cls, value: str) -> str:
        """Validate s3_arn to ensure it starts with 'arn:aws:s3:::'"""
        if not value.startswith("arn:aws:s3:::"):
            raise ValueError(f"Invalid S3 ARN: {value}. It should start with 'arn:aws:s3:::'")
        return value

    def s3_path(self) -> str:
        """
        Extracts the S3 path from the ARN.
        Example: arn:aws:s3:::my-bucket/my-prefix -> my-bucket/my-prefix
        """
        return self.s3_arn[len(ARN_PREFIX) :]

    def s3_uri(self) -> str:
        """
        Converts the S3 ARN to a URI format.
        Example: arn:aws:s3:::my-bucket/my-prefix -> s3://my-bucket/my-prefix
        """
        return f"s3://{self.s3_path()}"

    def bucket_name(self) -> str:
        """
        Extracts the bucket name from the S3 ARN.
        Example: arn:aws:s3:::my-bucket/my-prefix -> my-bucket
        """
        return self.s3_path().split("/")[0]

    def object_key(self) -> str:
        """
        Extracts the object key from the S3 ARN.
        Example: arn:aws:s3:::my-bucket/my-prefix -> my-prefix
        """
        return "/".join(self.s3_path().split("/")[1:])
