import json
import os
import re
import subprocess
import zipfile
from hashlib import sha256
from pathlib import Path
from typing import Dict, Optional

import boto3
from botocore.exceptions import ClientError

from hafnia.log import sys_logger, user_logger
from hafnia.platform import download_resource


def validate_trainer_package_format(path: Path) -> None:
    """Validate Hafnia Trainer Package Format submission"""
    hrf = zipfile.Path(path) if path.suffix == ".zip" else path
    required = {"src", "scripts", "Dockerfile"}
    errors = 0
    for rp in required:
        if not (hrf / rp).exists():
            user_logger.error(f"Required path {rp} not found in trainer package.")
            errors += 1
    if errors > 0:
        raise FileNotFoundError("Wrong trainer package structure")


def prepare_trainer_package(
    trainer_url: str, output_dir: Path, api_key: str, state_file: Optional[Path] = None
) -> Dict:
    resource = download_resource(trainer_url, output_dir.as_posix(), api_key)
    trainer_path = Path(resource["downloaded_files"][0])
    with zipfile.ZipFile(trainer_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    validate_trainer_package_format(output_dir)

    scripts_dir = output_dir / "scripts"
    if not any(scripts_dir.iterdir()):
        user_logger.warning("Scripts folder is empty")

    metadata = {
        "user_data": (output_dir / "src").as_posix(),
        "dockerfile": (output_dir / "Dockerfile").as_posix(),
        "digest": sha256(trainer_path.read_bytes()).hexdigest()[:8],
    }
    state_file = state_file if state_file else output_dir / "state.json"
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    return metadata


def buildx_available() -> bool:
    try:
        result = subprocess.run(["docker", "buildx", "version"], capture_output=True, text=True, check=True)
        return "buildx" in result.stdout.lower()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def build_dockerfile(dockerfile: str, docker_context: str, docker_tag: str) -> None:
    """
    Build a Docker image using the provided Dockerfile.

    Args:
        dockerfile (str): Path to the Dockerfile.
        docker_context (str): Path to the build context.
        docker_tag (str): Tag for the Docker image.
        meta_file (Optional[str]): File to store build metadata.
    """
    if not Path(dockerfile).exists():
        raise FileNotFoundError("Dockerfile not found.")

    cmd = ["docker", "build", "--platform", "linux/amd64", "-t", docker_tag, "-f", dockerfile]

    remote_cache = os.getenv("EXPERIMENT_CACHE_ECR")
    cloud_mode = os.getenv("HAFNIA_CLOUD", "false").lower() in ["true", "1", "yes"]

    if buildx_available():
        cmd.insert(1, "buildx")
        cmd += ["--build-arg", "BUILDKIT_INLINE_CACHE=1"]
        if cloud_mode:
            cmd += ["--push"]
        if remote_cache:
            cmd += [
                "--cache-from",
                f"type=registry,ref={remote_cache}:buildcache",
                "--cache-to",
                f"type=registry,ref={remote_cache}:buildcache,mode=max",
            ]
    cmd.append(docker_context)
    sys_logger.debug("Build cmd: `{}`".format(" ".join(cmd)))
    sys_logger.info(f"Building and pushing Docker image with BuildKit (buildx); cache repo: {remote_cache or 'none'}")
    result = None
    output = ""
    errors = []
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        output = (result.stdout or "") + (result.stderr or "")
    except subprocess.CalledProcessError as e:
        output = (e.stdout or "") + (e.stderr or "")
        error_pattern = r"ERROR: (.+?)(?:\n|$)"
        errors = re.findall(error_pattern, output)
        if not errors:
            raise RuntimeError(f"Docker build failed: {output}")
        if re.search(r"image tag '([^']+)' already exists", errors[-1]):
            sys_logger.warning("Image {} already exists in the registry.".format(docker_tag.rsplit("/")[-1]))
            return
        raise RuntimeError(f"Docker build failed: {output}")
    finally:
        stage_pattern = r"^.*\[\d+/\d+\][^\n]*"
        stages = re.findall(stage_pattern, output, re.MULTILINE)
        user_logger.info("\n".join(stages))
        sys_logger.debug(output)


def check_registry(docker_image: str) -> Optional[str]:
    """
    Returns the remote digest for TAG if it exists, otherwise None.
    """
    if "localhost" in docker_image:
        return None

    region = os.getenv("AWS_REGION")
    if not region:
        sys_logger.warning("AWS_REGION environment variable not set. Skip image exist check.")
        return None

    repo_name, image_tag = docker_image.rsplit(":")
    if "/" in repo_name:
        repo_name = repo_name.rsplit("/", 1)[-1]
    ecr = boto3.client("ecr", region_name=region)
    try:
        out = ecr.describe_images(repositoryName=repo_name, imageIds=[{"imageTag": image_tag}])
        return out["imageDetails"][0]["imageDigest"]
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        sys_logger.error(f"ECR client error: {error_code}")
        return None


def build_image(metadata: Dict, registry_repo: str, state_file: str = "state.json") -> None:
    docker_image = f"{registry_repo}:{metadata['digest']}"
    image_exists = check_registry(docker_image) is not None
    if image_exists:
        sys_logger.info("Image {} already exists in the registry.".format(docker_image.rsplit("/")[-1]))
    else:
        build_dockerfile(metadata["dockerfile"], Path(metadata["dockerfile"]).parent.as_posix(), docker_image)
    metadata.update({"image_tag": docker_image, "image_exists": image_exists})
    Path(state_file).write_text(json.dumps(metadata, indent=2))
