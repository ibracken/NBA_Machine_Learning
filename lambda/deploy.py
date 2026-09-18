#!/usr/bin/env python3
"""
Build, push, and deploy Lambda container images.

    python lambda/deploy.py minutes-projection injury-scraper
    python lambda/deploy.py --all

Each function directory under lambda/ with a Dockerfile is deployable. Images are tagged with a UTC
timestamp (and :latest) so a bad deploy can be rolled back by pointing the function at a previous tag.
Functions must already exist; creating one is a one-time console/CLI step (role, memory, timeout, env).
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_config import AWS_ACCOUNT_ID, AWS_REGION, validate_config  # noqa: E402

LAMBDA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LAMBDA_DIR.parent
FUNCTIONS = sorted(p.name for p in LAMBDA_DIR.iterdir() if (p / "Dockerfile").exists())
REGISTRY = f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com"


def resolve_command(name, candidates=()):
    found = shutil.which(name)
    if found:
        return found
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"'{name}' not found on PATH or in {[str(c) for c in candidates]}")


def ecr_login(docker, aws):
    password = subprocess.run(
        [aws, "ecr", "get-login-password", "--region", AWS_REGION],
        capture_output=True, text=True, check=True,
    ).stdout
    subprocess.run(
        [docker, "login", "--username", "AWS", "--password-stdin", REGISTRY],
        input=password, text=True, check=True, capture_output=True,
    )


def deploy(function_name, docker):
    repo = f"{REGISTRY}/{function_name}"
    tag = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    image_uri = f"{repo}:{tag}"

    print(f"\n=== {function_name}: building and pushing {image_uri}")
    subprocess.run(
        [
            docker, "buildx", "build",
            "--platform", "linux/amd64",
            "--provenance=false", "--sbom=false",
            "--push",
            "-t", image_uri, "-t", f"{repo}:latest",
            str(LAMBDA_DIR / function_name),
        ],
        check=True,
    )

    lambda_client = boto3.client("lambda", region_name=AWS_REGION)
    try:
        lambda_client.update_function_code(FunctionName=function_name, ImageUri=image_uri)
    except lambda_client.exceptions.ResourceNotFoundException:
        sys.exit(
            f"{function_name} does not exist. Create it once (PackageType=Image, ImageUri={image_uri}, "
            f"Role=arn:aws:iam::{AWS_ACCOUNT_ID}:role/lambda-execution-role) and rerun."
        )
    lambda_client.get_waiter("function_updated_v2").wait(FunctionName=function_name)
    config = lambda_client.get_function_configuration(FunctionName=function_name)
    print(f"=== {function_name}: deployed ({config['LastUpdateStatus']}), memory={config['MemorySize']}MB timeout={config['Timeout']}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("functions", nargs="*", help=f"one or more of: {', '.join(FUNCTIONS)}")
    parser.add_argument("--all", action="store_true", help="deploy every function")
    args = parser.parse_args()

    targets = FUNCTIONS if args.all else args.functions
    if not targets:
        parser.error("name at least one function or pass --all")
    unknown = [t for t in targets if t not in FUNCTIONS]
    if unknown:
        parser.error(f"unknown function(s) {unknown}; choose from {FUNCTIONS}")

    validate_config()
    aws = resolve_command("aws", [PROJECT_ROOT / ".venv" / "Scripts" / "aws.cmd", PROJECT_ROOT / ".venv" / "Scripts" / "aws.exe"])
    docker = resolve_command("docker", [Path(r"C:\Program Files\Docker\Docker\resources\bin\docker.exe")])

    ecr_login(docker, aws)
    for function_name in targets:
        deploy(function_name, docker)


if __name__ == "__main__":
    main()
