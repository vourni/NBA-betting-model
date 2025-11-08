#!/usr/bin/env python3
"""
Build and deploy the Cloud Run service with one Python command.

Usage:
  python deploy.py
  # or override defaults:
  python deploy.py --project-id nba-api-477520 --region us-central1 --service nba-api --api-token bigjgondoittoem
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime

def run(cmd: list[str], env: dict | None = None) -> None:
    pretty = " ".join(cmd)
    print(f"\n\033[1;34m$ {pretty}\033[0m")
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"\n\033[1;31mCommand failed with exit code {e.returncode}\033[0m")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Build & deploy to Cloud Run")
    parser.add_argument("--project-id", default="nba-api-477520")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--service", default="nba-api")
    parser.add_argument("--api-token", default="bigjgondoittoem")  # from your message
    parser.add_argument("--image-tag", default="latest",
                        help="Image tag to use (default: latest). You can set to a timestamp/commit SHA if you like.")
    args = parser.parse_args()

    # Validate gcloud is installed
    if shutil.which("gcloud") is None:
        print("\n\033[1;31mError: 'gcloud' CLI not found. Install Google Cloud SDK and ensure it's on PATH.\033[0m")
        sys.exit(127)

    # Compose the image path used by both build and deploy
    image = f"{args.region}-docker.pkg.dev/{args.project_id}/cloud-run-source-deploy/{args.service}:{args.image_tag}"

    # Helpful header
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n\033[1;32m=== Cloud Run Deploy ({now}) ===\033[0m")
    print(f" Project:      {args.project_id}")
    print(f" Region:       {args.region}")
    print(f" Service:      {args.service}")
    print(f" Image:        {image}")
    print(f" API_TOKEN:    (hidden)")

    # Inherit current env and set PROJECT_ID/REGION for any scripts that might read them
    env = os.environ.copy()
    env["PROJECT_ID"] = args.project_id
    env["REGION"] = args.region

    # 1) Build & push container to Artifact Registry
    run([
        "gcloud", "builds", "submit", ".",
        "--tag", image
    ], env=env)

    # 2) Deploy to Cloud Run
    run([
        "gcloud", "run", "deploy", args.service,
        "--image", image,
        "--region", args.region,
        "--platform", "managed",
        "--allow-unauthenticated",
        "--update-env-vars", f"API_TOKEN={args.api_token}"
    ], env=env)

    print("\n\033[1;32m✅ Deploy complete.\033[0m")
    print("Tip: run `gcloud run services describe {svc} --region {reg} --format='value(status.url)'` to get the URL."
          .format(svc=args.service, reg=args.region))

if __name__ == "__main__":
    main()
