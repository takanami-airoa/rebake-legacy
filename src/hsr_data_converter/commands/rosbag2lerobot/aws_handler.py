"""
AWS handling functions for rosbag to lerobot conversion
"""

import os
import shutil
import time
from pathlib import Path

from hsr_data_converter.commands.rosbag2lerobot.convert_config import ConvertConfig
from hsr_data_converter.utils.aws_helper import AWSHelper


def generate_upload_path(template_dir: str) -> str:
    """
    Generate S3 upload path with version information.

    Format: {template_dir}/{unix_timestamp}_{git_hash}

    Parameters
    ----------
    template_dir : str
        Template directory path (e.g., "{location} / {batch_id}_{storage_id} / {episode_id}")

    Returns
    -------
    str
        Upload path with version information

    Raises
    ------
    ValueError
        If GIT_HASH environment variable is not set
    """
    # Get Unix timestamp
    converted_at = int(time.time())

    # Get git hash from environment variable
    git_hash = os.environ.get("GIT_HASH", "")
    if not git_hash:
        raise ValueError("GIT_HASH environment variable is not set")

    # Clean template directory path
    template_dir_clean = template_dir.rstrip("/")

    # Generate upload path
    upload_prefix = f"{template_dir_clean}/{converted_at}_{git_hash}"

    return upload_prefix


def setup_aws_environment(cfg: ConvertConfig):
    """Setup AWS environment and download data."""
    # Validate AWS-required parameters
    if not all(
        [
            cfg.secret_name,
            cfg.rosbags_bucket_name,
            cfg.lerobot_bucket_name,
            cfg.template_dir,
        ]
    ):
        raise ValueError(
            "AWS mode requires: secret_name, rosbags_bucket_name, lerobot_bucket_name, template_dir"
        )

    # Setup AWS helper
    aws_helper = AWSHelper(cfg.secret_name)

    # Setup local working directories with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    work_dir = Path(f"tmp/hsr_conversion_{timestamp}")
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = work_dir / "rosbags"
    out_dir = work_dir / "lerobots"

    # Process single template directory
    template_dir = cfg.template_dir.strip("/")
    print(f"[INFO] Processing template directory: {template_dir}")

    # Check if meta.json exists in the template directory
    if not aws_helper.check_meta_file_exists(cfg.rosbags_bucket_name, template_dir):
        raise FileNotFoundError(f"meta.json not found in {template_dir}")

    # Create local directory for this template
    template_name = template_dir.split("/")[-1]
    local_template_dir = raw_dir / template_name

    # Download the template directory from S3
    print(f"[INFO] Downloading {template_dir} to {local_template_dir}")
    aws_helper.download_s3_directory(
        cfg.rosbags_bucket_name, template_dir, local_template_dir
    )

    # Return paths and cleanup function
    def cleanup():
        if work_dir.exists():
            shutil.rmtree(work_dir)
            print("[INFO] Cleaned up temporary files")

    return f"s3://{cfg.rosbags_bucket_name}/{template_dir}", raw_dir, out_dir, cleanup


def handle_aws_upload(cfg: ConvertConfig, out_dir: Path):
    """Handle AWS upload of results."""
    aws_helper = AWSHelper(cfg.secret_name)

    # Upload results to S3
    if out_dir.exists() and any(out_dir.iterdir()):
        # Generate upload path with version information
        upload_prefix = generate_upload_path(cfg.template_dir)

        # Find all dataset directories in out_dir
        dataset_dirs = [d for d in out_dir.iterdir() if d.is_dir()]
        print(f"[DEBUG] out_dir: {out_dir}")
        print(f"[DEBUG] dataset_dirs found: {[d.name for d in dataset_dirs]}")
        if dataset_dirs:
            for i, actual_dataset_dir in enumerate(dataset_dirs):
                # For multiple datasets, add index to upload prefix
                current_upload_prefix = upload_prefix
                if len(dataset_dirs) > 1:
                    current_upload_prefix = f"{upload_prefix}_{i}"

                print(
                    f"[INFO] Uploading dataset from {actual_dataset_dir} to {cfg.lerobot_bucket_name}/{current_upload_prefix}"
                )
                aws_helper.upload_directory_to_s3(
                    actual_dataset_dir, cfg.lerobot_bucket_name, current_upload_prefix
                )
        else:
            print("[WARN] No dataset directory found in output directory")
            return

        print(
            f"[INFO] Upload completed successfully to s3://{cfg.lerobot_bucket_name}/{upload_prefix}"
        )
    else:
        print("[WARN] No output directory or files found to upload")
