"""
Utility functions for rosbag to lerobot conversion
"""

import os
from pathlib import Path
from typing import Dict

import jsonlines


def update_last_jsonline(new_data: Dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)

    # read all lines
    if fpath.exists():
        with jsonlines.open(fpath, "r") as reader:
            lines = list(reader)
    else:
        raise FileNotFoundError(f"{fpath} does not exist")

    if not lines:
        raise ValueError(f"{fpath} is empty")

    # add new pairs of key and value
    lines[-1].update(new_data)

    # update the file
    with jsonlines.open(fpath, "w") as writer:
        writer.write_all(lines)


def update_episodes_jsonl(new_data: Dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)

    # read all lines
    if fpath.exists():
        with jsonlines.open(fpath, "r") as reader:
            lines = list(reader)
    else:
        raise FileNotFoundError(f"{fpath} does not exist")

    # add new pairs of key and value to each lines based on
    for line in lines:
        episode_index = line["episode_index"]
        new_info_per_episode = new_data[episode_index]
        line.update(new_info_per_episode)

    # update the file
    with jsonlines.open(fpath, "w") as writer:
        writer.write_all(lines)


def validate_git_information() -> None:
    git_hash = os.getenv("GIT_HASH")
    git_branch = os.getenv("GIT_BRANCH")
    git_url = os.getenv("GIT_URL")
    git_tag = os.getenv("GIT_TAG")

    if not git_hash or not git_branch or not git_url or not git_tag:
        raise ValueError(
            "Git information is not fully set in environment variables. (GIT_HASH, GIT_BRANCH, GIT_URL, GIT_TAG)"
        )


def get_git_information() -> Dict[str, str]:
    validate_git_information()

    git_hash = os.environ["GIT_HASH"]
    git_branch = os.environ["GIT_BRANCH"]
    git_url = os.environ["GIT_URL"]
    git_tag = os.environ["GIT_TAG"]

    return {
        "git_url": git_url,
        "git_hash": git_hash,
        "git_branch": git_branch,
        "git_tag": git_tag,
    }


def get_aws_job_information() -> Dict[str, str] | None:
    job_id = os.getenv("AWS_BATCH_JOB_ID")
    job_name = os.getenv("AWS_BATCH_JOB_NAME")
    return {"job_id": job_id, "job_name": job_name} if job_id and job_name else None


def get_marquez_url() -> str | None:
    return os.getenv("MARQUEZ_URL")


def is_lineage_enabled():
    lineage_enabled = os.environ.get("LINEAGE_ENABLED")
    return lineage_enabled is not None and lineage_enabled.lower() == "true"


def validate_lineage_enabled():
    if is_lineage_enabled() and not get_marquez_url():
        raise ValueError(
            "LINEAGE_ENABLED=true, but MARQUEZ_URL is not set in environment variables."
        )
