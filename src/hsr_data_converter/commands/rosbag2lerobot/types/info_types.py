from typing import Any, TypedDict
from enum import Enum


class InfoFeaturesDetailDtype(Enum):
    VIDEO = "video"
    FLOAT32 = "float32"
    BOOL = "bool"
    INT64 = "int64"


class InfoFeaturesDetailInterface(TypedDict):
    dtype: InfoFeaturesDetailDtype
    shape: list[int]
    names: list[str]
    info: dict[str, Any]
    description: str


class InfoInterface(TypedDict):
    codebase_version: str
    robot_type: str
    total_episodes: int
    total_frames: int
    total_tasks: int
    total_videos: int
    total_chunks: int
    chunks_size: int
    fps: int
    splits: dict[str, str]
    data_path: str
    video_path: str
    features: dict[str, InfoFeaturesDetailInterface]
