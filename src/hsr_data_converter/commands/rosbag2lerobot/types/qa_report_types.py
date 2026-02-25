from typing import Optional, TypedDict
from enum import Enum


class QaReportErrorType(Enum):
    NAN_VALUE = "nan_value_exception"
    TIMESTAMP_CONSISTENCY = "timestamp_consistency_exception"
    FRAME_SKIP = "frame_skip_exception"
    NOT_EXIST_VIDEO = "not_exist_video_exception"


class QaReportErrorLevel(Enum):
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"


class QaReportErrorInterface(TypedDict, total=False):
    error: str
    level: str
    item: Optional[str]
    frame: Optional[int]


class QaReportInterface(TypedDict):
    episode_index: int
    timestamp: str
    errors: list[QaReportErrorInterface]
