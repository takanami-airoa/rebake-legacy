import os
import json
from pathlib import Path
import pandas as pd

from .models.qa_report_models import QaReportList
from .models.info_models import Info
from .types.qa_report_types import QaReportErrorLevel, QaReportErrorType


class QaReportCreator:
    """
    Class for creating QA reports from dataset errors.

    Attributes:
        parquet_data_list (list[pd.DataFrame]): List of dataframes read from Parquet files.
        out_dir (str): Output directory where the QA report will be stored.
        out_path (str): Full path to the output QA report JSON file.
        qa_json_list (QaReportList): List of QA reports generated from the data.
        info (Info): Info object to retrieve video metadata.
    """

    def __init__(self, out_dir: str) -> None:
        """
        Initialize QaReportCreator by loading data from Parquet files in the specified directory.

        Args:
            out_dir (str): Directory path where Parquet files and info.json are located.

        Note:
            Parquet data frames may contain columns with conflicting data types.
            Conversion of specific columns to object type (`observation.image.head.is_fresh` and `observation.image.hand.is_fresh`)
            is performed to ensure compatibility and prevent data processing errors.
        """
        root_path = Path(out_dir)
        parquet_files = [str(p) for p in root_path.rglob("*.parquet")]

        self.parquet_data_list: list[pd.DataFrame] = []
        for parquet_path in parquet_files:
            parquet_data = pd.read_parquet(parquet_path)

            if "observation.image.head.is_fresh" in parquet_data.columns:
                parquet_data = parquet_data.astype(
                    {"observation.image.head.is_fresh": object}
                )

            if "observation.image.hand.is_fresh" in parquet_data.columns:
                parquet_data = parquet_data.astype(
                    {"observation.image.hand.is_fresh": object}
                )

            self.parquet_data_list.append(parquet_data)

        self.out_dir = out_dir
        self.out_path = f"{self.out_dir}/qa_report.json"
        self.qa_json_list = QaReportList(self.parquet_data_list)
        self.info = Info(self.out_dir)

    def save_to_json(self) -> None:
        """
        Save the QA report list to a JSON file at the specified output path.
        """
        with open(self.out_path, "w") as f:
            json.dump(self.qa_json_list.to_json(), f, indent=2)

    def report_nan_value_errors(self) -> None:
        """
        Detect and report errors related to NaN values in the dataset.
        Appends these errors to the QA report list.
        """
        for parquet_data in self.parquet_data_list:
            null_row_indices = parquet_data[parquet_data.isnull().any(axis=1)].index

            for idx in null_row_indices:
                row = parquet_data.loc[idx]
                null_columns = row[row.isnull()].index.tolist()
                frame_index = int(row["frame_index"])
                episode_index = int(row["episode_index"])

                for null_column in null_columns:
                    self.qa_json_list.append(
                        episode_index=episode_index,
                        error=QaReportErrorType.NAN_VALUE,
                        level=QaReportErrorLevel.ERROR,
                        item=null_column,
                        frame=frame_index,
                    )

    def report_timestamp_monotonicity_errors(self) -> None:
        """
        Detect and report errors related to non-monotonic timestamps in the dataset.
        Appends these errors to the QA report list.
        """
        for parquet_data in self.parquet_data_list:
            decreasing_indices = parquet_data.index[
                parquet_data["timestamp"].diff() < 0  # type: ignore[operator]
            ].tolist()
            for idx in decreasing_indices:
                row = parquet_data.loc[idx]
                frame_index = int(row["frame_index"])
                episode_index = int(row["episode_index"])
                self.qa_json_list.append(
                    episode_index=episode_index,
                    error=QaReportErrorType.TIMESTAMP_CONSISTENCY,
                    level=QaReportErrorLevel.ERROR,
                    item="timestamp",
                    frame=frame_index,
                )

    def report_frame_index_monotonicity_errors(self) -> None:
        """
        Detect and report errors related to non-monotonic frame indices and incorrect starting index
        in the dataset. Appends these errors to the QA report list.
        """
        for parquet_data in self.parquet_data_list:
            grouped = parquet_data.groupby("episode_index")

            for _, group in grouped:
                diff_values = group["frame_index"].diff().fillna(1)
                error_indices = group.index[diff_values != 1].tolist()
                for idx in error_indices:
                    row = group.loc[idx]
                    episode_index = int(row["episode_index"])
                    frame_index = int(row["frame_index"])
                    self.qa_json_list.append(
                        episode_index=episode_index,
                        error=QaReportErrorType.FRAME_SKIP,
                        level=QaReportErrorLevel.ERROR,
                        item="frame_index",
                        frame=frame_index,
                    )

    def report_missing_videos(self) -> None:
        """
        Verify and report errors for missing video files.
        Appends these errors to the QA report list.
        """
        video_path_list = self.info.get_video_path_list()
        for episode_index, video_path in video_path_list:
            if not os.path.exists(f"{self.out_dir}/{video_path}"):
                self.qa_json_list.append(
                    episode_index=episode_index,
                    error=QaReportErrorType.NOT_EXIST_VIDEO,
                    level=QaReportErrorLevel.ERROR,
                )
