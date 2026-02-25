import pandas as pd
from datetime import datetime, timezone
from hsr_data_converter.commands.rosbag2lerobot.types.qa_report_types import (
    QaReportErrorLevel,
    QaReportErrorType,
    QaReportErrorInterface,
    QaReportInterface,
)


class QaReportError:
    """
    Represents an error encountered during QA reporting.

    Attributes:
        error (QaReportErrorType): Type of error encountered.
        level (QaReportErrorLevel): Severity level of the error.
        item (str | None): Specific item related to the error, if applicable.
        frame (int | None): Frame index where the error was detected, if applicable.
    """

    def __init__(
        self,
        error: QaReportErrorType,
        level: QaReportErrorLevel,
        item: str | None,
        frame: int | None,
    ) -> None:
        """
        Initialize the QaReportError object.

        Args:
            error (QaReportErrorType): Type of error.
            level (QaReportErrorLevel): Level of severity for the error.
            item (str | None): Item related to the error.
            frame (int | None): Frame index where the error occurred.
        """
        self.error = error
        self.level = level
        self.item = item
        self.frame = frame

    def to_json(self) -> QaReportErrorInterface:
        """
        Convert the error details to JSON format.

        Returns:
            QaReportErrorInterface: JSON representation of the error details.
        """
        result: QaReportErrorInterface = {
            "error": self.error.value,
            "level": self.level.value,
        }
        if self.item:
            result["item"] = self.item
        if self.frame:
            result["frame"] = self.frame
        return result


class QaReport:
    """
    Represents a QA report for a specific episode.

    Attributes:
        episode_index (int): Index of the episode the report pertains to.
        timestamp (str): Timestamp indicating when the report was generated.
        errors (list[QaReportError]): List of errors identified in the episode.
    """

    def __init__(
        self, episode_index: int, timestamp: str, errors: list[QaReportError]
    ) -> None:
        """
        Initialize the QaReport object.

        Args:
            episode_index (int): Index of the episode.
            timestamp (str): Timestamp of report generation.
            errors (list[QaReportError]): List of errors.
        """
        self.episode_index = episode_index
        self.timestamp = timestamp
        self.errors = errors

    def to_json(self) -> QaReportInterface:
        """
        Convert the QA report to JSON format.

        Returns:
            QaReportInterface: JSON representation of the QA report.
        """
        return {
            "episode_index": self.episode_index,
            "timestamp": self.timestamp,
            "errors": [error.to_json() for error in self.errors],
        }

    def add_error(self, error: QaReportError | None) -> None:
        """
        Add an error to the QA report.

        Args:
            error (QaReportError | None): Error to be added, if applicable.
        """
        if error:
            self.errors.append(error)


class QaReportList:
    """
    Manages a list of QA reports, aggregating errors from multiple dataframes.

    Attributes:
        timestamp (str): Timestamp when the QA reports were initialized.
        qa_report_dict (dict[int, QaReport]): Dictionary of QA reports, keyed by episode index.
    """

    def __init__(self, dataframes: list[pd.DataFrame]) -> None:
        """
        Initialize the QaReportList object from given dataframes.

        Args:
            dataframes (list[pd.DataFrame]): List of dataframes containing QA data.
        """
        self.timestamp: str = datetime.now(timezone.utc).isoformat()
        self.qa_report_dict: dict[int, QaReport] = {}
        for dataframe in dataframes:
            for _, row in dataframe.iterrows():
                self.append(episode_index=int(row["episode_index"]))

    def to_json(self) -> list[QaReportInterface]:
        """
        Convert the list of QA reports to JSON format.

        Returns:
            list[QaReportInterface]: List of JSON representations of QA reports.
        """
        return [qa_json.to_json() for qa_json in self.qa_report_dict.values()]

    def append(
        self,
        episode_index: int,
        error: QaReportErrorType | None = None,
        level: QaReportErrorLevel | None = None,
        item: str | None = None,
        frame: int | None = None,
    ) -> None:
        """
        Append a new QA report or error to an existing report.

        If the report for the provided episode index does not exist, a new report
        is created and added to the dictionary. Otherwise, the error is appended
        to the existing report in the dictionary.

        Args:
            episode_index (int): Index of the episode.
            error (QaReportErrorType | None): Type of error, if applicable.
            level (QaReportErrorLevel | None): Level of severity, if applicable.
            item (str | None): Item related to error, if applicable.
            frame (int | None): Frame index for the error, if applicable.
        """
        if episode_index not in self.qa_report_dict.keys():
            self.qa_report_dict[episode_index] = QaReport(
                episode_index=episode_index,
                timestamp=self.timestamp,
                errors=[],
            )
        if error and level:
            qa_report_error = QaReportError(
                error=error, level=level, item=item, frame=frame
            )
            self.qa_report_dict[episode_index].add_error(qa_report_error)
