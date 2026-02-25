from unittest.mock import patch, MagicMock
from hsr_data_converter.commands.rosbag2lerobot.types.qa_report_types import (
    QaReportErrorType,
    QaReportErrorLevel,
)
from hsr_data_converter.commands.rosbag2lerobot.models.qa_report_models import (
    QaReportError,
    QaReport,
    QaReportList,
)


class TestQaReportErrorToJson:
    def test_standard_error_to_json(self):
        error = QaReportError(
            error=QaReportErrorType.NAN_VALUE,
            level=QaReportErrorLevel.WARNING,
            item="timestamp",
            frame=5,
        )
        expected_json = {
            "error": "nan_value_exception",
            "level": "WARNING",
            "item": "timestamp",
            "frame": 5,
        }
        assert error.to_json() == expected_json

    def test_error_level_critical_to_json(self):
        error = QaReportError(
            error=QaReportErrorType.FRAME_SKIP,
            level=QaReportErrorLevel.CRITICAL,
            item="timestamp",
            frame=15,
        )
        expected_json = {
            "error": "frame_skip_exception",
            "level": "CRITICAL",
            "item": "timestamp",
            "frame": 15,
        }
        assert error.to_json() == expected_json

    def test_error_level_error_to_json(self):
        error = QaReportError(
            error=QaReportErrorType.TIMESTAMP_CONSISTENCY,
            level=QaReportErrorLevel.ERROR,
            item="timestamp",
            frame=10,
        )
        expected_json = {
            "error": "timestamp_consistency_exception",
            "level": "ERROR",
            "item": "timestamp",
            "frame": 10,
        }
        assert error.to_json() == expected_json

    def test_item_is_null_and_frame_is_null(self):
        error = QaReportError(
            error=QaReportErrorType.NOT_EXIST_VIDEO,
            level=QaReportErrorLevel.WARNING,
            item=None,
            frame=None,
        )
        expected_json = {
            "error": "not_exist_video_exception",
            "level": "WARNING",
        }
        assert error.to_json() == expected_json


class TestQaReportToJson:
    def test_standard_report_to_json(self):
        error_mock = MagicMock()
        error_mock.to_json.return_value = {
            "error": "nan_value_exception",
            "level": "WARNING",
            "item": "timestamp",
            "frame": 5,
        }
        qa_report = QaReport(
            episode_index=1, timestamp="2023-01-01T00:00:00+00:00", errors=[error_mock]
        )
        expected_json = {
            "episode_index": 1,
            "timestamp": "2023-01-01T00:00:00+00:00",
            "errors": [error_mock.to_json.return_value],
        }
        assert qa_report.to_json() == expected_json
        error_mock.to_json.assert_called_once()

    def test_report_with_multiple_errors_to_json(self):
        error1_mock = MagicMock()
        error1_mock.to_json.return_value = {
            "error": "frame_skip_exception",
            "level": "CRITICAL",
            "item": "timestamp",
            "frame": 15,
        }

        error2_mock = MagicMock()
        error2_mock.to_json.return_value = {
            "error": "timestamp_consistency_exception",
            "level": "ERROR",
            "item": "timestamp",
            "frame": 10,
        }

        qa_report = QaReport(
            episode_index=2,
            timestamp="2023-01-02T00:00:00+00:00",
            errors=[error1_mock, error2_mock],
        )
        expected_json = {
            "episode_index": 2,
            "timestamp": "2023-01-02T00:00:00+00:00",
            "errors": [
                error1_mock.to_json.return_value,
                error2_mock.to_json.return_value,
            ],
        }
        assert qa_report.to_json() == expected_json
        error1_mock.to_json.assert_called_once()
        error2_mock.to_json.assert_called_once()

    def test_empty_errors_in_report_to_json(self):
        qa_report = QaReport(
            episode_index=3, timestamp="2023-01-03T00:00:00+00:00", errors=[]
        )
        expected_json = {
            "episode_index": 3,
            "timestamp": "2023-01-03T00:00:00+00:00",
            "errors": [],
        }
        assert qa_report.to_json() == expected_json


class TestQaReportAddError:
    def test_add_single_error(self):
        qa_report = QaReport(
            episode_index=1, timestamp="2023-01-01T00:00:00+00:00", errors=[]
        )

        error_mock = MagicMock()
        error_mock.to_json.return_value = {
            "error": "nan_value_exception",
            "level": "WARNING",
            "item": "timestamp",
            "frame": 5,
        }

        qa_report.add_error(error_mock)

        expected_json = {
            "episode_index": 1,
            "timestamp": "2023-01-01T00:00:00+00:00",
            "errors": [error_mock.to_json.return_value],
        }
        assert qa_report.to_json() == expected_json
        error_mock.to_json.assert_called_once()

    def test_add_multiple_errors(self):
        qa_report = QaReport(
            episode_index=2, timestamp="2023-01-02T00:00:00+00:00", errors=[]
        )

        error1_mock = MagicMock()
        error1_mock.to_json.return_value = {
            "error": "frame_skip_exception",
            "level": "CRITICAL",
            "item": "timestamp",
            "frame": 15,
        }

        error2_mock = MagicMock()
        error2_mock.to_json.return_value = {
            "error": "timestamp_consistency_exception",
            "level": "ERROR",
            "item": "timestamp",
            "frame": 10,
        }

        qa_report.add_error(error1_mock)
        qa_report.add_error(error2_mock)

        expected_json = {
            "episode_index": 2,
            "timestamp": "2023-01-02T00:00:00+00:00",
            "errors": [
                error1_mock.to_json.return_value,
                error2_mock.to_json.return_value,
            ],
        }
        assert qa_report.to_json() == expected_json
        error1_mock.to_json.assert_called_once()
        error2_mock.to_json.assert_called_once()

    def test_add_none_error(self):
        qa_report = QaReport(
            episode_index=3, timestamp="2023-01-03T00:00:00+00:00", errors=[]
        )
        qa_report.add_error(None)

        expected_json = {
            "episode_index": 3,
            "timestamp": "2023-01-03T00:00:00+00:00",
            "errors": [],
        }
        assert qa_report.to_json() == expected_json


class TestQaReportListConstructor:
    def test_constructor_with_mocked_dataframes(self):
        dataframe_mock1 = MagicMock()
        dataframe_mock1.iterrows.return_value = iter(
            [
                (0, {"episode_index": "1"}),
                (1, {"episode_index": "2"}),
            ]
        )

        dataframe_mock2 = MagicMock()
        dataframe_mock2.iterrows.return_value = iter(
            [
                (0, {"episode_index": "3"}),
            ]
        )

        with patch.object(QaReportList, "append") as mock_append:
            qa_report_list = QaReportList(dataframes=[dataframe_mock1, dataframe_mock2])
            assert qa_report_list.timestamp is not None
            mock_append.assert_any_call(episode_index=1)
            mock_append.assert_any_call(episode_index=2)
            mock_append.assert_any_call(episode_index=3)
            assert mock_append.call_count == 3


class TestQaReportListToJson:
    def test_to_json_with_mocked_reports(self):
        qa_report_mock1 = MagicMock()
        qa_report_mock1.to_json.return_value = {"qa_report": "mock1"}

        qa_report_mock2 = MagicMock()
        qa_report_mock2.to_json.return_value = {"qa_report": "mock2"}

        qa_list = QaReportList(dataframes=[])
        qa_list.qa_report_dict = {0: qa_report_mock1, 1: qa_report_mock2}

        expected_json = [
            qa_report_mock1.to_json.return_value,
            qa_report_mock2.to_json.return_value,
        ]
        assert qa_list.to_json() == expected_json

        qa_report_mock1.to_json.assert_called_once()
        qa_report_mock2.to_json.assert_called_once()


class TestQaReportListAppend:
    def test_add_new_report(self):
        qa_list = QaReportList(dataframes=[])
        qa_list.append(episode_index=1)

        assert len(qa_list.qa_report_dict) == 1
        assert qa_list.qa_report_dict[1].episode_index == 1
        assert qa_list.qa_report_dict[1].errors == []

    def test_add_error_to_existing_report(self):
        qa_report_mock = MagicMock(spec=QaReport)
        qa_report_mock.episode_index = 1
        qa_report_mock.errors = []

        qa_list = QaReportList(dataframes=[])
        qa_list.qa_report_dict[qa_report_mock.episode_index] = qa_report_mock

        error_type = MagicMock(spec=QaReportError)
        level_type = MagicMock()

        qa_list.append(episode_index=1, error=error_type, level=level_type)

        assert len(qa_report_mock.add_error.call_args_list) == 1

    def test_append_with_none_error_and_level(self):
        qa_list = QaReportList(dataframes=[])
        qa_list.append(episode_index=2, error=None, level=None)

        assert len(qa_list.qa_report_dict) == 1
        assert len(qa_list.qa_report_dict[2].errors) == 0

    def test_append_creates_new_report_when_no_matching_index(self):
        qa_report_mock = MagicMock(spec=QaReport)
        qa_report_mock.episode_index = 999

        qa_list = QaReportList(dataframes=[])
        qa_list.qa_report_dict[qa_report_mock.episode_index] = qa_report_mock

        qa_list.append(episode_index=2)

        assert len(qa_list.qa_report_dict.values()) == 2
        created_reports = [
            report
            for report in qa_list.qa_report_dict.values()
            if report.episode_index == 2
        ]
        assert len(created_reports) == 1
