import pandas as pd
from pathlib import Path
from unittest.mock import mock_open, patch, call, MagicMock
from hsr_data_converter.commands.rosbag2lerobot.qa_report_creator import (
    QaReportCreator,
)
from hsr_data_converter.commands.rosbag2lerobot.types.qa_report_types import (
    QaReportErrorType,
    QaReportErrorLevel,
)


class TestQaReportCreatorConstructor:
    @patch.object(Path, "rglob")
    @patch("pandas.read_parquet")
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.qa_report_models.QaReportList.__init__",
        return_value=None,
    )
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.info_models.Info.__init__",
        return_value=None,
    )
    def test_initialization_with_fresh_columns(
        self, mock_info_init, mock_qa_report_list_init, mock_read_parquet, mock_rglob
    ):
        mock_rglob.return_value = ["/some/test/path/file1.parquet"]
        mock_parquet_data = pd.DataFrame(
            {
                "frame_index": [1, 2, 3],
                "episode_index": [0, 0, 0],
                "observation.image.head.is_fresh": [True, False, True],
                "observation.image.hand.is_fresh": [True, True, False],
            }
        )
        mock_read_parquet.return_value = mock_parquet_data

        out_dir = "/some/test/path"
        creator = QaReportCreator(out_dir)

        assert creator.out_dir == "/some/test/path"
        assert creator.out_path == "/some/test/path/qa_report.json"
        assert len(creator.parquet_data_list) == 1
        assert all(isinstance(df, pd.DataFrame) for df in creator.parquet_data_list)

        mock_read_parquet.assert_called_once()
        mock_rglob.assert_called_once_with("*.parquet")

    @patch.object(Path, "rglob")
    @patch("pandas.read_parquet")
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.qa_report_models.QaReportList.__init__",
        return_value=None,
    )
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.info_models.Info.__init__",
        return_value=None,
    )
    def test_initialization_without_fresh_columns(
        self, mock_info_init, mock_qa_report_list_init, mock_read_parquet, mock_rglob
    ):
        mock_rglob.return_value = ["/some/test/path/file1.parquet"]
        mock_parquet_data = pd.DataFrame(
            {
                "frame_index": [1, 2, 3],
                "episode_index": [0, 0, 0],
            }
        )
        mock_read_parquet.return_value = mock_parquet_data

        out_dir = "/some/test/path"
        creator = QaReportCreator(out_dir)

        assert creator.out_dir == "/some/test/path"
        assert creator.out_path == "/some/test/path/qa_report.json"
        assert len(creator.parquet_data_list) == 1
        assert all(isinstance(df, pd.DataFrame) for df in creator.parquet_data_list)

        mock_read_parquet.assert_called_once()
        mock_rglob.assert_called_once_with("*.parquet")


class TestQaReportCreatorSaveToJson:
    @patch("builtins.open", new_callable=mock_open)
    @patch.object(Path, "rglob")
    @patch("pandas.read_parquet")
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.qa_report_models.QaReportList.__init__",
        return_value=None,
    )
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.info_models.Info.__init__",
        return_value=None,
    )
    def test_standard_save_to_json(
        self,
        mock_info_init,
        mock_qa_report_list_init,
        mock_read_parquet,
        mock_rglob,
        mock_builtin_open,
    ):
        mock_rglob.return_value = ["/some/test/path/file1.parquet"]
        mock_parquet_data = pd.DataFrame(
            {
                "frame_index": [1, 2, 3],
                "episode_index": [0, 0, 0],
            }
        )
        mock_read_parquet.return_value = mock_parquet_data

        mock_qa_report_list_instance = MagicMock()
        mock_qa_report_list_instance.to_json.return_value = '{"sample": "data"}'

        out_dir = "/some/test/path"
        creator = QaReportCreator(out_dir)

        creator.qa_json_list = mock_qa_report_list_instance

        creator.save_to_json()

        mock_builtin_open.assert_called_once_with("/some/test/path/qa_report.json", "w")
        mock_qa_report_list_instance.to_json.assert_called_once()
        mock_builtin_open().write.assert_called_once_with(
            '"{\\"sample\\": \\"data\\"}"'
        )


class TestQaReportCreatorReportNanValueErrors:
    @patch.object(Path, "rglob")
    @patch("pandas.read_parquet")
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.qa_report_models.QaReportList.__init__",
        return_value=None,
    )
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.info_models.Info.__init__",
        return_value=None,
    )
    def test_standard_nan_value_errors(
        self, mock_info_init, mock_qa_report_list_init, mock_read_parquet, mock_rglob
    ):
        mock_rglob.return_value = ["/some/test/path/file1.parquet"]

        mock_parquet_data = pd.DataFrame(
            {
                "frame_index": [1, 2, 3],
                "episode_index": [0, 0, 0],
                "some_value": [1.0, None, 3.0],
            }
        )
        mock_read_parquet.return_value = mock_parquet_data

        mock_qa_report_list_instance = MagicMock()

        out_dir = "/some/test/path"
        creator = QaReportCreator(out_dir)
        creator.qa_json_list = mock_qa_report_list_instance

        creator.report_nan_value_errors()

        expected_call = call(
            episode_index=0,
            error=QaReportErrorType.NAN_VALUE,
            level=QaReportErrorLevel.ERROR,
            item="some_value",
            frame=2,
        )
        mock_qa_report_list_instance.append.assert_has_calls([expected_call])

    @patch.object(Path, "rglob", return_value=[])
    @patch("pandas.read_parquet", return_value=pd.DataFrame())
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.qa_report_models.QaReportList.__init__",
        return_value=None,
    )
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.info_models.Info.__init__",
        return_value=None,
    )
    def test_no_nan_values(
        self, mock_info_init, mock_qa_report_list_init, mock_read_parquet, mock_rglob
    ):
        mock_qa_report_list_instance = MagicMock()

        out_dir = "/some/test/path"
        creator = QaReportCreator(out_dir)
        creator.qa_json_list = mock_qa_report_list_instance

        creator.report_nan_value_errors()

        mock_qa_report_list_instance.append.assert_not_called()

    @patch.object(Path, "rglob", return_value=[])
    @patch("pandas.read_parquet", return_value=pd.DataFrame())
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.qa_report_models.QaReportList.__init__",
        return_value=None,
    )
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.info_models.Info.__init__",
        return_value=None,
    )
    def test_empty_parquet_data_list(
        self, mock_info_init, mock_qa_report_list_init, mock_read_parquet, mock_rglob
    ):
        mock_qa_report_list_instance = MagicMock()

        out_dir = "/some/test/path"
        creator = QaReportCreator(out_dir)
        creator.qa_json_list = mock_qa_report_list_instance
        creator.parquet_data_list = []

        creator.report_nan_value_errors()

        mock_qa_report_list_instance.append.assert_not_called()


class TestQaReportCreatorReportTimestampMonotonicityErrors:
    @patch.object(Path, "rglob")
    @patch("pandas.read_parquet")
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.qa_report_models.QaReportList.__init__",
        return_value=None,
    )
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.info_models.Info.__init__",
        return_value=None,
    )
    def test_timestamp_decreasing_errors(
        self, mock_info_init, mock_qa_report_list_init, mock_read_parquet, mock_rglob
    ):
        mock_rglob.return_value = ["/some/test/path/file1.parquet"]

        mock_parquet_data = pd.DataFrame(
            {
                "frame_index": [1, 2, 3],
                "episode_index": [0, 0, 0],
                "timestamp": [1000, 950, 2000],
            }
        )
        mock_read_parquet.return_value = mock_parquet_data

        mock_qa_report_list_instance = MagicMock()

        out_dir = "/some/test/path"
        creator = QaReportCreator(out_dir)
        creator.qa_json_list = mock_qa_report_list_instance

        creator.report_timestamp_monotonicity_errors()

        expected_call = call(
            episode_index=0,
            error=QaReportErrorType.TIMESTAMP_CONSISTENCY,
            level=QaReportErrorLevel.ERROR,
            item="timestamp",
            frame=2,
        )
        mock_qa_report_list_instance.append.assert_has_calls([expected_call])

    @patch.object(Path, "rglob", return_value=[])
    @patch("pandas.read_parquet", return_value=pd.DataFrame())
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.qa_report_models.QaReportList.__init__",
        return_value=None,
    )
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.info_models.Info.__init__",
        return_value=None,
    )
    def test_no_timestamp_errors(
        self, mock_info_init, mock_qa_report_list_init, mock_read_parquet, mock_rglob
    ):
        mock_qa_report_list_instance = MagicMock()

        out_dir = "/some/test/path"
        creator = QaReportCreator(out_dir)
        creator.qa_json_list = mock_qa_report_list_instance

        creator.report_timestamp_monotonicity_errors()

        mock_qa_report_list_instance.append.assert_not_called()

    @patch.object(Path, "rglob", return_value=[])
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.qa_report_models.QaReportList.__init__",
        return_value=None,
    )
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.info_models.Info.__init__",
        return_value=None,
    )
    def test_empty_parquet_data_list(
        self, mock_info_init, mock_qa_report_list_init, mock_rglob
    ):
        mock_qa_report_list_instance = MagicMock()

        out_dir = "/some/test/path"
        creator = QaReportCreator(out_dir)

        creator.parquet_data_list = []
        creator.qa_json_list = mock_qa_report_list_instance

        creator.report_timestamp_monotonicity_errors()

        mock_qa_report_list_instance.append.assert_not_called()


class TestQaReportCreatorReportFrameIndexMonotonicityErrors:
    @patch.object(Path, "rglob")
    @patch("pandas.read_parquet")
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.qa_report_models.QaReportList.__init__",
        return_value=None,
    )
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.info_models.Info.__init__",
        return_value=None,
    )
    def test_report_non_monotonic_frame_index_errors(
        self, mock_info_init, mock_qa_report_list_init, mock_read_parquet, mock_rglob
    ):
        mock_rglob.return_value = ["/some/test/path/file1.parquet"]

        mock_parquet_data = pd.DataFrame(
            {
                "frame_index": [1, 3, 2],
                "episode_index": [0, 0, 0],
                "timestamp": [1000, 1500, 2000],
            }
        )
        mock_read_parquet.return_value = mock_parquet_data

        mock_qa_report_list_instance = MagicMock()

        out_dir = "/some/test/path"
        creator = QaReportCreator(out_dir)
        creator.qa_json_list = mock_qa_report_list_instance

        creator.report_frame_index_monotonicity_errors()

        expected_call = call(
            episode_index=0,
            error=QaReportErrorType.FRAME_SKIP,
            level=QaReportErrorLevel.ERROR,
            item="frame_index",
            frame=3,
        )
        mock_qa_report_list_instance.append.assert_has_calls([expected_call])

    @patch.object(Path, "rglob")
    @patch("pandas.read_parquet")
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.qa_report_models.QaReportList.__init__",
        return_value=None,
    )
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.info_models.Info.__init__",
        return_value=None,
    )
    def test_report_non_monotonic_frame_index_errors_with_multiple_episodes(
        self, mock_info_init, mock_qa_report_list_init, mock_read_parquet, mock_rglob
    ):
        mock_rglob.return_value = ["/some/test/path/file1.parquet"]

        mock_parquet_data = pd.DataFrame(
            {
                "frame_index": [0, 1, 2, 0, 2],
                "episode_index": [0, 0, 0, 1, 1],
                "timestamp": [1000, 1500, 2000, 2500, 3000],
            }
        )
        mock_read_parquet.return_value = mock_parquet_data

        mock_qa_report_list_instance = MagicMock()

        out_dir = "/some/test/path"
        creator = QaReportCreator(out_dir)
        creator.qa_json_list = mock_qa_report_list_instance

        creator.report_frame_index_monotonicity_errors()

        expected_call = call(
            episode_index=1,
            error=QaReportErrorType.FRAME_SKIP,
            level=QaReportErrorLevel.ERROR,
            item="frame_index",
            frame=2,
        )
        mock_qa_report_list_instance.append.assert_has_calls([expected_call])

    @patch.object(Path, "rglob", return_value=[])
    @patch("pandas.read_parquet", return_value=pd.DataFrame())
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.qa_report_models.QaReportList.__init__",
        return_value=None,
    )
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.info_models.Info.__init__",
        return_value=None,
    )
    def test_no_frame_index_errors_detected(
        self, mock_info_init, mock_qa_report_list_init, mock_read_parquet, mock_rglob
    ):
        mock_rglob.return_value = ["/some/test/path/file1.parquet"]

        mock_parquet_data = pd.DataFrame(
            {
                "frame_index": [0, 1, 2, 0, 1],
                "episode_index": [0, 0, 0, 1, 1],
                "timestamp": [1000, 1500, 2000, 2500, 3000],
            }
        )
        mock_read_parquet.return_value = mock_parquet_data

        mock_qa_report_list_instance = MagicMock()

        out_dir = "/some/test/path"
        creator = QaReportCreator(out_dir)
        creator.qa_json_list = mock_qa_report_list_instance

        creator.report_frame_index_monotonicity_errors()

        mock_qa_report_list_instance.append.assert_not_called()

    @patch.object(Path, "rglob", return_value=[])
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.qa_report_models.QaReportList.__init__",
        return_value=None,
    )
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.info_models.Info.__init__",
        return_value=None,
    )
    def test_report_no_errors_for_empty_parquet_data_list(
        self, mock_info_init, mock_qa_report_list_init, mock_rglob
    ):
        mock_qa_report_list_instance = MagicMock()

        out_dir = "/some/test/path"
        creator = QaReportCreator(out_dir)

        creator.parquet_data_list = []
        creator.qa_json_list = mock_qa_report_list_instance

        creator.report_frame_index_monotonicity_errors()

        mock_qa_report_list_instance.append.assert_not_called()


class TestQaReportCreatorReportMissingVideos:
    @patch.object(Path, "rglob", return_value=[])
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.qa_report_models.QaReportList.__init__",
        return_value=None,
    )
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.info_models.Info.__init__",
        return_value=None,
    )
    @patch("os.path.exists")
    def test_missing_videos(
        self, mock_exists, mock_info_init, mock_qa_report_list_init, mock_rglob
    ):
        mock_exists.side_effect = [False, True]

        mock_qa_report_list_instance = MagicMock()
        mock_info_instance = MagicMock()
        mock_info_instance.get_video_path_list.return_value = [
            (0, "video1.mp4"),
            (1, "video2.mp4"),
        ]

        out_dir = "/some/test/path"
        creator = QaReportCreator(out_dir)
        creator.qa_json_list = mock_qa_report_list_instance
        creator.info = mock_info_instance

        creator.report_missing_videos()

        expected_call = call(
            episode_index=0,
            error=QaReportErrorType.NOT_EXIST_VIDEO,
            level=QaReportErrorLevel.ERROR,
        )
        mock_qa_report_list_instance.append.assert_has_calls([expected_call])

    @patch.object(Path, "rglob", return_value=[])
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.qa_report_models.QaReportList.__init__",
        return_value=None,
    )
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.models.info_models.Info.__init__",
        return_value=None,
    )
    @patch("os.path.exists")
    def test_no_missing_videos(
        self, mock_exists, mock_info_init, mock_qa_report_list_init, mock_rglob
    ):
        mock_exists.return_value = True

        mock_qa_report_list_instance = MagicMock()
        mock_info_instance = MagicMock()
        mock_info_instance.get_video_path_list.return_value = [
            (0, "video1.mp4"),
            (1, "video2.mp4"),
        ]

        out_dir = "/some/test/path"
        creator = QaReportCreator(out_dir)
        creator.qa_json_list = mock_qa_report_list_instance
        creator.info = mock_info_instance

        creator.report_missing_videos()

        mock_qa_report_list_instance.append.assert_not_called()
