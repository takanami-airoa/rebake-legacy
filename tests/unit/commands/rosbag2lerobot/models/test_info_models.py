import pytest
from unittest.mock import mock_open, patch
import json
from hsr_data_converter.commands.rosbag2lerobot.types.info_types import (
    InfoFeaturesDetailDtype,
)
from hsr_data_converter.commands.rosbag2lerobot.models.info_models import (
    Info,
)


class TestInfoConstructor:
    @patch("builtins.open", new_callable=mock_open, read_data="{}")
    def test_standard_constructor(self, mock_open):
        Info("fake_dir")
        mock_open.assert_called_once_with("fake_dir/meta/info.json")

    @patch("builtins.open", new_callable=mock_open, read_data="{}")
    def test_error_file_not_found(self, mock_open):
        mock_open.side_effect = FileNotFoundError
        with pytest.raises(FileNotFoundError):
            Info("fake_dir")


class TestInfoGetVideoPath:
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=json.dumps({"video_path": "/path/to/video"}),
    )
    def test_standard_get_video_path(self, mock_open):
        info = Info("fake_dir")
        assert info.get_video_path() == "/path/to/video"


class TestInfoGetTotalChunks:
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=json.dumps({"total_chunks": 5}),
    )
    def test_standard_get_total_chunks(self, mock_open):
        info = Info("fake_dir")
        assert info.get_total_chunks() == 5


class TestInfoGetTotalVideos:
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=json.dumps({"total_videos": 10}),
    )
    def test_standard_get_total_videos(self, mock_open):
        info = Info("fake_dir")
        assert info.get_total_videos() == 10


class TestInfoGetTotalEpisodes:
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=json.dumps({"total_episodes": 3}),
    )
    def test_standard_get_total_episodes(self, mock_open):
        info = Info("fake_dir")
        assert info.get_total_episodes() == 3


class TestInfoGetFeatures:
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=json.dumps({"features": {"feature1": {}, "feature2": {}}}),
    )
    def test_standard_get_features(self, mock_open):
        info = Info("fake_dir")
        assert info.get_features() == {"feature1": {}, "feature2": {}}


class TestInfoGetVideoKeys:
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=json.dumps(
            {
                "features": {
                    "key1": {"dtype": InfoFeaturesDetailDtype.VIDEO.value},
                    "key2": {"dtype": InfoFeaturesDetailDtype.VIDEO.value},
                    "key3": {"dtype": InfoFeaturesDetailDtype.BOOL.value},
                    "key4": {"dtype": InfoFeaturesDetailDtype.INT64.value},
                }
            }
        ),
    )
    def test_standard_get_video_keys(self, mock_open):
        info = Info("fake_dir")
        assert info.get_video_keys() == ["key1", "key2"]


class TestInfoGetVideoPathList:
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=json.dumps(
            {
                "video_path": "/path/{episode_chunk:06d}/{video_key}/{episode_index:03d}.mp4",
                "total_chunks": 3,
                "total_episodes": 2,
                "features": {
                    "video_key1": {"dtype": InfoFeaturesDetailDtype.VIDEO.value},
                    "video_key2": {"dtype": InfoFeaturesDetailDtype.VIDEO.value},
                },
            }
        ),
    )
    def test_standard_get_video_path_list(self, mock_open):
        info = Info("fake_dir")
        actual = info.get_video_path_list()
        assert len(actual) == 12
        assert (0, "/path/000000/video_key1/000.mp4") in actual
        assert (0, "/path/000001/video_key1/000.mp4") in actual
        assert (0, "/path/000002/video_key1/000.mp4") in actual
        assert (0, "/path/000000/video_key2/000.mp4") in actual
        assert (0, "/path/000001/video_key2/000.mp4") in actual
        assert (0, "/path/000002/video_key2/000.mp4") in actual
        assert (1, "/path/000000/video_key1/001.mp4") in actual
        assert (1, "/path/000001/video_key1/001.mp4") in actual
        assert (1, "/path/000002/video_key1/001.mp4") in actual
        assert (1, "/path/000000/video_key2/001.mp4") in actual
        assert (1, "/path/000001/video_key2/001.mp4") in actual
        assert (1, "/path/000002/video_key2/001.mp4") in actual
