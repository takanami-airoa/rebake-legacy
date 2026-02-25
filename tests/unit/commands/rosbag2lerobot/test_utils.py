import os
from unittest.mock import patch

import jsonlines
import pytest

from hsr_data_converter.commands.rosbag2lerobot.utils import (
    update_last_jsonline,
    update_episodes_jsonl,
    validate_git_information,
    get_git_information,
    is_lineage_enabled,
    validate_lineage_enabled,
)


class TestUpdateLastJsonline:
    def test_standard_update(self, tmp_path):
        test_file = tmp_path / "test.jsonl"

        with jsonlines.open(test_file, "w") as writer:
            writer.write({"episode_index": 0, "task": "pick"})
            writer.write({"episode_index": 1, "task": "place"})

        new_data = {"success": True, "duration": 10.5}
        update_last_jsonline(new_data, test_file)

        with jsonlines.open(test_file, "r") as reader:
            lines = list(reader)

        assert len(lines) == 2
        assert lines[0] == {"episode_index": 0, "task": "pick"}
        assert lines[1] == {
            "episode_index": 1,
            "task": "place",
            "success": True,
            "duration": 10.5,
        }

    def test_error_file_not_found(self, tmp_path):
        nonexistent_file = tmp_path / "nonexistent.jsonl"

        with pytest.raises(FileNotFoundError, match="does not exist"):
            update_last_jsonline({"new_key": "new_value"}, nonexistent_file)

    def test_error_empty_file(self, tmp_path):
        test_file = tmp_path / "empty.jsonl"
        test_file.touch()

        with pytest.raises(ValueError, match="is empty"):
            update_last_jsonline({"new_key": "new_value"}, test_file)


class TestUpdateEpisodesJsonl:
    def test_standard_update(self, tmp_path):
        test_file = tmp_path / "episodes.jsonl"

        with jsonlines.open(test_file, "w") as writer:
            writer.write({"episode_index": 0, "task": "pick"})
            writer.write({"episode_index": 1, "task": "place"})
            writer.write({"episode_index": 2, "task": "stack"})

        new_data = {
            0: {"success": True, "duration": 8.0},
            1: {"success": False, "duration": 12.0},
            2: {"success": True, "duration": 15.0},
        }
        update_episodes_jsonl(new_data, test_file)

        with jsonlines.open(test_file, "r") as reader:
            lines = list(reader)

        assert len(lines) == 3
        assert lines[0] == {
            "episode_index": 0,
            "task": "pick",
            "success": True,
            "duration": 8.0,
        }
        assert lines[1] == {
            "episode_index": 1,
            "task": "place",
            "success": False,
            "duration": 12.0,
        }
        assert lines[2] == {
            "episode_index": 2,
            "task": "stack",
            "success": True,
            "duration": 15.0,
        }

    def test_error_file_not_found(self, tmp_path):
        nonexistent_file = tmp_path / "nonexistent.jsonl"

        with pytest.raises(FileNotFoundError, match="does not exist"):
            update_episodes_jsonl({1: {"new_key": "new_value"}}, nonexistent_file)


class TestValidateGitInformation:
    @patch.dict(
        os.environ,
        {
            "GIT_HASH": "abcd1234",
            "GIT_BRANCH": "main",
            "GIT_URL": "http://github.com/user/repo",
            "GIT_TAG": "v1.0.0",
        },
    )
    def test_standard(self):
        try:
            validate_git_information()
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

    @patch.dict(os.environ, {}, clear=True)
    @patch.dict(
        os.environ,
        {
            "GIT_BRANCH": "main",
            "GIT_URL": "http://github.com/user/repo",
            "GIT_TAG": "v1.0.0",
        },
    )
    def test_missing_git_hash(self):
        with pytest.raises(ValueError, match="Git information is not fully set"):
            validate_git_information()

    @patch.dict(os.environ, {}, clear=True)
    @patch.dict(
        os.environ,
        {
            "GIT_HASH": "abcd1234",
            "GIT_URL": "http://github.com/user/repo",
            "GIT_TAG": "v1.0.0",
        },
    )
    def test_missing_git_branch(self):
        with pytest.raises(ValueError, match="Git information is not fully set"):
            validate_git_information()

    @patch.dict(os.environ, {}, clear=True)
    @patch.dict(
        os.environ, {"GIT_HASH": "abcd1234", "GIT_BRANCH": "main", "GIT_TAG": "v1.0.0"}
    )
    def test_missing_git_url(self):
        with pytest.raises(ValueError, match="Git information is not fully set"):
            validate_git_information()

    @patch.dict(os.environ, {}, clear=True)
    @patch.dict(
        os.environ,
        {
            "GIT_HASH": "abcd1234",
            "GIT_BRANCH": "main",
            "GIT_URL": "http://github.com/user/repo",
        },
    )
    def test_missing_git_tag(self):
        with pytest.raises(ValueError, match="Git information is not fully set"):
            validate_git_information()


class TestGetGitInformation:
    @patch("hsr_data_converter.commands.rosbag2lerobot.utils.validate_git_information")
    @patch.dict(
        os.environ,
        {
            "GIT_HASH": "abcd1234",
            "GIT_BRANCH": "main",
            "GIT_URL": "http://github.com/user/repo",
            "GIT_TAG": "v1.0.0",
        },
    )
    def test_standard(self, mock_validate):
        git_info = get_git_information()
        mock_validate.assert_called_once()

        expected_info = {
            "git_url": "http://github.com/user/repo",
            "git_hash": "abcd1234",
            "git_branch": "main",
            "git_tag": "v1.0.0",
        }

        assert git_info == expected_info

    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.utils.validate_git_information",
        side_effect=ValueError("Git information is not fully set"),
    )
    @patch.dict(os.environ, {}, clear=True)
    def test_validation_failure(self, mock_validate):
        with pytest.raises(ValueError, match="Git information is not fully set"):
            get_git_information()
        mock_validate.assert_called_once()


class TestIsLineageEnabled:
    @patch.dict(
        os.environ,
        {},
    )
    def test_environment_is_none(self):
        actual = is_lineage_enabled()
        assert not actual

    @patch.dict(
        os.environ,
        {"LINEAGE_ENABLED": "true"},
    )
    def test_environment_is_true(self):
        actual = is_lineage_enabled()
        assert actual

    @patch.dict(
        os.environ,
        {"LINEAGE_ENABLED": "false"},
    )
    def test_environment_is_false(self):
        actual = is_lineage_enabled()
        assert not actual


class TestValidateLineageEnabled:
    @patch("hsr_data_converter.commands.rosbag2lerobot.utils.is_lineage_enabled")
    @patch("hsr_data_converter.commands.rosbag2lerobot.utils.get_marquez_url")
    def test_is_lineage_enabled_is_true_and_get_marquez_url_is_none(
        self, mock_get_marquez_url, mock_is_lineage_enabled
    ):
        mock_is_lineage_enabled.return_value = True
        mock_get_marquez_url.return_value = None
        with pytest.raises(ValueError) as e:
            validate_lineage_enabled()
        assert (
            str(e.value)
            == "LINEAGE_ENABLED=true, but MARQUEZ_URL is not set in environment variables."
        )

    @patch("hsr_data_converter.commands.rosbag2lerobot.utils.is_lineage_enabled")
    @patch("hsr_data_converter.commands.rosbag2lerobot.utils.get_marquez_url")
    def test_is_lineage_enabled_is_true_and_get_marquez_url_is_not_none(
        self, mock_get_marquez_url, mock_is_lineage_enabled
    ):
        mock_is_lineage_enabled.return_value = True
        mock_get_marquez_url.return_value = "https://example.com"
        try:
            validate_lineage_enabled()
            assert True
        except Exception:
            assert False

    @patch("hsr_data_converter.commands.rosbag2lerobot.utils.is_lineage_enabled")
    @patch("hsr_data_converter.commands.rosbag2lerobot.utils.get_marquez_url")
    def test_is_lineage_enabled_is_false_and_get_marquez_url_is_none(
        self, mock_get_marquez_url, mock_is_lineage_enabled
    ):
        mock_is_lineage_enabled.return_value = False
        mock_get_marquez_url.return_value = None
        try:
            validate_lineage_enabled()
            assert True
        except Exception:
            assert False

    @patch("hsr_data_converter.commands.rosbag2lerobot.utils.is_lineage_enabled")
    @patch("hsr_data_converter.commands.rosbag2lerobot.utils.get_marquez_url")
    def test_is_lineage_enabled_is_false_and_get_marquez_url_is_not_none(
        self, mock_get_marquez_url, mock_is_lineage_enabled
    ):
        mock_is_lineage_enabled.return_value = False
        mock_get_marquez_url.return_value = "https://example.com"
        try:
            validate_lineage_enabled()
            assert True
        except Exception:
            assert False
