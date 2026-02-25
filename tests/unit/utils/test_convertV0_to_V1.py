import unittest
import json
import pytest
from pathlib import Path, PosixPath
from unittest.mock import MagicMock, patch, mock_open, call

from hsr_data_converter.utils.convertV0_to_V1 import convert_dataset, CLIArgs


class TestConvertDataset(unittest.TestCase):
    @patch("hsr_data_converter.utils.convertV0_to_V1.MetadataV0_0.from_dict")
    @patch("pathlib.Path.mkdir")
    @patch("shutil.copy")
    @patch("json.load")
    def test_succeed_to_copy_the_bag_file(
        self, mock_load, mock_copy, mock_mkdir, mock_from_dict
    ):
        out_dir = "test_out_dir"
        raw_dir = "test_raw_dir"

        mock_json_data = [{"id": 1, "name": "test"}]
        m = mock_open(read_data=json.dumps(mock_json_data))

        mock_load.return_value = json.loads(json.dumps(mock_json_data))

        mock_copy.return_value = None

        mock_metadata_v0_0 = MagicMock()
        mock_metadata_v0_0.bag_path = "test.json"
        mock_from_dict.return_value = mock_metadata_v0_0

        mock_bag_file_names = ["1", "2"]
        mock_bag_file_pathes = [
            f"/dummy/path/{item}.bag" for item in mock_bag_file_names
        ]
        mock_bag_files = [Path(item) for item in mock_bag_file_pathes]

        args = CLIArgs(raw_dir=raw_dir, out_dir=out_dir)
        with (
            patch("builtins.open", m),
            patch.object(Path, "glob", return_value=mock_bag_files),
        ):
            convert_dataset(args=args)

        mock_load.assert_called_once()
        assert mock_from_dict.call_count == 1
        assert mock_mkdir.call_count == 3
        assert mock_copy.call_count == 2
        mock_copy.assert_has_calls(
            [
                call(
                    PosixPath(mock_bag_file_pathes[0]),
                    PosixPath(f"{out_dir}/{mock_bag_file_names[0]}"),
                ),
                call(
                    PosixPath(mock_bag_file_pathes[1]),
                    PosixPath(f"{out_dir}/{mock_bag_file_names[1]}"),
                ),
            ]
        )

    @patch("hsr_data_converter.utils.convertV0_to_V1.MetadataV0_0.from_dict")
    @patch("pathlib.Path.mkdir")
    @patch("shutil.copy")
    @patch("json.load")
    def test_not_exist_bag_files(
        self, mock_load, mock_copy, mock_mkdir, mock_from_dict
    ):
        out_dir = "test_out_dir"
        raw_dir = "test_raw_dir"

        mock_json_data = [{"id": 1, "name": "test"}]
        m = mock_open(read_data=json.dumps(mock_json_data))

        mock_load.return_value = json.loads(json.dumps(mock_json_data))

        mock_copy.return_value = None

        mock_metadata_v0_0 = MagicMock()
        mock_metadata_v0_0.bag_path = "test.json"
        mock_from_dict.return_value = mock_metadata_v0_0

        mock_bag_files = []

        args = CLIArgs(raw_dir=raw_dir, out_dir=out_dir)
        with (
            patch("builtins.open", m),
            patch.object(Path, "glob", return_value=mock_bag_files),
        ):
            convert_dataset(args=args)

        mock_load.assert_called_once()
        assert mock_from_dict.call_count == 1
        assert mock_mkdir.call_count == 1
        mock_copy.assert_not_called()

    @patch("hsr_data_converter.utils.convertV0_to_V1.MetadataV0_0.from_dict")
    @patch("pathlib.Path.mkdir")
    @patch("shutil.copy")
    @patch("json.load")
    def test_not_exist_meta_files(
        self, mock_load, mock_copy, mock_mkdir, mock_from_dict
    ):
        out_dir = "test_out_dir"
        raw_dir = "test_raw_dir"

        mock_json_data = []
        m = mock_open(read_data=json.dumps(mock_json_data))

        mock_load.return_value = json.loads(json.dumps(mock_json_data))

        mock_copy.return_value = None

        mock_metadata_v0_0 = MagicMock()
        mock_metadata_v0_0.bag_path = "test.json"
        mock_from_dict.return_value = mock_metadata_v0_0

        mock_bag_file_names = ["1", "2"]
        mock_bag_file_pathes = [
            f"/dummy/path/{item}.bag" for item in mock_bag_file_names
        ]
        mock_bag_files = [Path(item) for item in mock_bag_file_pathes]

        args = CLIArgs(raw_dir=raw_dir, out_dir=out_dir)
        with (
            patch("builtins.open", m),
            patch.object(Path, "glob", return_value=mock_bag_files),
        ):
            convert_dataset(args=args)

        mock_load.assert_called_once()
        mock_from_dict.assert_not_called()
        assert mock_mkdir.call_count == 1
        mock_copy.assert_not_called()

    @patch("hsr_data_converter.utils.convertV0_to_V1.MetadataV0_0.from_dict")
    @patch("pathlib.Path.mkdir")
    @patch("shutil.copy")
    @patch("json.load")
    def test_error_meta_file_is_not_json(
        self, mock_load, mock_copy, mock_mkdir, mock_from_dict
    ):
        out_dir = "test_out_dir"
        raw_dir = "test_raw_dir"

        mock_json_data = [{"id": 1, "name": "test"}]
        m = mock_open(read_data=json.dumps(mock_json_data))

        mock_load.side_effect = Exception("test error.")

        mock_copy.return_value = None

        mock_metadata_v0_0 = MagicMock()
        mock_metadata_v0_0.bag_path = "test.json"
        mock_from_dict.return_value = mock_metadata_v0_0

        mock_bag_file_names = ["1", "2"]
        mock_bag_file_pathes = [
            f"/dummy/path/{item}.bag" for item in mock_bag_file_names
        ]
        mock_bag_files = [Path(item) for item in mock_bag_file_pathes]

        args = CLIArgs(raw_dir=raw_dir, out_dir=out_dir)
        with pytest.raises(Exception) as e:
            with (
                patch("builtins.open", m),
                patch.object(Path, "glob", return_value=mock_bag_files),
            ):
                convert_dataset(args=args)
        assert str(e.value) == "test error."

        mock_load.assert_called_once()
        mock_from_dict.assert_not_called()
        assert mock_mkdir.call_count == 1
        mock_copy.assert_not_called()

    @patch("hsr_data_converter.utils.convertV0_to_V1.MetadataV0_0.from_dict")
    @patch("pathlib.Path.mkdir")
    @patch("shutil.copy")
    @patch("json.load")
    def test_error_faild_to_execute_metadata_from_dict(
        self, mock_load, mock_copy, mock_mkdir, mock_from_dict
    ):
        out_dir = "test_out_dir"
        raw_dir = "test_raw_dir"

        mock_json_data = [{"id": 1, "name": "test"}]
        m = mock_open(read_data=json.dumps(mock_json_data))

        mock_load.return_value = json.loads(json.dumps(mock_json_data))

        mock_copy.return_value = None

        mock_from_dict.side_effect = Exception("test error.")

        mock_bag_file_names = ["1", "2"]
        mock_bag_file_pathes = [
            f"/dummy/path/{item}.bag" for item in mock_bag_file_names
        ]
        mock_bag_files = [Path(item) for item in mock_bag_file_pathes]

        args = CLIArgs(raw_dir=raw_dir, out_dir=out_dir)
        with pytest.raises(Exception) as e:
            with (
                patch("builtins.open", m),
                patch.object(Path, "glob", return_value=mock_bag_files),
            ):
                convert_dataset(args=args)
        assert str(e.value) == "test error."

        mock_load.assert_called_once()
        assert mock_from_dict.call_count == 1
        assert mock_mkdir.call_count == 1
        mock_copy.assert_not_called()
