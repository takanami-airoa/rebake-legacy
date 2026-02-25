import json
import pandas as pd
from hsr_data_converter.commands.package.package import (
    PackageTool,
    ProcessJson,
    ProcessParquet,
    ProcessVideo,
)
import os
from unittest.mock import mock_open, patch, call, MagicMock
import numpy as np
import pytest
from pathlib import Path
from hsr_data_converter.utils.aws_helper import AWSHelper


class TestProcessJsonLoadJsonlData:
    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    def test_standard_json(self, mock_open):
        result = ProcessJson.load_jsonl_data("dummy/path.json", is_jsonl=False)
        expected = {"key": "value"}
        assert result == expected
        mock_open.assert_called_once_with("dummy/path.json", "r")

    @patch("builtins.open", new_callable=mock_open, read_data='[{"key": "value"}]')
    def test_standard_jsonl_array(self, mock_open):
        result = ProcessJson.load_jsonl_data("dummy/path.jsonl", is_jsonl=True)
        expected = [{"key": "value"}]
        assert result == expected
        mock_open.assert_called_once_with("dummy/path.jsonl")

    @patch("builtins.open", new_callable=mock_open, read_data="{key}")
    def test_error_json_decode_failure(self, mock_open):
        result = ProcessJson.load_jsonl_data("dummy/path.jsonl", is_jsonl=True)
        expected = []
        assert result == expected
        mock_open.assert_called()

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"key": "value"}\n{"key": "value2"}',
    )
    def test_jsonl_line_by_line(self, mock_open):
        result = ProcessJson.load_jsonl_data("dummy/path.jsonl", is_jsonl=True)
        expected = [{"key": "value"}, {"key": "value2"}]
        assert result == expected
        mock_open.assert_has_calls(
            [
                call("dummy/path.jsonl"),
                call().__enter__(),
                call().__iter__(),
                call().__exit__(None, None, None),
            ]
        )

    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    def test_error_loading_exception(self, mock_open):
        mock_open.side_effect = IOError("Unable to open file")
        result = ProcessJson.load_jsonl_data("dummy/path.jsonl", is_jsonl=True)
        expected = []
        assert result == expected
        mock_open.assert_has_calls([call("dummy/path.jsonl")])


class TestProcessJsonReadMeta:
    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    def test_standard_read_meta(self, mock_load_jsonl_data):
        mock_load_jsonl_data.side_effect = [
            [{"episode": 1}],  # episodes.jsonl
            [{"episode_stat": 1}],  # episodes_stats.jsonl
            {"info": "test"},  # info.json
            [{"task": 1}],  # tasks.jsonl
        ]

        result = ProcessJson.read_meta("dummy/dataset/path")

        expected = {
            "episodes": [{"episode": 1}],
            "episodes_stats": [{"episode_stat": 1}],
            "info": {"info": "test"},
            "tasks": [{"task": 1}],
        }

        assert result == expected
        mock_load_jsonl_data.assert_any_call(
            os.path.join("dummy/dataset/path", "meta", "episodes.jsonl"), is_jsonl=True
        )
        mock_load_jsonl_data.assert_any_call(
            os.path.join("dummy/dataset/path", "meta", "episodes_stats.jsonl"),
            is_jsonl=True,
        )
        mock_load_jsonl_data.assert_any_call(
            os.path.join("dummy/dataset/path", "meta", "info.json")
        )
        mock_load_jsonl_data.assert_any_call(
            os.path.join("dummy/dataset/path", "meta", "tasks.jsonl"), is_jsonl=True
        )


class TestProcessJsonSaveJsonl:
    @patch("builtins.open", new_callable=mock_open)
    def test_standard_save_jsonl(self, mock_open):
        mock_data = [{"key1": "value1"}, {"key2": "value2"}]

        ProcessJson.save_jsonl(mock_data, "dummy/path.jsonl")

        mock_open.assert_called_once_with("dummy/path.jsonl", "w")

        handle = mock_open()
        handle.write.assert_any_call('{"key1": "value1"}\n')
        handle.write.assert_any_call('{"key2": "value2"}\n')


class TestProcessParquetSetEpisodeIndex:
    @patch("pandas.read_parquet")
    def test_standard_set_episode_index(self, mock_read_parquet):
        mock_df = pd.DataFrame({"episode_index": [0, 0, 0]})
        mock_read_parquet.return_value = mock_df

        parquet_processor = ProcessParquet("dummy/path.parquet")
        parquet_processor.set_episode_index(episode_index=5)

        expected_df = pd.DataFrame({"episode_index": [5, 5, 5]})
        pd.testing.assert_frame_equal(parquet_processor.df, expected_df)


class TestProcessParquetSetIndex:
    @patch("pandas.read_parquet")
    def test_standard_set_index_with_mapping(self, mock_read_parquet):
        mock_df = pd.DataFrame({"index": [0, 1, 2]})
        mock_read_parquet.return_value = mock_df

        episode_to_frame_index = {2: 100}
        parquet_processor = ProcessParquet("dummy/path.parquet")
        parquet_processor.set_index(
            new_index=2, episode_to_frame_index=episode_to_frame_index
        )

        expected_df = pd.DataFrame({"index": [100, 101, 102]})
        pd.testing.assert_frame_equal(parquet_processor.df, expected_df)

    @patch("pandas.read_parquet")
    def test_standard_set_index_without_mapping(self, mock_read_parquet):
        mock_df = pd.DataFrame({"index": [0, 1, 2]})
        mock_read_parquet.return_value = mock_df

        parquet_processor = ProcessParquet("dummy/path.parquet")
        parquet_processor.set_index(new_index=2, episode_to_frame_index={})

        expected_df = pd.DataFrame({"index": [6, 7, 8]})
        pd.testing.assert_frame_equal(parquet_processor.df, expected_df)

    @patch("pandas.read_parquet")
    def test_no_index_column(self, mock_read_parquet):
        mock_df = pd.DataFrame({"some_other_column": [0, 1, 2]})
        mock_read_parquet.return_value = mock_df

        parquet_processor = ProcessParquet("dummy/path.parquet")
        parquet_processor.set_index(new_index=3, episode_to_frame_index={})

        expected_df = pd.DataFrame({"some_other_column": [0, 1, 2]})
        pd.testing.assert_frame_equal(parquet_processor.df, expected_df)


class TestProcessParquetSetTaskIndexMap:
    @patch("pandas.read_parquet")
    def test_standard_set_task_index_map(self, mock_read_parquet):
        mock_df = pd.DataFrame({"task_key": ["task1", "task2", "task3"]})
        mock_read_parquet.return_value = mock_df

        task_index_map = {"task1": 10, "task2": 20, "task3": 30}

        parquet_processor = ProcessParquet("dummy/path.parquet")
        parquet_processor.set_task_index_map("task_key", task_index_map)

        expected_df = pd.DataFrame({"task_key": [10, 20, 30]})
        pd.testing.assert_frame_equal(parquet_processor.df, expected_df)

    @patch("pandas.read_parquet")
    def test_no_task_key_column(self, mock_read_parquet):
        mock_df = pd.DataFrame({"another_column": ["task1", "task2", "task3"]})
        mock_read_parquet.return_value = mock_df

        task_index_map = {"task1": 10, "task2": 20, "task3": 30}

        parquet_processor = ProcessParquet("dummy/path.parquet")
        parquet_processor.set_task_index_map("task_key", task_index_map)

        expected_df = pd.DataFrame({"another_column": ["task1", "task2", "task3"]})
        pd.testing.assert_frame_equal(parquet_processor.df, expected_df)

    @patch("pandas.read_parquet")
    def test_empty_columns(self, mock_read_parquet):
        mock_df = pd.DataFrame(columns=[])
        mock_read_parquet.return_value = mock_df

        task_index_map = {"task1": 10, "task2": 20, "task3": 30}

        parquet_processor = ProcessParquet("dummy/path.parquet")
        parquet_processor.set_task_index_map("task_key", task_index_map)

        expected_df = pd.DataFrame(columns=[])
        pd.testing.assert_frame_equal(parquet_processor.df, expected_df)


class TestProcessParquetSetStringBasedMapping:
    @patch("pandas.read_parquet")
    def test_standard_set_string_based_mapping(self, mock_read_parquet):
        mock_df = pd.DataFrame({"task_key": [1, 2, 3]})
        mock_read_parquet.return_value = mock_df

        local_task_map = {1: "task1", 2: "task2", 3: "task3"}
        task_string_to_new_index = {"task1": 100, "task2": 200, "task3": 300}

        parquet_processor = ProcessParquet("dummy/path.parquet")
        parquet_processor.set_string_based_mapping(
            "task_key", local_task_map, task_string_to_new_index
        )

        expected_df = pd.DataFrame({"task_key": [100, 200, 300]})
        pd.testing.assert_frame_equal(parquet_processor.df, expected_df)

    @patch("pandas.read_parquet")
    def test_no_task_key_column(self, mock_read_parquet):
        mock_df = pd.DataFrame({"another_column": [1, 2, 3]})
        mock_read_parquet.return_value = mock_df

        local_task_map = {1: "task1", 2: "task2", 3: "task3"}
        task_string_to_new_index = {"task1": 100, "task2": 200, "task3": 300}

        parquet_processor = ProcessParquet("dummy/path.parquet")
        parquet_processor.set_string_based_mapping(
            "task_key", local_task_map, task_string_to_new_index
        )

        expected_df = pd.DataFrame({"another_column": [1, 2, 3]})
        pd.testing.assert_frame_equal(parquet_processor.df, expected_df)

    @patch("pandas.read_parquet")
    def test_value_not_in_mapping(self, mock_read_parquet):
        mock_df = pd.DataFrame({"task_key": [1, 4, 2]})
        mock_read_parquet.return_value = mock_df

        local_task_map = {1: "task1", 2: "task2", 3: "task3"}
        task_string_to_new_index = {"task1": 100, "task2": 200, "task3": 300}

        parquet_processor = ProcessParquet("dummy/path.parquet")
        parquet_processor.set_string_based_mapping(
            "task_key", local_task_map, task_string_to_new_index
        )

        expected_df = pd.DataFrame({"task_key": [100, -1, 200]})
        pd.testing.assert_frame_equal(parquet_processor.df, expected_df)

    @patch("pandas.read_parquet")
    def test_empty_columns(self, mock_read_parquet):
        mock_df = pd.DataFrame(columns=[])
        mock_read_parquet.return_value = mock_df

        local_task_map = {1: "task1", 2: "task2", 3: "task3"}
        task_string_to_new_index = {"task1": 100, "task2": 200, "task3": 300}

        parquet_processor = ProcessParquet("dummy/path.parquet")
        parquet_processor.set_string_based_mapping(
            "task_key", local_task_map, task_string_to_new_index
        )

        expected_df = pd.DataFrame(columns=[])
        pd.testing.assert_frame_equal(parquet_processor.df, expected_df)

    @patch("pandas.read_parquet")
    def test_non_null_column(self, mock_read_parquet):
        mock_df = pd.DataFrame({"task_key": [np.nan]})
        mock_read_parquet.return_value = mock_df

        local_task_map = {np.nan: "task_nan", 2: "task2", 3: "task3"}
        task_string_to_new_index = {"task_nan": -1, "task2": 200, "task3": 300}

        parquet_processor = ProcessParquet("dummy/path.parquet")
        parquet_processor.set_string_based_mapping(
            "task_key", local_task_map, task_string_to_new_index
        )

        expected_df = pd.DataFrame({"task_key": [np.nan]})
        pd.testing.assert_frame_equal(parquet_processor.df, expected_df)

    @patch("pandas.read_parquet")
    def test_ndarray_isinstance_true(self, mock_read_parquet):
        mock_df = pd.DataFrame(
            {"task_key": [np.array([1]), np.array([2]), np.array([3])]}
        )
        mock_read_parquet.return_value = mock_df

        local_task_map = {1: "task1", 2: "task2", 3: "task3"}
        task_string_to_new_index = {"task1": 100, "task2": 200, "task3": 300}

        parquet_processor = ProcessParquet("dummy/path.parquet")
        parquet_processor.set_string_based_mapping(
            "task_key", local_task_map, task_string_to_new_index
        )

        expected_df = pd.DataFrame({"task_key": [100, 200, 300]})
        pd.testing.assert_frame_equal(parquet_processor.df, expected_df)


class TestProcessParquetToParquet:
    @patch("pandas.read_parquet")
    @patch("os.makedirs")
    @patch("pandas.DataFrame.to_parquet")
    def test_standard_to_parquet(
        self, mock_to_parquet, mock_makedirs, mock_read_parquet
    ):
        mock_df = pd.DataFrame({"index": [0, 1, 2], "value": [10, 20, 30]})
        mock_read_parquet.return_value = mock_df

        parquet_processor = ProcessParquet("dummy/path.parquet")

        output_folder = "output/directory"
        dest_path = parquet_processor.to_parquet(
            new_index=0, chunks_size=5, output_folder=output_folder
        )

        expected_chunk_dir = os.path.join(output_folder, "data", "chunk-000")
        expected_dest_path = os.path.join(expected_chunk_dir, "episode_000000.parquet")

        assert dest_path == expected_dest_path
        mock_makedirs.assert_called_once_with(expected_chunk_dir, exist_ok=True)
        mock_to_parquet.assert_called_once_with(expected_dest_path, index=False)

    @patch("pandas.read_parquet")
    @patch("os.makedirs")
    @patch("pandas.DataFrame.to_parquet")
    def test_chunk_index_calculation(
        self, mock_to_parquet, mock_makedirs, mock_read_parquet
    ):
        mock_df = pd.DataFrame({"index": [0, 1, 2], "value": [10, 20, 30]})
        mock_read_parquet.return_value = mock_df

        parquet_processor = ProcessParquet("dummy/path.parquet")

        output_folder = "output/directory"
        dest_path = parquet_processor.to_parquet(
            new_index=12, chunks_size=5, output_folder=output_folder
        )

        expected_chunk_dir = os.path.join(output_folder, "data", "chunk-002")
        expected_dest_path = os.path.join(expected_chunk_dir, "episode_000012.parquet")

        assert dest_path == expected_dest_path
        mock_makedirs.assert_called_once_with(expected_chunk_dir, exist_ok=True)
        mock_to_parquet.assert_called_once_with(expected_dest_path, index=False)

    @patch("pandas.read_parquet")
    @patch("os.makedirs")
    @patch("pandas.DataFrame.to_parquet")
    def test_empty_dataframe(self, mock_to_parquet, mock_makedirs, mock_read_parquet):
        mock_df = pd.DataFrame(columns=["index", "value"])
        mock_read_parquet.return_value = mock_df

        parquet_processor = ProcessParquet("dummy/path.parquet")

        output_folder = "output/directory"
        dest_path = parquet_processor.to_parquet(
            new_index=0, chunks_size=5, output_folder=output_folder
        )

        expected_chunk_dir = os.path.join(output_folder, "data", "chunk-000")
        expected_dest_path = os.path.join(expected_chunk_dir, "episode_000000.parquet")

        assert dest_path == expected_dest_path
        mock_makedirs.assert_called_once_with(expected_chunk_dir, exist_ok=True)
        mock_to_parquet.assert_called_once_with(expected_dest_path, index=False)


class TestProcessVideoCopyVideos:
    @patch("os.makedirs")
    @patch("shutil.copy")
    def test_standard_copy_videos(self, mock_copy, mock_makedirs):
        source_video_path = "source/videos/video1.mp4"
        dest_video_path = "dest/videos/video1.mp4"

        result = ProcessVideo.copy_videos(source_video_path, dest_video_path)

        assert result is True
        mock_makedirs.assert_called_once_with(
            os.path.dirname(dest_video_path), exist_ok=True
        )
        mock_copy.assert_called_once_with(source_video_path, dest_video_path)

    @patch("os.makedirs")
    @patch("shutil.copy")
    def test_without_copy_video(self, mock_copy, mock_makedirs):
        source_video_path = "source/videos/video1.mp4"
        dest_video_path = "dest/videos/video1.mp4"

        result = ProcessVideo.copy_videos(
            source_video_path, dest_video_path, without_copy_video=True
        )

        assert result is False
        mock_makedirs.assert_called_once_with(
            os.path.dirname(dest_video_path), exist_ok=True
        )
        mock_copy.assert_not_called()

    @patch("os.makedirs")
    @patch("shutil.copy")
    def test_directory_already_exists(self, mock_copy, mock_makedirs):
        source_video_path = "source/videos/video1.mp4"
        dest_video_path = "dest/videos/video1.mp4"

        result = ProcessVideo.copy_videos(source_video_path, dest_video_path)

        assert result is True
        mock_makedirs.assert_called_once_with(
            os.path.dirname(dest_video_path), exist_ok=True
        )
        mock_copy.assert_called_once_with(source_video_path, dest_video_path)


class TestPackageToolInit:
    def test_standard(self):
        package_tool = PackageTool()
        assert package_tool.output_prefix == ""
        assert package_tool.aws_tmp_dir is None
        assert package_tool.robot_types is None
        assert package_tool.robot_ids is None
        assert package_tool.labels is None
        assert package_tool.location_names is None
        assert package_tool.short_horizon_tasks is None
        assert not package_tool.use_aws
        assert package_tool.secret == ""
        assert package_tool.source_bucket == ""
        assert package_tool.output_bucket == ""

    def test_use_aws_is_false(self):
        package_tool = PackageTool(
            robot_types="test_robot_types",
            robot_ids="test_robot_ids",
            labels="test_labels",
            location_names="test_location_names",
            short_horizon_tasks="test_short_horizon_tasks",
            use_aws=False,
            secret="test_secret",
            source_bucket="test_source_bucket",
            output_bucket="test_output_bucket",
        )
        assert package_tool.output_prefix == ""
        assert package_tool.aws_tmp_dir is None
        assert package_tool.robot_types == "test_robot_types"
        assert package_tool.robot_ids == "test_robot_ids"
        assert package_tool.labels == "test_labels"
        assert package_tool.location_names == "test_location_names"
        assert package_tool.short_horizon_tasks == "test_short_horizon_tasks"
        assert not package_tool.use_aws
        assert package_tool.secret == "test_secret"
        assert package_tool.source_bucket == "test_source_bucket"
        assert package_tool.output_bucket == "test_output_bucket"

    @patch("hsr_data_converter.commands.package.package.tempfile.mkdtemp")
    def test_use_aws_is_true(self, mock_mkdtemp):
        mock_mkdtemp.return_value = "./tmp/mkdtemp"
        package_tool = PackageTool(
            robot_types="test_robot_types",
            robot_ids="test_robot_ids",
            labels="test_labels",
            location_names="test_location_names",
            short_horizon_tasks="test_short_horizon_tasks",
            use_aws=True,
            secret="test_secret",
            source_bucket="test_source_bucket",
            output_bucket="test_output_bucket",
        )
        assert package_tool.output_prefix == ""
        assert package_tool.aws_tmp_dir == Path("./tmp/mkdtemp")
        assert package_tool.robot_types == "test_robot_types"
        assert package_tool.robot_ids == "test_robot_ids"
        assert package_tool.labels == "test_labels"
        assert package_tool.location_names == "test_location_names"
        assert package_tool.short_horizon_tasks == "test_short_horizon_tasks"
        assert package_tool.use_aws
        assert package_tool.secret == "test_secret"
        assert package_tool.source_bucket == "test_source_bucket"
        assert package_tool.output_bucket == "test_output_bucket"

    def test_error_secret_is_none(self):
        with pytest.raises(Exception) as e:
            PackageTool(
                robot_types="test_robot_types",
                robot_ids="test_robot_ids",
                labels="test_labels",
                location_names="test_location_names",
                short_horizon_tasks="test_short_horizon_tasks",
                use_aws=True,
                secret=None,
                source_bucket="test_source_bucket",
                output_bucket="test_output_bucket",
            )
        assert str(e.value) == "if use_aws is True, secret is Required."

    def test_error_source_bucket_is_none(self):
        with pytest.raises(Exception) as e:
            PackageTool(
                robot_types="test_robot_types",
                robot_ids="test_robot_ids",
                labels="test_labels",
                location_names="test_location_names",
                short_horizon_tasks="test_short_horizon_tasks",
                use_aws=True,
                secret="test_secret",
                source_bucket=None,
                output_bucket="test_output_bucket",
            )
        assert str(e.value) == "if use_aws is True, source_bucket is Required."

    def test_error_output_bucket_is_none(self):
        with pytest.raises(Exception) as e:
            PackageTool(
                robot_types="test_robot_types",
                robot_ids="test_robot_ids",
                labels="test_labels",
                location_names="test_location_names",
                short_horizon_tasks="test_short_horizon_tasks",
                use_aws=True,
                secret="test_secret",
                source_bucket="test_source_bucket",
                output_bucket=None,
            )
        assert str(e.value) == "if use_aws is True, output_bucket is Required."


class TestPackageToolFinalize:
    @patch("hsr_data_converter.commands.package.package.tempfile.mkdtemp")
    def test_standard_finalize(self, mock_mkdtemp):
        mock_mkdtemp.return_value = "./tmp/mkdtemp"
        package_tool = PackageTool(
            use_aws=True,
            secret="my_secret",
            source_bucket="my_bucket",
            output_bucket="output_bucket",
        )
        package_tool.aws_tmp_dir = Path("/tmp/some_temp_dir")
        with patch("shutil.rmtree") as mock_rmtree:
            package_tool.finalize()
            mock_rmtree.assert_called_once_with(package_tool.aws_tmp_dir)

    def test_finalize_without_aws(self):
        package_tool = PackageTool(use_aws=False)
        package_tool.aws_tmp_dir = None
        with patch("shutil.rmtree") as mock_rmtree:
            package_tool.finalize()
            assert not mock_rmtree.called

    @patch("hsr_data_converter.commands.package.package.tempfile.mkdtemp")
    def test_finalize_without_temp_dir(self, mock_mkdtemp):
        mock_mkdtemp.return_value = "./tmp/mkdtemp"
        package_tool = PackageTool(
            use_aws=True,
            secret="my_secret",
            source_bucket="my_bucket",
            output_bucket="output_bucket",
        )
        package_tool.aws_tmp_dir = None
        with patch("shutil.rmtree") as mock_rmtree:
            package_tool.finalize()
            assert not mock_rmtree.called


class TestPackageToolInitSourceFolders:
    def test_standard_init_source_folders_without_aws(self):
        package_tool = PackageTool(use_aws=False)
        source_folders = ["folder1", "folder2"]
        output_folder = "output"
        actual_source_folders, actual_output_folder = package_tool._init_source_folders(
            source_folders, output_folder
        )
        assert actual_source_folders == source_folders
        assert actual_output_folder == output_folder

    @patch("hsr_data_converter.commands.package.package.tempfile.mkdtemp")
    @patch("pathlib.Path.mkdir")
    @patch("hsr_data_converter.utils.aws_helper.AWSHelper.download_s3_directory")
    def test_init_source_folders_with_aws(
        self, mock_download_s3_directory, mock_mkdir, mock_mkdtemp
    ):
        mock_mkdtemp.return_value = "./tmp/mkdtemp"
        package_tool = PackageTool(
            use_aws=True,
            secret="my_secret",
            source_bucket="my_bucket",
            output_bucket="output_bucket",
        )
        package_tool.aws_helper = AWSHelper(package_tool.secret)

        source_folders = ["folder1", "folder2"]
        output_folder = "output"
        expected_source_folders = [
            str(package_tool.aws_tmp_dir / folder) for folder in source_folders
        ]

        actual_source_folders, actual_output_folder = package_tool._init_source_folders(
            source_folders, output_folder
        )

        assert actual_source_folders == expected_source_folders
        assert isinstance(actual_output_folder, str)
        mock_download_s3_directory.assert_any_call(
            bucket_name=package_tool.source_bucket,
            s3_prefix="folder1",
            local_dir=Path(package_tool.aws_tmp_dir / "folder1"),
        )
        mock_download_s3_directory.assert_any_call(
            bucket_name=package_tool.source_bucket,
            s3_prefix="folder2",
            local_dir=Path(package_tool.aws_tmp_dir / "folder2"),
        )


class TestPackageToolUploadMergedDatasets:
    @patch("hsr_data_converter.commands.package.package.tempfile.mkdtemp")
    @patch("hsr_data_converter.utils.aws_helper.AWSHelper.upload_directory_to_s3")
    def test_standard_upload(self, mock_upload_directory_to_s3, mock_mkdtemp):
        mock_mkdtemp.return_value = "./tmp/mkdtemp"
        package_tool = PackageTool(
            use_aws=True,
            secret="my_secret",
            source_bucket="my_bucket",
            output_bucket="output_bucket",
        )
        package_tool.aws_helper = MagicMock()
        output_folder = "/path/to/output"
        package_tool._upload_merged_datasets(output_folder)
        package_tool.aws_helper.upload_directory_to_s3.assert_called_once_with(
            local_dir=Path(output_folder),
            bucket_name="output_bucket",
            s3_prefix=package_tool.output_prefix,
        )

    @patch("hsr_data_converter.utils.aws_helper.AWSHelper.upload_directory_to_s3")
    def test_upload_without_aws(self, mock_upload_directory_to_s3):
        package_tool = PackageTool(use_aws=False)
        package_tool.aws_helper = MagicMock()
        output_folder = "/path/to/output"
        package_tool._upload_merged_datasets(output_folder)
        mock_upload_directory_to_s3.assert_not_called()


class TestPackageToolValidateFPS:
    """Test _validate_fps method"""

    def test_consistent_fps(self, tmp_path):
        """Test with consistent FPS across datasets"""
        # Create test datasets with same FPS
        for i in range(2):
            dataset_path = tmp_path / f"dataset{i}"
            meta_path = dataset_path / "meta"
            meta_path.mkdir(parents=True)

            info = {"fps": 10}
            with open(meta_path / "info.json", "w") as f:
                json.dump(info, f)

        package_tool = PackageTool()
        is_valid, fps, error_msg = package_tool._validate_fps(
            [str(tmp_path / "dataset0"), str(tmp_path / "dataset1")]
        )

        assert is_valid is True
        assert fps == 10
        assert error_msg is None

    def test_inconsistent_fps(self, tmp_path):
        """Test with inconsistent FPS across datasets"""
        # Create test datasets with different FPS
        for i, fps_val in enumerate([10, 20]):
            dataset_path = tmp_path / f"dataset{i}"
            meta_path = dataset_path / "meta"
            meta_path.mkdir(parents=True)

            info = {"fps": fps_val}
            with open(meta_path / "info.json", "w") as f:
                json.dump(info, f)

        package_tool = PackageTool()
        is_valid, fps, error_msg = package_tool._validate_fps(
            [str(tmp_path / "dataset0"), str(tmp_path / "dataset1")]
        )

        assert is_valid is False
        assert fps is None
        assert "Inconsistent FPS" in error_msg

    def test_missing_info_json(self, tmp_path):
        """Test with missing info.json"""
        dataset_path = tmp_path / "dataset0"
        dataset_path.mkdir(parents=True)

        package_tool = PackageTool()
        is_valid, fps, error_msg = package_tool._validate_fps([str(dataset_path)])

        assert is_valid is False
        assert fps is None
        assert "info.json not found" in error_msg

    def test_missing_fps_field(self, tmp_path):
        """Test with missing fps field in info.json"""
        dataset_path = tmp_path / "dataset0"
        meta_path = dataset_path / "meta"
        meta_path.mkdir(parents=True)

        info = {}  # No fps field
        with open(meta_path / "info.json", "w") as f:
            json.dump(info, f)

        package_tool = PackageTool()
        is_valid, fps, error_msg = package_tool._validate_fps([str(dataset_path)])

        assert is_valid is False
        assert fps is None
        assert "FPS not specified" in error_msg


class TestPackageToolValidateFeatureShapes:
    """Test _validate_feature_shapes method"""

    def test_consistent_shapes(self, tmp_path):
        """Test with consistent action.absolute shapes"""
        for i in range(2):
            dataset_path = tmp_path / f"dataset{i}"
            meta_path = dataset_path / "meta"
            meta_path.mkdir(parents=True)

            info = {
                "features": {"action.absolute": {"shape": [14], "dtype": "float32"}}
            }
            with open(meta_path / "info.json", "w") as f:
                json.dump(info, f)

        package_tool = PackageTool()
        is_valid, error_msg = package_tool._validate_feature_shapes(
            [str(tmp_path / "dataset0"), str(tmp_path / "dataset1")]
        )

        assert is_valid is True
        assert error_msg is None

    def test_inconsistent_shapes(self, tmp_path):
        """Test with inconsistent action.absolute shapes"""
        for i, shape in enumerate([[14], [18]]):
            dataset_path = tmp_path / f"dataset{i}"
            meta_path = dataset_path / "meta"
            meta_path.mkdir(parents=True)

            info = {
                "features": {"action.absolute": {"shape": shape, "dtype": "float32"}}
            }
            with open(meta_path / "info.json", "w") as f:
                json.dump(info, f)

        package_tool = PackageTool()
        is_valid, error_msg = package_tool._validate_feature_shapes(
            [str(tmp_path / "dataset0"), str(tmp_path / "dataset1")]
        )

        assert is_valid is False
        assert "Inconsistent action.absolute shapes" in error_msg

    def test_missing_features(self, tmp_path):
        """Test with missing features field"""
        dataset_path = tmp_path / "dataset0"
        meta_path = dataset_path / "meta"
        meta_path.mkdir(parents=True)

        info = {}  # No features field
        with open(meta_path / "info.json", "w") as f:
            json.dump(info, f)

        package_tool = PackageTool()
        is_valid, error_msg = package_tool._validate_feature_shapes([str(dataset_path)])

        assert is_valid is False
        assert "Features not specified" in error_msg

    def test_missing_action_absolute(self, tmp_path):
        """Test with missing action absolute field"""
        dataset_path = tmp_path / "dataset"
        meta_path = dataset_path / "meta"
        meta_path.mkdir(parents=True)

        info = {"features": {}}
        with open(meta_path / "info.json", "w") as f:
            json.dump(info, f)

        package_tool = PackageTool()
        is_valid, error_msg = package_tool._validate_feature_shapes(
            [str(tmp_path / "dataset")]
        )

        assert is_valid is False
        assert "action.absolute feature not found in " in error_msg
        assert "/meta/info.json" in error_msg

    def test_not_exist_info_json(self, tmp_path):
        """Test with not exist info.json"""
        package_tool = PackageTool()
        is_valid, error_msg = package_tool._validate_feature_shapes(
            [str(tmp_path / "dataset0"), str(tmp_path / "dataset1")]
        )

        assert is_valid is False
        assert "info.json not found in " in error_msg

    def test_throw_exception(self, tmp_path):
        """Test with throw exception"""
        dataset_path = tmp_path / "dataset"
        meta_path = dataset_path / "meta"
        meta_path.mkdir(parents=True)

        with open(meta_path / "info.json", "w") as f:
            f.write("error")

        package_tool = PackageTool()
        is_valid, error_msg = package_tool._validate_feature_shapes(
            [str(tmp_path / "dataset")]
        )

        assert not is_valid
        assert "Error reading " in error_msg
        assert "/meta/info.json: " in error_msg


class TestPackageToolGetErrorEpisodesFromQAReport:
    """Test _get_error_episodes_from_qa_report method"""

    def test_with_errors(self, tmp_path):
        """Test with qa_report.json containing errors"""
        qa_report = [
            {"episode_index": 0, "errors": []},
            {"episode_index": 1, "errors": ["error1"]},
            {"episode_index": 2, "errors": []},
            {"episode_index": 3, "errors": ["error2", "error3"]},
        ]

        qa_report_path = tmp_path / "qa_report.json"
        with open(qa_report_path, "w") as f:
            json.dump(qa_report, f)

        package_tool = PackageTool()
        error_indices = package_tool._get_error_episodes_from_qa_report(str(tmp_path))

        assert set(error_indices) == {1, 3}

    def test_no_errors(self, tmp_path):
        """Test with qa_report.json containing no errors"""
        qa_report = [
            {"episode_index": 0, "errors": []},
            {"episode_index": 1, "errors": []},
        ]

        qa_report_path = tmp_path / "qa_report.json"
        with open(qa_report_path, "w") as f:
            json.dump(qa_report, f)

        package_tool = PackageTool()
        error_indices = package_tool._get_error_episodes_from_qa_report(str(tmp_path))

        assert error_indices == []

    def test_missing_file(self, tmp_path):
        """Test with missing qa_report.json"""
        package_tool = PackageTool()
        error_indices = package_tool._get_error_episodes_from_qa_report(str(tmp_path))

        assert error_indices == []

    def test_not_json_file(self, tmp_path):
        """Test with qa_report.json is not json file"""
        qa_report_path = tmp_path / "qa_report.json"
        with open(qa_report_path, "w") as f:
            f.write("error")
        package_tool = PackageTool()
        error_indices = package_tool._get_error_episodes_from_qa_report(str(tmp_path))

        assert error_indices == []

    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    def test_other_error(self, mock_load_jsonl_data, tmp_path):
        """Test with other error"""
        qa_report = [
            {"episode_index": 0, "errors": []},
            {"episode_index": 1, "errors": []},
        ]

        qa_report_path = tmp_path / "qa_report.json"
        with open(qa_report_path, "w") as f:
            json.dump(qa_report, f)
        mock_load_jsonl_data.side_effect = Exception("error")
        package_tool = PackageTool()
        error_indices = package_tool._get_error_episodes_from_qa_report(str(tmp_path))

        assert error_indices == []


class TestPackageToolGetErrorEpisodesFromParquet:
    """Test _get_error_episodes_from_parquet method"""

    def test_with_failures(self, tmp_path):
        """Test with parquet files containing success_primitive_action=False"""
        data_dir = tmp_path / "data" / "chunk-000"
        data_dir.mkdir(parents=True)

        # Create parquet files
        # Episode 0: all success
        df0 = pd.DataFrame(
            {
                "episode_index": [0, 0, 0],
                "success_primitive_action": [True, True, True],
            }
        )
        df0.to_parquet(data_dir / "episode_000000.parquet", index=False)

        # Episode 1: has failure
        df1 = pd.DataFrame(
            {
                "episode_index": [1, 1, 1],
                "success_primitive_action": [True, False, True],
            }
        )
        df1.to_parquet(data_dir / "episode_000001.parquet", index=False)

        # Episode 2: all success
        df2 = pd.DataFrame(
            {
                "episode_index": [2, 2],
                "success_primitive_action": [True, True],
            }
        )
        df2.to_parquet(data_dir / "episode_000002.parquet", index=False)

        package_tool = PackageTool()
        error_indices = package_tool._get_error_episodes_from_parquet(str(tmp_path))

        assert set(error_indices) == {1}

    def test_all_success(self, tmp_path):
        """Test with all episodes successful"""
        data_dir = tmp_path / "data" / "chunk-000"
        data_dir.mkdir(parents=True)

        df = pd.DataFrame(
            {
                "episode_index": [0, 0, 0],
                "success_primitive_action": [True, True, True],
            }
        )
        df.to_parquet(data_dir / "episode_000000.parquet", index=False)

        package_tool = PackageTool()
        error_indices = package_tool._get_error_episodes_from_parquet(str(tmp_path))

        assert error_indices == []

    def test_missing_data_dir(self, tmp_path):
        """Test with missing data directory"""
        package_tool = PackageTool()
        error_indices = package_tool._get_error_episodes_from_parquet(str(tmp_path))

        assert error_indices == []

    @patch("hsr_data_converter.commands.package.package.pd.read_parquet")
    def test_read_parquet_error(self, mock_pd_read_parquet, tmp_path):
        """Test with read parquet error"""
        data_dir = tmp_path / "data" / "chunk-000"
        data_dir.mkdir(parents=True)

        df = pd.DataFrame(
            {
                "episode_index": [0, 0, 0],
                "success_primitive_action": [True, True, True],
            }
        )
        df.to_parquet(data_dir / "episode_000000.parquet", index=False)
        mock_pd_read_parquet.side_effect = Exception("error")
        package_tool = PackageTool()
        error_indices = package_tool._get_error_episodes_from_parquet(str(tmp_path))

        assert error_indices == []


class TestPackageToolGetFilteredEpisodeIndicesExcludingRobotTypes:
    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    def test_standard_robot_type_filter(self, mock_load_jsonl_data):
        mock_load_jsonl_data.side_effect = [
            {"robot_type": "HSR"},
            [{"episode_index": 0}, {"episode_index": 1}],
        ]

        tool = PackageTool(robot_types=["ROBOT_A"])
        result = tool._get_filtered_episode_indices_excluding_robot_types(
            "fake_dataset_path"
        )
        assert result == [0, 1]
        mock_load_jsonl_data.assert_any_call(
            os.path.join("fake_dataset_path", "meta", "info.json")
        )
        mock_load_jsonl_data.assert_any_call(
            os.path.join("fake_dataset_path", "meta", "episodes.jsonl"), True
        )

    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    def test_excluding_when_robot_type_is_included(self, mock_load_jsonl_data):
        mock_load_jsonl_data.return_value = {"robot_type": "HSR"}

        tool = PackageTool(robot_types=["HSR"])
        result = tool._get_filtered_episode_indices_excluding_robot_types(
            "fake_dataset_path"
        )
        assert result == []
        mock_load_jsonl_data.assert_called_once_with(
            os.path.join("fake_dataset_path", "meta", "info.json")
        )

    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    def test_error_loading_info_json(self, mock_load_jsonl_data):
        mock_load_jsonl_data.side_effect = FileNotFoundError()

        tool = PackageTool(robot_types=["ROBOT_A"])
        with pytest.raises(FileNotFoundError):
            tool._get_filtered_episode_indices_excluding_robot_types(
                "fake_dataset_path"
            )
        mock_load_jsonl_data.assert_called_once_with(
            os.path.join("fake_dataset_path", "meta", "info.json")
        )

    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    def test_missing_robot_type_in_info(self, mock_load_jsonl_data):
        mock_load_jsonl_data.return_value = {}

        tool = PackageTool(robot_types=["ROBOT_A"])
        result = tool._get_filtered_episode_indices_excluding_robot_types(
            "fake_dataset_path"
        )
        assert result == []

    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    def test_no_robot_type_provided(self, mock_load_jsonl_data):
        mock_load_jsonl_data.return_value = {"robot_type": "HSR"}

        tool = PackageTool(robot_types=None)
        result = tool._get_filtered_episode_indices_excluding_robot_types(
            "fake_dataset_path"
        )
        assert result == []


class TestPackageToolGetFilteredEpisodeIndicesByRobotIds:
    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    def test_standard_robot_id_filter(self, mock_load_jsonl_data):
        mock_load_jsonl_data.return_value = [
            {"episode_index": 0, "hsr_id": "ID_123"},
            {"episode_index": 1, "hsr_id": "ID_456"},
        ]

        tool = PackageTool(robot_ids=["ID_123"])
        result = tool._get_filtered_episode_indices_by_robot_ids("fake_dataset_path")
        assert result == [1]
        mock_load_jsonl_data.assert_called_once_with(
            os.path.join("fake_dataset_path", "meta", "episodes.jsonl"), True
        )

    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    def test_empty_robot_id_list(self, mock_load_jsonl_data):
        mock_load_jsonl_data.return_value = [
            {"episode_index": 0, "hsr_id": "ID_123"},
            {"episode_index": 1, "hsr_id": "ID_456"},
        ]

        tool = PackageTool(robot_ids=None)
        result = tool._get_filtered_episode_indices_by_robot_ids("fake_dataset_path")
        assert result == []
        assert not mock_load_jsonl_data.called

    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    def test_no_hsr_id_key_in_episode(self, mock_load_jsonl_data):
        mock_load_jsonl_data.return_value = [{"episode_index": 0}, {"episode_index": 1}]

        tool = PackageTool(robot_ids=["ID_123"])
        result = tool._get_filtered_episode_indices_by_robot_ids("fake_dataset_path")
        assert result == [0, 1]
        mock_load_jsonl_data.assert_called_once_with(
            os.path.join("fake_dataset_path", "meta", "episodes.jsonl"), True
        )

    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    def test_error_loading_episodes_jsonl(self, mock_load_jsonl_data):
        mock_load_jsonl_data.side_effect = FileNotFoundError()

        tool = PackageTool(robot_ids=["ID_123"])
        with pytest.raises(FileNotFoundError):
            tool._get_filtered_episode_indices_by_robot_ids("fake_dataset_path")
        mock_load_jsonl_data.assert_called_once_with(
            os.path.join("fake_dataset_path", "meta", "episodes.jsonl"), True
        )


class TestPackageToolGetFilteredEpisodeIndicesByLabels:
    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    def test_standard_label_filter(self, mock_load_jsonl_data):
        mock_load_jsonl_data.return_value = [
            {"episode_index": 0, "label": "label_1"},
            {"episode_index": 1, "label": "label_2"},
        ]

        tool = PackageTool(labels=["label_1"])
        result = tool._get_filtered_episode_indices_by_labels("fake_dataset_path")
        assert result == [1]
        mock_load_jsonl_data.assert_called_once_with(
            os.path.join("fake_dataset_path", "meta", "episodes.jsonl"), True
        )

    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    def test_empty_label_list(self, mock_load_jsonl_data):
        mock_load_jsonl_data.return_value = [
            {"episode_index": 0, "label": "label_1"},
            {"episode_index": 1, "label": "label_2"},
        ]

        tool = PackageTool(labels=None)
        result = tool._get_filtered_episode_indices_by_labels("fake_dataset_path")
        assert result == []
        assert not mock_load_jsonl_data.called

    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    def test_no_label_key_in_episode(self, mock_load_jsonl_data):
        mock_load_jsonl_data.return_value = [{"episode_index": 0}, {"episode_index": 1}]

        tool = PackageTool(labels=["label_1"])
        result = tool._get_filtered_episode_indices_by_labels("fake_dataset_path")
        assert result == [0, 1]
        mock_load_jsonl_data.assert_called_once_with(
            os.path.join("fake_dataset_path", "meta", "episodes.jsonl"), True
        )

    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    def test_error_loading_episodes_jsonl(self, mock_load_jsonl_data):
        mock_load_jsonl_data.side_effect = FileNotFoundError()

        tool = PackageTool(labels=["label_1"])
        with pytest.raises(FileNotFoundError):
            tool._get_filtered_episode_indices_by_labels("fake_dataset_path")
        mock_load_jsonl_data.assert_called_once_with(
            os.path.join("fake_dataset_path", "meta", "episodes.jsonl"), True
        )


class TestPackageToolGetFilteredEpisodeIndicesByLocationNames:
    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    def test_standard_location_name_filter(self, mock_load_jsonl_data):
        mock_load_jsonl_data.return_value = [
            {"episode_index": 0, "location_name": "location_1"},
            {"episode_index": 1, "location_name": "location_2"},
        ]

        tool = PackageTool(location_names=["location_1"])
        result = tool._get_filtered_episode_indices_by_location_names(
            "fake_dataset_path"
        )
        assert result == [1]
        mock_load_jsonl_data.assert_called_once_with(
            os.path.join("fake_dataset_path", "meta", "episodes.jsonl"), True
        )

    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    def test_empty_location_list(self, mock_load_jsonl_data):
        mock_load_jsonl_data.return_value = [
            {"episode_index": 0, "location_name": "location_1"},
            {"episode_index": 1, "location_name": "location_2"},
        ]

        tool = PackageTool(location_names=None)
        result = tool._get_filtered_episode_indices_by_location_names(
            "fake_dataset_path"
        )
        assert result == []
        assert not mock_load_jsonl_data.called

    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    def test_no_location_name_key_in_episode(self, mock_load_jsonl_data):
        mock_load_jsonl_data.return_value = [{"episode_index": 0}, {"episode_index": 1}]

        tool = PackageTool(location_names=["location_1"])
        result = tool._get_filtered_episode_indices_by_location_names(
            "fake_dataset_path"
        )
        assert result == [0, 1]
        mock_load_jsonl_data.assert_called_once_with(
            os.path.join("fake_dataset_path", "meta", "episodes.jsonl"), True
        )

    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    def test_error_loading_episodes_jsonl(self, mock_load_jsonl_data):
        mock_load_jsonl_data.side_effect = FileNotFoundError()

        tool = PackageTool(location_names=["location_1"])
        with pytest.raises(FileNotFoundError):
            tool._get_filtered_episode_indices_by_location_names("fake_dataset_path")
        mock_load_jsonl_data.assert_called_once_with(
            os.path.join("fake_dataset_path", "meta", "episodes.jsonl"), True
        )


class TestPackageToolGetFilteredEpisodeIndicesByShortHorizonTasks:
    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    def test_standard_short_horizon_task_filter(self, mock_load_jsonl_data):
        mock_load_jsonl_data.return_value = [
            {"episode_index": 0, "short_horizon_task": "task_1"},
            {"episode_index": 1, "short_horizon_task": "task_2"},
        ]

        tool = PackageTool(short_horizon_tasks=["task_1"])
        result = tool._get_filtered_episode_indices_by_short_horizon_tasks(
            "fake_dataset_path"
        )
        assert result == [1]
        mock_load_jsonl_data.assert_called_once_with(
            os.path.join("fake_dataset_path", "meta", "episodes.jsonl"), True
        )

    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    def test_empty_short_horizon_task_list(self, mock_load_jsonl_data):
        mock_load_jsonl_data.return_value = [
            {"episode_index": 0, "short_horizon_task": "task_1"},
            {"episode_index": 1, "short_horizon_task": "task_2"},
        ]

        tool = PackageTool(short_horizon_tasks=None)
        result = tool._get_filtered_episode_indices_by_short_horizon_tasks(
            "fake_dataset_path"
        )
        assert result == []
        assert not mock_load_jsonl_data.called

    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    def test_no_short_horizon_task_key_in_episode(self, mock_load_jsonl_data):
        mock_load_jsonl_data.return_value = [{"episode_index": 0}, {"episode_index": 1}]

        tool = PackageTool(short_horizon_tasks=["task_1"])
        result = tool._get_filtered_episode_indices_by_short_horizon_tasks(
            "fake_dataset_path"
        )
        assert result == [0, 1]
        mock_load_jsonl_data.assert_called_once_with(
            os.path.join("fake_dataset_path", "meta", "episodes.jsonl"), True
        )

    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    def test_error_loading_episodes_jsonl(self, mock_load_jsonl_data):
        mock_load_jsonl_data.side_effect = FileNotFoundError()

        tool = PackageTool(short_horizon_tasks=["task_1"])
        with pytest.raises(FileNotFoundError):
            tool._get_filtered_episode_indices_by_short_horizon_tasks(
                "fake_dataset_path"
            )
        mock_load_jsonl_data.assert_called_once_with(
            os.path.join("fake_dataset_path", "meta", "episodes.jsonl"), True
        )


class TestPackageToolGetFilteredEpisodeByRobotTypes:
    def test_standard_robot_type_filtering(self):
        package_tool = PackageTool()
        metas = {
            "episodes": [
                {"episode_index": 0, "hsr_id": "robot_a"},
                {"episode_index": 1, "hsr_id": "robot_b"},
                {"episode_index": 2, "hsr_id": "robot_c"},
            ]
        }
        parquet_file_paths = ["dummy_path"]
        filtered_robot_types = ["robot_a", "robot_c"]

        filtered_indices = package_tool._get_filtered_episode_by_robot_types(
            parquet_file_paths, metas, filtered_robot_types
        )
        assert filtered_indices == [0, 2]

    def test_error_no_matching_robot_type(self):
        package_tool = PackageTool()
        metas = {
            "episodes": [
                {"episode_index": 0, "hsr_id": "robot_a"},
                {"episode_index": 1, "hsr_id": "robot_b"},
                {"episode_index": 2, "hsr_id": "robot_c"},
            ]
        }
        parquet_file_paths = ["dummy_path"]
        filtered_robot_types = ["robot_d"]

        filtered_indices = package_tool._get_filtered_episode_by_robot_types(
            parquet_file_paths, metas, filtered_robot_types
        )
        assert filtered_indices == []


class TestPackageToolGetFilteredEpisodeByPrimitiveAction:
    @patch("hsr_data_converter.commands.package.package.pd.read_parquet")
    def test_standard_primitive_action_filtering(self, mock_read_parquet):
        mock_read_parquet.return_value = pd.DataFrame(
            {"episode_index": [0, 1, 2], "primitive_action_index": [100, 200, 300]}
        )

        package_tool = PackageTool()
        metas = {
            "tasks": [
                {"task_index": 100, "task": "action_a"},
                {"task_index": 200, "task": "action_b"},
                {"task_index": 300, "task": "action_c"},
            ]
        }
        parquet_file_paths = ["dummy_path_a", "dummy_path_c"]
        filtered_primitive_actions = ["action_a", "action_c"]

        filtered_indices = package_tool._get_filtered_episode_by_primitive_action(
            parquet_file_paths, metas, filtered_primitive_actions
        )
        assert filtered_indices == [0, 0]
        mock_read_parquet.assert_has_calls([call("dummy_path_a"), call("dummy_path_c")])

    @patch("hsr_data_converter.commands.package.package.pd.read_parquet")
    def test_error_no_matching_primitive_action(self, mock_read_parquet):
        mock_read_parquet.return_value = pd.DataFrame(
            {"episode_index": [0, 1, 2], "primitive_action_index": [100, 200, 300]}
        )

        package_tool = PackageTool()
        metas = {
            "tasks": [
                {"task_index": 100, "task": "action_a"},
                {"task_index": 200, "task": "action_b"},
                {"task_index": 300, "task": "action_c"},
            ]
        }
        parquet_file_paths = ["dummy_path"]
        filtered_primitive_actions = ["action_d"]

        filtered_indices = package_tool._get_filtered_episode_by_primitive_action(
            parquet_file_paths, metas, filtered_primitive_actions
        )
        assert filtered_indices == []
        mock_read_parquet.assert_called_once_with("dummy_path")


class TestPackageToolGetFilteredEpisodeByShortHorizonTask:
    @patch("hsr_data_converter.commands.package.package.pd.read_parquet")
    def test_standard_short_horizon_task_filtering(self, mock_read_parquet):
        mock_read_parquet.return_value = pd.DataFrame(
            {"episode_index": [0, 1, 2], "short_horizon_task_index": [10, 20, 30]}
        )

        package_tool = PackageTool()
        metas = {
            "tasks": [
                {"task_index": 10, "task": "task_a"},
                {"task_index": 20, "task": "task_b"},
                {"task_index": 30, "task": "task_c"},
            ]
        }
        parquet_file_paths = ["dummy_path_a", "dummy_path_c"]
        filtered_short_horizon_tasks = ["task_b", "task_c"]

        filtered_indices = package_tool._get_filtered_episode_by_short_horizon_task(
            parquet_file_paths, metas, filtered_short_horizon_tasks
        )
        assert filtered_indices == [0, 0]
        mock_read_parquet.assert_has_calls([call("dummy_path_a"), call("dummy_path_c")])

    @patch("hsr_data_converter.commands.package.package.pd.read_parquet")
    def test_error_no_matching_short_horizon_task(self, mock_read_parquet):
        mock_read_parquet.return_value = pd.DataFrame(
            {"episode_index": [0, 1, 2], "short_horizon_task_index": [10, 20, 30]}
        )

        package_tool = PackageTool()
        metas = {
            "tasks": [
                {"task_index": 10, "task": "task_a"},
                {"task_index": 20, "task": "task_b"},
                {"task_index": 30, "task": "task_c"},
            ]
        }
        parquet_file_paths = ["dummy_path"]
        filtered_short_horizon_tasks = ["task_d"]

        filtered_indices = package_tool._get_filtered_episode_by_short_horizon_task(
            parquet_file_paths, metas, filtered_short_horizon_tasks
        )
        assert filtered_indices == []
        mock_read_parquet.assert_called_once_with("dummy_path")


class TestPackageToolFilterEpisodes:
    """Test _filter_episodes method"""

    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._get_filtered_episode_indices_excluding_robot_types"
    )
    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._get_filtered_episode_indices_by_robot_ids"
    )
    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._get_filtered_episode_indices_by_labels"
    )
    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._get_filtered_episode_indices_by_location_names"
    )
    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._get_filtered_episode_indices_by_short_horizon_tasks"
    )
    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._get_error_episodes_from_qa_report"
    )
    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._get_error_episodes_from_parquet"
    )
    def test_filter_with_qa_report(
        self,
        mock_get_error_episodes_from_parquet,
        mock_get_error_episodes_from_qa_report,
        mock_get_filtered_episode_indices_by_short_horizon_tasks,
        mock_get_filtered_episode_indices_by_location_names,
        mock_get_filtered_episode_indices_by_labels,
        mock_get_filtered_episode_indices_by_robot_ids,
        mock_get_filtered_episode_indices_excluding_robot_types,
        tmp_path,
    ):
        dataset_path = tmp_path / "dataset0"
        dataset_path.mkdir(parents=True)

        qa_report = [
            {"episode_index": 0, "errors": []},
            {"episode_index": 1, "errors": ["error1"]},
        ]
        with open(dataset_path / "qa_report.json", "w") as f:
            json.dump(qa_report, f)

        mock_get_filtered_episode_indices_excluding_robot_types.return_value = []
        mock_get_filtered_episode_indices_by_robot_ids.return_value = []
        mock_get_filtered_episode_indices_by_labels.return_value = []
        mock_get_filtered_episode_indices_by_location_names.return_value = []
        mock_get_filtered_episode_indices_by_short_horizon_tasks.return_value = []
        mock_get_error_episodes_from_qa_report.return_value = [1]
        mock_get_error_episodes_from_parquet.return_value = []

        package_tool = PackageTool()
        filtered_map = package_tool._filter_episodes([str(dataset_path)])

        assert str(dataset_path) in filtered_map
        assert 1 in filtered_map[str(dataset_path)]
        assert 0 not in filtered_map[str(dataset_path)]

    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._get_filtered_episode_indices_excluding_robot_types"
    )
    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._get_filtered_episode_indices_by_robot_ids"
    )
    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._get_filtered_episode_indices_by_labels"
    )
    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._get_filtered_episode_indices_by_location_names"
    )
    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._get_filtered_episode_indices_by_short_horizon_tasks"
    )
    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._get_error_episodes_from_qa_report"
    )
    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._get_error_episodes_from_parquet"
    )
    def test_filter_with_parquet_errors(
        self,
        mock_get_error_episodes_from_parquet,
        mock_get_error_episodes_from_qa_report,
        mock_get_filtered_episode_indices_by_short_horizon_tasks,
        mock_get_filtered_episode_indices_by_location_names,
        mock_get_filtered_episode_indices_by_labels,
        mock_get_filtered_episode_indices_by_robot_ids,
        mock_get_filtered_episode_indices_excluding_robot_types,
        tmp_path,
    ):
        dataset_path = tmp_path / "dataset0"
        data_dir = dataset_path / "data" / "chunk-000"
        data_dir.mkdir(parents=True)

        # Create parquet with failure
        df = pd.DataFrame(
            {
                "episode_index": [0, 0],
                "success_primitive_action": [True, False],
            }
        )
        df.to_parquet(data_dir / "episode_000000.parquet", index=False)

        mock_get_filtered_episode_indices_excluding_robot_types.return_value = []
        mock_get_filtered_episode_indices_by_robot_ids.return_value = []
        mock_get_filtered_episode_indices_by_labels.return_value = []
        mock_get_filtered_episode_indices_by_location_names.return_value = []
        mock_get_filtered_episode_indices_by_short_horizon_tasks.return_value = []
        mock_get_error_episodes_from_qa_report.return_value = []
        mock_get_error_episodes_from_parquet.return_value = [0]

        package_tool = PackageTool()
        filtered_map = package_tool._filter_episodes([str(dataset_path)])

        assert str(dataset_path) in filtered_map
        assert 0 in filtered_map[str(dataset_path)]


class TestPackageToolMergeStats:
    """Test _merge_stats method"""

    def test_merge_simple_stats(self):
        """Test merging simple min/max/mean statistics"""
        stats_list = [
            {
                "observation.state": {
                    "max": [1.0, 2.0],
                    "mean": [0.5, 1.0],
                    "min": [0.0, 0.0],
                }
            },
            {
                "observation.state": {
                    "max": [2.0, 3.0],
                    "mean": [1.0, 1.5],
                    "min": [-1.0, -0.5],
                }
            },
        ]

        package_tool = PackageTool()
        merged = package_tool._merge_stats(stats_list)

        assert merged["observation.state"]["max"] == [2.0, 3.0]
        assert merged["observation.state"]["min"] == [-1.0, -0.5]

    def test_merge_multiple_features(self):
        """Test merging statistics with multiple features"""
        stats_list = [
            {
                "observation.state": {
                    "max": [1.0, 2.0],
                    "mean": [0.5, 1.0],
                    "min": [0.0, 0.0],
                },
                "action": {
                    "max": [5.0],
                    "mean": [2.5],
                    "min": [0.0],
                },
            },
            {
                "observation.state": {
                    "max": [2.0, 3.0],
                    "mean": [1.0, 1.5],
                    "min": [-1.0, -0.5],
                },
                "action": {
                    "max": [10.0],
                    "mean": [5.0],
                    "min": [-2.0],
                },
            },
        ]

        package_tool = PackageTool()
        merged = package_tool._merge_stats(stats_list)

        assert merged["observation.state"]["max"] == [2.0, 3.0]
        assert merged["observation.state"]["min"] == [-1.0, -0.5]
        assert merged["action"]["max"] == [10.0]
        assert merged["action"]["min"] == [-2.0]


class TestPackageToolCopyVideos:
    @patch("builtins.open", new_callable=mock_open)
    @patch("hsr_data_converter.commands.package.package.json.load")
    @patch("hsr_data_converter.commands.package.package.os.path.exists")
    @patch("hsr_data_converter.commands.package.package.os.makedirs")
    @patch("hsr_data_converter.commands.package.package.shutil.copy")
    def test_standard_copy_videos(
        self, mock_copy, mock_makedirs, mock_exists, mock_json_load, mock_open
    ):
        mock_exists.side_effect = lambda path: True if "episode" in path else False
        mock_json_load.return_value = {
            "chunks_size": 1,
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.video": {"dtype": "video"},
                "other_feature": {"dtype": "other"},
            },
        }

        package_tool = PackageTool()
        source_folders = ["/path/to/source1", "/path/to/source2"]
        output_folder = "/path/to/output"
        episode_mapping = [("/path/to/source1", 0, 100), ("/path/to/source2", 1, 101)]

        package_tool._copy_videos(source_folders, output_folder, episode_mapping)

        mock_makedirs.assert_called()
        mock_copy.assert_called()

    @patch("builtins.open", new_callable=mock_open)
    @patch("hsr_data_converter.commands.package.package.json.load")
    @patch("hsr_data_converter.commands.package.package.os.path.exists")
    @patch("hsr_data_converter.commands.package.package.os.makedirs")
    @patch("hsr_data_converter.commands.package.package.os.walk")
    @patch("hsr_data_converter.commands.package.package.shutil.copy")
    def test_standard_copy_videos_when_source_video_path_not_found(
        self,
        mock_copy,
        mock_walk,
        mock_makedirs,
        mock_exists,
        mock_json_load,
        mock_open,
    ):
        mock_exists.return_value = False
        mock_json_load.return_value = {
            "chunks_size": 1,
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {"observation.rgb_camera": {"dtype": "video"}},
        }

        mock_walk.return_value = [
            ("/path/to/source6/videos", ("chunk-000",), ()),
            (
                "/path/to/source6/videos/chunk-000/observation.rgb_camera",
                (),
                ("episode_000000.mp4",),
            ),
        ]

        package_tool = PackageTool()
        source_folders = ["/path/to/source6"]
        output_folder = "/path/to/output"
        episode_mapping = [("/path/to/source6", 0, 400)]

        package_tool._copy_videos(source_folders, output_folder, episode_mapping)

        mock_makedirs.assert_called()
        mock_copy.assert_called()

    @patch("builtins.open", new_callable=mock_open)
    @patch("hsr_data_converter.commands.package.package.json.load")
    @patch("hsr_data_converter.commands.package.package.os.path.exists")
    @patch("hsr_data_converter.commands.package.package.os.makedirs")
    @patch("hsr_data_converter.commands.package.package.os.walk")
    @patch("hsr_data_converter.commands.package.package.shutil.copy")
    def test_copy_videos_not_found(
        self,
        mock_copy,
        mock_walk,
        mock_makedirs,
        mock_exists,
        mock_json_load,
        mock_open,
    ):
        mock_exists.return_value = False
        mock_json_load.return_value = {
            "chunks_size": 1,
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {"observation.rgb_camera": {"dtype": "video"}},
        }

        mock_walk.return_value = []

        package_tool = PackageTool()
        source_folders = ["/path/to/source6"]
        output_folder = "/path/to/output"
        episode_mapping = [("/path/to/source6", 0, 400)]

        package_tool._copy_videos(source_folders, output_folder, episode_mapping)

        mock_makedirs.assert_not_called()
        mock_copy.assert_not_called()


class TestPackageToolCopyDataFiles:
    @patch("hsr_data_converter.commands.package.package.pd.read_parquet")
    @patch("hsr_data_converter.commands.package.package.os.makedirs")
    @patch("pandas.DataFrame.to_parquet")
    @patch("hsr_data_converter.commands.package.package.os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data='{"fps": 30}')
    def test_standard_copy_data_files(
        self, mock_open, mock_exists, mock_to_parquet, mock_makedirs, mock_read_parquet
    ):
        mock_exists.return_value = True
        mock_read_parquet.return_value = pd.DataFrame(
            {
                "episode_index": [0],
                "observation.state": [[1, 2, 3]],
                "action": [[1, 2]],
                "index": [0],
            }
        )

        package_tool = PackageTool()
        source_folders = ["/path/to/source8"]
        output_folder = "/path/to/output"
        episode_mapping = [("/path/to/source8", 0, 600)]

        result = package_tool._copy_data_files(
            source_folders=source_folders,
            output_folder=output_folder,
            episode_mapping=episode_mapping,
            max_dim=32,
            chunks_size=100,
            default_fps=24,
        )

        assert result is True
        mock_makedirs.assert_called()
        mock_read_parquet.assert_called()
        mock_to_parquet.assert_called()

    @patch("hsr_data_converter.commands.package.package.pd.read_parquet")
    @patch("hsr_data_converter.commands.package.package.os.makedirs")
    @patch("pandas.DataFrame.to_parquet")
    @patch("hsr_data_converter.commands.package.package.os.path.exists")
    @patch("hsr_data_converter.commands.package.package.os.walk")
    def test_no_parquet_file_found(
        self, mock_walk, mock_exists, mock_to_parquet, mock_makedirs, mock_read_parquet
    ):
        mock_exists.return_value = False

        mock_walk.return_value = [("/path/to/source9", (), ("episode_000000.parquet",))]

        package_tool = PackageTool()
        source_folders = ["/path/to/source9"]
        output_folder = "/path/to/output"
        episode_mapping = [("/path/to/source9", 0, 601)]
        folder_to_task_strings = {"/path/to/source9": ""}
        task_string_to_new_index = 1

        result = package_tool._copy_data_files(
            source_folders=source_folders,
            output_folder=output_folder,
            episode_mapping=episode_mapping,
            folder_to_task_strings=folder_to_task_strings,
            task_string_to_new_index=task_string_to_new_index,
            max_dim=32,
            chunks_size=100,
            default_fps=24,
        )

        assert result
        mock_makedirs.assert_called()
        mock_read_parquet.assert_called()
        mock_to_parquet.assert_not_called()
        mock_walk.assert_called_once_with("/path/to/source9")

    @patch("hsr_data_converter.commands.package.package.pd.read_parquet")
    @patch("hsr_data_converter.commands.package.package.os.makedirs")
    @patch("pandas.DataFrame.to_parquet")
    @patch("hsr_data_converter.commands.package.package.os.path.exists")
    def test_existing_file_copy_failure(
        self, mock_exists, mock_to_parquet, mock_makedirs, mock_read_parquet
    ):
        mock_exists.return_value = True
        mock_read_parquet.side_effect = Exception("Read failure")

        package_tool = PackageTool()
        source_folders = ["/path/to/source10"]
        output_folder = "/path/to/output"
        episode_mapping = [("/path/to/source10", 0, 602)]

        result = package_tool._copy_data_files(
            source_folders=source_folders,
            output_folder=output_folder,
            episode_mapping=episode_mapping,
            max_dim=32,
            fps=24,
            chunks_size=100,
            default_fps=24,
        )

        assert result is False
        mock_makedirs.assert_not_called()
        mock_read_parquet.assert_called()
        mock_to_parquet.assert_not_called()

    @patch("hsr_data_converter.commands.package.package.pd.read_parquet")
    @patch("hsr_data_converter.commands.package.package.os.makedirs")
    @patch("pandas.DataFrame.to_parquet")
    @patch("hsr_data_converter.commands.package.package.os.path.exists")
    def test_string_based_mapping_application(
        self, mock_exists, mock_to_parquet, mock_makedirs, mock_read_parquet
    ):
        mock_exists.return_value = True
        mock_read_parquet.return_value = pd.DataFrame(
            {
                "episode_index": [0],
                "task_index": ["task_a"],
                "primitive_action_index": ["action_a"],
                "short_horizon_task_index": ["task_b"],
            }
        )

        package_tool = PackageTool()
        source_folders = ["/path/to/source11"]
        output_folder = "/path/to/output"
        episode_mapping = [("/path/to/source11", 0, 603)]

        folder_to_task_strings = {
            "/path/to/source11": {
                "task_a": "task_new_a",
                "task_b": "task_new_b",
                "action_a": "action_new_a",
            }
        }
        task_string_to_new_index = {"task_new_a": 0, "task_new_b": 1, "action_new_a": 2}

        result = package_tool._copy_data_files(
            source_folders=source_folders,
            output_folder=output_folder,
            episode_mapping=episode_mapping,
            max_dim=32,
            fps=24,
            chunks_size=100,
            default_fps=24,
            folder_to_task_strings=folder_to_task_strings,
            task_string_to_new_index=task_string_to_new_index,
        )

        assert result is True
        mock_makedirs.assert_called()
        mock_read_parquet.assert_called()
        mock_to_parquet.assert_called()

    @patch("hsr_data_converter.commands.package.package.pd.read_parquet")
    @patch("hsr_data_converter.commands.package.package.os.makedirs")
    @patch("hsr_data_converter.commands.package.package.ProcessParquet.to_parquet")
    @patch("hsr_data_converter.commands.package.package.os.path.exists")
    @patch("hsr_data_converter.commands.package.package.os.walk")
    def test_no_parquet_file_found1(
        self, mock_walk, mock_exists, mock_to_parquet, mock_makedirs, mock_read_parquet
    ):
        mock_exists.return_value = False

        mock_walk.return_value = [("/path/to/source9", (), ("episode_000000.parquet",))]
        mock_to_parquet.side_effect = Exception("test")

        package_tool = PackageTool()
        source_folders = ["/path/to/source9"]
        output_folder = "/path/to/output"
        episode_mapping = [("/path/to/source9", 0, 601)]
        folder_to_task_strings = {"/path/to/source9": ""}
        task_string_to_new_index = 1

        result = package_tool._copy_data_files(
            source_folders=source_folders,
            output_folder=output_folder,
            episode_mapping=episode_mapping,
            folder_to_task_strings=folder_to_task_strings,
            task_string_to_new_index=task_string_to_new_index,
            max_dim=32,
            chunks_size=100,
            default_fps=24,
        )

        assert not result
        mock_makedirs.assert_not_called()
        mock_read_parquet.assert_called()
        mock_to_parquet.assert_called()
        mock_walk.assert_called_once_with("/path/to/source9")


class TestPackageToolValidateTimestamps:
    @patch("hsr_data_converter.commands.package.package.pd.read_parquet")
    @patch("builtins.open", new_callable=mock_open, read_data='{"fps": 30}')
    @patch("hsr_data_converter.commands.package.package.os.path.exists")
    @patch("hsr_data_converter.commands.package.package.os.walk")
    def test_validate_timestamps_with_consistent_fps_and_timestamps(
        self, mock_walk, mock_exists, mock_open, mock_read_parquet
    ):
        mock_exists.side_effect = lambda path: path.endswith("info.json")
        mock_read_parquet.return_value = pd.DataFrame(
            {"timestamp": [0.0, 0.033, 0.067], "some_data": [1, 2, 3]}
        )
        mock_walk.return_value = [
            ("/path/to/source6/videos", (), ("episode_000001.parquet",))
        ]

        package_tool = PackageTool()
        source_folders = ["/path/to/source13"]

        issues, fps_values = package_tool._validate_timestamps(source_folders)

        assert not issues
        assert fps_values == [30]
        mock_open.assert_called_once_with("/path/to/source13/meta/info.json")
        mock_read_parquet.assert_called_once()

    @patch("hsr_data_converter.commands.package.package.pd.read_parquet")
    @patch("builtins.open", new_callable=mock_open, read_data='{"fps": 30}')
    @patch("hsr_data_converter.commands.package.package.os.path.exists")
    @patch("hsr_data_converter.commands.package.package.os.walk")
    def test_validate_timestamps_warns_when_no_timestamp_columns(
        self, mock_walk, mock_exists, mock_open, mock_read_parquet
    ):
        mock_exists.side_effect = lambda path: path.endswith("info.json")
        mock_read_parquet.return_value = pd.DataFrame({})
        mock_walk.side_effect = [
            [],
            [("/path/to/source6/videos", (), ("episode_000001.parquet",))],
        ]

        package_tool = PackageTool()
        source_folders = ["/path/to/source13"]

        issues, fps_values = package_tool._validate_timestamps(source_folders)

        assert "Warning: Dataset /path/to/source13 has no timestamp columns" in issues
        assert fps_values == [30]
        mock_open.assert_called_once_with("/path/to/source13/meta/info.json")
        mock_read_parquet.assert_called_once()

    @patch("hsr_data_converter.commands.package.package.pd.read_parquet")
    @patch("builtins.open", new_callable=mock_open)
    @patch("hsr_data_converter.commands.package.package.os.path.exists")
    @patch("hsr_data_converter.commands.package.package.os.walk")
    @patch("hsr_data_converter.commands.package.package.json.load")
    def test_validate_timestamps_warns_on_inconsistent_fps(
        self, mock_json_load, mock_walk, mock_exists, mock_open, mock_read_parquet
    ):
        mock_exists.side_effect = lambda path: path.endswith("info.json")
        mock_json_load.side_effect = [{"fps": 30}, {"fps": 20}]
        mock_read_parquet.return_value = pd.DataFrame(
            {"timestamp": [0.0, 0.033, 0.067], "some_data": [1, 2, 3]}
        )
        mock_walk.return_value = [
            ("/path/to/source6/videos", (), ("episode_000001.parquet",))
        ]

        package_tool = PackageTool()
        source_folders = ["/path/to/source13", "/path/to/source14"]

        issues, fps_values = package_tool._validate_timestamps(source_folders)

        assert "Warning: Inconsistent FPS across datasets: [30, 20]" in issues
        assert fps_values == [30, 20]
        mock_open.call_count == 2
        mock_read_parquet.call_count == 2

    @patch("hsr_data_converter.commands.package.package.pd.read_parquet")
    @patch("builtins.open", new_callable=mock_open, read_data='{"fps": 30}')
    @patch("hsr_data_converter.commands.package.package.os.path.exists")
    @patch("hsr_data_converter.commands.package.package.os.walk")
    def test_validate_timestamps_warns_when_no_parquet_file_found(
        self, mock_walk, mock_exists, mock_open, mock_read_parquet
    ):
        mock_exists.side_effect = lambda path: path.endswith("info.json")
        mock_read_parquet.return_value = pd.DataFrame(
            {"timestamp": [0.0, 0.033, 0.067], "some_data": [1, 2, 3]}
        )
        mock_walk.return_value = []

        package_tool = PackageTool()
        source_folders = ["/path/to/source13"]

        issues, fps_values = package_tool._validate_timestamps(source_folders)

        assert "Warning: No parquet files found in dataset /path/to/source13" in issues
        assert fps_values == [30]
        mock_open.assert_called_once_with("/path/to/source13/meta/info.json")
        mock_read_parquet.assert_not_called()

    @patch("hsr_data_converter.commands.package.package.pd.read_parquet")
    @patch("builtins.open", new_callable=mock_open, read_data='{"fps": 30}')
    @patch("hsr_data_converter.commands.package.package.os.path.exists")
    @patch("hsr_data_converter.commands.package.package.os.walk")
    def test_validate_timestamps_handles_generic_errors(
        self, mock_walk, mock_exists, mock_open, mock_read_parquet
    ):
        mock_exists.side_effect = Exception("test")
        mock_read_parquet.return_value = pd.DataFrame(
            {"timestamp": [0.0, 0.033, 0.067], "some_data": [1, 2, 3]}
        )
        mock_walk.return_value = [
            ("/path/to/source6/videos", (), ("episode_000001.parquet",))
        ]

        package_tool = PackageTool()
        source_folders = ["/path/to/source13"]

        issues, fps_values = package_tool._validate_timestamps(source_folders)

        assert "Error: Failed to validate dataset /path/to/source13: test" in issues
        assert fps_values == []
        mock_open.assert_not_called()
        mock_read_parquet.assert_not_called()


class TestPackageToolPadParquetData:
    @patch("hsr_data_converter.commands.package.package.pd.read_parquet")
    @patch("hsr_data_converter.commands.package.package.pd.DataFrame.to_parquet")
    @patch("hsr_data_converter.commands.package.package.os.makedirs")
    def test_standard_pad_parquet_data(
        self, mock_makedirs, mock_to_parquet, mock_read_parquet
    ):
        mock_read_parquet.return_value = pd.DataFrame(
            {
                "episode_index": [0],
                "observation.state": [[1, 2, 3]],
                "action": [[1, 2]],
            }
        )

        package_tool = PackageTool()
        source_path = "/path/to/source.parquet"
        target_path = "/path/to/target.parquet"

        padded_df = package_tool._pad_parquet_data(
            source_path, target_path, original_dim=3, target_dim=5
        )

        assert padded_df is not None
        assert padded_df["observation.state"].iloc[0] == [1, 2, 3, 0, 0]
        assert padded_df["action"].iloc[0] == [1, 2, 0, 0, 0]

        mock_makedirs.assert_called_once_with("/path/to", exist_ok=True)
        mock_to_parquet.assert_called_once_with(target_path, index=False)
        mock_read_parquet.assert_called_once_with(source_path)

    @patch("hsr_data_converter.commands.package.package.pd.read_parquet")
    @patch("hsr_data_converter.commands.package.package.pd.DataFrame.to_parquet")
    @patch("hsr_data_converter.commands.package.package.os.makedirs")
    def test_pad_parquet_data_no_padding_needed(
        self, mock_makedirs, mock_to_parquet, mock_read_parquet
    ):
        mock_read_parquet.return_value = pd.DataFrame(
            {
                "episode_index": [0],
                "observation.state": [[1, 2, 3, 4, 5]],
                "action": [[1, 2, 3, 4, 5]],
            }
        )

        package_tool = PackageTool()
        source_path = "/path/to/source_no_padding.parquet"
        target_path = "/path/to/target_no_padding.parquet"

        padded_df = package_tool._pad_parquet_data(
            source_path, target_path, original_dim=5, target_dim=5
        )

        assert padded_df["observation.state"].iloc[0] == [1, 2, 3, 4, 5]
        assert padded_df["action"].iloc[0] == [1, 2, 3, 4, 5]

        mock_makedirs.assert_called_once_with("/path/to", exist_ok=True)
        mock_to_parquet.assert_called_once_with(target_path, index=False)
        mock_read_parquet.assert_called_once_with(source_path)

    @patch("hsr_data_converter.commands.package.package.pd.read_parquet")
    @patch("hsr_data_converter.commands.package.package.pd.DataFrame.to_parquet")
    @patch("hsr_data_converter.commands.package.package.os.makedirs")
    def test_pad_parquet_data_with_original_dim_greater_than_target_dim(
        self, mock_makedirs, mock_to_parquet, mock_read_parquet
    ):
        mock_read_parquet.return_value = pd.DataFrame(
            {
                "episode_index": [0],
                "observation.state": [[1, 2, 3, 4, 5]],
                "action": [[1, 2, 3, 4, 5]],
            }
        )

        package_tool = PackageTool()
        source_path = "/path/to/source_greater_than_target.parquet"
        target_path = "/path/to/target_greater_than_target.parquet"

        padded_df = package_tool._pad_parquet_data(
            source_path, target_path, original_dim=5, target_dim=3
        )

        assert padded_df["observation.state"].iloc[0] == [1, 2, 3, 4, 5]
        assert padded_df["action"].iloc[0] == [1, 2, 3, 4, 5]

        mock_makedirs.assert_called_once_with("/path/to", exist_ok=True)
        mock_to_parquet.assert_called_once_with(target_path, index=False)
        mock_read_parquet.assert_called_once_with(source_path)


class TestPackageToolValidateVersions:
    @patch("hsr_data_converter.commands.package.package.json.load")
    @patch("hsr_data_converter.commands.package.package.os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_versions_are_unique_or_not_set(
        self, mock_builtins_open, mock_exists, mock_json_load
    ):
        mock_exists.return_value = True
        mock_json_load.side_effect = [{"codebase_version": "v2.1"}]

        package_tool = PackageTool()
        source_folders = ["folder1"]
        result = package_tool._validate_versions(source_folders)

        assert result

    @patch("hsr_data_converter.commands.package.package.json.load")
    @patch("hsr_data_converter.commands.package.package.os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_some_versions_none_or_duplicate(
        self, mock_builtins_open, mock_exists, mock_json_load
    ):
        mock_exists.return_value = True
        mock_json_load.side_effect = [
            {"codebase_version": "v2.1"},
            {"codebase_version": "v2.1"},
            {},
        ]

        package_tool = PackageTool()
        source_folders = ["folder1", "folder2", "folder3"]
        result = package_tool._validate_versions(source_folders)

        assert result

    @patch("hsr_data_converter.commands.package.package.json.load")
    @patch("hsr_data_converter.commands.package.package.os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_all_versions_none(self, mock_builtins_open, mock_exists, mock_json_load):
        mock_exists.return_value = True
        mock_json_load.side_effect = [{}, {}, {}]

        package_tool = PackageTool()
        source_folders = ["folder1", "folder2", "folder3"]
        result = package_tool._validate_versions(source_folders)

        assert result

    @patch("hsr_data_converter.commands.package.package.json.load")
    @patch("hsr_data_converter.commands.package.package.os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_versions_are_different(
        self, mock_builtins_open, mock_exists, mock_json_load
    ):
        mock_exists.return_value = True
        mock_json_load.side_effect = [
            {"codebase_version": "v2.1"},
            {"codebase_version": "v2.2"},
            {"codebase_version": "v2.1"},
        ]

        package_tool = PackageTool()
        source_folders = ["folder1", "folder2", "folder3"]
        result = package_tool._validate_versions(source_folders)

        assert not result


class TestPackageToolMergeDatasets:
    @patch("hsr_data_converter.commands.package.package.pd.read_parquet")
    @patch("hsr_data_converter.commands.package.package.PackageTool._merge_stats")
    @patch("hsr_data_converter.commands.package.package.PackageTool._copy_videos")
    @patch("hsr_data_converter.commands.package.package.PackageTool._copy_data_files")
    @patch("hsr_data_converter.commands.package.package.ProcessJson.load_jsonl_data")
    @patch("hsr_data_converter.commands.package.package.ProcessJson.save_jsonl")
    @patch("builtins.open", new_callable=mock_open)
    @patch("hsr_data_converter.commands.package.package.os.makedirs")
    @patch("hsr_data_converter.commands.package.package.os.path.exists")
    @patch("hsr_data_converter.commands.package.package.os.walk")
    @patch("hsr_data_converter.commands.package.package.json.load")
    @patch("hsr_data_converter.commands.package.package.json.dump")
    def test_standard_merge_datasets(
        self,
        mock_json_dump,
        mock_json_load,
        mock_walk,
        mock_path_exist,
        mock_makedirs,
        mock_builtins_open,
        mock_save_jsonl,
        mock_load_jsonl_data,
        mock_copy_data_files,
        mock_copy_videos,
        mock_merge_stats,
        mock_read_parquet,
    ):
        mock_json_load.side_effect = [
            {"fps": 30}  # stats_path
        ]
        mock_path_exist.return_value = True
        mock_load_jsonl_data.side_effect = [
            [{"task": "aaa", "task_index": "2"}],  # tasks_path
            {"chunks_size": 1},  # info_path
            {"total_videos": 1},  # folder_info_path
            [{"episode_index": 1}],  # episodes_path
            [{"episode_index": 1}],  # episodes_stats_path
            {
                "chunks_size": 1,
                "features": {"action": {"shape": "shape1"}},
            },  # info_path
        ]
        mock_walk.return_value = [
            ("/path/to/source6/videos", (), ("episode_000001.parquet",))
        ]
        mock_save_jsonl.return_value = None
        mock_read_parquet.return_value = pd.DataFrame(
            {
                "episode_index": [0],
                "observation.state": [[1, 2, 3, 4, 5]],
                "action": [[1, 2, 3, 4, 5]],
            }
        )

        package_tool = PackageTool()

        source_folders = ["/path/to/source1"]
        output_folder = "/path/to/output"
        filtered_episodes_map = {}

        package_tool._merge_datasets(
            source_folders=source_folders,
            output_folder=output_folder,
            filtered_episodes_map=filtered_episodes_map,
            max_dim=32,
            fps=30,
            chunk_size=1000,
        )

        mock_makedirs.assert_any_call("/path/to/output", exist_ok=True)
        mock_makedirs.assert_any_call("/path/to/output/meta", exist_ok=True)

        mock_copy_videos.assert_called_once()
        mock_copy_data_files.assert_called_once()


class TestPackageToolPackage:
    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._init_source_folders"
    )
    @patch("hsr_data_converter.commands.package.package.PackageTool._validate_versions")
    @patch("hsr_data_converter.commands.package.package.PackageTool._validate_fps")
    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._validate_feature_shapes"
    )
    @patch("hsr_data_converter.commands.package.package.PackageTool._filter_episodes")
    @patch("hsr_data_converter.commands.package.package.PackageTool._merge_datasets")
    def test_standard_package(
        self,
        mock_merge_datasets,
        mock_filter_episodes,
        mock_validate_feature_shapes,
        mock_validate_fps,
        mock_validate_versions,
        mock_init_source_folders,
    ):
        mock_validate_fps.return_value = (True, 30, None)
        mock_validate_feature_shapes.return_value = (True, None)
        mock_filter_episodes.return_value = {}
        mock_validate_versions.return_value = True

        package_tool = PackageTool()
        source_folders = ["/path/to/source1", "/path/to/source2"]
        output_folder = "/path/to/output"
        mock_init_source_folders.return_value = source_folders, output_folder

        success = package_tool.package(
            source_folders=source_folders,
            output_folder=output_folder,
            fps=20,
            max_dim=32,
            chunk_size=1000,
            ignore_check_version=False,
        )

        assert success is True
        mock_merge_datasets.assert_called_once_with(
            source_folders,
            output_folder,
            filtered_episodes_map={},
            max_dim=32,
            fps=30,
            chunk_size=1000,
        )

    @patch("hsr_data_converter.commands.package.package.tempfile.mkdtemp")
    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._upload_merged_datasets"
    )
    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._init_source_folders"
    )
    @patch("hsr_data_converter.commands.package.package.PackageTool._validate_versions")
    @patch("hsr_data_converter.commands.package.package.PackageTool._validate_fps")
    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._validate_feature_shapes"
    )
    @patch("hsr_data_converter.commands.package.package.PackageTool._filter_episodes")
    @patch("hsr_data_converter.commands.package.package.PackageTool._merge_datasets")
    def test_use_aws_is_true(
        self,
        mock_merge_datasets,
        mock_filter_episodes,
        mock_validate_feature_shapes,
        mock_validate_fps,
        mock_validate_versions,
        mock_init_source_folders,
        mock_upload_merged_datasets,
        mock_mkdtemp,
    ):
        mock_validate_fps.return_value = (True, 30, None)
        mock_validate_feature_shapes.return_value = (True, None)
        mock_filter_episodes.return_value = {}
        mock_validate_versions.return_value = True
        mock_mkdtemp.return_value = "./tmp/mkdtemp"

        package_tool = PackageTool(
            use_aws=True,
            secret="test_secret",
            source_bucket="test_source_bucket",
            output_bucket="test_output_bucket",
        )
        source_folders = ["/path/to/source1", "/path/to/source2"]
        output_folder = "/path/to/output"
        mock_init_source_folders.return_value = source_folders, output_folder

        success = package_tool.package(
            source_folders=source_folders,
            output_folder=output_folder,
            fps=20,
            max_dim=32,
            chunk_size=1000,
            ignore_check_version=False,
        )

        assert success is True
        mock_merge_datasets.assert_called_once_with(
            source_folders,
            output_folder,
            filtered_episodes_map={},
            max_dim=32,
            fps=30,
            chunk_size=1000,
        )

    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._init_source_folders"
    )
    @patch("hsr_data_converter.commands.package.package.PackageTool._validate_versions")
    @patch("hsr_data_converter.commands.package.package.PackageTool._validate_fps")
    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._validate_feature_shapes"
    )
    @patch("hsr_data_converter.commands.package.package.PackageTool._filter_episodes")
    @patch("hsr_data_converter.commands.package.package.PackageTool._merge_datasets")
    def test_skip_validate_versions(
        self,
        mock_merge_datasets,
        mock_filter_episodes,
        mock_validate_feature_shapes,
        mock_validate_fps,
        mock_validate_versions,
        mock_init_source_folders,
    ):
        mock_validate_fps.return_value = (True, 30, None)
        mock_validate_feature_shapes.return_value = (True, None)
        mock_filter_episodes.return_value = {}
        mock_validate_versions.return_value = True

        package_tool = PackageTool()
        source_folders = ["/path/to/source1", "/path/to/source2"]
        output_folder = "/path/to/output"
        mock_init_source_folders.return_value = source_folders, output_folder

        success = package_tool.package(
            source_folders=source_folders,
            output_folder=output_folder,
            fps=20,
            max_dim=32,
            chunk_size=1000,
            ignore_check_version=True,
        )

        assert success is True
        mock_merge_datasets.assert_called_once_with(
            source_folders,
            output_folder,
            filtered_episodes_map={},
            max_dim=32,
            fps=30,
            chunk_size=1000,
        )
        mock_validate_versions.assert_not_called()

    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._init_source_folders"
    )
    @patch("hsr_data_converter.commands.package.package.PackageTool._validate_fps")
    def test_fps_validation_failure(self, mock_validate_fps, mock_init_source_folders):
        mock_validate_fps.return_value = (False, None, "FPS validation failed")

        package_tool = PackageTool()
        source_folders = ["/path/to/source3"]
        output_folder = "/path/to/output"
        mock_init_source_folders.return_value = source_folders, output_folder

        success = package_tool.package(
            source_folders=source_folders,
            output_folder=output_folder,
            fps=None,
            max_dim=32,
            chunk_size=1000,
            ignore_check_version=False,
        )

        assert success is False

    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._init_source_folders"
    )
    @patch("hsr_data_converter.commands.package.package.PackageTool._validate_fps")
    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._validate_feature_shapes"
    )
    def test_feature_shape_validation_failure(
        self, mock_validate_feature_shapes, mock_validate_fps, mock_init_source_folders
    ):
        mock_validate_fps.return_value = (True, 30, None)
        mock_validate_feature_shapes.return_value = (
            False,
            "Feature shape validation failed",
        )

        package_tool = PackageTool()
        source_folders = ["/path/to/source3"]
        output_folder = "/path/to/output"
        mock_init_source_folders.return_value = source_folders, output_folder

        success = package_tool.package(
            source_folders=source_folders,
            output_folder=output_folder,
            fps=None,
            max_dim=32,
            chunk_size=1000,
            ignore_check_version=False,
        )

        assert success is False

    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._init_source_folders"
    )
    @patch("hsr_data_converter.commands.package.package.PackageTool._validate_versions")
    @patch("hsr_data_converter.commands.package.package.PackageTool._validate_fps")
    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._validate_feature_shapes"
    )
    @patch("hsr_data_converter.commands.package.package.PackageTool._filter_episodes")
    @patch("hsr_data_converter.commands.package.package.PackageTool._merge_datasets")
    def test_versions_validation_failure(
        self,
        mock_merge_datasets,
        mock_filter_episodes,
        mock_validate_feature_shapes,
        mock_validate_fps,
        mock_validate_versions,
        mock_init_source_folders,
    ):
        mock_validate_fps.return_value = (True, 30, None)
        mock_validate_feature_shapes.return_value = (True, None)
        mock_validate_versions.return_value = False

        package_tool = PackageTool()
        source_folders = ["/path/to/source1", "/path/to/source2"]
        output_folder = "/path/to/output"
        mock_init_source_folders.return_value = source_folders, output_folder

        success = package_tool.package(
            source_folders=source_folders,
            output_folder=output_folder,
            fps=20,
            max_dim=32,
            chunk_size=1000,
            ignore_check_version=False,
        )

        assert success is False
        mock_merge_datasets.assert_not_called()

    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._init_source_folders"
    )
    @patch("hsr_data_converter.commands.package.package.PackageTool._validate_versions")
    @patch("hsr_data_converter.commands.package.package.PackageTool._validate_fps")
    @patch(
        "hsr_data_converter.commands.package.package.PackageTool._validate_feature_shapes"
    )
    @patch("hsr_data_converter.commands.package.package.PackageTool._filter_episodes")
    @patch("hsr_data_converter.commands.package.package.PackageTool._merge_datasets")
    def test_error(
        self,
        mock_merge_datasets,
        mock_filter_episodes,
        mock_validate_feature_shapes,
        mock_validate_fps,
        mock_validate_versions,
        mock_init_source_folders,
    ):
        mock_validate_fps.return_value = (True, 30, None)
        mock_validate_feature_shapes.return_value = (True, None)
        mock_filter_episodes.return_value = {}
        mock_validate_versions.return_value = True
        mock_merge_datasets.side_effect = Exception("test")

        package_tool = PackageTool()
        source_folders = ["/path/to/source1", "/path/to/source2"]
        output_folder = "/path/to/output"
        mock_init_source_folders.return_value = source_folders, output_folder

        success = package_tool.package(
            source_folders=source_folders,
            output_folder=output_folder,
            fps=20,
            max_dim=32,
            chunk_size=1000,
            ignore_check_version=False,
        )

        assert not success
        mock_merge_datasets.assert_called_once_with(
            source_folders,
            output_folder,
            filtered_episodes_map={},
            max_dim=32,
            fps=30,
            chunk_size=1000,
        )
