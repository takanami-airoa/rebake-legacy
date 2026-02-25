import argparse
import contextlib
import glob
import json
import os
import shutil
import traceback
import tempfile
import numpy as np
import pandas as pd
from typing import Any
from hsr_data_converter.utils.aws_helper import AWSHelper
from pathlib import Path
from uuid import uuid4


class ProcessJson:
    @staticmethod
    def load_jsonl_data(file_path, is_jsonl=False):
        """
        Load data from a JSON or JSONL file.

        Args:
            file_path (str): Path to the file.
            is_jsonl (bool): Indicates if the file is JSONL format.

        Returns:
            dict or list: The loaded JSON object(s).
        """
        data = []

        if is_jsonl:
            try:
                with open(file_path) as f:
                    content = f.read()
                    # Determine if the content is an array or line-by-line JSONL
                    if content.strip().startswith("[") and content.strip().endswith(
                        "]"
                    ):
                        return json.loads(content)
                    else:
                        try:
                            return json.loads("[" + content + "]")
                        except json.JSONDecodeError:
                            pass

                # Fall back to line-by-line parsing if properly formatted JSON array fails
                with open(file_path) as f:
                    for line in f:
                        with contextlib.suppress(json.JSONDecodeError):
                            data.append(json.loads(line))

                return data
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                return data
        else:
            with open(file_path, "r") as f:
                return json.load(f)

    @staticmethod
    def read_meta(input_dataset_path):
        """
        Read meta data from the dataset directory.

        Args:
            input_dataset_path (str): Path to the input dataset.

        Returns:
            dict: Metadata dictionary containing episodes, episodes_stats, info, and tasks.
        """
        episodes_meta = ProcessJson.load_jsonl_data(
            os.path.join(input_dataset_path, "meta", "episodes.jsonl"), is_jsonl=True
        )
        episodes_stats_meta = ProcessJson.load_jsonl_data(
            os.path.join(input_dataset_path, "meta", "episodes_stats.jsonl"),
            is_jsonl=True,
        )
        info_meta = ProcessJson.load_jsonl_data(
            os.path.join(input_dataset_path, "meta", "info.json")
        )
        tasks_meta = ProcessJson.load_jsonl_data(
            os.path.join(input_dataset_path, "meta", "tasks.jsonl"), is_jsonl=True
        )

        metas = {
            "episodes": episodes_meta,
            "episodes_stats": episodes_stats_meta,
            "info": info_meta,
            "tasks": tasks_meta,
        }
        return metas

    @staticmethod
    def save_jsonl(data, file_path):
        """
        Save data in JSONL format

        Args:
            data (list): List of JSON objects to save
            file_path (str): Path to the output file
        """
        with open(file_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")


class ProcessParquet:
    def __init__(self, parquet_file_path: str):
        self.df = pd.read_parquet(parquet_file_path)

    def set_episode_index(self, episode_index: int) -> None:
        if "episode_index" in self.df.columns:
            self.df["episode_index"] = episode_index

    def set_index(self, new_index: int, episode_to_frame_index: dict) -> None:
        if "index" in self.df.columns:
            if episode_to_frame_index and new_index in episode_to_frame_index:
                # Use pre-calculated frame index start value
                first_index = episode_to_frame_index[new_index]
                print(
                    f"Update index column, start value: {first_index} (using global cumulative frame count)"
                )
            else:
                # If no mapping provided, use current calculation as fallback
                first_index = new_index * len(self.df)
                print(
                    f"Update index column, start value: {first_index} (using episode index multiplied by length)"
                )
            self.df["index"] = [first_index + i for i in range(len(self.df))]

    def set_task_index_map(self, key: str, task_index_map: dict) -> None:
        if key in self.df.columns:
            self.df[key] = self.df[key].map(task_index_map)

    def set_string_based_mapping(
        self, key: str, local_task_map: dict, task_string_to_new_index: int
    ) -> None:
        if key in self.df.columns:
            original_col = self.df[key]
            if original_col.isnull().all():
                return

            # Pre-process column to handle unhashable types like numpy arrays
            def to_hashable(val):
                if isinstance(val, np.ndarray):
                    return val.item(0) if val.size > 0 else None
                return val

            processed_col = original_col.apply(to_hashable)

            task_strings_col = processed_col.map(local_task_map)
            new_col = task_strings_col.map(task_string_to_new_index)

            # Report any indices that were not found in the mapping
            unmapped_mask = new_col.isnull() & processed_col.notnull()
            if unmapped_mask.any():
                unique_unmapped_indices = processed_col[unmapped_mask].unique()
                print(
                    f"Warning: For column `{key}`, no mapping found for task indices: {list(unique_unmapped_indices)}. "
                    "These values will be set to -1."
                )
                new_col = new_col.fillna(-1).astype("int64")

            self.df[key] = new_col
            print(
                f"Column `{key}` was updated with new task indices following string-based mapping."
            )

    def to_parquet(self, new_index: int, chunks_size: int, output_folder: str) -> str:
        chunk_index = new_index // chunks_size
        chunk_dir = os.path.join(output_folder, "data", f"chunk-{chunk_index:03d}")
        os.makedirs(chunk_dir, exist_ok=True)
        dest_path = os.path.join(chunk_dir, f"episode_{new_index:06d}.parquet")
        self.df.to_parquet(dest_path, index=False)
        return dest_path


class ProcessVideo:
    @staticmethod
    def copy_videos(
        source_video_path: str,
        dest_video_path: str,
        without_copy_video: bool = False,
    ) -> bool:
        os.makedirs(
            os.path.dirname(dest_video_path),
            exist_ok=True,
        )
        if without_copy_video:
            return False

        print(f"Copying video: {source_video_path} -> {dest_video_path}")
        shutil.copy(source_video_path, dest_video_path)
        return not without_copy_video


class PackageTool:
    """
    Unified tool for processing datasets with filtering and merging capabilities.

    This tool integrates the following workflow:
    1. Load metadata from source datasets
    2. Validate FPS and feature shape consistency
    3. Filter error episodes
    4. Merge datasets into a single output
    """

    def __init__(
        self,
        robot_types: list[str] | None = None,
        robot_ids: list[str] | None = None,
        labels: list[str] | None = None,
        location_names: list[str] | None = None,
        short_horizon_tasks: list[str] | None = None,
        use_aws: bool = False,
        secret: str = "",
        source_bucket: str = "",
        output_bucket: str = "",
    ):
        """
        Initialize the PackageTool with filtering criteria.

        Args:
            robot_types (list[str] | None): List of allowed robot types. Episodes with different types will be filtered out.
            robot_ids (list[str] | None): List of allowed robot IDs. Episodes with different IDs will be filtered out.
            labels (list[str] | None): List of allowed labels. Episodes with different labels will be filtered out.
            location_names (list[str] | None): List of allowed location names. Episodes with different location names will be filtered out.
            short_horizon_tasks (list[str] | None): List of allowed short horizon tasks. Episodes with different tasks will be filtered out.
            use_aws (bool): Flag indicating whether to use AWS for input and output. Default is False.
            secret (str): AWS secret key. Required if use_aws is True.
            source_bucket (str): Name of the bucket storing input datasets. Required if use_aws is True.
            output_bucket (str): Name of the output bucket. Required if use_aws is True.
        """
        self.aws_tmp_dir: Path | None = None
        if use_aws:
            if not secret:
                raise Exception("if use_aws is True, secret is Required.")
            if not source_bucket:
                raise Exception("if use_aws is True, source_bucket is Required.")
            if not output_bucket:
                raise Exception("if use_aws is True, output_bucket is Required.")
            self.aws_tmp_dir = Path(tempfile.mkdtemp(dir="./tmp"))

        self.use_aws = use_aws
        self.secret = secret
        self.source_bucket = source_bucket
        self.output_bucket = output_bucket
        self.output_prefix = ""
        self.aws_helper = AWSHelper(self.secret)

        self.robot_types = robot_types
        self.robot_ids = robot_ids
        self.labels = labels
        self.location_names = location_names
        self.short_horizon_tasks = short_horizon_tasks

    def finalize(self):
        """
        Clean up resources after processing.

        This method is responsible for cleaning up any temporary directories
        and files that were created during the processing steps, particularly
        when using AWS temporary directories.
        """
        if self.use_aws and self.aws_tmp_dir is not None:
            shutil.rmtree(self.aws_tmp_dir)

    def _init_source_folders(
        self, source_folders: list[str], output_folder: str
    ) -> tuple[list[str], str]:
        """
        Initialize source folders, especially for AWS use cases.

        If AWS is utilized, this method downloads dataset directories
        from the specified S3 bucket into a temporary local directory.

        Args:
            source_folders (list[str]): List of source folder paths.
            output_folder (str): Path to the output folder.

        Returns:
            tuple[list[str], str]: A tuple containing a list of local directory paths where the folders have been initialized
            and the path to the new output folder.
        """
        if not self.use_aws:
            return source_folders, output_folder
        else:
            self.output_prefix = output_folder
            replaced_source_folders: list[str] = []
            for folder in source_folders:
                if self.aws_tmp_dir is not None:
                    local_dir: Path = self.aws_tmp_dir / folder
                    local_dir.mkdir(parents=True)
                    self.aws_helper.download_s3_directory(
                        bucket_name=self.source_bucket,
                        s3_prefix=folder,
                        local_dir=local_dir,
                    )
                    replaced_source_folders.append(str(local_dir))
            new_output_folder = (
                str(self.aws_tmp_dir / str(uuid4()))
                if self.aws_tmp_dir
                else output_folder
            )
            return replaced_source_folders, new_output_folder

    def _upload_merged_datasets(self, output_folder: str) -> None:
        """
        Upload merged datasets to an AWS S3 bucket.

        This method uploads the contents of the local output folder to the specified
        output S3 bucket using the configured AWSHelper.

        Args:
            output_folder (str): The local path to the folder containing merged datasets
            ready to be uploaded to S3.
        """
        self.aws_helper.upload_directory_to_s3(
            local_dir=Path(output_folder),
            bucket_name=self.output_bucket,
            s3_prefix=self.output_prefix,
        )

    def _validate_fps(self, source_folders):
        """
        Validate that all datasets have the same FPS

        Args:
            source_folders (list): List of source dataset folder paths

        Returns:
            tuple: (is_valid, fps_value, error_message)
        """
        fps_values = []
        fps_per_folder = {}

        for folder in source_folders:
            info_path = os.path.join(folder, "meta", "info.json")
            if not os.path.exists(info_path):
                return False, None, f"info.json not found in {folder}"

            try:
                with open(info_path) as f:
                    info = json.load(f)
                    if "fps" not in info:
                        return (
                            False,
                            None,
                            f"FPS not specified in {folder}/meta/info.json",
                        )

                    fps = info["fps"]
                    fps_values.append(fps)
                    fps_per_folder[folder] = fps
            except Exception as e:
                return False, None, f"Error reading {folder}/meta/info.json: {e}"

        # Check if all FPS values are the same
        unique_fps = set(fps_values)
        if len(unique_fps) > 1:
            error_msg = "Inconsistent FPS across datasets:\n"
            for folder, fps in fps_per_folder.items():
                error_msg += f"  {folder}: {fps}\n"
            return False, None, error_msg.strip()

        return True, fps_values[0], None

    def _validate_feature_shapes(self, source_folders):
        """
        Validate that all datasets have consistent feature shapes
        Specifically checks action.absolute shape consistency

        Args:
            source_folders (list): List of source dataset folder paths

        Returns:
            tuple: (is_valid, error_message)
        """
        feature_shapes_per_folder = {}

        for folder in source_folders:
            info_path = os.path.join(folder, "meta", "info.json")
            if not os.path.exists(info_path):
                return False, f"info.json not found in {folder}"

            try:
                with open(info_path) as f:
                    info = json.load(f)
                    if "features" not in info:
                        return (
                            False,
                            f"Features not specified in {folder}/meta/info.json",
                        )

                    # Extract action.absolute feature shape
                    features = info["features"]
                    if "action.absolute" not in features:
                        return (
                            False,
                            f"action.absolute feature not found in {folder}/meta/info.json",
                        )

                    action_absolute_shape = features["action.absolute"].get(
                        "shape", None
                    )
                    feature_shapes_per_folder[folder] = action_absolute_shape
            except Exception as e:
                return False, f"Error reading {folder}/meta/info.json: {e}"

        # Check if all action.absolute shapes are the same
        shapes = list(feature_shapes_per_folder.values())
        if not all(shape == shapes[0] for shape in shapes):
            error_msg = "Inconsistent action.absolute shapes across datasets:\n"
            for folder, shape in feature_shapes_per_folder.items():
                error_msg += f"  {folder}: {shape}\n"
            return False, error_msg.strip()

        return True, None

    def _get_error_episodes_from_qa_report(self, dataset_path):
        """
        Get error episode indices from qa_report.json

        Args:
            dataset_path (str): Dataset folder path

        Returns:
            list[int]: List of error episode indices
        """
        qa_report_file = os.path.join(dataset_path, "qa_report.json")
        try:
            qa_report_data_list = ProcessJson.load_jsonl_data(qa_report_file)
            return [
                qa_report_data["episode_index"]
                for qa_report_data in qa_report_data_list
                if qa_report_data.get("errors")
            ]
        except FileNotFoundError:
            print(f"Warning: No qa_report.json found in {dataset_path}")
            return []
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode JSON from {qa_report_file}")
            return []
        except Exception as e:
            print(f"Warning: Error reading {qa_report_file}: {e}")
            return []

    def _get_error_episodes_from_parquet(self, dataset_path):
        """
        Get error episode indices from parquet files
        (episodes where success_primitive_action = False)

        Args:
            dataset_path (str): Dataset folder path

        Returns:
            list[int]: List of error episode indices
        """
        error_indices = []
        parquet_file_paths = sorted(
            glob.glob(
                os.path.join(dataset_path, "data", "chunk-*", "episode_*.parquet")
            )
        )

        for parquet_file_path in parquet_file_paths:
            try:
                df = pd.read_parquet(parquet_file_path)
                if "success_primitive_action" in df.columns:
                    success_primitive_action = df["success_primitive_action"].unique()
                    if False in success_primitive_action:
                        error_indices.append(int(df["episode_index"].iloc[0]))
            except Exception as e:
                print(f"Warning: Error reading {parquet_file_path}: {e}")
                continue

        return error_indices

    def _get_filtered_episode_indices_excluding_robot_types(
        self, dataset_path: str
    ) -> list[int]:
        """
        Get indices of episodes that do not match the specified robot types.

        Args:
            dataset_path (str): Path to the dataset directory.

        Returns:
            list[int]: List of episode indices to filter out. These are the indices of episodes
                    whose robot types do not match the specified list of robot types.
        """
        if self.robot_types is None:
            return []

        info_path = os.path.join(dataset_path, "meta", "info.json")
        info: dict = ProcessJson.load_jsonl_data(info_path)
        robot_type: str = info.get("robot_type", "")
        if robot_type not in self.robot_types:
            episodes_path = os.path.join(dataset_path, "meta", "episodes.jsonl")
            episodes: list[dict[str, Any]] = ProcessJson.load_jsonl_data(
                episodes_path, True
            )
            return [int(episode["episode_index"]) for episode in episodes]

        return []

    def _get_filtered_episode_indices_by_robot_ids(
        self, dataset_path: str
    ) -> list[int]:
        """
        Get indices of episodes that do not match the specified robot IDs.

        Args:
            dataset_path (str): Path to the dataset directory.

        Returns:
            list[int]: List of episode indices to filter out. These are the indices of episodes
                    whose robot IDs do not match the specified list of robot IDs.
        """
        if self.robot_ids is None:
            return []
        return self._get_filtered_episode_indices_from_episodes_jsonl(
            dataset_path=dataset_path, key="hsr_id", filtered_items=self.robot_ids
        )

    def _get_filtered_episode_indices_by_labels(self, dataset_path: str) -> list[int]:
        """
        Get indices of episodes that do not match the specified labels.

        Args:
            dataset_path (str): Path to the dataset directory.

        Returns:
            list[int]: List of episode indices to filter out. These are the indices of episodes
                    whose labels do not match the specified list of labels.
        """
        if self.labels is None:
            return []
        return self._get_filtered_episode_indices_from_episodes_jsonl(
            dataset_path=dataset_path, key="label", filtered_items=self.labels
        )

    def _get_filtered_episode_indices_by_location_names(
        self, dataset_path: str
    ) -> list[int]:
        """
        Get indices of episodes that do not match the specified location names.

        Args:
            dataset_path (str): Path to the dataset directory.

        Returns:
            list[int]: List of episode indices to filter out. These are the indices of episodes
                    whose location names do not match the specified list of location names.
        """
        if self.location_names is None:
            return []
        return self._get_filtered_episode_indices_from_episodes_jsonl(
            dataset_path=dataset_path,
            key="location_name",
            filtered_items=self.location_names,
        )

    def _get_filtered_episode_indices_by_short_horizon_tasks(
        self, dataset_path: str
    ) -> list[int]:
        """
        Get indices of episodes that do not match the specified short horizon tasks.

        Args:
            dataset_path (str): Path to the dataset directory.

        Returns:
            list[int]: List of episode indices to filter out. These are the indices of episodes
                    whose short horizon tasks do not match the specified list of short horizon tasks.
        """
        if self.short_horizon_tasks is None:
            return []
        return self._get_filtered_episode_indices_from_episodes_jsonl(
            dataset_path=dataset_path,
            key="short_horizon_task",
            filtered_items=self.short_horizon_tasks,
        )

    def _get_filtered_episode_indices_from_episodes_jsonl(
        self, dataset_path: str, key: str, filtered_items: list[str]
    ) -> list[int]:
        """
        Get indices of episodes that do not match specified key-value pairs from episodes.jsonl.

        Args:
            dataset_path (str): Path to the dataset directory.
            key (str): The key to be checked within each episode's metadata.
            filtered_items (list[str]): The list of values that are allowed. Episodes containing any other value will be filtered out.

        Returns:
            list[int]: List of episode indices to filter out. These are the indices of episodes
                    whose specified key-values do not match the specified list of filtered items.
        """
        filtered_indices: list[int] = []
        episodes_path = os.path.join(dataset_path, "meta", "episodes.jsonl")
        episodes: list[dict[str, Any]] = ProcessJson.load_jsonl_data(
            episodes_path, True
        )
        for episode in episodes:
            if key not in episode or episode.get(key) not in filtered_items:
                filtered_indices.append(int(episode["episode_index"]))

        return filtered_indices

    def _get_filtered_episode_by_robot_types(
        self, parquet_file_paths, metas, filtered_robot_types
    ):
        """
        Filter episodes by robot types

        Args:
            parquet_file_paths (list): List of parquet file paths
            metas (dict): Metadata dictionary
            filtered_robot_types (list): List of robot types to filter

        Returns:
            list[int]: List of filtered episode indices
        """
        filtered_episode_indices = []
        episodes_meta = metas["episodes"]
        for episode_meta in episodes_meta:
            if episode_meta["hsr_id"] in filtered_robot_types:
                filtered_episode_indices.append(episode_meta["episode_index"])
        return filtered_episode_indices

    def _get_filtered_episode_by_primitive_action(
        self, parquet_file_paths, metas, filtered_primitive_actions
    ):
        """
        Filter episodes by primitive action

        Args:
            parquet_file_paths (list): List of parquet file paths
            metas (dict): Metadata dictionary
            filtered_primitive_actions (list): List of primitive actions to filter

        Returns:
            list[int]: List of filtered episode indices
        """
        filtered_primitive_action_indices = []
        tasks_meta = metas["tasks"]
        for task in tasks_meta:
            if task["task"] in filtered_primitive_actions:
                filtered_primitive_action_indices.append(task["task_index"])

        filtered_episode_indices = []
        for parquet_file_path in parquet_file_paths:
            df = pd.read_parquet(parquet_file_path)
            if (
                df["primitive_action_index"]
                .isin(filtered_primitive_action_indices)
                .any()
            ):
                filtered_episode_indices.append(int(df["episode_index"].iloc[0]))
        return filtered_episode_indices

    def _get_filtered_episode_by_short_horizon_task(
        self, parquet_file_paths, metas, filtered_short_horizon_tasks
    ):
        """
        Filter episodes by short horizon task

        Args:
            parquet_file_paths (list): List of parquet file paths
            metas (dict): Metadata dictionary
            filtered_short_horizon_tasks (list): List of short horizon tasks to filter

        Returns:
            list[int]: List of filtered episode indices
        """
        filtered_short_horizon_task_indices = []
        tasks_meta = metas["tasks"]
        for task in tasks_meta:
            if task["task"] in filtered_short_horizon_tasks:
                filtered_short_horizon_task_indices.append(task["task_index"])

        filtered_episode_indices = []
        for parquet_file_path in parquet_file_paths:
            df = pd.read_parquet(parquet_file_path)
            if (
                df["short_horizon_task_index"]
                .isin(filtered_short_horizon_task_indices)
                .any()
            ):
                filtered_episode_indices.append(int(df["episode_index"].iloc[0]))
        return filtered_episode_indices

    def _filter_episodes(self, source_folders):
        """
        Filter error episodes from all source datasets

        This method applies the following filtering criteria for each dataset:
        1. Exclude episodes whose robot types do not match the specified list (`robot_types`).
        2. Exclude episodes whose robot IDs do not match the specified list (`robot_ids`).
        3. Exclude episodes whose labels do not match the specified list (`labels`).
        4. Exclude episodes whose location names do not match the specified list (`location_names`).
        5. Exclude episodes whose short horizon tasks do not match the specified list (`short_horizon_tasks`).
        6. Exclude episodes with errors specified in `qa_report.json` (QA process detected errors).
        7. Exclude episodes with errors found in parquet files (where `success_primitive_action` is False).

        Args:
            source_folders (list): List of source dataset folder paths

        Returns:
            dict: {folder_path: set(error_episode_indices)}
        """
        filtered_episodes_map = {}

        for folder in source_folders:
            error_indices = set()

            # Filter based on info.json
            info_filtered_indices = (
                self._get_filtered_episode_indices_excluding_robot_types(folder)
            )
            error_indices.update(info_filtered_indices)
            print(
                f"  {folder}: Found {len(info_filtered_indices)} episodes excluded in info.json"
            )

            # Filter based on episodes.jsonl
            episodes_filtered_indices = self._get_filtered_episode_indices_by_robot_ids(
                folder
            )
            error_indices.update(episodes_filtered_indices)
            print(
                f"  {folder}: Found {len(episodes_filtered_indices)} episodes excluded by robot ids in episodes.jsonl"
            )

            episodes_filtered_indices = self._get_filtered_episode_indices_by_labels(
                folder
            )
            error_indices.update(episodes_filtered_indices)
            print(
                f"  {folder}: Found {len(episodes_filtered_indices)} episodes excluded by labels in episodes.jsonl"
            )

            episodes_filtered_indices = (
                self._get_filtered_episode_indices_by_location_names(folder)
            )
            error_indices.update(episodes_filtered_indices)
            print(
                f"  {folder}: Found {len(episodes_filtered_indices)} episodes excluded by location names in episodes.jsonl"
            )

            episodes_filtered_indices = (
                self._get_filtered_episode_indices_by_short_horizon_tasks(folder)
            )
            error_indices.update(episodes_filtered_indices)
            print(
                f"  {folder}: Found {len(episodes_filtered_indices)} episodes excluded by short horizon tasks in episodes.jsonl"
            )

            # Filter based on qa_report.json
            qa_errors = self._get_error_episodes_from_qa_report(folder)
            error_indices.update(qa_errors)
            if qa_errors:
                print(f"  {folder}: Found {len(qa_errors)} errors in qa_report.json")

            # Filter based on parquet files
            parquet_errors = self._get_error_episodes_from_parquet(folder)
            error_indices.update(parquet_errors)
            if parquet_errors:
                print(
                    f"  {folder}: Found {len(parquet_errors)} errors in parquet files"
                )

            filtered_episodes_map[folder] = error_indices
            if error_indices:
                print(f"  {folder}: Total {len(error_indices)} episodes to filter")

        total_filtered = sum(len(errors) for errors in filtered_episodes_map.values())
        print(f"Total episodes to filter across all datasets: {total_filtered}")

        return filtered_episodes_map

    def _merge_stats(self, stats_list):
        """
        Merge statistics from multiple datasets, ensuring dimensional consistency

        Args:
            stats_list (list): List of dictionaries containing statistics for each dataset

        Returns:
            dict: Merged statistics
        """
        # Initialize merged stats with the structure of the first stats
        merged_stats = {}

        # Find common features across all stats
        common_features = set(stats_list[0].keys())
        for stats in stats_list[1:]:
            common_features = common_features.intersection(set(stats.keys()))

        # Process features in the order they appear in the first stats file
        for feature in stats_list[0]:
            if feature not in common_features:
                continue

            merged_stats[feature] = {}

            # Find common stat types for this feature
            common_stat_types = []
            for stat_type in ["mean", "std", "max", "min"]:
                if all(stat_type in stats[feature] for stats in stats_list):
                    common_stat_types.append(stat_type)

            # Determine the original shape of each value
            original_shapes = []
            for stats in stats_list:
                if "mean" in stats[feature]:
                    shape = np.array(stats[feature]["mean"]).shape
                    original_shapes.append(shape)

            # Special handling for image features to preserve nested structure
            if feature.startswith("observation.images."):
                for stat_type in common_stat_types:
                    try:
                        values = [stats[feature][stat_type] for stats in stats_list]

                        # For image features, we need to preserve the nested structure
                        result = []

                        # For RGB channels
                        for channel_idx in range(len(values[0])):
                            channel_result = []

                            for pixel_idx in range(len(values[0][channel_idx])):
                                pixel_result = []

                                # For each pixel value
                                for value_idx in range(
                                    len(values[0][channel_idx][pixel_idx])
                                ):
                                    # Calculate statistic based on type
                                    if stat_type == "mean":
                                        avg = sum(
                                            values[i][channel_idx][pixel_idx][value_idx]
                                            for i in range(len(values))
                                        ) / len(values)
                                        pixel_result.append(avg)
                                    elif stat_type == "std":
                                        avg = sum(
                                            values[i][channel_idx][pixel_idx][value_idx]
                                            for i in range(len(values))
                                        ) / len(values)
                                        pixel_result.append(avg)
                                    elif stat_type == "max":
                                        max_val = max(
                                            values[i][channel_idx][pixel_idx][value_idx]
                                            for i in range(len(values))
                                        )
                                        pixel_result.append(max_val)
                                    elif stat_type == "min":
                                        min_val = min(
                                            values[i][channel_idx][pixel_idx][value_idx]
                                            for i in range(len(values))
                                        )
                                        pixel_result.append(min_val)

                                channel_result.append(pixel_result)

                            result.append(channel_result)

                        merged_stats[feature][stat_type] = result
                    except Exception as e:
                        print(
                            f"Warning: Error processing image feature {feature}.{stat_type}: {e}"
                        )
                        # Fallback to first value
                        merged_stats[feature][stat_type] = values[0]
            # If all shapes are the same, no need for special handling
            elif len({str(shape) for shape in original_shapes}) == 1:
                # All shapes are the same, use standard merging
                for stat_type in common_stat_types:
                    values = [stats[feature][stat_type] for stats in stats_list]

                    try:
                        # Calculate the new statistic based on the type
                        if stat_type == "mean":
                            if all("count" in stats[feature] for stats in stats_list):
                                counts = [
                                    stats[feature]["count"][0] for stats in stats_list
                                ]
                                total_count = sum(counts)
                                weighted_values = [
                                    np.array(val) * count / total_count
                                    for val, count in zip(values, counts, strict=False)
                                ]
                                merged_stats[feature][stat_type] = np.sum(
                                    weighted_values, axis=0
                                ).tolist()
                            else:
                                merged_stats[feature][stat_type] = np.mean(
                                    np.array(values), axis=0
                                ).tolist()

                        elif stat_type == "std":
                            if all("count" in stats[feature] for stats in stats_list):
                                counts = [
                                    stats[feature]["count"][0] for stats in stats_list
                                ]
                                total_count = sum(counts)
                                variances = [np.array(std) ** 2 for std in values]
                                weighted_variances = [
                                    var * count / total_count
                                    for var, count in zip(
                                        variances, counts, strict=False
                                    )
                                ]
                                merged_stats[feature][stat_type] = np.sqrt(
                                    np.sum(weighted_variances, axis=0)
                                ).tolist()
                            else:
                                merged_stats[feature][stat_type] = np.mean(
                                    np.array(values), axis=0
                                ).tolist()

                        elif stat_type == "max":
                            merged_stats[feature][stat_type] = np.maximum.reduce(
                                np.array(values)
                            ).tolist()

                        elif stat_type == "min":
                            merged_stats[feature][stat_type] = np.minimum.reduce(
                                np.array(values)
                            ).tolist()
                    except Exception as e:
                        print(f"Warning: Error processing {feature}.{stat_type}: {e}")
                        continue
            else:
                # Shapes are different, need special handling for state vectors
                if feature in ["observation.state", "action"]:
                    # For state vectors, we need to handle different dimensions
                    max_dim = max(
                        len(np.array(stats[feature]["mean"]).flatten())
                        for stats in stats_list
                        if "mean" in stats[feature]
                    )

                    for stat_type in common_stat_types:
                        try:
                            # Get values and their original dimensions
                            values_with_dims = []
                            for stats in stats_list:
                                val = np.array(stats[feature][stat_type]).flatten()
                                dim = len(val)
                                values_with_dims.append((val, dim))

                            # Initialize result array with zeros
                            result = np.zeros(max_dim)

                            # Calculate statistics for each dimension separately
                            if stat_type == "mean":
                                if all(
                                    "count" in stats[feature] for stats in stats_list
                                ):
                                    counts = [
                                        stats[feature]["count"][0]
                                        for stats in stats_list
                                    ]
                                    total_count = sum(counts)

                                    # For each dimension, calculate weighted mean of available values
                                    for d in range(max_dim):
                                        dim_values = []
                                        dim_weights = []
                                        for (val, dim), count in zip(
                                            values_with_dims, counts, strict=False
                                        ):
                                            if (
                                                d < dim
                                            ):  # Only use values that have this dimension
                                                dim_values.append(val[d])
                                                dim_weights.append(count)

                                        if (
                                            dim_values
                                        ):  # If we have values for this dimension
                                            weighted_sum = sum(
                                                v * w
                                                for v, w in zip(
                                                    dim_values,
                                                    dim_weights,
                                                    strict=False,
                                                )
                                            )
                                            result[d] = weighted_sum / sum(dim_weights)
                                else:
                                    # Simple average for each dimension
                                    for d in range(max_dim):
                                        dim_values = [
                                            val[d]
                                            for val, dim in values_with_dims
                                            if d < dim
                                        ]
                                        if dim_values:
                                            result[d] = sum(dim_values) / len(
                                                dim_values
                                            )

                            elif stat_type == "std":
                                if all(
                                    "count" in stats[feature] for stats in stats_list
                                ):
                                    counts = [
                                        stats[feature]["count"][0]
                                        for stats in stats_list
                                    ]
                                    total_count = sum(counts)

                                    # For each dimension, calculate weighted variance
                                    for d in range(max_dim):
                                        dim_variances = []
                                        dim_weights = []
                                        for (val, dim), count in zip(
                                            values_with_dims, counts, strict=False
                                        ):
                                            if (
                                                d < dim
                                            ):  # Only use values that have this dimension
                                                dim_variances.append(
                                                    val[d] ** 2
                                                )  # Square for variance
                                                dim_weights.append(count)

                                        if (
                                            dim_variances
                                        ):  # If we have values for this dimension
                                            weighted_var = sum(
                                                v * w
                                                for v, w in zip(
                                                    dim_variances,
                                                    dim_weights,
                                                    strict=False,
                                                )
                                            ) / sum(dim_weights)
                                            result[d] = np.sqrt(
                                                weighted_var
                                            )  # Take sqrt for std
                                else:
                                    # Simple average of std for each dimension
                                    for d in range(max_dim):
                                        dim_values = [
                                            val[d]
                                            for val, dim in values_with_dims
                                            if d < dim
                                        ]
                                        if dim_values:
                                            result[d] = sum(dim_values) / len(
                                                dim_values
                                            )

                            elif stat_type == "max":
                                # For each dimension, take the maximum of available values
                                for d in range(max_dim):
                                    dim_values = [
                                        val[d]
                                        for val, dim in values_with_dims
                                        if d < dim
                                    ]
                                    if dim_values:
                                        result[d] = max(dim_values)

                            elif stat_type == "min":
                                # For each dimension, take the minimum of available values
                                for d in range(max_dim):
                                    dim_values = [
                                        val[d]
                                        for val, dim in values_with_dims
                                        if d < dim
                                    ]
                                    if dim_values:
                                        result[d] = min(dim_values)

                            # Convert result to list and store
                            merged_stats[feature][stat_type] = result.tolist()

                        except Exception as e:
                            print(
                                f"Warning: Error processing {feature}.{stat_type} with different dimensions: {e}"
                            )
                            continue
                else:
                    # For other features with different shapes, use the first shape as template
                    template_shape = original_shapes[0]
                    print(f"Using shape {template_shape} as template for {feature}")

                    for stat_type in common_stat_types:
                        try:
                            # Use the first stats as template
                            merged_stats[feature][stat_type] = stats_list[0][feature][
                                stat_type
                            ]
                        except Exception as e:
                            print(
                                f"Warning: Error processing {feature}.{stat_type} with shape {template_shape}: {e}"
                            )
                            continue

            # Add count if available in all stats
            if all("count" in stats[feature] for stats in stats_list):
                try:
                    merged_stats[feature]["count"] = [
                        sum(stats[feature]["count"][0] for stats in stats_list)
                    ]
                except Exception as e:
                    print(f"Warning: Error processing {feature}.count: {e}")

        return merged_stats

    def _copy_videos(self, source_folders, output_folder, episode_mapping):
        """
        Copy video files from source folders to output folder, maintaining correct indices and structure

        Args:
            source_folders (list): List of source dataset folder paths
            output_folder (str): Output folder path
            episode_mapping (list): List of tuples containing (old_folder, old_index, new_index)
        """
        # Get info.json to determine video structure
        info_path = os.path.join(source_folders[0], "meta", "info.json")
        with open(info_path) as f:
            info = json.load(f)

        video_path_template = info["video_path"]

        # Identify video keys from the template
        # Example: "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        video_keys = []
        for feature_name, feature_info in info["features"].items():
            if feature_info.get("dtype") == "video":
                # Use the full feature name as the video key
                video_keys.append(feature_name)

        print(f"Found video keys: {video_keys}")

        # Copy videos for each episode
        for old_folder, old_index, new_index in episode_mapping:
            # Determine episode chunk (usually 0 for small datasets)
            episode_chunk = old_index // info["chunks_size"]
            new_episode_chunk = new_index // info["chunks_size"]

            for video_key in video_keys:
                # Try different possible source paths
                source_patterns = [
                    # Standard path with the episode index from metadata
                    os.path.join(
                        old_folder,
                        video_path_template.format(
                            episode_chunk=episode_chunk,
                            video_key=video_key,
                            episode_index=old_index,
                        ),
                    ),
                    # Try with 0-based indexing
                    os.path.join(
                        old_folder,
                        video_path_template.format(
                            episode_chunk=0, video_key=video_key, episode_index=0
                        ),
                    ),
                    # Try with different formatting
                    os.path.join(
                        old_folder,
                        f"videos/chunk-{episode_chunk:03d}/{video_key}/episode_{old_index}.mp4",
                    ),
                    os.path.join(
                        old_folder, f"videos/chunk-000/{video_key}/episode_000000.mp4"
                    ),
                ]

                # Find the first existing source path
                source_video_path = None
                for pattern in source_patterns:
                    if os.path.exists(pattern):
                        source_video_path = pattern
                        break

                if source_video_path:
                    dest_video_path = os.path.join(
                        output_folder,
                        video_path_template.format(
                            episode_chunk=new_episode_chunk,
                            video_key=video_key,
                            episode_index=new_index,
                        ),
                    )
                    ProcessVideo.copy_videos(source_video_path, dest_video_path)
                else:
                    # If no file is found, search the directory recursively
                    found = False
                    for root, _, files in os.walk(os.path.join(old_folder, "videos")):
                        for file in files:
                            if file.endswith(".mp4") and video_key in root:
                                source_video_path = os.path.join(root, file)

                                # Construct destination path
                                dest_video_path = os.path.join(
                                    output_folder,
                                    video_path_template.format(
                                        episode_chunk=new_episode_chunk,
                                        video_key=video_key,
                                        episode_index=new_index,
                                    ),
                                )
                                ProcessVideo.copy_videos(
                                    source_video_path, dest_video_path
                                )
                                found = True
                                break
                        if found:
                            break

                    if not found:
                        print(
                            f"Warning: Video file not found for {video_key}, episode {old_index} in {old_folder}"
                        )

    def _copy_data_files(
        self,
        source_folders,
        output_folder,
        episode_mapping,
        max_dim=18,
        fps=None,
        episode_to_frame_index=None,
        folder_to_task_strings=None,
        task_string_to_new_index=None,
        chunks_size=1000,
        default_fps=20,
    ):
        """
        Copy and process parquet data files, including dimension padding and index updates

        Args:
            source_folders (list): List of source dataset folder paths
            output_folder (str): Output folder path
            episode_mapping (list): List of tuples containing (old_folder, old_index, new_index)
            max_dim (int): Maximum dimension for vectors
            fps (float, optional): Frame rate
            episode_to_frame_index (dict, optional): Mapping of episode index to frame index
            folder_to_task_strings (dict, optional): Task string mappings
            task_string_to_new_index (dict, optional): New task index mappings
            chunks_size (int): Number of episodes per chunk
            default_fps (float): Default frame rate

        Returns:
            bool: Whether at least one file was successfully copied
        """
        if fps is None:
            info_path = os.path.join(source_folders[0], "meta", "info.json")
            if os.path.exists(info_path):
                with open(info_path) as f:
                    info = json.load(f)
                    fps = info.get("fps", default_fps)
            else:
                fps = default_fps

        print(f"Using FPS={fps}")

        total_copied = 0
        total_failed = 0
        failed_files = []

        for i, (old_folder, old_index, new_index) in enumerate(episode_mapping):
            episode_str = f"episode_{old_index:06d}.parquet"
            source_paths = [
                os.path.join(old_folder, "parquet", episode_str),
                os.path.join(old_folder, "data", episode_str),
            ]

            source_path = None
            for path in source_paths:
                if os.path.exists(path):
                    source_path = path
                    break

            if source_path:
                try:
                    process_parquet = ProcessParquet(source_path)
                    process_parquet.set_episode_index(new_index)
                    process_parquet.set_index(new_index, episode_to_frame_index)

                    if (
                        folder_to_task_strings
                        and task_string_to_new_index
                        and old_folder in folder_to_task_strings
                    ):
                        local_task_map = folder_to_task_strings[old_folder]
                        for col_name in [
                            "task_index",
                            "primitive_action_index",
                            "short_horizon_task_index",
                        ]:
                            process_parquet.set_string_based_mapping(
                                col_name, local_task_map, task_string_to_new_index
                            )

                    dest_path = process_parquet.to_parquet(
                        new_index, chunks_size, output_folder
                    )

                    total_copied += 1
                    print(f"Processed and saved: {dest_path}")

                except Exception as e:
                    error_msg = f"Processing {source_path} failed: {e}"
                    print(error_msg)
                    traceback.print_exc()
                    failed_files.append(
                        {"file": source_path, "reason": str(e), "episode": old_index}
                    )
                    total_failed += 1
            else:
                # Recursive search
                found = False
                for root, _, files in os.walk(old_folder):
                    for file in files:
                        if (
                            file.endswith(".parquet")
                            and f"episode_{old_index:06d}" in file
                        ):
                            try:
                                source_path = os.path.join(root, file)
                                process_parquet = ProcessParquet(source_path)
                                process_parquet.set_episode_index(new_index)
                                process_parquet.set_index(
                                    new_index, episode_to_frame_index
                                )

                                if (
                                    folder_to_task_strings
                                    and task_string_to_new_index
                                    and old_folder in folder_to_task_strings
                                ):
                                    local_task_map = folder_to_task_strings[old_folder]
                                    for col_name in [
                                        "task_index",
                                        "primitive_action_index",
                                        "short_horizon_task_index",
                                    ]:
                                        process_parquet.set_string_based_mapping(
                                            col_name,
                                            local_task_map,
                                            task_string_to_new_index,
                                        )

                                dest_path = process_parquet.to_parquet(
                                    new_index, chunks_size, output_folder
                                )

                                total_copied += 1
                                found = True
                                print(f"Processed and saved: {dest_path}")
                                break
                            except Exception as e:
                                error_msg = f"Processing {source_path} failed: {e}"
                                print(error_msg)
                                traceback.print_exc()
                                failed_files.append(
                                    {
                                        "file": source_path,
                                        "reason": str(e),
                                        "episode": old_index,
                                    }
                                )
                                total_failed += 1
                        if found:
                            break

                if not found:
                    error_msg = f"Could not find parquet file for episode {old_index}, source folder: {old_folder}"
                    print(error_msg)
                    failed_files.append(
                        {
                            "file": f"episode_{old_index:06d}.parquet",
                            "reason": "File not found",
                            "folder": old_folder,
                        }
                    )
                    total_failed += 1

        print(f"Copied {total_copied} data files, {total_failed} failed")

        if failed_files:
            print("\nDetails of failed files:")
            for i, failed in enumerate(failed_files):
                print(f"{i + 1}. File: {failed['file']}")
                if "folder" in failed:
                    print(f"   Folder: {failed['folder']}")
                if "episode" in failed:
                    print(f"   Episode index: {failed['episode']}")
                print(f"   Reason: {failed['reason']}")
                print("---")

        return total_copied > 0

    def _validate_timestamps(self, source_folders, tolerance_s=1e-4):
        """
        Validate timestamp structure of source datasets, identify potential issues

        Args:
            source_folders (list): List of source dataset folder paths
            tolerance_s (float): Tolerance for timestamp discontinuities in seconds

        Returns:
            tuple: (issues, fps_values) - List of issues and list of detected FPS values
        """
        issues = []
        fps_values = []

        for folder in source_folders:
            try:
                # Try to get FPS from info.json
                info_path = os.path.join(folder, "meta", "info.json")
                if os.path.exists(info_path):
                    with open(info_path) as f:
                        info = json.load(f)
                        if "fps" in info:
                            fps = info["fps"]
                            fps_values.append(fps)
                            print(f"Dataset {folder} FPS={fps}")

                # Check if any parquet files contain timestamps
                parquet_path = None
                for root, _, files in os.walk(os.path.join(folder, "parquet")):
                    for file in files:
                        if file.endswith(".parquet"):
                            parquet_path = os.path.join(root, file)
                            break
                    if parquet_path:
                        break

                if not parquet_path:
                    for root, _, files in os.walk(os.path.join(folder, "data")):
                        for file in files:
                            if file.endswith(".parquet"):
                                parquet_path = os.path.join(root, file)
                                break
                        if parquet_path:
                            break

                if parquet_path:
                    df = pd.read_parquet(parquet_path)
                    timestamp_cols = [
                        col for col in df.columns if "timestamp" in col or "time" in col
                    ]
                    if timestamp_cols:
                        print(
                            f"Dataset {folder} contains timestamp columns: {timestamp_cols}"
                        )
                    else:
                        issues.append(
                            f"Warning: Dataset {folder} has no timestamp columns"
                        )
                else:
                    issues.append(
                        f"Warning: No parquet files found in dataset {folder}"
                    )

            except Exception as e:
                issues.append(f"Error: Failed to validate dataset {folder}: {e}")
                print(f"Validation error: {e}")
                traceback.print_exc()

        # Check if FPS values are consistent
        if len(set(fps_values)) > 1:
            issues.append(f"Warning: Inconsistent FPS across datasets: {fps_values}")

        return issues, fps_values

    def _pad_parquet_data(
        self, source_path, target_path, original_dim=14, target_dim=18
    ):
        """
        Extend parquet data from original dimension to target dimension by zero-padding

        Args:
            source_path (str): Source parquet file path
            target_path (str): Target parquet file path
            original_dim (int): Original vector dimension
            target_dim (int): Target vector dimension

        Returns:
            DataFrame: Padded DataFrame
        """
        # Read parquet file
        df = pd.read_parquet(source_path)

        # Print column names for debugging
        print(f"Columns in {source_path}: {df.columns.tolist()}")

        # Create a new DataFrame to store padded data
        new_df = df.copy()

        # Check if observation.state and action columns exist
        if "observation.state" in df.columns:
            # Check the first row of data to confirm if it is a vector
            first_state = df["observation.state"].iloc[0]
            print(
                f"First observation.state type: {type(first_state)}, value: {first_state}"
            )

            # If it's a vector (list or numpy array)
            if isinstance(first_state, (list, np.ndarray)):
                # Check dimension
                state_dim = len(first_state)
                print(f"observation.state dimension: {state_dim}")

                if state_dim < target_dim:
                    # Pad vector
                    print(
                        f"Padding observation.state from {state_dim} to {target_dim} dimensions"
                    )
                    new_df["observation.state"] = df["observation.state"].apply(
                        lambda x: np.pad(
                            x, (0, target_dim - len(x)), "constant"
                        ).tolist()
                    )

        # Process action column similarly
        if "action" in df.columns:
            # Check first row data
            first_action = df["action"].iloc[0]
            print(f"First action type: {type(first_action)}, value: {first_action}")

            # If it's a vector
            if isinstance(first_action, (list, np.ndarray)):
                # Check dimension
                action_dim = len(first_action)
                print(f"action dimension: {action_dim}")

                if action_dim < target_dim:
                    # Pad vector
                    print(
                        f"Padding action from {action_dim} to {target_dim} dimensions"
                    )
                    new_df["action"] = df["action"].apply(
                        lambda x: np.pad(
                            x, (0, target_dim - len(x)), "constant"
                        ).tolist()
                    )

        # Ensure target directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # Save to a new parquet file
        new_df.to_parquet(target_path, index=False)

        print(f"Processed {source_path} and saved to {target_path}")

        return new_df

    def _validate_versions(self, source_folders: list[str]) -> bool:
        """
        Validate that all datasets have either a unique codebase version or are not set.

        This method checks the `info.json` file in each source dataset folder to retrieve
        the `codebase_version`. It ensures that either all non-null versions are the same, or none is set.

        Args:
            source_folders (list[str]): A list of paths to the source dataset folders.

        Returns:
            bool: True if all existing versions are either the same, one or none is set,
                False if there are multiple differing versions.
        """
        versions: list[str | None] = []
        for folder in source_folders:
            info_path = os.path.join(folder, "meta", "info.json")
            if os.path.exists(info_path):
                with open(info_path) as f:
                    info: dict = json.load(f)
                    versions.append(info.get("codebase_version"))

        filtered_versions = [version for version in versions if version is not None]
        return len(set(filtered_versions)) <= 1

    def _merge_datasets(
        self,
        source_folders,
        output_folder,
        filtered_episodes_map,
        max_dim=32,
        fps=20,
        chunk_size=1000,
    ):
        """
        Merge multiple datasets into a single output

        This method handles:
        1. Merging all episodes, tasks and stats
        2. Renumbering all indices to maintain continuity
        3. Pads vector dimensions for consistency
        4. Updates metadata files
        5. Copies and processes data and video files

        Args:
            source_folders (list): List of source dataset folder paths
            output_folder (str): Output folder path
            filtered_episodes_map (dict): Map of {folder_path: set(error_episode_indices)}
            max_dim (int): Maximum dimension for vectors
            fps (int): Frame rate
            chunk_size (int): Number of episodes per chunk
        """
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, "meta"), exist_ok=True)

        print(f"Using default FPS value: {fps}")

        # Initialize collections
        all_episodes = []
        all_episodes_stats = []
        all_tasks = []
        folder_to_task_strings = {}
        task_string_to_new_index = {}
        all_unique_tasks = []

        # Build task mappings across all datasets
        for folder in source_folders:
            tasks_path = os.path.join(folder, "meta", "tasks.jsonl")
            if not os.path.exists(tasks_path):
                continue

            folder_tasks = ProcessJson.load_jsonl_data(tasks_path, True)
            current_folder_map = {
                task["task_index"]: task["task"] for task in folder_tasks
            }
            folder_to_task_strings[folder] = current_folder_map

            for task in folder_tasks:
                task_desc = task["task"]
                if task_desc not in task_string_to_new_index:
                    new_index = len(all_unique_tasks)
                    task_string_to_new_index[task_desc] = new_index
                    all_unique_tasks.append(
                        {"task_index": new_index, "task": task_desc}
                    )

        all_tasks = all_unique_tasks

        total_frames = 0
        total_episodes = 0
        total_videos = 0

        # Keep track of episode mapping (old_folder, old_index, new_index)
        episode_mapping = []

        # Collect all stats for proper merging
        all_stats_data = []

        # Track dimensions for each folder
        folder_dimensions = {}

        # Add a variable to track cumulative frames
        cumulative_frame_count = 0

        # Create a mapping to store the starting frame index for each new episode index
        episode_to_frame_index = {}

        # Get chunks_size from first dataset
        info_path = os.path.join(source_folders[0], "meta", "info.json")
        chunks_size = chunk_size
        if os.path.exists(info_path):
            info = ProcessJson.load_jsonl_data(info_path, False)
            chunks_size = info.get("chunks_size", chunk_size)

        # Process each source folder
        for folder in source_folders:
            try:
                # Get total_videos from info.json
                folder_info_path = os.path.join(folder, "meta", "info.json")
                if os.path.exists(folder_info_path):
                    folder_info = ProcessJson.load_jsonl_data(folder_info_path, False)
                    if "total_videos" in folder_info:
                        folder_videos = folder_info["total_videos"]
                        total_videos += folder_videos
                        print(
                            f"Read video count from {folder}'s info.json: {folder_videos}"
                        )

                # Detect dimensions
                folder_dim = max_dim

                # Try to find a parquet file to determine dimensions
                for root, _dirs, files in os.walk(folder):
                    for file in files:
                        if file.endswith(".parquet"):
                            try:
                                df = pd.read_parquet(os.path.join(root, file))
                                if "observation.state" in df.columns:
                                    first_state = df["observation.state"].iloc[0]
                                    if isinstance(first_state, (list, np.ndarray)):
                                        folder_dim = len(first_state)
                                        print(
                                            f"Detected {folder_dim} dimensions in {folder}"
                                        )
                                        break
                            except Exception as e:
                                print(f"Error checking dimensions in {folder}: {e}")
                            break
                    if folder_dim != max_dim:
                        break

                folder_dimensions[folder] = folder_dim

                # Load episodes
                episodes_path = os.path.join(folder, "meta", "episodes.jsonl")
                if not os.path.exists(episodes_path):
                    print(f"Warning: Episodes file not found in {folder}, skipping")
                    continue

                episodes = ProcessJson.load_jsonl_data(episodes_path, True)

                # Load episode stats
                episodes_stats_path = os.path.join(
                    folder, "meta", "episodes_stats.jsonl"
                )
                episodes_stats = []
                if os.path.exists(episodes_stats_path):
                    episodes_stats = ProcessJson.load_jsonl_data(
                        episodes_stats_path, True
                    )

                # Create stats mapping
                stats_map = {}
                for stat in episodes_stats:
                    if "episode_index" in stat:
                        stats_map[stat["episode_index"]] = stat

                # Get error indices for this folder
                error_indices = (filtered_episodes_map or {}).get(folder, set())

                # Process episodes
                for episode in episodes:
                    old_index = episode["episode_index"]
                    if old_index in error_indices:
                        continue
                    new_index = total_episodes

                    episode["episode_index"] = new_index
                    all_episodes.append(episode)

                    # Update stats if available
                    if old_index in stats_map:
                        stats = stats_map[old_index]
                        stats["episode_index"] = new_index
                        all_episodes_stats.append(stats)

                        if "stats" in stats:
                            all_stats_data.append(stats["stats"])

                    # Add to mapping
                    episode_mapping.append((folder, old_index, new_index))

                    # Update counters
                    total_episodes += 1
                    total_frames += episode["length"]

                    episode_to_frame_index[new_index] = cumulative_frame_count
                    cumulative_frame_count += episode["length"]

            except Exception as e:
                print(f"Error processing folder {folder}: {e}")
                continue

        print(f"Processed {total_episodes} episodes from {len(source_folders)} folders")

        # Save metadata
        ProcessJson.save_jsonl(
            all_episodes, os.path.join(output_folder, "meta", "episodes.jsonl")
        )
        ProcessJson.save_jsonl(
            all_episodes_stats,
            os.path.join(output_folder, "meta", "episodes_stats.jsonl"),
        )
        ProcessJson.save_jsonl(
            all_tasks, os.path.join(output_folder, "meta", "tasks.jsonl")
        )

        # Merge and save stats
        stats_list = []
        for folder in source_folders:
            stats_path = os.path.join(folder, "meta", "stats.json")
            if os.path.exists(stats_path):
                with open(stats_path) as f:
                    stats = json.load(f)
                    stats_list.append(stats)

        if stats_list:
            merged_stats = self._merge_stats(stats_list)

            # Update with episode-specific stats if available
            if all_stats_data:
                for feature in merged_stats:
                    if feature in all_stats_data[0]:
                        values = [
                            stat[feature] for stat in all_stats_data if feature in stat
                        ]

                        max_dim_feat = max(
                            len(np.array(val.get("mean", [0])).flatten())
                            for val in values
                            if "mean" in val
                        )

                        # Update statistics with padding
                        if "count" in merged_stats[feature]:
                            merged_stats[feature]["count"] = [
                                sum(
                                    stat.get("count", [0])[0]
                                    for stat in values
                                    if "count" in stat
                                )
                            ]

                        # Update min/max/mean/std with padding
                        for stat_type in ["min", "max", "mean", "std"]:
                            if stat_type in merged_stats[feature] and all(
                                stat_type in stat for stat in values
                            ):
                                padded_values = []
                                for val in values:
                                    val_array = np.array(val[stat_type])
                                    val_flat = val_array.flatten()
                                    if len(val_flat) < max_dim_feat:
                                        padded = np.zeros(max_dim_feat)
                                        padded[: len(val_flat)] = val_flat
                                        padded_values.append(padded)
                                    else:
                                        padded_values.append(val_flat)

                                if stat_type == "min":
                                    merged_stats[feature][stat_type] = (
                                        np.minimum.reduce(padded_values).tolist()
                                    )
                                elif stat_type == "max":
                                    merged_stats[feature][stat_type] = (
                                        np.maximum.reduce(padded_values).tolist()
                                    )
                                elif stat_type == "mean":
                                    if all("count" in stat for stat in values):
                                        counts = [stat["count"][0] for stat in values]
                                        total_count = sum(counts)
                                        weighted_means = [
                                            mean * count / total_count
                                            for mean, count in zip(
                                                padded_values, counts, strict=False
                                            )
                                        ]
                                        merged_stats[feature]["mean"] = np.sum(
                                            weighted_means, axis=0
                                        ).tolist()
                                    else:
                                        merged_stats[feature]["mean"] = np.mean(
                                            padded_values, axis=0
                                        ).tolist()
                                elif stat_type == "std":
                                    if all("count" in stat for stat in values):
                                        counts = [stat["count"][0] for stat in values]
                                        total_count = sum(counts)
                                        variances = [std**2 for std in padded_values]
                                        weighted_variances = [
                                            var * count / total_count
                                            for var, count in zip(
                                                variances, counts, strict=False
                                            )
                                        ]
                                        merged_stats[feature]["std"] = np.sqrt(
                                            np.sum(weighted_variances, axis=0)
                                        ).tolist()
                                    else:
                                        merged_stats[feature]["std"] = np.mean(
                                            padded_values, axis=0
                                        ).tolist()

            with open(os.path.join(output_folder, "meta", "stats.json"), "w") as f:
                json.dump(merged_stats, f, indent=4)

        # Update info.json
        info = ProcessJson.load_jsonl_data(info_path, False)
        info["total_episodes"] = total_episodes
        info["total_frames"] = total_frames
        info["total_tasks"] = len(all_tasks)
        info["total_chunks"] = (total_episodes + info["chunks_size"] - 1) // info[
            "chunks_size"
        ]
        info["splits"] = {"train": f"0:{total_episodes}"}

        # Update feature dimensions
        if "features" in info:
            actual_max_dim = max_dim
            for _folder, dim in folder_dimensions.items():
                actual_max_dim = max(actual_max_dim, dim)

            for feature_name in ["observation.state", "action"]:
                if (
                    feature_name in info["features"]
                    and "shape" in info["features"][feature_name]
                ):
                    info["features"][feature_name]["shape"] = [actual_max_dim]
                    print(f"Updated {feature_name} shape to {actual_max_dim}")

        info["total_videos"] = total_videos
        print(f"Update total videos to: {total_videos}")

        with open(os.path.join(output_folder, "meta", "info.json"), "w") as f:
            json.dump(info, f, indent=4)

        # Copy files
        self._copy_videos(source_folders, output_folder, episode_mapping)
        self._copy_data_files(
            source_folders,
            output_folder,
            episode_mapping,
            max_dim=max_dim,
            fps=fps,
            episode_to_frame_index=episode_to_frame_index,
            folder_to_task_strings=folder_to_task_strings,
            task_string_to_new_index=task_string_to_new_index,
            chunks_size=chunks_size,
        )

        print(
            f"Merged {total_episodes} episodes with {total_frames} frames into {output_folder}"
        )

    def package(
        self,
        source_folders,
        output_folder,
        ignore_check_version,
        fps=None,
        max_dim=32,
        chunk_size=1000,
    ):
        """
        Package multiple datasets into a single unified dataset

        This method executes the following workflow:
        1. Validate FPS consistency across all datasets
        2. Validate feature shape consistency
        3. Filter error episodes from each dataset (both qa_report and parquet)
        4. Merge all datasets into a single output

        Args:
            source_folders (list): List of source dataset folder paths
            output_folder (str): Output folder path
            fps (int, optional): Expected FPS value. If not provided, will use the FPS from datasets
            max_dim (int): Maximum dimension for vectors (default: 32)
            chunk_size (int): Number of episodes per chunk (default: 1000)

        Returns:
            bool: True if packaging succeeded, False otherwise
        """
        print(f"Starting package process for {len(source_folders)} datasets...")

        # Step 1: Preparing package
        print("\n=== Step 1: Preparing package ===")
        source_folders, output_folder = self._init_source_folders(
            source_folders, output_folder
        )

        # Step 2: Validate FPS
        print("\n=== Step 2: Validating FPS ===")
        is_valid, detected_fps, error_msg = self._validate_fps(source_folders)
        if not is_valid:
            print(f"FPS validation failed: {error_msg}")
            return False

        if fps is not None and fps != detected_fps:
            print(
                f"Warning: Specified FPS ({fps}) differs from detected FPS ({detected_fps})"
            )
            print(f"Using detected FPS: {detected_fps}")
        fps = detected_fps
        print(f"✓ All datasets have consistent FPS: {fps}")

        # Step 3: Validate feature shapes
        print("\n=== Step 3: Validating Feature Shapes ===")
        is_valid, error_msg = self._validate_feature_shapes(source_folders)
        if not is_valid:
            print(f"Feature shape validation failed: {error_msg}")
            return False
        print("✓ All datasets have consistent feature shapes")

        # Step 4: Validate Versions
        print("\n=== Step 4: Validating versions ===")
        if not ignore_check_version:
            if not self._validate_versions(source_folders):
                print("Versions validation failed")
                return False
            print("✓ All existing versions are either the same")
        else:
            print("✓ Skip versions validation")

        # Step 5: Filter error episodes (both qa_report and parquet)
        print("\n=== Step 5: Filtering Error Episodes ===")
        filtered_episodes_map = self._filter_episodes(source_folders)

        # Step 6: Merge datasets
        print("\n=== Step 6: Merging Datasets ===")
        try:
            self._merge_datasets(
                source_folders,
                output_folder,
                filtered_episodes_map=filtered_episodes_map,
                max_dim=max_dim,
                fps=fps,
                chunk_size=chunk_size,
            )
            print(f"\n✓ Successfully packaged datasets to {output_folder}")

            # Step 7: If use AWS, upload merged datasets
            if self.use_aws:
                print("\n=== Step 7: If use AWS, upload merged datasets ===")
                self._upload_merged_datasets(output_folder=output_folder)

            return True
        except Exception as e:
            print(f"Error during merge: {e}")
            traceback.print_exc()
            return False


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Package multiple datasets with filtering and validation."
    )
    parser.add_argument(
        "--sources", nargs="+", required=True, help="List of source folder paths"
    )
    parser.add_argument("--output", required=True, help="Output folder path")
    parser.add_argument(
        "--fps", type=int, help="Expected FPS (if not provided, will be detected)"
    )
    parser.add_argument(
        "--max_dim", type=int, default=32, help="Maximum dimension (default: 32)"
    )
    parser.add_argument(
        "--use_aws",
        action="store_true",
        help="Whether to use AWS for input/output (default: false)",
    )
    parser.add_argument(
        "--secret",
        type=str,
        default="",
        help="AWS secret, required if use_aws is true",
    )
    parser.add_argument(
        "--source_bucket",
        type=str,
        default="",
        help="Name of the bucket storing the input datasets, required if use_aws is true",
    )
    parser.add_argument(
        "--output_bucket",
        type=str,
        default="",
        help="Name of the bucket for output, required if use_aws is true",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Episodes per chunk (default: 1000)",
    )
    parser.add_argument(
        "--ignore_check_version",
        type=str,
        default="false",
        help="ignore check version (default: False)",
    )

    parser.add_argument(
        "--robot_types",
        type=str,
        nargs="+",
        default=None,
        help="List of robot types separated by space",
    )
    parser.add_argument(
        "--robot_ids",
        type=str,
        nargs="+",
        default=None,
        help="List of robot ids separated by space",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="List of labels separated by space",
    )
    parser.add_argument(
        "--location_names",
        type=str,
        nargs="+",
        default=None,
        help="List of location names separated by space",
    )
    parser.add_argument(
        "--short_horizon_tasks",
        type=str,
        nargs="+",
        default=None,
        help="List of short horizon tasks separated by space",
    )

    # Parse arguments
    args = parser.parse_args()

    # Execute package
    package_tool = PackageTool(
        robot_types=args.robot_types,
        robot_ids=args.robot_ids,
        labels=args.labels,
        location_names=args.location_names,
        short_horizon_tasks=args.short_horizon_tasks,
        use_aws=args.use_aws,
        secret=args.secret,
        source_bucket=args.source_bucket,
        output_bucket=args.output_bucket,
    )
    try:
        success = package_tool.package(
            source_folders=args.sources,
            output_folder=args.output,
            fps=args.fps,
            max_dim=args.max_dim,
            chunk_size=args.chunk_size,
            ignore_check_version=args.ignore_check_version == "true",
        )
    finally:
        package_tool.finalize()

    exit(0 if success else 1)
