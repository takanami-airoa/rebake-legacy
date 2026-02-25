import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from airoa_metadata.versions.v1_0 import MetadataV1_0
from hsr_data_converter.commands.rosbag2lerobot.core_converter import (
    _convert_aggregate_rosbag_to_lerobot_format,
    _convert_individual_rosbag_to_lerobot_format,
    _convert_metadata_to_v1_0,
    _create_episodes_info,
    perform_conversion,
)


class TestPerformConversion:
    @pytest.fixture
    def mock_typestore(self):
        with patch("rosbags.typesys.store.Typestore", autospec=True) as typestore:
            yield typestore

    @pytest.fixture
    def mock_config(self):
        cfg = MagicMock()
        cfg.conversion_type = "individual"
        cfg.repo_id = "example_repo_id"
        cfg.fps = 30
        cfg.robot_type = "example_robot"
        cfg.separate_per_primitive = True
        return cfg

    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter._convert_individual_rosbag_to_lerobot_format"
    )
    @patch("rosbags.typesys.get_typestore")
    def test_standard_individual_conversion(
        self, mock_get_typestore, mock_convert_individual, mock_config, mock_typestore
    ):
        mock_get_typestore.return_value = mock_typestore
        raw_dir = Path("/path/to/raw")
        out_dir = Path("/path/to/out")

        perform_conversion(mock_config, raw_dir, out_dir, "source_dir")

        mock_convert_individual.assert_called()

    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter._convert_aggregate_rosbag_to_lerobot_format"
    )
    @patch("rosbags.typesys.get_typestore")
    def test_standard_aggregate_conversion(
        self, mock_get_typestore, mock_convert_aggregate, mock_config, mock_typestore
    ):
        mock_get_typestore.return_value = mock_typestore
        mock_config.conversion_type = "aggregate"
        raw_dir = Path("/path/to/raw")
        out_dir = Path("/path/to/out")

        perform_conversion(mock_config, raw_dir, out_dir, "source_dir")

        mock_convert_aggregate.assert_called()

    def test_error_invalid_conversion_type(self, mock_config):
        mock_config.conversion_type = "invalid_type"
        raw_dir = Path("/path/to/raw")
        out_dir = Path("/path/to/out")

        with pytest.raises(ValueError, match="Invalid conversion type: invalid_type"):
            perform_conversion(mock_config, raw_dir, out_dir, "source_dir")


class TestConvertIndividualRosbagToLeRobotFormat:
    @pytest.fixture
    def mock_dataset(self):
        with patch(
            "lerobot.datasets.lerobot_dataset.LeRobotDataset.create", autospec=True
        ) as create_mock:
            dataset_mock = MagicMock()
            create_mock.return_value = dataset_mock
            yield dataset_mock

    @pytest.fixture
    def mock_load_hsr_episodes(self):
        with patch(
            "hsr_data_converter.commands.rosbag2lerobot.core_converter.load_hsr_episodes",
            autospec=True,
        ) as load_mock:
            yield load_mock

    @pytest.fixture
    def mock_update_episodes_jsonl(self):
        with patch(
            "hsr_data_converter.commands.rosbag2lerobot.core_converter.update_episodes_jsonl",
            autospec=True,
        ) as update_mock:
            yield update_mock

    @pytest.fixture
    def mock_metadata_loader(self):
        with patch(
            "airoa_metadata.core.loader.MetadataLoader.load_from_file", autospec=True
        ) as load_mock:
            metadata_mock = MagicMock()
            metadata_mock.run.segments = [{"is_composite": False}]
            metadata_mock.run.instructions = [{"text": ["short task"]}]
            load_mock.return_value = metadata_mock
            yield load_mock

    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter._create_episodes_info",
        return_value={},
    )
    @patch(
        "airoa_metadata.versions.v0_0.MetadataV0_0.convert", return_value=MagicMock()
    )
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter.time.time",
        side_effect=[0, 5, 10],
    )
    @patch("hsr_data_converter.commands.rosbag2lerobot.core_converter.shutil.rmtree")
    def test_standard_conversion(
        self,
        mock_rmtree,
        mock_time,
        mock_convert,
        mock_create_episodes_info,
        mock_dataset,
        mock_load_hsr_episodes,
        mock_update_episodes_jsonl,
        mock_metadata_loader,
    ):
        cfg = MagicMock()
        cfg.repo_id = "example_repo_id"
        cfg.fps = 30
        cfg.robot_type = "example_robot"
        cfg.separate_per_primitive = True

        raw_dir = MagicMock(spec=Path)
        out_dir = MagicMock(spec=Path)
        source_dir = "source_dir"
        typestore = MagicMock()

        rosbag_dir = MagicMock(spec=Path)
        rosbag_dir.name = "rosbag_1"
        rosbag_dir.iterdir.return_value = [rosbag_dir]
        raw_dir.iterdir.return_value = [rosbag_dir]

        meta_path = rosbag_dir / "meta.json"
        meta_path.exists.return_value = True

        mock_load_hsr_episodes.return_value = (
            [[], [{"success_primitive_action": [1]}]],
            [[], ["task"]],
            [[], [123456789]],
            1234567890,
        )

        with patch.dict(
            "os.environ",
            {"LINEAGE_ENABLED": "true", "MARQUEZ_URL": "http://example.com"},
        ):
            result = _convert_individual_rosbag_to_lerobot_format(
                cfg, raw_dir, out_dir, source_dir, typestore
            )

        assert result == mock_dataset
        mock_load_hsr_episodes.assert_called_once()
        mock_update_episodes_jsonl.assert_called_once()
        mock_dataset.add_frame.assert_called()
        mock_dataset.save_episode.assert_called_once()
        mock_create_episodes_info.assert_called_once()

    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter.load_hsr_episodes",
        side_effect=Exception("Failed to load episodes"),
    )
    @patch("airoa_metadata.core.loader.MetadataLoader.load_from_file", autospec=True)
    @patch(
        "airoa_metadata.versions.v0_0.MetadataV0_0.convert", return_value=MagicMock()
    )
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter.time.time",
        side_effect=[0, 5, 10],
    )
    def test_error_failed_loading_episodes(
        self,
        mock_time,
        mock_convert,
        mock_metadata_loader,
        mock_load_hsr_episodes,
        mock_dataset,
    ):
        cfg = MagicMock()
        cfg.repo_id = "example_repo_id"
        cfg.fps = 30
        cfg.robot_type = "example_robot"
        cfg.separate_per_primitive = True
        cfg.silent_load = False

        raw_dir = MagicMock(spec=Path)
        out_dir = MagicMock(spec=Path)
        typestore = MagicMock()

        rosbag_dir = MagicMock(spec=Path)
        rosbag_dir.name = "rosbag_1"
        rosbag_dir.iterdir.return_value = [rosbag_dir]
        raw_dir.iterdir.return_value = [rosbag_dir]

        meta_path = rosbag_dir / "meta.json"
        meta_path.exists.return_value = True

        mock_metadata_loader.return_value = MagicMock()

        with pytest.raises(Exception, match="Failed to load episodes"):
            _convert_individual_rosbag_to_lerobot_format(
                cfg, raw_dir, out_dir, "", typestore
            )

        mock_dataset.add_frame.assert_not_called()
        mock_dataset.save_episode.assert_not_called()

    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter.time.time",
        side_effect=[0, 5, 10],
    )
    def test_error_no_rosbag_dirs(self, mock_time, mock_dataset):
        cfg = MagicMock()
        cfg.repo_id = "example_repo_id"
        cfg.fps = 30
        cfg.robot_type = "example_robot"
        cfg.separate_per_primitive = True

        raw_dir = MagicMock(spec=Path)
        out_dir = MagicMock(spec=Path)
        typestore = MagicMock()

        raw_dir.iterdir.return_value = []

        result = _convert_individual_rosbag_to_lerobot_format(
            cfg, raw_dir, out_dir, "", typestore
        )

        assert result is None
        mock_dataset.add_frame.assert_not_called()
        mock_dataset.save_episode.assert_not_called()

    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter.time.time",
        side_effect=[0, 5, 10],
    )
    @patch("hsr_data_converter.commands.rosbag2lerobot.core_converter.shutil.rmtree")
    def test_error_no_meta_json(self, mock_rmtree, mock_time, mock_dataset):
        cfg = MagicMock()
        cfg.repo_id = "example_repo_id"
        cfg.fps = 30
        cfg.robot_type = "example_robot"
        cfg.separate_per_primitive = True

        raw_dir = MagicMock(spec=Path)
        out_dir = MagicMock(spec=Path)
        typestore = MagicMock()

        rosbag_dir = MagicMock(spec=Path)
        rosbag_dir.name = "rosbag_1"
        rosbag_dir.iterdir.return_value = [rosbag_dir]
        raw_dir.iterdir.return_value = [rosbag_dir]

        meta_path = rosbag_dir / "meta.json"
        meta_path.exists.return_value = False

        result = _convert_individual_rosbag_to_lerobot_format(
            cfg, raw_dir, out_dir, "", typestore
        )

        assert result == mock_dataset
        mock_dataset.add_frame.assert_not_called()
        mock_dataset.save_episode.assert_not_called()


class TestConvertAggregateRosbagToLeRobotFormat:
    @pytest.fixture
    def mock_dataset(self):
        with patch(
            "lerobot.datasets.lerobot_dataset.LeRobotDataset.create", autospec=True
        ) as create_mock:
            dataset_mock = MagicMock()
            create_mock.return_value = dataset_mock
            yield dataset_mock

    @pytest.fixture
    def mock_load_hsr_episodes(self):
        with patch(
            "hsr_data_converter.commands.rosbag2lerobot.core_converter.load_hsr_episodes",
            autospec=True,
        ) as load_mock:
            yield load_mock

    @pytest.fixture
    def mock_update_episodes_jsonl(self):
        with patch(
            "hsr_data_converter.commands.rosbag2lerobot.core_converter.update_episodes_jsonl",
            autospec=True,
        ) as update_mock:
            yield update_mock

    @pytest.fixture
    def mock_metadata_loader(self):
        with patch(
            "airoa_metadata.core.loader.MetadataLoader.load_from_file", autospec=True
        ) as load_mock:
            metadata_mock = MagicMock()
            metadata_mock.run.segments = [{"is_composite": False}]
            metadata_mock.run.instructions = [{"text": ["short task"]}]
            load_mock.return_value = metadata_mock
            yield metadata_mock

    @patch("hsr_data_converter.commands.rosbag2lerobot.core_converter.QaReportCreator")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"files": [{"name": "example_bag"}]}',
    )
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter._create_episodes_info",
        return_value={},
    )
    @patch(
        "airoa_metadata.versions.v0_0.MetadataV0_0.convert", return_value=MagicMock()
    )
    @patch("json.load", return_value=[{"files": [{}]}])
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter.time.time",
        side_effect=[0, 5, 10],
    )
    def test_standard_aggregate_conversion(
        self,
        mock_time,
        mock_json_load,
        mock_convert,
        mock_create_episodes_info,
        mock_builtins_open,
        mock_qa_report_creator,
        mock_dataset,
        mock_load_hsr_episodes,
        mock_update_episodes_jsonl,
        mock_metadata_loader,
    ):
        cfg = MagicMock()
        cfg.repo_id = "example_repo_id"
        cfg.fps = 30
        cfg.robot_type = "example_robot"
        cfg.separate_per_primitive = True

        raw_dir = MagicMock(spec=Path)
        out_dir = MagicMock(spec=Path)
        typestore = MagicMock()

        rosbag_dir = MagicMock(spec=Path)
        rosbag_dir.name = "rosbag_1"
        rosbag_dir.is_dir.return_value = True
        raw_dir.iterdir.return_value = [rosbag_dir]

        meta_path = rosbag_dir / "meta.json"
        meta_path.exists.return_value = True

        mock_qa_report_creator.report_nan_value_errors.return_value = None
        mock_qa_report_creator.report_timestamp_monotonicity_errors.return_value = None
        mock_qa_report_creator.report_frame_index_monotonicity_errors.return_value = (
            None
        )
        mock_qa_report_creator.report_missing_videos.return_value = None
        mock_qa_report_creator.save_to_json.return_value = None

        mock_load_hsr_episodes.return_value = (
            [[], [{"success_primitive_action": [1]}]],
            [[], ["task"]],
            [[], [123456789]],
            1234567890,
        )

        result = _convert_aggregate_rosbag_to_lerobot_format(
            cfg, raw_dir, out_dir, "", typestore
        )

        assert result == mock_dataset
        mock_load_hsr_episodes.assert_called_once()
        mock_update_episodes_jsonl.assert_called_once()
        mock_dataset.add_frame.assert_called()
        mock_dataset.save_episode.assert_called_once()
        mock_create_episodes_info.assert_called_once()
        mock_qa_report_creator.assert_called_once()

    @patch("hsr_data_converter.commands.rosbag2lerobot.core_converter.QaReportCreator")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"files": [{"name": "example_bag"}]}',
    )
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter._create_episodes_info",
        return_value={},
    )
    @patch(
        "airoa_metadata.versions.v0_0.MetadataV0_0.convert", return_value=MagicMock()
    )
    @patch("json.load", return_value={"files": {}})
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter.time.time",
        side_effect=[0, 5, 10],
    )
    def test_meta_dataset_not_a_list(
        self,
        mock_time,
        mock_json_load,
        mock_convert,
        mock_create_episodes_info,
        mock_builtins_open,
        mock_qa_report_creator,
        mock_dataset,
        mock_load_hsr_episodes,
        mock_update_episodes_jsonl,
        mock_metadata_loader,
    ):
        cfg = MagicMock()
        cfg.repo_id = "example_repo_id"
        cfg.fps = 30
        cfg.robot_type = "example_robot"
        cfg.separate_per_primitive = True

        raw_dir = MagicMock(spec=Path)
        out_dir = MagicMock(spec=Path)
        typestore = MagicMock()

        rosbag_dir = MagicMock(spec=Path)
        rosbag_dir.name = "rosbag_1"
        rosbag_dir.is_dir.return_value = True
        raw_dir.iterdir.return_value = [rosbag_dir]

        meta_path = rosbag_dir / "meta.json"
        meta_path.exists.return_value = True

        mock_qa_report_creator.report_nan_value_errors.return_value = None
        mock_qa_report_creator.report_timestamp_monotonicity_errors.return_value = None
        mock_qa_report_creator.report_frame_index_monotonicity_errors.return_value = (
            None
        )
        mock_qa_report_creator.report_missing_videos.return_value = None
        mock_qa_report_creator.save_to_json.return_value = None

        mock_load_hsr_episodes.return_value = (
            [[{"success_primitive_action": [1]}]],
            [["task"]],
            [[123456789]],
            1234567890,
        )

        result = _convert_aggregate_rosbag_to_lerobot_format(
            cfg, raw_dir, out_dir, "", typestore
        )

        assert result == mock_dataset
        mock_load_hsr_episodes.assert_called_once()
        mock_update_episodes_jsonl.assert_called_once()
        mock_dataset.add_frame.assert_called()
        mock_dataset.save_episode.assert_called_once()
        mock_qa_report_creator.assert_called_once()

    @patch("hsr_data_converter.commands.rosbag2lerobot.core_converter.QaReportCreator")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"files": [{"name": "example_bag"}]}',
    )
    @patch("json.load", return_value=[{"files": [{}]}])
    @patch("hsr_data_converter.commands.rosbag2lerobot.core_converter.shutil.rmtree")
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter.time.time",
        side_effect=[0, 5, 10],
    )
    def test_error_no_rosbag_dirs(
        self,
        mock_time,
        mock_rmtree,
        mock_json_load,
        mock_builtins_open,
        mock_qa_report_creator,
        mock_dataset,
        mock_update_episodes_jsonl,
    ):
        cfg = MagicMock()
        cfg.repo_id = "example_repo_id"
        cfg.fps = 30
        cfg.robot_type = "example_robot"
        cfg.separate_per_primitive = True

        raw_dir = MagicMock(spec=Path)
        out_dir = MagicMock(spec=Path)
        typestore = MagicMock()

        raw_dir.iterdir.return_value = []

        mock_qa_report_creator.report_nan_value_errors.return_value = None
        mock_qa_report_creator.report_timestamp_monotonicity_errors.return_value = None
        mock_qa_report_creator.report_frame_index_monotonicity_errors.return_value = (
            None
        )
        mock_qa_report_creator.report_missing_videos.return_value = None
        mock_qa_report_creator.save_to_json.return_value = None

        result = _convert_aggregate_rosbag_to_lerobot_format(
            cfg, raw_dir, out_dir, "", typestore
        )

        assert result is not None
        mock_update_episodes_jsonl.assert_called_once()
        mock_dataset.add_frame.assert_not_called()
        mock_dataset.save_episode.assert_not_called()
        mock_qa_report_creator.assert_called_once()

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"files": [{"name": "example_bag"}]}',
    )
    @patch("json.load", side_effect=json.JSONDecodeError("Expecting value", "", 0))
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter.time.time",
        side_effect=[0, 5, 10],
    )
    def test_error_json_decode_error(
        self, mock_time, mock_json_load, mock_builtins_open, mock_dataset
    ):
        cfg = MagicMock()
        cfg.repo_id = "example_repo_id"
        cfg.fps = 30
        cfg.robot_type = "example_robot"
        cfg.separate_per_primitive = True

        raw_dir = MagicMock(spec=Path)
        out_dir = MagicMock(spec=Path)
        typestore = MagicMock()

        rosbag_dir = MagicMock(spec=Path)
        rosbag_dir.name = "rosbag_1"
        rosbag_dir.is_dir.return_value = True
        raw_dir.iterdir.return_value = [rosbag_dir]

        meta_path = rosbag_dir / "meta.json"
        meta_path.exists.return_value = True

        with pytest.raises(json.JSONDecodeError):
            _convert_aggregate_rosbag_to_lerobot_format(
                cfg, raw_dir, out_dir, "", typestore
            )

        mock_dataset.add_frame.assert_not_called()
        mock_dataset.save_episode.assert_not_called()

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"files": [{"name": "example_bag"}]}',
    )
    @patch("json.load", return_value=[{"files": [{}]}])
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter.time.time",
        side_effect=[0, 5, 10],
    )
    def test_error_no_meta_json(
        self,
        mock_time,
        mock_json_load,
        mock_builtins_open,
        mock_dataset,
        mock_update_episodes_jsonl,
    ):
        cfg = MagicMock()
        cfg.repo_id = "example_repo_id"
        cfg.fps = 30
        cfg.robot_type = "example_robot"
        cfg.separate_per_primitive = True

        raw_dir = MagicMock(spec=Path)
        out_dir = MagicMock(spec=Path)
        typestore = MagicMock()

        rosbag_dir = MagicMock(spec=Path)
        rosbag_dir.name = "rosbag_1"
        rosbag_dir.is_dir.return_value = True
        raw_dir.iterdir.return_value = [rosbag_dir]

        meta_path = rosbag_dir / "meta.json"
        meta_path.exists.return_value = False

        result = _convert_aggregate_rosbag_to_lerobot_format(
            cfg, raw_dir, out_dir, "", typestore
        )

        assert result is not None
        mock_update_episodes_jsonl.assert_called_once()
        mock_dataset.add_frame.assert_not_called()
        mock_dataset.save_episode.assert_not_called()

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"files": [{"name": "example_bag"}]}',
    )
    @patch("json.load", return_value=[{"files": [{}]}, {"files": [{}]}])
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter.time.time",
        side_effect=[0, 5, 10],
    )
    def test_error_meta_dataset_list_greater_than_one(
        self, mock_time, mock_json_load, mock_builtins_open, mock_dataset
    ):
        cfg = MagicMock()
        cfg.repo_id = "example_repo_id"
        cfg.fps = 30
        cfg.robot_type = "example_robot"
        cfg.separate_per_primitive = True

        raw_dir = MagicMock(spec=Path)
        out_dir = MagicMock(spec=Path)
        typestore = MagicMock()

        rosbag_dir = MagicMock(spec=Path)
        rosbag_dir.name = "rosbag_1"
        rosbag_dir.is_dir.return_value = True
        raw_dir.iterdir.return_value = [rosbag_dir]

        meta_path = rosbag_dir / "meta.json"
        meta_path.exists.return_value = True

        with pytest.raises(
            ValueError, match="meta.json shoud have only one MetaDataset, but got 2"
        ):
            _convert_aggregate_rosbag_to_lerobot_format(
                cfg, raw_dir, out_dir, "", typestore
            )

        mock_dataset.add_frame.assert_not_called()
        mock_dataset.save_episode.assert_not_called()


@dataclass
class MockContext:
    some_field: str = "some_value"


class TestCreateEpisodesInfo:
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter.get_git_information"
    )
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter._convert_metadata_to_v1_0"
    )
    def test_single_instruction(self, mock_convert_metadata, mock_git_info):
        metadata_v1_0 = MagicMock()
        metadata_v1_0.bag_path = "path/to/example_bag"
        metadata_v1_0.version = "1.0"
        metadata_v1_0.location_name = "example_location"
        metadata_v1_0.interface = "example_interface"
        metadata_v1_0.git_hash = "abc123"
        metadata_v1_0.git_branch = "main"
        metadata_v1_0.interface_git_hash = "def456"
        metadata_v1_0.interface_git_branch = "dev"
        metadata_v1_0.label = "example_label"
        metadata_v1_0.hsr_id = "hsr_1"

        mock_convert_metadata.return_value = metadata_v1_0

        mock_git_info.return_value = {
            "git_hash": "ghi789",
            "git_branch": "release",
            "git_url": "https://example.com/repo.git",
            "git_tag": "v1.0.0",
        }

        metadata = MagicMock()
        metadata.run.instructions = [MagicMock(text=["action"])]
        metadata.run.segments = [MagicMock(success=True)]
        metadata.uuid = "uuid-example"
        metadata.context = MockContext()

        task_type = "PA"
        task_success = True
        source_dir = "source_dir"

        expected_output = {
            "bag_path": "path/to/example_bag",
            "version": "1.0",
            "location_name": "example_location",
            "interface": "example_interface",
            "git_hash": "abc123",
            "git_branch": "main",
            "interface_git_hash": "def456",
            "interface_git_branch": "dev",
            "pipeline_git_hash": "ghi789",
            "pipeline_git_branch": "release",
            "pipeline_git_url": "https://example.com/repo.git",
            "pipeline_git_tag": "v1.0.0",
            "label": "example_label",
            "hsr_id": "hsr_1",
            "task_type": task_type,
            "task_success": task_success,
            "short_horizon_task": [],
            "primitive_action": ["action"],
            "success_short_horizon_task": True,
            "uuid": "uuid-example",
            "context": {"some_field": "some_value"},
            "source_location": source_dir,
        }

        result = _create_episodes_info(metadata, task_type, task_success, source_dir)
        assert result == expected_output
        mock_convert_metadata.assert_called_once_with(metadata)
        mock_git_info.assert_called_once()

    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter.get_git_information"
    )
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter._convert_metadata_to_v1_0"
    )
    def test_multiple_instructions(self, mock_convert_metadata, mock_git_info):
        metadata_v1_0 = MagicMock()
        metadata_v1_0.bag_path = "path/to/example_bag"
        metadata_v1_0.version = "1.0"
        metadata_v1_0.location_name = "example_location"
        metadata_v1_0.interface = "example_interface"
        metadata_v1_0.git_hash = "abc123"
        metadata_v1_0.git_branch = "main"
        metadata_v1_0.interface_git_hash = "def456"
        metadata_v1_0.interface_git_branch = "dev"
        metadata_v1_0.label = "example_label"
        metadata_v1_0.hsr_id = "hsr_1"

        mock_convert_metadata.return_value = metadata_v1_0

        mock_git_info.return_value = {
            "git_hash": "ghi789",
            "git_branch": "release",
            "git_url": "https://example.com/repo.git",
            "git_tag": "v1.0.0",
        }

        metadata = MagicMock()
        metadata.run.instructions = [
            MagicMock(text=["action1"]),
            MagicMock(text=["action2"]),
        ]
        metadata.run.segments = [MagicMock(success=True), MagicMock(success=True)]
        metadata.uuid = "uuid-example"
        metadata.context = MockContext()

        task_type = "PA"
        task_success = True
        source_dir = "source_dir"

        expected_output = {
            "bag_path": "path/to/example_bag",
            "version": "1.0",
            "location_name": "example_location",
            "interface": "example_interface",
            "git_hash": "abc123",
            "git_branch": "main",
            "interface_git_hash": "def456",
            "interface_git_branch": "dev",
            "pipeline_git_hash": "ghi789",
            "pipeline_git_branch": "release",
            "pipeline_git_url": "https://example.com/repo.git",
            "pipeline_git_tag": "v1.0.0",
            "label": "example_label",
            "hsr_id": "hsr_1",
            "task_type": task_type,
            "task_success": task_success,
            "short_horizon_task": "action2",
            "primitive_action": ["action1"],
            "success_short_horizon_task": True,
            "uuid": "uuid-example",
            "context": {"some_field": "some_value"},
            "source_location": source_dir,
        }

        result = _create_episodes_info(metadata, task_type, task_success, source_dir)
        assert result == expected_output
        mock_convert_metadata.assert_called_once_with(metadata)
        mock_git_info.assert_called_once()

    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter.get_git_information"
    )
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter._convert_metadata_to_v1_0"
    )
    def test_error_empty_instructions(self, mock_convert_metadata, mock_git_info):
        metadata_v1_0 = MagicMock()
        metadata_v1_0.bag_path = "path/to/example_bag"
        metadata_v1_0.version = "1.0"
        metadata_v1_0.location_name = "example_location"
        metadata_v1_0.interface = "example_interface"
        metadata_v1_0.git_hash = "abc123"
        metadata_v1_0.git_branch = "main"
        metadata_v1_0.interface_git_hash = "def456"
        metadata_v1_0.interface_git_branch = "dev"
        metadata_v1_0.label = "example_label"
        metadata_v1_0.hsr_id = "hsr_1"

        mock_convert_metadata.return_value = metadata_v1_0
        metadata = MagicMock()
        metadata.run.instructions = []
        metadata.run.segments = []

        task_type = "PA"
        task_success = True
        source_dir = "source_dir"

        with pytest.raises(IndexError):
            _create_episodes_info(metadata, task_type, task_success, source_dir)

        mock_convert_metadata.assert_called_once_with(metadata)
        mock_git_info.assert_called_once()


class TestConvertMetadataToV1_0:
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter.MetadataLoader.load_from_dict"
    )
    def test_standard_conversion(self, mock_loader):
        metadata = MagicMock()
        metadata.to_json.return_value = json.dumps(
            {
                "files": [{"name": "example_bag"}],
                "context": {
                    "entities": [
                        {"role": "robot", "id": "robot_1"},
                        {"role": "location", "name": "location_1"},
                        {"role": "operator", "id": "operator_1"},
                        {
                            "role": "task",
                            "template": {"description": "task_description"},
                        },
                    ],
                    "components": [
                        {
                            "role": "interface",
                            "name": "interface_1",
                            "source": {"git": {"hash": "def456", "branch": "dev"}},
                        },
                        {
                            "role": "data_collection",
                            "source": {"git": {"hash": "abc123", "branch": "main"}},
                        },
                    ],
                },
                "run": {
                    "instructions": [
                        {"text": ["short task"]},
                        {"text": ["primitive action"]},
                    ],
                    "segments": [{"success": True, "start_time": 100, "end_time": 200}],
                },
            }
        )

        expected_output = MetadataV1_0(
            bag_path="example_bag",
            location_name="location_1",
            interface="interface_1",
            git_hash="abc123",
            git_branch="main",
            interface_git_hash="def456",
            interface_git_branch="dev",
            label="operator_1",
            hsr_id="robot_1",
        )
        mock_loader.return_value = expected_output

        result = _convert_metadata_to_v1_0(metadata)

        assert (
            result.bag_path == expected_output.bag_path
            and result.version == expected_output.version
            and result.location_name == expected_output.location_name
            and result.interface == expected_output.interface
            and result.git_hash == expected_output.git_hash
            and result.git_branch == expected_output.git_branch
            and result.interface_git_hash == expected_output.interface_git_hash
            and result.interface_git_branch == expected_output.interface_git_branch
            and result.label == expected_output.label
            and result.hsr_id == expected_output.hsr_id
        )

        mock_loader.assert_called_once_with(
            {
                "bag_path": "example_bag",
                "hsr_id": "robot_1",
                "version": "1.0",
                "location_name": "location_1",
                "interface": "interface_1",
                "instructions": [["short task"], ["primitive action"]],
                "segments": [
                    {
                        "start_time": 100,
                        "end_time": 200,
                        "instructions_index": 0,
                        "has_suboptimal": False,
                        "is_directed": True,
                    }
                ],
                "label": "operator_1",
                "git_hash": "abc123",
                "git_branch": "main",
                "interface_git_hash": "def456",
                "interface_git_branch": "dev",
            }
        )

    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter.MetadataLoader.load_from_dict"
    )
    def test_large_segment_times(self, mock_loader):
        metadata = MagicMock()
        metadata.to_json.return_value = json.dumps(
            {
                "files": [{"name": "example_bag"}],
                "context": {
                    "entities": [
                        {"role": "robot", "id": "robot_1"},
                        {"role": "location", "name": "location_1"},
                        {"role": "operator", "id": "operator_1"},
                        {
                            "role": "task",
                            "template": {"description": "task_description"},
                        },
                    ],
                    "components": [
                        {
                            "role": "interface",
                            "name": "interface_1",
                            "source": {"git": {"hash": "def456", "branch": "dev"}},
                        },
                        {
                            "role": "data_collection",
                            "source": {"git": {"hash": "abc123", "branch": "main"}},
                        },
                    ],
                },
                "run": {
                    "instructions": [
                        {"text": ["short task"]},
                        {"text": ["primitive action"]},
                    ],
                    "segments": [
                        {"success": True, "start_time": 100001, "end_time": 100}
                    ],
                },
            }
        )

        expected_output = MetadataV1_0(
            bag_path="example_bag",
            location_name="location_1",
            interface="interface_1",
            git_hash="abc123",
            git_branch="main",
            interface_git_hash="def456",
            interface_git_branch="dev",
            label="operator_1",
            hsr_id="robot_1",
        )
        mock_loader.return_value = expected_output

        result = _convert_metadata_to_v1_0(metadata)

        assert (
            result.bag_path == "example_bag"
            and result.version == "1.0"
            and result.location_name == "location_1"
            and result.interface == "interface_1"
            and result.git_hash == "abc123"
            and result.git_branch == "main"
            and result.interface_git_hash == "def456"
            and result.interface_git_branch == "dev"
            and result.label == "operator_1"
            and result.hsr_id == "robot_1"
        )

        mock_loader.assert_called_once_with(
            {
                "bag_path": "example_bag",
                "hsr_id": "robot_1",
                "version": "1.0",
                "location_name": "location_1",
                "interface": "interface_1",
                "instructions": [
                    ["short task"],
                    ["primitive action"],
                    ["task_description"],
                ],
                "segments": [
                    {
                        "start_time": 100001,
                        "end_time": 100,
                        "instructions_index": 0,
                        "has_suboptimal": False,
                        "is_directed": True,
                    },
                    {
                        "start_time": 100001.0,
                        "end_time": 100.0,
                        "instructions_index": 2,
                        "has_suboptimal": False,
                        "is_directed": True,
                    },
                ],
                "label": "operator_1",
                "git_hash": "abc123",
                "git_branch": "main",
                "interface_git_hash": "def456",
                "interface_git_branch": "dev",
            }
        )

    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.core_converter.MetadataLoader.load_from_dict"
    )
    def test_incorrect_metadata_conversion(self, mock_loader):
        metadata = MagicMock()
        metadata.to_json.return_value = json.dumps(
            {
                "files": [{"name": "example_bag"}],
                "context": {
                    "entities": [
                        {"role": "robot", "id": "robot_1"},
                        {"role": "location", "name": "location_1"},
                        {"role": "operator", "id": "operator_1"},
                        {
                            "role": "task",
                            "template": {"description": "task_description"},
                        },
                    ],
                    "components": [
                        {
                            "role": "interface",
                            "name": "interface_1",
                            "source": {"git": {"hash": "def456", "branch": "dev"}},
                        },
                        {
                            "role": "data_collection",
                            "source": {"git": {"hash": "abc123", "branch": "main"}},
                        },
                    ],
                },
                "run": {
                    "instructions": [
                        {"text": ["short task"]},
                        {"text": ["primitive action"]},
                    ],
                    "segments": [{"success": True, "start_time": 100, "end_time": 200}],
                },
            }
        )

        expected_output = MagicMock()
        mock_loader.return_value = expected_output
        with pytest.raises(ValueError, match="Failed to convert metadata to V1.0."):
            _convert_metadata_to_v1_0(metadata)

        mock_loader.assert_called_once_with(
            {
                "bag_path": "example_bag",
                "hsr_id": "robot_1",
                "version": "1.0",
                "location_name": "location_1",
                "interface": "interface_1",
                "instructions": [["short task"], ["primitive action"]],
                "segments": [
                    {
                        "start_time": 100,
                        "end_time": 200,
                        "instructions_index": 0,
                        "has_suboptimal": False,
                        "is_directed": True,
                    }
                ],
                "label": "operator_1",
                "git_hash": "abc123",
                "git_branch": "main",
                "interface_git_hash": "def456",
                "interface_git_branch": "dev",
            }
        )
