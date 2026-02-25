import unittest
import os
import json
import shutil
from hsr_data_converter.commands.package.package import PackageTool


# Expected dataset information for filter_episodes tests
FILTER_DATASET_INFO = {
    "dataset1": {
        "hsr_id": "robot008",
        "length": 401,
        "short_horizon_task": "Pull the chain to turn the desk lamp on or off",
        "task_type": "SHT",
    },
    "dataset2": {
        "hsr_id": "robot002",
        "length": 890,
        "short_horizon_task": "Bake a toast",
        "task_type": "SHT",
    },
    "dataset3": {
        "hsr_id": "robot002",
        "length": 2745,
        "short_horizon_task": "Make coffee",
        "task_type": "SHT",
    },
    "dataset4": {
        "hsr_id": "robot001",
        "length": 4201,
        "short_horizon_task": "Washing dishes in the dishwasher",
        "task_type": "SHT",
        "has_errors": True,  # Has success_primitive_action=False
    },
    "dataset5": {
        "hsr_id": "robot002",
        "length": 863,
        "short_horizon_task": "Open the towel stand and hang the towel.",
        "task_type": "SHT",
    },
    "dataset6": {
        "hsr_id": "robot004",
        "length": 1113,
        "short_horizon_task": "Stand the slippers in the slipper rack",
        "task_type": "SHT",
    },
    "dataset7": {
        "hsr_id": "robot006",
        "length": 79,
        "short_horizon_task": ("Press the button to turn the desk lamp on and off"),
        "task_type": "SHT",
    },
}

# Expected dataset information for merge_dataset tests
MERGE_DATASET_INFO = {
    "dataset1": {
        "hsr_id": "robot008",
        "length": 401,
        "short_horizon_task": "Pull the chain to turn the desk lamp on or off",
        "task_type": "SHT",
    },
    "dataset2": {
        "hsr_id": "robot002",
        "length": 890,
        "short_horizon_task": "Bake a toast",
        "task_type": "SHT",
    },
    "dataset3": {
        "hsr_id": "robot002",
        "length": 2745,
        "short_horizon_task": "Make coffee",
        "task_type": "SHT",
        "has_errors": True,  # Has qa_report.json with errors
    },
    "dataset4": {
        "hsr_id": "robot001",
        "length": 4201,
        "short_horizon_task": "Washing dishes in the dishwasher",
        "task_type": "SHT",
    },
    "dataset5": {
        "hsr_id": "robot002",
        "length": 863,
        "short_horizon_task": "Open the towel stand and hang the towel.",
        "task_type": "SHT",
    },
    "dataset6": {
        "hsr_id": "robot004",
        "length": 1113,
        "short_horizon_task": "Stand the slippers in the slipper rack",
        "task_type": "SHT",
    },
    "dataset7": {
        "hsr_id": "robot006",
        "length": 79,
        "short_horizon_task": ("Press the button to turn the desk lamp on and off"),
        "task_type": "SHT",
    },
}


class TestPackageTool(unittest.TestCase):
    def setUp(self):
        test_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
        merge_dataset_path = os.path.join(test_root, "test_data", "merge_dataset")
        self.dataset1 = os.path.join(merge_dataset_path, "dataset1")
        self.dataset2 = os.path.join(merge_dataset_path, "dataset2")
        self.dataset4 = os.path.join(merge_dataset_path, "dataset4")
        self.dataset5 = os.path.join(merge_dataset_path, "dataset5")
        self.output_dir = os.path.join(test_root, "output_package_tool")

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_package_with_validation(self):
        """Test package command with FPS and feature shape validation"""
        package_tool = PackageTool()
        success = package_tool.package(
            source_folders=[self.dataset1, self.dataset2],
            output_folder=self.output_dir,
            fps=10,
            max_dim=32,
            chunk_size=1000,
            ignore_check_version=True,
        )

        self.assertTrue(success)

        # Verify output exists
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "meta", "info.json"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "meta", "episodes.jsonl"))
        )

        # Verify episodes
        with open(os.path.join(self.output_dir, "meta", "episodes.jsonl")) as f:
            episodes = [json.loads(line) for line in f]

        # Should have 2 episodes (dataset1 and dataset2)
        self.assertEqual(len(episodes), 2)

        # Verify info.json
        with open(os.path.join(self.output_dir, "meta", "info.json")) as f:
            info = json.load(f)

        self.assertEqual(info["fps"], 10)
        self.assertEqual(info["total_episodes"], 2)

    def test_package_multiple_datasets(self):
        """Test package command with multiple datasets"""
        package_tool = PackageTool()
        success = package_tool.package(
            source_folders=[
                self.dataset1,
                self.dataset2,
                self.dataset4,
                self.dataset5,
            ],
            output_folder=self.output_dir,
            max_dim=32,
            ignore_check_version=True,
        )

        self.assertTrue(success)

        # Verify episodes
        with open(os.path.join(self.output_dir, "meta", "episodes.jsonl")) as f:
            episodes = [json.loads(line) for line in f]

        # Should have 3 episodes (dataset4 is filtered out due to success_primitive_action=False)
        self.assertEqual(len(episodes), 3)

    def test_validate_fps_success(self):
        """Test FPS validation with consistent FPS"""
        package_tool = PackageTool()
        is_valid, fps, error_msg = package_tool._validate_fps(
            [self.dataset1, self.dataset2]
        )
        self.assertTrue(is_valid)
        self.assertEqual(fps, 10)
        self.assertIsNone(error_msg)

    def test_validate_feature_shapes_success(self):
        """Test feature shape validation with consistent shapes"""
        package_tool = PackageTool()
        is_valid, error_msg = package_tool._validate_feature_shapes(
            [self.dataset1, self.dataset2]
        )
        self.assertTrue(is_valid)
        self.assertIsNone(error_msg)

    def test_filter_episodes_both_methods(self):
        """Test filtering episodes using both qa_report and parquet methods"""
        test_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
        merge_dataset_path = os.path.join(test_root, "test_data", "merge_dataset")
        filter_dataset_path = os.path.join(test_root, "test_data", "filter_episodes")

        dataset3 = os.path.join(merge_dataset_path, "dataset3")  # Has qa_report errors
        dataset4 = os.path.join(
            filter_dataset_path, "dataset4"
        )  # Has success_primitive_action=False

        package_tool = PackageTool()

        # Test with dataset3 (has qa_report errors)
        filtered_map = package_tool._filter_episodes([dataset3])
        self.assertIn(dataset3, filtered_map)
        self.assertGreater(len(filtered_map[dataset3]), 0)

        # Test with dataset4 (has parquet errors)
        filtered_map = package_tool._filter_episodes([dataset4])
        self.assertIn(dataset4, filtered_map)
        self.assertGreater(len(filtered_map[dataset4]), 0)
