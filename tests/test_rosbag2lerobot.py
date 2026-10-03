import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import jsonlines
import numpy as np
import torch
from moma_metadata import Metadata
from moma_metadata.versions.v1_3 import (
    ContextV1_3,
    FileV1_3,
    InstructionV1_3,
    RunV1_3,
    SegmentV1_3,
)

from hsr_data_converter.convert_config import ConvertConfig
from hsr_data_converter.hsr_utils import GRIPPER_CLOSE_ACTION, GRIPPER_OPEN_ACTION
from hsr_data_converter.recorded_topics import RecordedTopics
from hsr_data_converter.rosbag2lerobot.action_calculator import (
    calc_initial_subaction,
)
from hsr_data_converter.rosbag2lerobot.episode_loader import (
    check_skip_condition,
    get_updated_features_from_topics,
    load_hsr_episodes,
)
from hsr_data_converter.rosbag2lerobot.utils import (
    update_episodes_jsonl,
    update_last_jsonline,
)


def mock_lerobot_modules():
    """Mock all lerobot-related modules"""
    mocks = {
        "lerobot": MagicMock(),
        "lerobot.datasets": MagicMock(),
        "lerobot.datasets.lerobot_dataset": MagicMock(),
        "lerobot.configs": MagicMock(),
        "lerobot.configs.parser": MagicMock(),
        "rosbags": MagicMock(),
        "rosbags.highlevel": MagicMock(),
        "rosbags.typesys": MagicMock(),
        "rosbags.typesys.store": MagicMock(),
    }
    return patch.dict("sys.modules", mocks)


class TestRosbag2Lerobot(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        self.mock_patcher = mock_lerobot_modules()
        self.mock_patcher.start()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.mock_patcher.stop()

    def test_get_updated_features_from_topics(self):
        with mock_lerobot_modules():
            received_topics = MagicMock(spec=RecordedTopics)

            mock_hand_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            mock_head_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            received_topics.hand_rgb = (1000000000, mock_hand_image)
            received_topics.head_rgb = (2000000000, mock_head_image)

            joint_names = [
                "arm_lift_joint",
                "arm_flex_joint",
                "arm_roll_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
                "hand_motor_joint",
                "head_pan_joint",
                "head_tilt_joint",
            ]
            joint_positions = np.array([0.1, -0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8])

            received_topics.joint_positions = (1500000000, joint_positions)
            received_topics.joint_names = joint_names
            received_topics.wrist_wrench = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

            received_topics.abs_arm_action = (
                1600000000,
                np.array([0.2, -0.1, 0.4, -0.3, 0.6]),
            )
            received_topics.abs_arm_action_names = [
                "arm_lift_joint",
                "arm_flex_joint",
                "arm_roll_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
            ]

            received_topics.abs_gripper_action = (
                1700000000,
                np.array([GRIPPER_OPEN_ACTION]),
            )
            received_topics.abs_gripper_name = ["hand_motor_joint"]

            received_topics.abs_head_action = (1800000000, np.array([-0.8, 0.9]))
            received_topics.abs_head_action_names = [
                "head_pan_joint",
                "head_tilt_joint",
            ]

            received_topics.delta_base_action = (1900000000, np.array([0.1, 0.0, 0.05]))
            received_topics.delta_base_action_names = ["base_x", "base_y", "base_t"]

            last_frame_timestamp = 500000000

            updated_features = get_updated_features_from_topics(
                received_topics, last_frame_timestamp
            )

            np.testing.assert_array_equal(
                updated_features["observation.image.hand"], mock_hand_image
            )
            np.testing.assert_array_equal(
                updated_features["observation.image.head"], mock_head_image
            )

            np.testing.assert_array_equal(
                updated_features["observation.state"],
                joint_positions.astype(np.float32),
            )

            expected_wrench = torch.tensor(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32
            )
            torch.testing.assert_close(
                updated_features["observation.wrist.wrench"], expected_wrench
            )

            expected_arm_action = torch.tensor(
                [0.2, -0.1, 0.4, -0.3, 0.6], dtype=torch.float32
            )
            torch.testing.assert_close(
                updated_features["action.arm"], expected_arm_action
            )

            expected_gripper_action = torch.tensor(
                [GRIPPER_OPEN_ACTION], dtype=torch.float32
            )
            torch.testing.assert_close(
                updated_features["action.gripper"], expected_gripper_action
            )

            expected_head_action = torch.tensor([-0.8, 0.9], dtype=torch.float32)
            torch.testing.assert_close(
                updated_features["action.head"], expected_head_action
            )

            expected_base_action = torch.tensor([0.1, 0.0, 0.05], dtype=torch.float32)
            torch.testing.assert_close(
                updated_features["action.base"], expected_base_action
            )

            self.assertTrue(updated_features["observation.image.hand.is_fresh"].all())
            self.assertTrue(updated_features["observation.image.head.is_fresh"].all())
            self.assertTrue(updated_features["observation.state.is_fresh"].all())
            self.assertTrue(updated_features["action.arm.is_fresh"].all())
            self.assertTrue(updated_features["action.gripper.is_fresh"].all())
            self.assertTrue(updated_features["action.head.is_fresh"].all())
            self.assertTrue(updated_features["action.base.is_fresh"].all())

    def test_calc_initial_subaction(self):
        with mock_lerobot_modules():
            frame = {
                "observation.state": torch.tensor(
                    [0.1, -0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8], dtype=torch.float32
                ),
                "action.arm": None,
                "action.gripper": None,
                "action.head": None,
                "action.base": None,
            }

            initial_subaction = calc_initial_subaction(frame)

            expected_arm = torch.tensor(
                [0.1, -0.2, 0.3, -0.4, 0.5], dtype=torch.float32
            )
            torch.testing.assert_close(initial_subaction["action.arm"], expected_arm)
            self.assertFalse(initial_subaction["action.arm.is_fresh"].any())

            expected_gripper = torch.tensor([GRIPPER_OPEN_ACTION], dtype=torch.float32)
            torch.testing.assert_close(
                initial_subaction["action.gripper"], expected_gripper
            )
            self.assertFalse(initial_subaction["action.gripper.is_fresh"].any())

            expected_head = torch.tensor([-0.7, 0.8], dtype=torch.float32)
            torch.testing.assert_close(initial_subaction["action.head"], expected_head)
            self.assertFalse(initial_subaction["action.head.is_fresh"].any())

            expected_base = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
            torch.testing.assert_close(initial_subaction["action.base"], expected_base)
            self.assertFalse(initial_subaction["action.base.is_fresh"].any())

    def test_calc_initial_subaction_gripper_close(self):
        with mock_lerobot_modules():
            frame = {
                "observation.state": torch.tensor(
                    [0.1, -0.2, 0.3, -0.4, 0.5, -0.1, -0.7, 0.8], dtype=torch.float32
                ),
                "action.arm": None,
                "action.gripper": None,
                "action.head": None,
                "action.base": None,
            }

            initial_subaction = calc_initial_subaction(frame)

            expected_gripper = torch.tensor([GRIPPER_CLOSE_ACTION], dtype=torch.float32)
            torch.testing.assert_close(
                initial_subaction["action.gripper"], expected_gripper
            )

    def test_check_skip_condition(self):
        with mock_lerobot_modules():
            received_topics_no_movement = MagicMock()
            received_topics_no_movement.joint_velocities = (
                0,
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            )

            self.assertTrue(check_skip_condition(received_topics_no_movement))

            received_topics_moving = MagicMock()
            received_topics_moving.joint_velocities = (
                0,
                np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            )

            self.assertFalse(check_skip_condition(received_topics_moving))

    def test_update_last_jsonline(self):
        with mock_lerobot_modules():
            test_file = self.temp_path / "test.jsonl"

            with jsonlines.open(test_file, "w") as writer:
                writer.write({"episode_index": 0, "task": "pick"})
                writer.write({"episode_index": 1, "task": "place"})

            new_data = {"success": True, "duration": 10.5}
            update_last_jsonline(new_data, test_file)

            with jsonlines.open(test_file, "r") as reader:
                lines = list(reader)

            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0], {"episode_index": 0, "task": "pick"})
            self.assertEqual(
                lines[1],
                {
                    "episode_index": 1,
                    "task": "place",
                    "success": True,
                    "duration": 10.5,
                },
            )

    def test_update_episodes_jsonl(self):
        with mock_lerobot_modules():
            test_file = self.temp_path / "episodes.jsonl"

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

            self.assertEqual(len(lines), 3)
            self.assertEqual(
                lines[0],
                {"episode_index": 0, "task": "pick", "success": True, "duration": 8.0},
            )
            self.assertEqual(
                lines[1],
                {
                    "episode_index": 1,
                    "task": "place",
                    "success": False,
                    "duration": 12.0,
                },
            )
            self.assertEqual(
                lines[2],
                {
                    "episode_index": 2,
                    "task": "stack",
                    "success": True,
                    "duration": 15.0,
                },
            )

    @patch("hsr_data_converter.rosbag2lerobot.episode_loader.check_skip_condition")
    @patch(
        "hsr_data_converter.rosbag2lerobot.episode_loader.get_updated_features_from_topics"
    )
    @patch("hsr_data_converter.rosbag2lerobot.action_calculator.calc_initial_subaction")
    @patch("hsr_data_converter.rosbag2lerobot.action_calculator.calc_absolute_action")
    @patch("hsr_data_converter.rosbag2lerobot.action_calculator.calc_delta_action")
    @patch("hsr_data_converter.rosbag2lerobot.episode_loader.AnyReader")
    @patch("hsr_data_converter.tf_buffer.TFBuffer")
    @patch("hsr_data_converter.recorded_topics.RecordedTopics")
    def test_load_hsr_episodes_with_frames(
        self,
        mock_recorded_topics,
        mock_tf_buffer,
        mock_any_reader,
        mock_calc_delta_action,
        mock_calc_absolute_action,
        mock_calc_initial_subaction,
        mock_get_updated_features,
        mock_check_skip_condition,
    ):
        with mock_lerobot_modules():
            test_bag_path = self.temp_path / "test.bag"
            test_bag_path.touch()

            cfg = ConvertConfig()
            cfg.fps = 10
            cfg.separate_per_primitive = False

            metadataset = Metadata(
                uuid=str(uuid.uuid4()),
                files=[FileV1_3(type="rosbag", name=str(test_bag_path))],
                context=ContextV1_3(),
                run=RunV1_3(
                    total_time_s=2.0,
                    instructions=[InstructionV1_3(idx=0, text=["pick_object"])],
                    segments=[
                        SegmentV1_3(
                            start_time=0,
                            end_time=2,
                            instruction_idx=0,
                            success=True,
                            controlled_by="test_controller",
                        )
                    ],
                ),
            )

            meta_dir = self.temp_path
            typestore = MagicMock()
            dataset = MagicMock()
            dataset.meta.get_task_index.return_value = 0
            dataset.meta.add_task.return_value = None

            mock_tf_buffer_instance = MagicMock()
            mock_tf_buffer.return_value = mock_tf_buffer_instance
            mock_tf_buffer_instance.lookup_transform.return_value = np.eye(4)

            mock_recorded_topics_instance = MagicMock()
            mock_recorded_topics.return_value = mock_recorded_topics_instance

            call_count = [0]

            def mock_update_topics(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    mock_recorded_topics_instance.is_operating = False
                    mock_recorded_topics_instance.is_valid = False
                else:
                    mock_recorded_topics_instance.is_operating = True
                    mock_recorded_topics_instance.is_valid = True

            mock_recorded_topics_instance.update_topics = mock_update_topics

            mock_hand_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            mock_head_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            mock_recorded_topics_instance.hand_rgb = (1000000000, mock_hand_image)
            mock_recorded_topics_instance.head_rgb = (1000000000, mock_head_image)

            joint_names = [
                "arm_lift_joint",
                "arm_flex_joint",
                "arm_roll_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
                "hand_motor_joint",
                "head_pan_joint",
                "head_tilt_joint",
            ]
            joint_positions = np.array([0.1, -0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8])
            mock_recorded_topics_instance.joint_positions = (
                1000000000,
                joint_positions,
            )
            mock_recorded_topics_instance.joint_names = joint_names
            mock_recorded_topics_instance.wrist_wrench = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

            mock_reader_instance = MagicMock()
            mock_any_reader.return_value.__enter__.return_value = mock_reader_instance
            mock_any_reader.return_value.__exit__.return_value = None

            mock_connection_tf_static = MagicMock()
            mock_connection_tf_static.topic = "/tf_static"
            mock_connection_tf_static.msgtype = "tf2_msgs/TFMessage"

            mock_connection_joint_states = MagicMock()
            mock_connection_joint_states.topic = "/hsrb/joint_states"
            mock_connection_joint_states.msgtype = "sensor_msgs/JointState"

            mock_reader_instance.connections = [
                mock_connection_tf_static,
                mock_connection_joint_states,
            ]
            mock_reader_instance.deserialize.return_value = MagicMock()

            static_messages = []

            data_messages = [
                (mock_connection_joint_states, 1000000000, b"mock_data_1"),
                (mock_connection_joint_states, 1100000000, b"mock_data_2"),
                (mock_connection_joint_states, 1200000000, b"mock_data_3"),
            ]

            call_counter = [0]

            def mock_messages(*args, **kwargs):
                call_counter[0] += 1
                if call_counter[0] == 1:
                    return iter(static_messages)
                else:
                    return iter(data_messages)

            mock_reader_instance.messages = mock_messages

            mock_check_skip_condition.return_value = False

            mock_get_updated_features.return_value = {
                "observation.image.hand": mock_hand_image,
                "observation.image.head": mock_head_image,
                "observation.state": torch.tensor(joint_positions, dtype=torch.float32),
                "observation.wrist.wrench": torch.tensor(
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32
                ),
                "observation.image.hand.is_fresh": torch.tensor([True]),
                "observation.image.head.is_fresh": torch.tensor([True]),
                "observation.state.is_fresh": torch.tensor(
                    [True, True, True, True, True, True, True, True]
                ),
            }

            mock_calc_initial_subaction.return_value = {
                "action.arm": torch.tensor(
                    [0.1, -0.2, 0.3, -0.4, 0.5], dtype=torch.float32
                ),
                "action.gripper": torch.tensor(
                    [GRIPPER_OPEN_ACTION], dtype=torch.float32
                ),
                "action.head": torch.tensor([-0.7, 0.8], dtype=torch.float32),
                "action.base": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                "action.arm.is_fresh": torch.tensor(
                    [False, False, False, False, False]
                ),
                "action.gripper.is_fresh": torch.tensor([False]),
                "action.head.is_fresh": torch.tensor([False, False]),
                "action.base.is_fresh": torch.tensor([False, False, False]),
            }

            mock_calc_absolute_action.return_value = (
                torch.tensor(
                    [0.1, -0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8], dtype=torch.float32
                ),
                torch.tensor([True]),
            )

            mock_calc_delta_action.return_value = (
                torch.tensor(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32
                ),
                torch.tensor([True]),
            )

            episodes, all_tasks, all_timestamps, last_timestamp = load_hsr_episodes(
                cfg, metadataset, meta_dir, typestore, dataset
            )

            self.assertIsInstance(episodes, list)
            self.assertIsInstance(all_tasks, list)
            self.assertIsInstance(all_timestamps, list)
            self.assertIsInstance(last_timestamp, int)

            self.assertGreater(len(episodes), 0, "Should have at least one episode")

            for episode in episodes:
                self.assertIsInstance(episode, list)
                if len(episode) > 0:
                    frame = episode[0]
                    self.assertIn("observation.state", frame)
                    self.assertIn("action.arm", frame)
                    self.assertIn("next.done", frame)

            self.assertGreater(len(all_tasks), 0, "Should have tasks")
            for task_list in all_tasks:
                self.assertIsInstance(task_list, list)
                if len(task_list) > 0:
                    self.assertEqual(task_list[0], "pick_object")

            self.assertGreater(len(all_timestamps), 0, "Should have timestamps")
            for timestamp_list in all_timestamps:
                self.assertIsInstance(timestamp_list, list)
                for ts in timestamp_list:
                    self.assertIsInstance(ts, torch.Tensor)

            self.assertGreaterEqual(last_timestamp, 0)


if __name__ == "__main__":
    unittest.main()
