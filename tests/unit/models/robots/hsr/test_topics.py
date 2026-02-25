import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch
from rosbags.typesys.stores.ros1_noetic import (
    geometry_msgs__msg__Twist,
    trajectory_msgs__msg__JointTrajectory,
)
from hsr_data_converter.models.robots.hsr.topics import RecordedTopics
from rosbags.typesys.stores.ros1_noetic import (
    sensor_msgs__msg__JointState,
    sensor_msgs__msg__CompressedImage,
    std_msgs__msg__String,
)
from hsr_data_converter.models.robots.hsr.config import (
    GRIPPER_OPEN_ACTION,
    GRIPPER_CLOSE_ACTION,
)
from rosbags.typesys.stores.ros1_noetic import (
    geometry_msgs__msg__Vector3,
)
from rosbags.typesys.stores.ros2_dashing import tf2_msgs__msg__TFMessage


class TestRecordedTopicsIsOperating:
    def test_standard_operating(self):
        recorded_topics = RecordedTopics(_arm_trajectory_controller_command=MagicMock())
        assert recorded_topics.is_operating

    def test_no_action_operating(self):
        recorded_topics = RecordedTopics(
            _arm_trajectory_controller_command=None,
            _head_trajectory_controller_command=None,
            _gripper_controller_command=None,
            _gripper_controller_grasp_command=None,
            _command_velocity=None,
        )
        assert not recorded_topics.is_operating

    def test_command_velocity_operating(self):
        recorded_topics = RecordedTopics(
            _command_velocity=MagicMock(spec=geometry_msgs__msg__Twist)
        )
        assert recorded_topics.is_operating

    def test_gripper_command_operating(self):
        recorded_topics = RecordedTopics(
            _gripper_controller_command=MagicMock(
                spec=trajectory_msgs__msg__JointTrajectory
            )
        )
        assert recorded_topics.is_operating


class TestRecordedTopicsIsValid:
    def test_standard_valid(self):
        recorded_topics = RecordedTopics(
            _joint_states=MagicMock(spec=sensor_msgs__msg__JointState),
            _hand_camera_image_raw_compressed=MagicMock(
                spec=sensor_msgs__msg__CompressedImage
            ),
            _head_rgbd_sensor_rgb_image_rect_color_compressed=MagicMock(
                spec=sensor_msgs__msg__CompressedImage
            ),
            _control_mode_topic=MagicMock(spec=std_msgs__msg__String),
        )
        assert recorded_topics.is_valid

    def test_partial_topics_valid(self):
        recorded_topics = RecordedTopics(
            _joint_states=None,
            _hand_camera_image_raw_compressed=MagicMock(
                spec=sensor_msgs__msg__CompressedImage
            ),
            _head_rgbd_sensor_rgb_image_rect_color_compressed=MagicMock(
                spec=sensor_msgs__msg__CompressedImage
            ),
            _control_mode_topic=MagicMock(spec=std_msgs__msg__String),
        )
        assert not recorded_topics.is_valid

    def test_no_topics_valid(self):
        recorded_topics = RecordedTopics(
            _joint_states=None,
            _hand_camera_image_raw_compressed=None,
            _head_rgbd_sensor_rgb_image_rect_color_compressed=None,
            _control_mode_topic=None,
        )
        assert not recorded_topics.is_valid


class TestRecordedTopicsHeadRgb:
    @patch("cv2.imdecode", return_value=np.zeros((480, 640, 3), dtype=np.uint8))
    @patch("cv2.cvtColor", return_value=np.zeros((480, 640, 3), dtype=np.uint8))
    def test_standard_head_rgb(self, mock_cvtColor, mock_imdecode):
        mock_data = np.frombuffer(b"\x00" * 307200, dtype=np.uint8)  # Mock image data
        mock_timestamp = 123456789

        recorded_topics = RecordedTopics(
            _head_rgbd_sensor_rgb_image_rect_color_compressed=MagicMock(
                spec=sensor_msgs__msg__CompressedImage, data=mock_data
            ),
            _head_rgbd_sensor_rgb_image_rect_color_compressed_timestamp=mock_timestamp,
        )

        timestamp, rgb_image = recorded_topics.head_rgb
        assert timestamp == mock_timestamp
        assert np.array_equal(rgb_image, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_imdecode.assert_called_once_with(mock_data, cv2.IMREAD_COLOR)

        assert mock_cvtColor.call_args[0][1] == cv2.COLOR_BGR2RGB
        assert np.array_equal(
            mock_cvtColor.call_args[0][0], np.zeros((480, 640, 3), dtype=np.uint8)
        )

    def test_error_head_rgb_none(self):
        recorded_topics = RecordedTopics(
            _head_rgbd_sensor_rgb_image_rect_color_compressed=None
        )
        with pytest.raises(
            ValueError, match="head_rgbd_sensor_rgb_image_rect_color_compressed is None"
        ):
            recorded_topics.head_rgb

    def test_error_head_rgb_timestamp_none(self):
        mock_data = np.frombuffer(b"\x00" * 307200, dtype=np.uint8)  # Mock image data
        recorded_topics = RecordedTopics(
            _head_rgbd_sensor_rgb_image_rect_color_compressed=MagicMock(
                spec=sensor_msgs__msg__CompressedImage, data=mock_data
            ),
            _head_rgbd_sensor_rgb_image_rect_color_compressed_timestamp=None,
        )

        with pytest.raises(
            ValueError,
            match="head_rgbd_sensor_rgb_image_rect_color_compressed_timestamp is None",
        ):
            recorded_topics.head_rgb


class TestRecordedTopicsHandRgb:
    @patch("cv2.imdecode", return_value=np.zeros((480, 640, 3), dtype=np.uint8))
    @patch("cv2.cvtColor", return_value=np.zeros((480, 640, 3), dtype=np.uint8))
    def test_standard_hand_rgb(self, mock_cvtColor, mock_imdecode):
        mock_data = np.frombuffer(b"\x00" * 307200, dtype=np.uint8)  # Mock image data
        mock_timestamp = 987654321

        recorded_topics = RecordedTopics(
            _hand_camera_image_raw_compressed=MagicMock(
                spec=sensor_msgs__msg__CompressedImage, data=mock_data
            ),
            _hand_camera_image_raw_compressed_timestamp=mock_timestamp,
        )

        timestamp, rgb_image = recorded_topics.hand_rgb
        assert timestamp == mock_timestamp
        assert np.array_equal(rgb_image, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_imdecode.assert_called_once_with(mock_data, cv2.IMREAD_COLOR)

        assert mock_cvtColor.call_args[0][1] == cv2.COLOR_BGR2RGB
        assert np.array_equal(
            mock_cvtColor.call_args[0][0], np.zeros((480, 640, 3), dtype=np.uint8)
        )

    def test_error_hand_rgb_none(self):
        recorded_topics = RecordedTopics(_hand_camera_image_raw_compressed=None)
        with pytest.raises(
            ValueError, match="hand_camera_image_raw_compressed is None"
        ):
            recorded_topics.hand_rgb

    def test_error_hand_rgb_timestamp_none(self):
        mock_data = np.frombuffer(b"\x00" * 307200, dtype=np.uint8)  # Mock image data
        recorded_topics = RecordedTopics(
            _hand_camera_image_raw_compressed=MagicMock(
                spec=sensor_msgs__msg__CompressedImage, data=mock_data
            ),
            _hand_camera_image_raw_compressed_timestamp=None,
        )

        with pytest.raises(
            ValueError, match="hand_camera_image_raw_compressed_timestamp is None"
        ):
            recorded_topics.hand_rgb


class TestRecordedTopicsJointPositions:
    def test_standard_joint_positions(self):
        mock_position = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mock_timestamp = 1122334455

        recorded_topics = RecordedTopics(
            _joint_states=MagicMock(
                spec=sensor_msgs__msg__JointState, position=mock_position
            ),
            _joint_states_timestamp=mock_timestamp,
        )

        timestamp, positions = recorded_topics.joint_positions
        assert timestamp == mock_timestamp
        assert np.array_equal(positions, mock_position)

    def test_error_joint_positions_none(self):
        recorded_topics = RecordedTopics(_joint_states=None)
        with pytest.raises(ValueError, match="joint_states is None"):
            recorded_topics.joint_positions

    def test_error_joint_positions_timestamp_none(self):
        mock_position = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        recorded_topics = RecordedTopics(
            _joint_states=MagicMock(
                spec=sensor_msgs__msg__JointState, position=mock_position
            ),
            _joint_states_timestamp=None,
        )

        with pytest.raises(ValueError, match="joint_states_timestamp is None"):
            recorded_topics.joint_positions


class TestRecordedTopicsJointVelocities:
    def test_standard_joint_velocities(self):
        mock_velocity = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_timestamp = 9988776655

        recorded_topics = RecordedTopics(
            _joint_states=MagicMock(
                spec=sensor_msgs__msg__JointState, velocity=mock_velocity
            ),
            _joint_states_timestamp=mock_timestamp,
        )

        timestamp, velocities = recorded_topics.joint_velocities
        assert timestamp == mock_timestamp
        assert np.array_equal(velocities, mock_velocity)

    def test_error_joint_velocities_none(self):
        recorded_topics = RecordedTopics(_joint_states=None)
        with pytest.raises(ValueError, match="joint_states is None"):
            recorded_topics.joint_velocities

    def test_error_joint_velocities_timestamp_none(self):
        mock_velocity = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        recorded_topics = RecordedTopics(
            _joint_states=MagicMock(
                spec=sensor_msgs__msg__JointState, velocity=mock_velocity
            ),
            _joint_states_timestamp=None,
        )

        with pytest.raises(ValueError, match="joint_states_timestamp is None"):
            recorded_topics.joint_velocities


class TestRecordedTopicsJointNames:
    def test_standard_joint_names(self):
        mock_names = ["joint1", "joint2", "joint3"]

        mock_joint_state = MagicMock()
        mock_joint_state.name = mock_names

        recorded_topics = RecordedTopics(_joint_states=mock_joint_state)

        joint_names = recorded_topics.joint_names
        assert joint_names == mock_names

    def test_error_joint_names_none(self):
        recorded_topics = RecordedTopics(_joint_states=None)
        with pytest.raises(ValueError, match="joint_states is None"):
            recorded_topics.joint_names


class TestRecordedTopicsWristWrench:
    def test_standard_wrist_wrench(self):
        mock_wrench = MagicMock()
        mock_wrench.wrench.force.x = 1.0
        mock_wrench.wrench.force.y = 2.0
        mock_wrench.wrench.force.z = 3.0
        mock_wrench.wrench.torque.x = 0.1
        mock_wrench.wrench.torque.y = 0.2
        mock_wrench.wrench.torque.z = 0.3

        recorded_topics = RecordedTopics(_wrist_wrench_raw=mock_wrench)

        wrench = recorded_topics.wrist_wrench
        expected_wrench = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float32)
        assert np.array_equal(wrench, expected_wrench)

    def test_wrist_wrench_none(self):
        recorded_topics = RecordedTopics(_wrist_wrench_raw=None)

        wrench = recorded_topics.wrist_wrench
        expected_wrench = np.zeros(6, dtype=np.float32)
        assert np.array_equal(wrench, expected_wrench)


class TestRecordedTopicsAbsArmAction:
    def test_standard_abs_arm_action(self):
        mock_positions = np.array([0.5, 1.0, 1.5], dtype=np.float32)
        mock_timestamp = 123456789

        mock_joint_trajectory = MagicMock(spec=trajectory_msgs__msg__JointTrajectory)
        mock_joint_trajectory.points = [MagicMock(positions=mock_positions)]

        recorded_topics = RecordedTopics(
            _arm_trajectory_controller_command=mock_joint_trajectory,
            _arm_trajectory_controller_command_timestamp=mock_timestamp,
        )

        timestamp, action = recorded_topics.abs_arm_action
        assert timestamp == mock_timestamp
        assert np.array_equal(action, mock_positions)

    def test_abs_arm_action_none(self):
        recorded_topics = RecordedTopics(_arm_trajectory_controller_command=None)

        timestamp, action = recorded_topics.abs_arm_action
        assert timestamp is None
        assert action is None


class TestRecordedTopicsAbsArmActionNames:
    def test_standard_abs_arm_action_names(self):
        mock_joint_names = ["joint1", "joint2", "joint3"]

        mock_joint_trajectory = MagicMock(spec=trajectory_msgs__msg__JointTrajectory)
        mock_joint_trajectory.joint_names = mock_joint_names

        recorded_topics = RecordedTopics(
            _arm_trajectory_controller_command=mock_joint_trajectory
        )

        joint_names = recorded_topics.abs_arm_action_names
        assert joint_names == mock_joint_names

    def test_abs_arm_action_names_none(self):
        recorded_topics = RecordedTopics(_arm_trajectory_controller_command=None)

        joint_names = recorded_topics.abs_arm_action_names
        assert joint_names == []


class TestRecordedTopicsAbsGripperAction:
    def test_abs_gripper_open_action(self):
        mock_timestamp = 123456789

        mock_gripper_command = MagicMock(spec=trajectory_msgs__msg__JointTrajectory)

        recorded_topics = RecordedTopics(
            _gripper_controller_command=mock_gripper_command,
            _gripper_controller_command_timestamp=mock_timestamp,
        )

        timestamp, action = recorded_topics.abs_gripper_action
        expected_action = np.array([GRIPPER_OPEN_ACTION])
        assert timestamp == mock_timestamp
        assert np.array_equal(action, expected_action)

    def test_abs_gripper_close_action(self):
        mock_timestamp = 987654321

        mock_grasp_command = MagicMock()

        recorded_topics = RecordedTopics(
            _gripper_controller_command=None,
            _gripper_controller_grasp_command=mock_grasp_command,
            _gripper_controller_grasp_command_timestamp=mock_timestamp,
        )

        timestamp, action = recorded_topics.abs_gripper_action
        expected_action = np.array([GRIPPER_CLOSE_ACTION])
        assert timestamp == mock_timestamp
        assert np.array_equal(action, expected_action)

    def test_abs_gripper_action_none(self):
        recorded_topics = RecordedTopics(
            _gripper_controller_command=None, _gripper_controller_grasp_command=None
        )

        timestamp, action = recorded_topics.abs_gripper_action
        assert timestamp is None
        assert action is None


class TestRecordedTopicsAbsGripperName:
    def test_abs_gripper_open_name(self):
        mock_joint_names = ["gripper_joint1", "gripper_joint2"]

        mock_gripper_command = MagicMock(spec=trajectory_msgs__msg__JointTrajectory)
        mock_gripper_command.joint_names = mock_joint_names

        recorded_topics = RecordedTopics(
            _gripper_controller_command=mock_gripper_command
        )

        joint_names = recorded_topics.abs_gripper_name
        assert joint_names == mock_joint_names

    def test_abs_gripper_close_name(self):
        mock_grasp_command = MagicMock()

        recorded_topics = RecordedTopics(
            _gripper_controller_command=None,
            _gripper_controller_grasp_command=mock_grasp_command,
        )

        joint_names = recorded_topics.abs_gripper_name
        expected_names = ["hand_motor_joint"]
        assert joint_names == expected_names

    def test_abs_gripper_name_none(self):
        recorded_topics = RecordedTopics(
            _gripper_controller_command=None, _gripper_controller_grasp_command=None
        )

        joint_names = recorded_topics.abs_gripper_name
        assert joint_names is None


class TestRecordedTopicsAbsHeadAction:
    def test_standard_abs_head_action(self):
        mock_positions = np.array([0.1, 0.2], dtype=np.float32)
        mock_timestamp = 135792468

        mock_head_trajectory = MagicMock(spec=trajectory_msgs__msg__JointTrajectory)
        mock_head_trajectory.points = [MagicMock(positions=mock_positions)]

        recorded_topics = RecordedTopics(
            _head_trajectory_controller_command=mock_head_trajectory,
            _head_trajectory_controller_command_timestamp=mock_timestamp,
        )

        timestamp, action = recorded_topics.abs_head_action
        assert timestamp == mock_timestamp
        assert np.array_equal(action, mock_positions)

    def test_abs_head_action_none(self):
        recorded_topics = RecordedTopics(_head_trajectory_controller_command=None)

        timestamp, action = recorded_topics.abs_head_action
        assert timestamp is None
        assert action is None


class TestRecordedTopicsAbsHeadActionNames:
    def test_standard_abs_head_action_names(self):
        mock_joint_names = ["head_joint1", "head_joint2"]

        mock_head_trajectory = MagicMock(spec=trajectory_msgs__msg__JointTrajectory)
        mock_head_trajectory.joint_names = mock_joint_names

        recorded_topics = RecordedTopics(
            _head_trajectory_controller_command=mock_head_trajectory
        )

        joint_names = recorded_topics.abs_head_action_names
        assert joint_names == mock_joint_names

    def test_abs_head_action_names_none(self):
        recorded_topics = RecordedTopics(_head_trajectory_controller_command=None)

        joint_names = recorded_topics.abs_head_action_names
        assert joint_names == []


class TestRecordedTopicsControlMode:
    def test_standard_control_mode(self):
        mock_mode_data = "AUTO"
        mock_timestamp = 246813579

        mock_control_mode = MagicMock(spec=std_msgs__msg__String)
        mock_control_mode.data = mock_mode_data

        recorded_topics = RecordedTopics(
            _control_mode_topic=mock_control_mode,
            _control_mode_topic_timestamp=mock_timestamp,
        )

        timestamp, mode = recorded_topics.control_mode
        assert timestamp == mock_timestamp
        assert mode == mock_mode_data

    def test_control_mode_none(self):
        recorded_topics = RecordedTopics(_control_mode_topic=None)

        timestamp, mode = recorded_topics.control_mode
        assert timestamp is None
        assert mode == "None"


class TestRecordedTopicsDeltaBaseAction:
    def test_standard_delta_base_action(self):
        mock_linear_x = 1.0
        mock_linear_y = 2.0
        mock_angular_z = 3.0
        mock_timestamp = 1212121212

        mock_twist = MagicMock(spec=geometry_msgs__msg__Twist)
        mock_linear = MagicMock(spec=geometry_msgs__msg__Vector3)
        mock_angular = MagicMock(spec=geometry_msgs__msg__Vector3)

        # Simulate the message structure
        mock_linear.x = mock_linear_x
        mock_linear.y = mock_linear_y
        mock_angular.z = mock_angular_z

        mock_twist.linear = mock_linear
        mock_twist.angular = mock_angular

        recorded_topics = RecordedTopics(
            _command_velocity=mock_twist, _command_velocity_timestamp=mock_timestamp
        )

        timestamp, action = recorded_topics.delta_base_action
        expected_action = np.array([mock_linear_x, mock_linear_y, mock_angular_z])
        assert timestamp == mock_timestamp
        assert np.array_equal(action, expected_action)

    def test_delta_base_action_none(self):
        recorded_topics = RecordedTopics(_command_velocity=None)

        timestamp, action = recorded_topics.delta_base_action
        expected_action = np.array([0.0, 0.0, 0.0])
        assert timestamp is None
        assert np.array_equal(action, expected_action)


class TestRecordedTopicsDeltaBaseActionNames:
    def test_standard_delta_base_action_names(self):
        recorded_topics = RecordedTopics()

        action_names = recorded_topics.delta_base_action_names
        expected_names = ["base_x", "base_y", "base_t"]
        assert action_names == expected_names


class TestRecordedTopicsUpdateTopics:
    def test_standard_arm_trajectory_update(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/hsrb/arm_trajectory_controller/command"
        mock_topic_data = trajectory_msgs__msg__JointTrajectory(
            header=MagicMock(), joint_names=MagicMock(), points=MagicMock()
        )

        recorded_topics = RecordedTopics()

        last_timestamp = 123456789

        recorded_topics.update_topics(mock_connection, last_timestamp, mock_topic_data)

        assert recorded_topics._arm_trajectory_controller_command == mock_topic_data
        assert (
            recorded_topics._arm_trajectory_controller_command_timestamp
            == last_timestamp
        )

    def test_error_invalid_topic_type(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/hsrb/arm_trajectory_controller/command"
        mock_topic_data = MagicMock()

        recorded_topics = RecordedTopics()

        last_timestamp = 123456789

        with pytest.raises(
            ValueError,
            match="Invalid topic type for /hsrb/arm_trajectory_controller/command",
        ):
            recorded_topics.update_topics(
                mock_connection, last_timestamp, mock_topic_data
            )

    def test_standard_gripper_command_update(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/hsrb/gripper_controller/command"
        mock_topic_data = trajectory_msgs__msg__JointTrajectory(
            header=MagicMock(), joint_names=MagicMock(), points=MagicMock()
        )

        recorded_topics = RecordedTopics()

        last_timestamp = 987654321

        recorded_topics.update_topics(mock_connection, last_timestamp, mock_topic_data)

        assert recorded_topics._gripper_controller_command == mock_topic_data
        assert recorded_topics._gripper_controller_command_timestamp == last_timestamp

    def test_error_invalid_gripper_topic_type(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/hsrb/gripper_controller/command"
        mock_topic_data = MagicMock()

        recorded_topics = RecordedTopics()

        last_timestamp = 987654321

        with pytest.raises(
            ValueError, match="Invalid topic type for /hsrb/gripper_controller/command"
        ):
            recorded_topics.update_topics(
                mock_connection, last_timestamp, mock_topic_data
            )

    def test_standard_gripper_grasp_goal_update(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/hsrb/gripper_controller/grasp/goal"
        mock_topic_data = MagicMock()

        recorded_topics = RecordedTopics()

        last_timestamp = 1122334455

        recorded_topics.update_topics(mock_connection, last_timestamp, mock_topic_data)

        assert recorded_topics._gripper_controller_grasp_command == mock_topic_data
        assert (
            recorded_topics._gripper_controller_grasp_command_timestamp
            == last_timestamp
        )

    def test_error_invalid_gripper_grasp_goal_type(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/hsrb/gripper_controller/grasp/goal"
        mock_topic_data = 123

        recorded_topics = RecordedTopics()

        last_timestamp = 1122334455

        recorded_topics.update_topics(mock_connection, last_timestamp, mock_topic_data)

        assert recorded_topics._gripper_controller_grasp_command == mock_topic_data
        assert (
            recorded_topics._gripper_controller_grasp_command_timestamp
            == last_timestamp
        )

    def test_standard_head_trajectory_update(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/hsrb/head_trajectory_controller/command"
        mock_topic_data = trajectory_msgs__msg__JointTrajectory(
            header=MagicMock(), joint_names=MagicMock(), points=MagicMock()
        )

        recorded_topics = RecordedTopics()

        last_timestamp = 2211443366

        recorded_topics.update_topics(mock_connection, last_timestamp, mock_topic_data)

        assert recorded_topics._head_trajectory_controller_command == mock_topic_data
        assert (
            recorded_topics._head_trajectory_controller_command_timestamp
            == last_timestamp
        )

    def test_error_invalid_head_trajectory_topic_type(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/hsrb/head_trajectory_controller/command"
        mock_topic_data = MagicMock()

        recorded_topics = RecordedTopics()

        last_timestamp = 2211443366

        with pytest.raises(
            ValueError,
            match="Invalid topic type for /hsrb/head_trajectory_controller/command",
        ):
            recorded_topics.update_topics(
                mock_connection, last_timestamp, mock_topic_data
            )

    def test_standard_joint_states_update(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/hsrb/joint_states"
        mock_topic_data = sensor_msgs__msg__JointState(
            header=MagicMock(),
            name=MagicMock(),
            position=MagicMock(),
            velocity=MagicMock(),
            effort=MagicMock(),
        )

        recorded_topics = RecordedTopics()

        last_timestamp = 3344556677

        recorded_topics.update_topics(mock_connection, last_timestamp, mock_topic_data)

        assert recorded_topics._joint_states == mock_topic_data
        assert recorded_topics._joint_states_timestamp == last_timestamp

    def test_error_invalid_joint_states_topic_type(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/hsrb/joint_states"
        mock_topic_data = MagicMock()

        recorded_topics = RecordedTopics()

        last_timestamp = 3344556677

        with pytest.raises(
            ValueError, match="Invalid topic type for /hsrb/joint_states"
        ):
            recorded_topics.update_topics(
                mock_connection, last_timestamp, mock_topic_data
            )

    def test_standard_hand_camera_compressed_update(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/hsrb/hand_camera/image_raw/compressed"
        mock_topic_data = sensor_msgs__msg__CompressedImage(
            header=MagicMock(), format="jpeg", data=MagicMock()
        )

        recorded_topics = RecordedTopics()

        last_timestamp = 5566778899

        recorded_topics.update_topics(mock_connection, last_timestamp, mock_topic_data)

        assert recorded_topics._hand_camera_image_raw_compressed == mock_topic_data
        assert (
            recorded_topics._hand_camera_image_raw_compressed_timestamp
            == last_timestamp
        )

    def test_error_invalid_hand_camera_compressed_topic_type(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/hsrb/hand_camera/image_raw/compressed"
        mock_topic_data = MagicMock()

        recorded_topics = RecordedTopics()

        last_timestamp = 5566778899

        with pytest.raises(
            ValueError,
            match="Invalid topic type for /hsrb/hand_camera/image_raw/compressed",
        ):
            recorded_topics.update_topics(
                mock_connection, last_timestamp, mock_topic_data
            )

    def test_standard_head_rgbd_sensor_rgb_image_update(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed"
        mock_topic_data = sensor_msgs__msg__CompressedImage(
            header=MagicMock(), format="jpeg", data=MagicMock()
        )

        recorded_topics = RecordedTopics()

        last_timestamp = 6677889900

        recorded_topics.update_topics(mock_connection, last_timestamp, mock_topic_data)

        assert (
            recorded_topics._head_rgbd_sensor_rgb_image_rect_color_compressed
            == mock_topic_data
        )
        assert (
            recorded_topics._head_rgbd_sensor_rgb_image_rect_color_compressed_timestamp
            == last_timestamp
        )

    def test_error_invalid_head_rgbd_sensor_rgb_image_type(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed"
        mock_topic_data = MagicMock()

        recorded_topics = RecordedTopics()

        last_timestamp = 6677889900

        with pytest.raises(
            ValueError,
            match="Invalid topic type for /hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed",
        ):
            recorded_topics.update_topics(
                mock_connection, last_timestamp, mock_topic_data
            )

    def test_standard_command_velocity_update(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/hsrb/command_velocity"
        mock_topic_data = geometry_msgs__msg__Twist(
            linear=MagicMock(), angular=MagicMock()
        )

        recorded_topics = RecordedTopics()

        last_timestamp = 7788990011

        recorded_topics.update_topics(mock_connection, last_timestamp, mock_topic_data)

        assert recorded_topics._command_velocity == mock_topic_data
        assert recorded_topics._command_velocity_timestamp == last_timestamp

    def test_error_invalid_command_velocity_topic_type(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/hsrb/command_velocity"
        mock_topic_data = MagicMock()

        recorded_topics = RecordedTopics()

        last_timestamp = 7788990011

        with pytest.raises(
            ValueError, match="Invalid topic type for /hsrb/command_velocity"
        ):
            recorded_topics.update_topics(
                mock_connection, last_timestamp, mock_topic_data
            )

    def test_standard_control_mode_update(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/control_mode"
        mock_topic_data = std_msgs__msg__String(data="AUTO")

        recorded_topics = RecordedTopics()

        last_timestamp = 8899001122

        recorded_topics.update_topics(mock_connection, last_timestamp, mock_topic_data)

        assert recorded_topics._control_mode_topic == mock_topic_data
        assert recorded_topics._control_mode_topic_timestamp == last_timestamp

    def test_error_invalid_control_mode_topic_type(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/control_mode"
        mock_topic_data = MagicMock()

        recorded_topics = RecordedTopics()

        last_timestamp = 8899001122

        with pytest.raises(ValueError, match="Invalid topic type for /control_mode"):
            recorded_topics.update_topics(
                mock_connection, last_timestamp, mock_topic_data
            )

    def test_standard_tf_update(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/tf"
        mock_topic_data = tf2_msgs__msg__TFMessage(transforms=MagicMock())

        recorded_topics = RecordedTopics()

        last_timestamp = 9900112233

        recorded_topics.update_topics(mock_connection, last_timestamp, mock_topic_data)

        assert recorded_topics._tf == mock_topic_data

    def test_error_invalid_tf_topic_type(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/tf"
        mock_topic_data = MagicMock()

        recorded_topics = RecordedTopics()

        last_timestamp = 9900112233

        with pytest.raises(ValueError, match="Invalid topic type for /tf"):
            recorded_topics.update_topics(
                mock_connection, last_timestamp, mock_topic_data
            )

    def test_standard_tf_static_update(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/tf_static"
        mock_topic_data = tf2_msgs__msg__TFMessage(transforms=MagicMock())

        recorded_topics = RecordedTopics()

        last_timestamp = 9999111122

        recorded_topics.update_topics(mock_connection, last_timestamp, mock_topic_data)

        assert recorded_topics._tf_static == mock_topic_data

    def test_error_invalid_tf_static_topic_type(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/tf_static"
        mock_topic_data = MagicMock()

        recorded_topics = RecordedTopics()

        last_timestamp = 9999111122

        with pytest.raises(ValueError, match="Invalid topic type for /tf_static"):
            recorded_topics.update_topics(
                mock_connection, last_timestamp, mock_topic_data
            )

    def test_standard_wrist_wrench_raw_update(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/hsrb/wrist_wrench/raw"
        mock_topic_data = MagicMock()

        recorded_topics = RecordedTopics()

        last_timestamp = 1010101010

        recorded_topics.update_topics(mock_connection, last_timestamp, mock_topic_data)

        assert recorded_topics._wrist_wrench_raw == mock_topic_data

    def test_assigning_non_magic_mock_to_wrist_wrench_raw(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/hsrb/wrist_wrench/raw"
        mock_topic_data = "Unexpected Type"

        recorded_topics = RecordedTopics()

        last_timestamp = 1010101010

        recorded_topics.update_topics(mock_connection, last_timestamp, mock_topic_data)

        assert recorded_topics._wrist_wrench_raw == mock_topic_data

    def test_unknown_topic_name(self):
        mock_connection = MagicMock()
        mock_connection.topic = "/unknown/topic"
        mock_topic_data = MagicMock()

        recorded_topics = RecordedTopics()

        last_timestamp = 1234567890

        with pytest.raises(ValueError, match=r"Unknown topic name: /unknown/topic"):
            recorded_topics.update_topics(
                mock_connection, last_timestamp, mock_topic_data
            )
