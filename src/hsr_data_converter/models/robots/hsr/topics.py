from dataclasses import dataclass
from typing import Any, List, Tuple, cast

import cv2
import numpy as np
from rosbags.interfaces import Connection
from rosbags.typesys.stores.ros1_noetic import (
    geometry_msgs__msg__Twist,
    sensor_msgs__msg__CompressedImage,
    sensor_msgs__msg__JointState,
    std_msgs__msg__String,
    trajectory_msgs__msg__JointTrajectory,
)
from rosbags.typesys.stores.ros2_dashing import tf2_msgs__msg__TFMessage

from hsr_data_converter.models.robots.hsr.config import (
    GRIPPER_CLOSE_ACTION,
    GRIPPER_OPEN_ACTION,
)


@dataclass
class RecordedTopics:
    # variables to store topics
    _arm_trajectory_controller_command: trajectory_msgs__msg__JointTrajectory | None = (
        None
    )
    _head_trajectory_controller_command: (
        trajectory_msgs__msg__JointTrajectory | None
    ) = None
    _gripper_controller_command: trajectory_msgs__msg__JointTrajectory | None = None
    _gripper_controller_grasp_command: object = None  # CAUTION: this topic is tmc original message, so we do not specify a message type
    _command_velocity: geometry_msgs__msg__Twist | None = None
    _joint_states: sensor_msgs__msg__JointState | None = None
    _servo_states: Any = None
    _hand_camera_image_raw_compressed: sensor_msgs__msg__CompressedImage | None = None
    _head_rgbd_sensor_rgb_image_rect_color_compressed: (
        sensor_msgs__msg__CompressedImage | None
    ) = None
    _control_mode_topic: std_msgs__msg__String | None = None
    _tf: tf2_msgs__msg__TFMessage | None = None
    _tf_static: tf2_msgs__msg__TFMessage | None = None
    _wrist_wrench_raw: Any = None

    # variables to store timestamp [ns]
    _arm_trajectory_controller_command_timestamp: int | None = None
    _head_trajectory_controller_command_timestamp: int | None = None
    _gripper_controller_command_timestamp: int | None = None
    _gripper_controller_grasp_command_timestamp: int | None = None
    _command_velocity_timestamp: int | None = None
    _joint_states_timestamp: int | None = None
    _servo_states_timestamp: int | None = None
    _hand_camera_image_raw_compressed_timestamp: int | None = None
    _head_rgbd_sensor_rgb_image_rect_color_compressed_timestamp: int | None = None
    _control_mode_topic_timestamp: int | None = None

    @property
    def is_operating(self) -> bool:
        """
        Check if any action topics are being published

        Returns
        -------
        bool
            True if any action topics are being published
        """
        action_topics = [
            self._arm_trajectory_controller_command,
            self._head_trajectory_controller_command,
            self._gripper_controller_command,
            self._gripper_controller_grasp_command,
            self._command_velocity,
        ]
        return any([x is not None for x in action_topics])

    @property
    def is_valid(self) -> bool:
        """
        Check if all necessary topics to make a frame are available

        Returns
        -------
        bool
            True if all necessary topics are available
        """
        return all(
            [
                # NOTE: excluding action topics because they are not necessary to make a frame
                self._joint_states,
                self._hand_camera_image_raw_compressed,
                self._head_rgbd_sensor_rgb_image_rect_color_compressed,
                self._control_mode_topic,
            ]
        )

    @property
    def head_rgb(self) -> Tuple[int, np.ndarray]:
        """
        Get the head RGB image from the compressed image topic

        Returns
        -------
        Tuple[int, np.ndarray]
            The timestamp and the RGB image at the timestamp

        Raises
        ------
        ValueError
            If the head RGB image is not available
        """
        if self._head_rgbd_sensor_rgb_image_rect_color_compressed is None:
            raise ValueError("head_rgbd_sensor_rgb_image_rect_color_compressed is None")
        img = cv2.imdecode(
            self._head_rgbd_sensor_rgb_image_rect_color_compressed.data,
            cv2.IMREAD_COLOR,
        )
        timestamp = self._head_rgbd_sensor_rgb_image_rect_color_compressed_timestamp
        if timestamp is None:
            raise ValueError(
                "head_rgbd_sensor_rgb_image_rect_color_compressed_timestamp is None"
            )
        return timestamp, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @property
    def hand_rgb(self) -> Tuple[int, np.ndarray]:
        if self._hand_camera_image_raw_compressed is None:
            raise ValueError("hand_camera_image_raw_compressed is None")
        img = cv2.imdecode(
            self._hand_camera_image_raw_compressed.data, cv2.IMREAD_COLOR
        )
        timestamp = self._hand_camera_image_raw_compressed_timestamp
        if timestamp is None:
            raise ValueError("hand_camera_image_raw_compressed_timestamp is None")
        return timestamp, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @property
    def joint_positions(self) -> Tuple[int, np.ndarray]:
        if self._joint_states is None:
            raise ValueError("joint_states is None")
        timestamp = self._joint_states_timestamp
        if timestamp is None:
            raise ValueError("joint_states_timestamp is None")
        return timestamp, self._joint_states.position

    @property
    def joint_velocities(self) -> Tuple[int, np.ndarray]:
        if self._joint_states is None:
            raise ValueError("joint_states is None")
        timestamp = self._joint_states_timestamp
        if timestamp is None:
            raise ValueError("joint_states_timestamp is None")
        return timestamp, self._joint_states.velocity

    @property
    def joint_names(self) -> List[str]:
        if self._joint_states is None:
            raise ValueError("joint_states is None")
        return self._joint_states.name

    @property
    def wrist_wrench(self) -> np.ndarray:
        if self._wrist_wrench_raw is None:
            # Return a zero array if no wrench data is available
            return np.zeros(6, dtype=np.float32)

        return np.array(
            [
                self._wrist_wrench_raw.wrench.force.x,
                self._wrist_wrench_raw.wrench.force.y,
                self._wrist_wrench_raw.wrench.force.z,
                self._wrist_wrench_raw.wrench.torque.x,
                self._wrist_wrench_raw.wrench.torque.y,
                self._wrist_wrench_raw.wrench.torque.z,
            ],
            dtype=np.float32,
        )

    @property
    def abs_arm_action(self) -> Tuple[int | None, np.ndarray | None]:
        if self._arm_trajectory_controller_command is None:
            return None, None
        timestamp = self._arm_trajectory_controller_command_timestamp
        action = self._arm_trajectory_controller_command.points[0].positions
        return timestamp, action

    @property
    def abs_arm_action_names(self) -> List[str]:
        if self._arm_trajectory_controller_command is None:
            return []
        joint_names = self._arm_trajectory_controller_command.joint_names
        return joint_names

    @property
    def abs_gripper_action(self) -> Tuple[int | None, np.ndarray | None]:
        # CAUTION: prioritize open action over close action
        if (
            self._gripper_controller_command is None
            and self._gripper_controller_grasp_command is None
        ):  # open and close action are None
            return None, None
        elif self._gripper_controller_command is not None:  # open action is available
            timestamp = self._gripper_controller_command_timestamp
            return timestamp, np.array([GRIPPER_OPEN_ACTION])
        elif (
            self._gripper_controller_grasp_command is not None
        ):  # close action is available
            timestamp = self._gripper_controller_grasp_command_timestamp
            return timestamp, np.array([GRIPPER_CLOSE_ACTION])

        # This code should never be reached due to the conditions above
        raise RuntimeError("Unexpected state in abs_gripper_action")

    @property
    def abs_gripper_name(self) -> List[str] | None:
        if (
            self._gripper_controller_command is None
            and self._gripper_controller_grasp_command is None
        ):  # open and close action are None
            return None
        elif self._gripper_controller_command is not None:  # open action is available
            return self._gripper_controller_command.joint_names
        else:
            return [
                "hand_motor_joint"
            ]  # return hard-coded name for close action because the gripper_controller_grasp_command does not have joint_name.

    @property
    def abs_head_action(self) -> Tuple[int | None, np.ndarray | None]:
        if self._head_trajectory_controller_command is None:
            return None, None
        timestamp = self._head_trajectory_controller_command_timestamp
        action = self._head_trajectory_controller_command.points[0].positions
        return timestamp, action

    @property
    def abs_head_action_names(self) -> List[str]:
        if self._head_trajectory_controller_command is None:
            return []
        return self._head_trajectory_controller_command.joint_names

    @property
    def servo_states_names(self) -> List[str]:
        if self._servo_states is None:
            return []
        return self._servo_states.name

    @property
    def servo_states_command_position(self) -> Tuple[int | None, np.ndarray | None]:
        if self._servo_states is None:
            return None, None
        timestamp = self._servo_states_timestamp
        position = self._servo_states.command_position
        return timestamp, position

    @property
    def servo_states_present_motor_shaft_position(
        self,
    ) -> Tuple[int | None, np.ndarray | None]:
        if self._servo_states is None:
            return None, None
        timestamp = self._servo_states_timestamp
        position = self._servo_states.present_motor_shaft_position
        return timestamp, position

    @property
    def servo_states_present_driven_shaft_position(
        self,
    ) -> Tuple[int | None, np.ndarray | None]:
        if self._servo_states is None:
            return None, None
        timestamp = self._servo_states_timestamp
        position = self._servo_states.present_driven_shaft_position
        return timestamp, position

    @property
    def servo_states_present_velocity(self) -> Tuple[int | None, np.ndarray | None]:
        if self._servo_states is None:
            return None, None
        timestamp = self._servo_states_timestamp
        velocity = self._servo_states.present_velocity
        return timestamp, velocity

    @property
    def servo_states_command_velocity(self) -> Tuple[int | None, np.ndarray | None]:
        if self._servo_states is None:
            return None, None
        timestamp = self._servo_states_timestamp
        velocity = self._servo_states.command_velocity
        return timestamp, velocity

    @property
    def servo_states_present_effort(self) -> Tuple[int | None, np.ndarray | None]:
        if self._servo_states is None:
            return None, None
        timestamp = self._servo_states_timestamp
        effort = self._servo_states.present_effort
        return timestamp, effort

    @property
    def servo_states_present_temperature(self) -> Tuple[int | None, np.ndarray | None]:
        if self._servo_states is None:
            return None, None
        timestamp = self._servo_states_timestamp
        temperature = self._servo_states.present_temperature
        return timestamp, temperature

    @property
    def servo_states_present_current(self) -> Tuple[int | None, np.ndarray | None]:
        if self._servo_states is None:
            return None, None
        timestamp = self._servo_states_timestamp
        current = self._servo_states.present_current
        return timestamp, current

    @property
    def control_mode(self) -> Tuple[int | None, str]:
        if self._control_mode_topic is None:
            return None, "None"
        timestamp = self._control_mode_topic_timestamp
        return timestamp, self._control_mode_topic.data

    @property
    def delta_base_action(self) -> Tuple[int | None, np.ndarray | None]:
        if self._command_velocity is None:
            return None, np.array(
                [0.0, 0.0, 0.0]
            )  # Stop the base if the command is not sent
        timestamp = self._command_velocity_timestamp
        action = np.array(
            [
                self._command_velocity.linear.x,
                self._command_velocity.linear.y,
                self._command_velocity.angular.z,
            ]
        )
        return timestamp, action

    @property
    def delta_base_action_names(self) -> List[str]:
        return ["base_x", "base_y", "base_t"]

    def update_topics(self, connection: Connection, last_timestamp: int, topic: object):
        """
        Update a topic from rosbags

        Parameters
        ----------
        connection : Connection
            information about topic
        last_timestamp : int
            last timestamp of the topic in nanoseconds
        topic : object
            topic data (contents depend on the topic)

        Raises
        ------
        ValueError
            If the topic name is unknown or not supported
        """

        if connection.topic == "/hsrb/arm_trajectory_controller/command":
            if "JointTrajectory" not in str(type(topic)):
                raise ValueError(f"Invalid topic type for {connection.topic}")
            self._arm_trajectory_controller_command = cast(
                trajectory_msgs__msg__JointTrajectory, topic
            )
            self._arm_trajectory_controller_command_timestamp = last_timestamp
        elif connection.topic == "/hsrb/gripper_controller/command":
            if "JointTrajectory" not in str(type(topic)):
                raise ValueError(f"Invalid topic type for {connection.topic}")
            self._gripper_controller_command = cast(
                trajectory_msgs__msg__JointTrajectory, topic
            )
            self._gripper_controller_command_timestamp = last_timestamp
        elif connection.topic == "/hsrb/gripper_controller/grasp/goal":
            self._gripper_controller_grasp_command = topic
            self._gripper_controller_grasp_command_timestamp = last_timestamp
        elif connection.topic == "/hsrb/head_trajectory_controller/command":
            if "JointTrajectory" not in str(type(topic)):
                raise ValueError(f"Invalid topic type for {connection.topic}")
            self._head_trajectory_controller_command = cast(
                trajectory_msgs__msg__JointTrajectory, topic
            )
            self._head_trajectory_controller_command_timestamp = last_timestamp
        elif connection.topic == "/hsrb/joint_states":
            if "JointState" not in str(type(topic)):
                raise ValueError(f"Invalid topic type for {connection.topic}")
            self._joint_states = cast(sensor_msgs__msg__JointState, topic)
            self._joint_states_timestamp = last_timestamp
        elif connection.topic == "/hsrb/servo_states":
            self._servo_states = topic
            self._servo_states_timestamp = last_timestamp
        elif connection.topic == "/hsrb/hand_camera/image_raw/compressed":
            if "CompressedImage" not in str(type(topic)):
                raise ValueError(f"Invalid topic type for {connection.topic}")
            self._hand_camera_image_raw_compressed = cast(
                sensor_msgs__msg__CompressedImage, topic
            )
            self._hand_camera_image_raw_compressed_timestamp = last_timestamp
        elif (
            connection.topic == "/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed"
        ):
            if "CompressedImage" not in str(type(topic)):
                raise ValueError(f"Invalid topic type for {connection.topic}")
            self._head_rgbd_sensor_rgb_image_rect_color_compressed = cast(
                sensor_msgs__msg__CompressedImage, topic
            )
            self._head_rgbd_sensor_rgb_image_rect_color_compressed_timestamp = (
                last_timestamp
            )
        elif connection.topic == "/hsrb/command_velocity":
            if "Twist" not in str(type(topic)):
                raise ValueError(f"Invalid topic type for {connection.topic}")
            self._command_velocity = cast(geometry_msgs__msg__Twist, topic)
            self._command_velocity_timestamp = last_timestamp
        elif connection.topic == "/control_mode":
            if "String" not in str(type(topic)):
                raise ValueError(f"Invalid topic type for {connection.topic}")
            self._control_mode_topic = cast(std_msgs__msg__String, topic)
            self._control_mode_topic_timestamp = last_timestamp
        elif connection.topic == "/tf":
            if "TFMessage" not in str(type(topic)):
                raise ValueError(f"Invalid topic type for {connection.topic}")
            self._tf = cast(tf2_msgs__msg__TFMessage, topic)
        elif connection.topic == "/tf_static":
            if "TFMessage" not in str(type(topic)):
                raise ValueError(f"Invalid topic type for {connection.topic}")
            self._tf_static = cast(tf2_msgs__msg__TFMessage, topic)
        elif connection.topic == "/hsrb/wrist_wrench/raw":
            self._wrist_wrench_raw = topic
        else:
            raise ValueError(f"Unknown topic name: {connection.topic}")
