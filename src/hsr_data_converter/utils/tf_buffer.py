from typing import cast

import numpy as np
from rosbags.typesys.stores.ros1_noetic import geometry_msgs__msg__Transform
from rosbags.typesys.stores.ros2_dashing import tf2_msgs__msg__TFMessage

from hsr_data_converter.utils.transformations import (
    euler_from_matrix,
    quaternion_matrix,
    translation_from_matrix,
)


def transform_to_matrix(transform: geometry_msgs__msg__Transform) -> np.ndarray:
    """
    Converts a ROS Transform message to a 4x4 NumPy matrix using transformations.py.
    """
    # The ROS quaternion is (x, y, z, w)
    # The transformations.py library expects a quaternion as [x, y, z, w]
    ros_quat = transform.rotation
    quat = [ros_quat.x, ros_quat.y, ros_quat.z, ros_quat.w]

    # Create a 4x4 rotation matrix from the quaternion
    rotation_matrix = quaternion_matrix(quat)

    # Get the translation vector
    ros_trans = transform.translation
    trans = [ros_trans.x, ros_trans.y, ros_trans.z]

    # Set the translation part of the matrix
    rotation_matrix[:3, 3] = trans

    return rotation_matrix


def matrix_to_xyz_rpy(matrix: np.ndarray) -> np.ndarray:
    """
    Converts a 4x4 NumPy matrix to a 6D pose array (x, y, z, roll, pitch, yaw)
    using transformations.py.
    """
    # Extract the translation vector from the matrix
    translation = translation_from_matrix(matrix)

    # Extract Euler angles (roll, pitch, yaw) from the matrix.
    # 'sxyz' specifies a static frame rotation sequence: first around X (roll),
    # then Y (pitch), then Z (yaw). This is a common convention.
    rpy = euler_from_matrix(matrix, axes="sxyz")

    # Concatenate translation and rotation into a single 6D vector
    return np.concatenate([translation, rpy]).astype(np.float32)


class TFBuffer:
    """A buffer to store and look up TF transformations from a ROS bag.

    Follows ROS2 tf2 convention: lookup_transform(target, source, time) returns
    T_source_target matrix that transforms points from target frame to source frame.
    """

    def __init__(self) -> None:
        # For transforms from /tf_static
        # {child_frame: (parent_frame, matrix)}
        self.static_transforms: dict[str, tuple[str, np.ndarray]] = {}

        # For transforms from /tf
        # {child_frame: [(timestamp, parent_frame, matrix)]}
        self.dynamic_transforms: dict[str, list[tuple[int, str, np.ndarray]]] = {}

    def add_transform_msg(self, msg: tf2_msgs__msg__TFMessage, is_static: bool) -> None:
        for transform_stamped in msg.transforms:
            child_frame = transform_stamped.child_frame_id.lstrip("/")
            parent_frame = transform_stamped.header.frame_id.lstrip("/")
            matrix = transform_to_matrix(
                cast(geometry_msgs__msg__Transform, transform_stamped.transform)
            )

            if is_static:
                self.static_transforms[child_frame] = (parent_frame, matrix)
            else:
                if child_frame not in self.dynamic_transforms:
                    self.dynamic_transforms[child_frame] = []

                timestamp = (
                    transform_stamped.header.stamp.sec * 1_000_000_000
                    + transform_stamped.header.stamp.nanosec
                )
                self.dynamic_transforms[child_frame].append(
                    (timestamp, parent_frame, matrix)
                )

    def _find_transform(
        self, child_frame: str, timestamp: int
    ) -> tuple[str | None, np.ndarray | None]:
        """Finds the transform for a child_frame at a specific time."""
        if child_frame in self.static_transforms:
            return self.static_transforms[child_frame]

        if child_frame in self.dynamic_transforms:
            transforms = self.dynamic_transforms[child_frame]
            # Find the latest transform at or before the given timestamp
            best_match = None
            for ts, parent, matrix in reversed(transforms):
                if ts <= timestamp:
                    best_match = (parent, matrix)
                    break
            if best_match:
                return best_match

        return None, None

    def lookup_transform(
        self, target_frame: str, source_frame: str, timestamp: int
    ) -> np.ndarray | None:
        """
        Calculates the full transformation from source_frame to target_frame.
        Returns T_source_target matrix (transforms points from target to source frame)
        to match ROS2 tf2 convention.
        """
        # Chain from source up to the target_frame
        chain_to_target = []
        current_frame = source_frame

        # Traverse up the tree from the source to the target
        while current_frame != target_frame:
            parent_frame, matrix = self._find_transform(current_frame, timestamp)
            if matrix is None:
                print(
                    f"[WARN] Could not find transform for '{current_frame}' at timestamp {timestamp}"
                )
                return None
            chain_to_target.append(matrix)
            if parent_frame is None:
                # TODO: Error here?
                return None

            current_frame = parent_frame

        # Multiply transforms down the chain
        if not chain_to_target:
            return np.eye(4)

        chain_to_source = list(reversed(chain_to_target))
        total_transform = chain_to_source.pop()
        while chain_to_source:
            total_transform = total_transform @ chain_to_source.pop()

        # Return inverse to match ROS2 tf2 convention (T_source_target)
        return np.linalg.inv(total_transform)
