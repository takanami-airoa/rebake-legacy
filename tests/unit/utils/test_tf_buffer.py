import unittest
import pytest
from unittest.mock import MagicMock, patch

import numpy as np

from rosbags.typesys.stores.ros1_noetic import geometry_msgs__msg__Transform
from rosbags.typesys.stores.ros2_dashing import tf2_msgs__msg__TFMessage

from hsr_data_converter.utils.tf_buffer import (
    TFBuffer,
    transform_to_matrix,
    matrix_to_xyz_rpy,
)


class TestTransformToMatrix(unittest.TestCase):
    @patch("hsr_data_converter.utils.tf_buffer.quaternion_matrix")
    def test_standard_translation(self, mock_quaternion_matrix):
        mock_quaternion_matrix.return_value = np.identity(4)

        mock_transform = MagicMock()
        mock_transform.rotation.x = 0.0
        mock_transform.rotation.y = 0.0
        mock_transform.rotation.z = 0.0
        mock_transform.rotation.w = 1.0
        mock_transform.translation.x = 1.0
        mock_transform.translation.y = 2.0
        mock_transform.translation.z = 3.0

        result_matrix = transform_to_matrix(mock_transform)
        expected_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        assert np.allclose(result_matrix, expected_matrix)
        mock_quaternion_matrix.assert_called_once_with([0.0, 0.0, 0.0, 1.0])

    @patch("hsr_data_converter.utils.tf_buffer.quaternion_matrix")
    def test_rotation_quaternion(self, mock_quaternion_matrix):
        mock_quaternion_matrix.return_value = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        mock_transform = MagicMock()
        mock_transform.rotation.x = 0.0
        mock_transform.rotation.y = 0.7071
        mock_transform.rotation.z = 0.0
        mock_transform.rotation.w = 0.7071
        mock_transform.translation.x = 0.0
        mock_transform.translation.y = 0.0
        mock_transform.translation.z = 0.0

        result_matrix = transform_to_matrix(mock_transform)
        expected_matrix = mock_quaternion_matrix.return_value
        assert np.allclose(result_matrix, expected_matrix)
        mock_quaternion_matrix.assert_called_once_with([0.0, 0.7071, 0.0, 0.7071])

    @patch("hsr_data_converter.utils.tf_buffer.quaternion_matrix")
    def test_zero_translation(self, mock_quaternion_matrix):
        mock_quaternion_matrix.return_value = np.identity(4)

        mock_transform = MagicMock()
        mock_transform.rotation.x = 0.0
        mock_transform.rotation.y = 0.0
        mock_transform.rotation.z = 0.0
        mock_transform.rotation.w = 1.0
        mock_transform.translation.x = 0.0
        mock_transform.translation.y = 0.0
        mock_transform.translation.z = 0.0

        result_matrix = transform_to_matrix(mock_transform)
        expected_matrix = np.identity(4)
        assert np.allclose(result_matrix, expected_matrix)
        mock_quaternion_matrix.assert_called_once_with([0.0, 0.0, 0.0, 1.0])

    @patch("hsr_data_converter.utils.tf_buffer.quaternion_matrix")
    def test_large_translation(self, mock_quaternion_matrix):
        mock_quaternion_matrix.return_value = np.identity(4)

        mock_transform = MagicMock()
        mock_transform.rotation.x = 0.0
        mock_transform.rotation.y = 0.0
        mock_transform.rotation.z = 0.0
        mock_transform.rotation.w = 1.0
        mock_transform.translation.x = 1000.0
        mock_transform.translation.y = 2000.0
        mock_transform.translation.z = 3000.0

        result_matrix = transform_to_matrix(mock_transform)
        expected_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 1000.0],
                [0.0, 1.0, 0.0, 2000.0],
                [0.0, 0.0, 1.0, 3000.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        assert np.allclose(result_matrix, expected_matrix)
        mock_quaternion_matrix.assert_called_once_with([0.0, 0.0, 0.0, 1.0])

    @patch("hsr_data_converter.utils.tf_buffer.quaternion_matrix")
    def test_zero_quaternion(self, mock_quaternion_matrix):
        mock_quaternion_matrix.return_value = np.identity(4)

        mock_transform = MagicMock()
        mock_transform.rotation.x = 0.0
        mock_transform.rotation.y = 0.0
        mock_transform.rotation.z = 0.0
        mock_transform.rotation.w = 0.0
        mock_transform.translation.x = 1.0
        mock_transform.translation.y = 2.0
        mock_transform.translation.z = 3.0

        result_matrix = transform_to_matrix(mock_transform)
        expected_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        assert np.allclose(result_matrix, expected_matrix)
        mock_quaternion_matrix.assert_called_once_with([0.0, 0.0, 0.0, 0.0])

    @patch("hsr_data_converter.utils.tf_buffer.quaternion_matrix")
    def test_error_quaternion_matrix_failure(self, mock_quaternion_matrix):
        mock_quaternion_matrix.side_effect = Exception(
            "Failed to compute quaternion matrix"
        )

        mock_transform = MagicMock()
        mock_transform.rotation.x = 0.0
        mock_transform.rotation.y = 0.0
        mock_transform.rotation.z = 0.0
        mock_transform.rotation.w = 1.0
        mock_transform.translation.x = 1.0
        mock_transform.translation.y = 2.0
        mock_transform.translation.z = 3.0

        with pytest.raises(Exception, match="Failed to compute quaternion matrix"):
            transform_to_matrix(mock_transform)
        mock_quaternion_matrix.assert_called_once_with([0.0, 0.0, 0.0, 1.0])


class TestMatrixToXyzRpy:
    @patch("hsr_data_converter.utils.tf_buffer.translation_from_matrix")
    @patch("hsr_data_converter.utils.tf_buffer.euler_from_matrix")
    def test_standard_conversion(
        self, mock_euler_from_matrix, mock_translation_from_matrix
    ):
        mock_translation_from_matrix.return_value = np.array([1.0, 2.0, 3.0])
        mock_euler_from_matrix.return_value = np.array([0.1, 0.2, 0.3])

        mock_matrix = np.identity(4)

        result = matrix_to_xyz_rpy(mock_matrix)
        expected_result = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float32)

        assert np.allclose(result, expected_result)
        mock_translation_from_matrix.assert_called_once_with(mock_matrix)
        mock_euler_from_matrix.assert_called_once_with(mock_matrix, axes="sxyz")

    @patch("hsr_data_converter.utils.tf_buffer.translation_from_matrix")
    @patch("hsr_data_converter.utils.tf_buffer.euler_from_matrix")
    def test_zero_matrix(self, mock_euler_from_matrix, mock_translation_from_matrix):
        mock_translation_from_matrix.return_value = np.array([0.0, 0.0, 0.0])
        mock_euler_from_matrix.return_value = np.array([0.0, 0.0, 0.0])

        mock_matrix = np.zeros((4, 4))

        result = matrix_to_xyz_rpy(mock_matrix)
        expected_result = np.zeros(6, dtype=np.float32)

        assert np.allclose(result, expected_result)
        mock_translation_from_matrix.assert_called_once_with(mock_matrix)
        mock_euler_from_matrix.assert_called_once_with(mock_matrix, axes="sxyz")

    @patch(
        "hsr_data_converter.utils.tf_buffer.translation_from_matrix",
        side_effect=Exception("Failed to get translation"),
    )
    @patch("hsr_data_converter.utils.tf_buffer.euler_from_matrix")
    def test_error_translation_failure(
        self, mock_euler_from_matrix, mock_translation_from_matrix
    ):
        mock_matrix = np.identity(4)

        with pytest.raises(Exception, match="Failed to get translation"):
            matrix_to_xyz_rpy(mock_matrix)
        mock_translation_from_matrix.assert_called_once_with(mock_matrix)
        mock_euler_from_matrix.assert_not_called()

    @patch("hsr_data_converter.utils.tf_buffer.translation_from_matrix")
    @patch(
        "hsr_data_converter.utils.tf_buffer.euler_from_matrix",
        side_effect=Exception("Failed to get Euler angles"),
    )
    def test_error_euler_failure(
        self, mock_euler_from_matrix, mock_translation_from_matrix
    ):
        mock_translation_from_matrix.return_value = np.array([1.0, 2.0, 3.0])
        mock_matrix = np.identity(4)

        with pytest.raises(Exception, match="Failed to get Euler angles"):
            matrix_to_xyz_rpy(mock_matrix)
        mock_translation_from_matrix.assert_called_once_with(mock_matrix)
        mock_euler_from_matrix.assert_called_once_with(mock_matrix, axes="sxyz")


class TestTfBufferConstructor(unittest.TestCase):
    def test_standard_initialization(self):
        buffer = TFBuffer()
        assert buffer.static_transforms == {}
        assert buffer.dynamic_transforms == {}


class TestTFBufferAddTransformMsg(unittest.TestCase):
    @patch("hsr_data_converter.utils.tf_buffer.transform_to_matrix")
    def test_standard_add_static_transform(self, mock_transform_to_matrix):
        mock_transform_to_matrix.return_value = np.eye(4)

        buffer = TFBuffer()
        msg = MagicMock(spec=tf2_msgs__msg__TFMessage)
        transform_stamped = MagicMock()
        transform_stamped.child_frame_id = "/child"
        transform_stamped.header.frame_id = "/parent"
        transform_stamped.transform = MagicMock(spec=geometry_msgs__msg__Transform)
        msg.transforms = [transform_stamped]

        buffer.add_transform_msg(msg, is_static=True)

        assert buffer.static_transforms["child"][0] == "parent"
        assert np.array_equal(buffer.static_transforms["child"][1], np.eye(4))
        mock_transform_to_matrix.assert_called_once_with(transform_stamped.transform)

    @patch("hsr_data_converter.utils.tf_buffer.transform_to_matrix")
    def test_standard_add_dynamic_transform(self, mock_transform_to_matrix):
        mock_transform_to_matrix.return_value = np.eye(4)

        buffer = TFBuffer()
        msg = MagicMock(spec=tf2_msgs__msg__TFMessage)
        transform_stamped = MagicMock()
        transform_stamped.child_frame_id = "/child"
        transform_stamped.header.frame_id = "/parent"
        transform_stamped.transform = MagicMock(spec=geometry_msgs__msg__Transform)
        transform_stamped.header.stamp.sec = 0
        transform_stamped.header.stamp.nanosec = 1_000_000
        msg.transforms = [transform_stamped]

        buffer.add_transform_msg(msg, is_static=False)

        timestamp = (
            transform_stamped.header.stamp.sec * 1_000_000_000
            + transform_stamped.header.stamp.nanosec
        )
        assert buffer.dynamic_transforms["child"][0][0] == timestamp
        assert buffer.dynamic_transforms["child"][0][1] == "parent"
        assert np.array_equal(buffer.dynamic_transforms["child"][0][2], np.eye(4))
        mock_transform_to_matrix.assert_called_once_with(transform_stamped.transform)

    @patch("hsr_data_converter.utils.tf_buffer.transform_to_matrix")
    def test_error_transform_to_matrix(self, mock_transform_to_matrix):
        mock_transform_to_matrix.side_effect = Exception("error")

        buffer = TFBuffer()
        msg = MagicMock(spec=tf2_msgs__msg__TFMessage)
        transform_stamped = MagicMock()
        transform_stamped.child_frame_id = "/child"
        transform_stamped.header.frame_id = "/parent"
        transform_stamped.transform = MagicMock(spec=geometry_msgs__msg__Transform)
        msg.transforms = [transform_stamped]

        with pytest.raises(Exception, match="error"):
            buffer.add_transform_msg(msg, is_static=False)

        assert not buffer.dynamic_transforms.get("child")
        assert buffer.static_transforms == {}
        mock_transform_to_matrix.assert_called_once_with(transform_stamped.transform)


class TestTFBufferPrivateFindTransform(unittest.TestCase):
    def setUp(self):
        self.buffer = TFBuffer()

    def test_standard_find_static_transform(self):
        self.buffer.static_transforms["child"] = ("parent", np.eye(4))
        parent, matrix = self.buffer._find_transform("child", 0)
        assert parent == "parent"
        assert np.array_equal(matrix, np.eye(4))

    def test_standard_find_dynamic_transform(self):
        timestamp = 1_000_000_000
        self.buffer.dynamic_transforms["child"] = [(timestamp, "parent", np.eye(4))]
        parent, matrix = self.buffer._find_transform("child", timestamp)
        assert parent == "parent"
        assert np.array_equal(matrix, np.eye(4))

    def test_no_matching_transform(self):
        parent, matrix = self.buffer._find_transform("unknown_child", 0)
        assert parent is None
        assert matrix is None

    def test_find_dynamic_transform_before_timestamp(self):
        self.buffer.dynamic_transforms["child"] = [
            (1_000_000_000, "parent1", np.identity(4)),
            (2_000_000_000, "parent2", 2 * np.identity(4)),
        ]
        parent, matrix = self.buffer._find_transform("child", 1_500_000_000)
        assert parent == "parent1"
        assert np.array_equal(matrix, np.identity(4))

    def test_find_static_transform_only(self):
        self.buffer.static_transforms["child"] = ("parent", np.identity(4))
        self.buffer.dynamic_transforms["child"] = [
            (1_000_000_000, "parent1", np.identity(4))
        ]
        parent, matrix = self.buffer._find_transform("child", 0)
        assert parent == "parent"
        assert np.array_equal(matrix, np.identity(4))


class TestTFBufferLookupTransform(unittest.TestCase):
    def setUp(self):
        self.buffer = TFBuffer()
        self.identity_matrix = np.eye(4)
        self.transformation_matrix = np.array(
            [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
        )

    @patch.object(TFBuffer, "_find_transform")
    def test_standard_lookup_same_frame(self, mock_find_transform):
        mock_find_transform.return_value = (None, self.identity_matrix)
        result = self.buffer.lookup_transform("root", "root", 0)
        assert np.array_equal(result, self.identity_matrix)

    @patch.object(TFBuffer, "_find_transform")
    def test_standard_lookup_transform(self, mock_find_transform):
        mock_find_transform.side_effect = [
            ("intermediate", self.transformation_matrix),
            ("root", self.identity_matrix),
        ]
        result = self.buffer.lookup_transform("root", "leaf", 0)
        expected_transform = np.linalg.inv(self.transformation_matrix)
        assert np.allclose(result, expected_transform)

    @patch.object(TFBuffer, "_find_transform")
    def test_error_no_transform_found(self, mock_find_transform):
        mock_find_transform.return_value = (None, None)
        result = self.buffer.lookup_transform("root", "unknown", 0)
        assert result is None

    @patch("builtins.print")
    @patch.object(TFBuffer, "_find_transform")
    def test_error_transform_chain_broken(self, mock_find_transform, mock_print):
        mock_find_transform.side_effect = [
            ("intermediate", self.transformation_matrix),
            (None, None),
        ]
        result = self.buffer.lookup_transform("root", "leaf", 0)
        assert result is None
        mock_print.assert_called_once_with(
            "[WARN] Could not find transform for 'intermediate' at timestamp 0"
        )

    @patch.object(TFBuffer, "_find_transform")
    def test_no_intermediate_transform(self, mock_find_transform):
        mock_find_transform.side_effect = [("root", self.transformation_matrix)]
        result = self.buffer.lookup_transform("root", "leaf", 0)
        expected_transform = np.linalg.inv(self.transformation_matrix)
        assert np.allclose(result, expected_transform)

    @patch.object(TFBuffer, "_find_transform")
    def test_error_parent_frame_none(self, mock_find_transform):
        mock_find_transform.return_value = (None, self.transformation_matrix)
        result = self.buffer.lookup_transform("root", "leaf", 0)
        assert result is None
