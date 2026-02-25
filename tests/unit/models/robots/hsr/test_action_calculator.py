import pytest
import torch
from unittest.mock import MagicMock
from hsr_data_converter.models.robots.hsr.action_calculator import (
    calc_initial_subaction,
    calc_delta_action,
    calc_absolute_action,
    enforce_joint_limits,
)
from hsr_data_converter.models.robots.hsr.config import HSR_JOINT_LIMITS


class TestCalcInitialSubaction:
    def test_standard_initial_subaction(self, monkeypatch):
        monkeypatch.setattr(
            "hsr_data_converter.models.robots.hsr.config.GRIPPER_CLOSE_ACTION",
            0.0,
        )
        monkeypatch.setattr(
            "hsr_data_converter.models.robots.hsr.config.GRIPPER_OPEN_ACTION",
            1.0,
        )

        frame = {
            "observation.state": torch.tensor(
                [0.5, -0.5, 0.5, -0.5, 0.5, -0.1, 0.1, -0.1]
            ),
            "action.arm": None,
            "action.gripper": None,
            "action.head": None,
            "action.base": None,
        }

        result = calc_initial_subaction(frame)

        expected_arm_action = torch.tensor([0.5, -0.5, 0.5, -0.5, 0.5])
        expected_gripper_action = torch.tensor([0.0])
        expected_head_action = torch.tensor([0.1, -0.1])
        expected_base_action = torch.tensor([0.0, 0.0, 0.0])

        assert torch.equal(result["action.arm"], expected_arm_action)
        assert torch.all(result["action.arm.is_fresh"] == torch.full([5], False))
        assert torch.equal(result["action.gripper"], expected_gripper_action)
        assert torch.all(result["action.gripper.is_fresh"] == torch.full([1], False))
        assert torch.equal(result["action.head"], expected_head_action)
        assert torch.all(result["action.head.is_fresh"] == torch.full([2], False))
        assert torch.equal(result["action.base"], expected_base_action)
        assert torch.all(result["action.base.is_fresh"] == torch.full([3], False))

    def test_error_action_already_set(self, monkeypatch):
        monkeypatch.setattr(
            "hsr_data_converter.models.robots.hsr.config.GRIPPER_CLOSE_ACTION",
            0.0,
        )
        monkeypatch.setattr(
            "hsr_data_converter.models.robots.hsr.config.GRIPPER_OPEN_ACTION",
            1.0,
        )

        initial_subaction = calc_initial_subaction(
            {
                "observation.state": torch.tensor(
                    [0.5, -0.5, 0.5, -0.5, 0.5, 0.1, 0.1, -0.1]
                ),
                "action.arm": MagicMock(),
                "action.gripper": MagicMock(),
                "action.head": MagicMock(),
                "action.base": MagicMock(),
            }
        )

        assert "action.arm" not in initial_subaction
        assert "action.gripper" not in initial_subaction
        assert "action.head" not in initial_subaction
        assert "action.base" not in initial_subaction


class TestCalcDeltaAction:
    def test_standard_delta_action(self):
        frame = {
            "observation.state": torch.tensor(
                [1.0, 2.0, 3.0, 4.0, 5.0, 0.5, -0.5, 0.5], dtype=torch.float32
            ),
            "action.arm": torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5], dtype=torch.float32),
            "action.arm.is_fresh": torch.tensor([True, True, True, False, False]),
            "action.gripper": torch.tensor([1.0], dtype=torch.float32),
            "action.gripper.is_fresh": torch.tensor([True]),
            "action.head": torch.tensor([0.3, -0.3], dtype=torch.float32),
            "action.head.is_fresh": torch.tensor([True, False]),
            "action.base": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            "action.base.is_fresh": torch.tensor([True, True, True]),
            "observation.state.is_fresh": torch.tensor(
                [True, True, True, True, True, True, True, True], dtype=torch.bool
            ),
        }

        delta_action, delta_is_fresh = calc_delta_action(frame)

        expected_delta_action = torch.tensor(
            [0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 0.8, -0.8, 0.1, 0.2, 0.3],
            dtype=torch.float32,
        )
        expected_delta_is_fresh = torch.tensor(
            [True, True, True, False, False, True, True, False, True, True, True]
        )

        assert torch.equal(delta_action, expected_delta_action)
        assert torch.equal(delta_is_fresh, expected_delta_is_fresh)

    def test_error_inconsistent_arm_length(self):
        frame = {
            "observation.state": torch.tensor(
                [1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32
            ),
            "action.arm": torch.tensor([1.5, 2.5, 3.5, 4.5]),
            "action.arm.is_fresh": torch.tensor([True, True, True, False]),
            "action.gripper": torch.tensor([1.0], dtype=torch.float32),
            "action.gripper.is_fresh": torch.tensor([True]),
            "action.head": torch.tensor([0.3, -0.3], dtype=torch.float32),
            "action.head.is_fresh": torch.tensor([True, False]),
            "action.base": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            "action.base.is_fresh": torch.tensor([True, True, True]),
            "observation.state.is_fresh": torch.tensor(
                [True, True, True, True, True], dtype=torch.bool
            ),
        }

        with pytest.raises(RuntimeError):
            calc_delta_action(frame)


class TestCalcAbsoluteAction:
    def test_standard_absolute_action(self):
        frame = {
            "action.arm": torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5], dtype=torch.float32),
            "action.arm.is_fresh": torch.tensor([True, True, True, False, False]),
            "action.gripper": torch.tensor([1.0], dtype=torch.float32),
            "action.gripper.is_fresh": torch.tensor([True]),
            "action.head": torch.tensor([0.3, -0.3], dtype=torch.float32),
            "action.head.is_fresh": torch.tensor([True, False]),
        }

        abs_action, abs_is_fresh = calc_absolute_action(frame)

        expected_abs_action = torch.tensor(
            [1.5, 2.5, 3.5, 4.5, 5.5, 1.0, 0.3, -0.3], dtype=torch.float32
        )
        expected_abs_is_fresh = torch.tensor(
            [True, True, True, False, False, True, True, False]
        )

        assert torch.equal(abs_action, expected_abs_action)
        assert torch.equal(abs_is_fresh, expected_abs_is_fresh)

    def test_error_none_input(self):
        frame = {
            "action.arm": None,
            "action.arm.is_fresh": torch.tensor([True, True, True, False, False]),
            "action.gripper": None,
            "action.gripper.is_fresh": torch.tensor([True]),
            "action.head": None,
            "action.head.is_fresh": torch.tensor([True, False]),
        }

        with pytest.raises(TypeError):
            calc_absolute_action(frame)


class TestEnforceJointLimits:
    def test_clamp_above_max(self):
        """Test clamping values above maximum limits"""
        joint_names = (
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "hand_motor_joint",
            "head_pan_joint",
            "head_tilt_joint",
        )

        # Create values above max for each joint by adding offset
        abs_action = torch.tensor(
            [
                HSR_JOINT_LIMITS["arm_lift_joint"][1] + 0.5,
                HSR_JOINT_LIMITS["arm_flex_joint"][1] + 0.5,
                HSR_JOINT_LIMITS["arm_roll_joint"][1] + 0.5,
                HSR_JOINT_LIMITS["wrist_flex_joint"][1] + 0.5,
                HSR_JOINT_LIMITS["wrist_roll_joint"][1] + 0.5,
                HSR_JOINT_LIMITS["hand_motor_joint"][1] + 0.5,
                HSR_JOINT_LIMITS["head_pan_joint"][1] + 0.5,
                HSR_JOINT_LIMITS["head_tilt_joint"][1] + 0.5,
            ],
            dtype=torch.float32,
        )

        clamped = enforce_joint_limits(abs_action, joint_names, HSR_JOINT_LIMITS)

        # Expected: all values clamped to their max limits
        expected = torch.tensor(
            [
                HSR_JOINT_LIMITS["arm_lift_joint"][1],
                HSR_JOINT_LIMITS["arm_flex_joint"][1],
                HSR_JOINT_LIMITS["arm_roll_joint"][1],
                HSR_JOINT_LIMITS["wrist_flex_joint"][1],
                HSR_JOINT_LIMITS["wrist_roll_joint"][1],
                HSR_JOINT_LIMITS["hand_motor_joint"][1],
                HSR_JOINT_LIMITS["head_pan_joint"][1],
                HSR_JOINT_LIMITS["head_tilt_joint"][1],
            ],
            dtype=torch.float32,
        )

        assert torch.allclose(clamped, expected, atol=1e-6)

    def test_clamp_below_min(self):
        """Test clamping values below minimum limits"""
        joint_names = (
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "hand_motor_joint",
            "head_pan_joint",
            "head_tilt_joint",
        )

        # Create values below min for each joint by subtracting offset
        abs_action = torch.tensor(
            [
                HSR_JOINT_LIMITS["arm_lift_joint"][0] - 0.5,
                HSR_JOINT_LIMITS["arm_flex_joint"][0] - 0.5,
                HSR_JOINT_LIMITS["arm_roll_joint"][0] - 0.5,
                HSR_JOINT_LIMITS["wrist_flex_joint"][0] - 0.5,
                HSR_JOINT_LIMITS["wrist_roll_joint"][0] - 0.5,
                HSR_JOINT_LIMITS["hand_motor_joint"][0] - 0.5,
                HSR_JOINT_LIMITS["head_pan_joint"][0] - 0.5,
                HSR_JOINT_LIMITS["head_tilt_joint"][0] - 0.5,
            ],
            dtype=torch.float32,
        )

        clamped = enforce_joint_limits(abs_action, joint_names, HSR_JOINT_LIMITS)

        # Expected: all values clamped to their min limits
        expected = torch.tensor(
            [
                HSR_JOINT_LIMITS["arm_lift_joint"][0],
                HSR_JOINT_LIMITS["arm_flex_joint"][0],
                HSR_JOINT_LIMITS["arm_roll_joint"][0],
                HSR_JOINT_LIMITS["wrist_flex_joint"][0],
                HSR_JOINT_LIMITS["wrist_roll_joint"][0],
                HSR_JOINT_LIMITS["hand_motor_joint"][0],
                HSR_JOINT_LIMITS["head_pan_joint"][0],
                HSR_JOINT_LIMITS["head_tilt_joint"][0],
            ],
            dtype=torch.float32,
        )

        assert torch.allclose(clamped, expected, atol=1e-6)

    def test_within_limits(self):
        """Test that values within limits remain unchanged"""
        joint_names = (
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "hand_motor_joint",
            "head_pan_joint",
            "head_tilt_joint",
        )

        # Create values within valid ranges (midpoint between min and max)
        abs_action = torch.tensor(
            [
                (
                    HSR_JOINT_LIMITS["arm_lift_joint"][0]
                    + HSR_JOINT_LIMITS["arm_lift_joint"][1]
                )
                / 2,
                (
                    HSR_JOINT_LIMITS["arm_flex_joint"][0]
                    + HSR_JOINT_LIMITS["arm_flex_joint"][1]
                )
                / 2,
                (
                    HSR_JOINT_LIMITS["arm_roll_joint"][0]
                    + HSR_JOINT_LIMITS["arm_roll_joint"][1]
                )
                / 2,
                (
                    HSR_JOINT_LIMITS["wrist_flex_joint"][0]
                    + HSR_JOINT_LIMITS["wrist_flex_joint"][1]
                )
                / 2,
                (
                    HSR_JOINT_LIMITS["wrist_roll_joint"][0]
                    + HSR_JOINT_LIMITS["wrist_roll_joint"][1]
                )
                / 2,
                (
                    HSR_JOINT_LIMITS["hand_motor_joint"][0]
                    + HSR_JOINT_LIMITS["hand_motor_joint"][1]
                )
                / 2,
                (
                    HSR_JOINT_LIMITS["head_pan_joint"][0]
                    + HSR_JOINT_LIMITS["head_pan_joint"][1]
                )
                / 2,
                (
                    HSR_JOINT_LIMITS["head_tilt_joint"][0]
                    + HSR_JOINT_LIMITS["head_tilt_joint"][1]
                )
                / 2,
            ],
            dtype=torch.float32,
        )

        clamped = enforce_joint_limits(abs_action, joint_names, HSR_JOINT_LIMITS)

        # Expected: no changes
        assert torch.equal(clamped, abs_action)

    def test_mixed_clamping(self):
        """Test mixed scenario with some values needing clamping and others not"""
        joint_names = (
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "hand_motor_joint",
            "head_pan_joint",
            "head_tilt_joint",
        )

        # Mix of values: some valid (midpoint), some too high (max+0.5), some too low (min-0.5)
        valid_1 = (
            HSR_JOINT_LIMITS["arm_lift_joint"][0]
            + HSR_JOINT_LIMITS["arm_lift_joint"][1]
        ) / 2
        valid_2 = (
            HSR_JOINT_LIMITS["wrist_flex_joint"][0]
            + HSR_JOINT_LIMITS["wrist_flex_joint"][1]
        ) / 2
        valid_3 = (
            HSR_JOINT_LIMITS["hand_motor_joint"][0]
            + HSR_JOINT_LIMITS["hand_motor_joint"][1]
        ) / 2

        abs_action = torch.tensor(
            [
                valid_1,  # Within limits
                HSR_JOINT_LIMITS["arm_flex_joint"][0] - 0.5,  # Below min
                HSR_JOINT_LIMITS["arm_roll_joint"][1] + 0.5,  # Above max
                valid_2,  # Within limits
                HSR_JOINT_LIMITS["wrist_roll_joint"][0] - 0.5,  # Below min
                valid_3,  # Within limits
                HSR_JOINT_LIMITS["head_pan_joint"][1] + 0.5,  # Above max
                HSR_JOINT_LIMITS["head_tilt_joint"][0] - 0.5,  # Below min
            ],
            dtype=torch.float32,
        )

        clamped = enforce_joint_limits(abs_action, joint_names, HSR_JOINT_LIMITS)

        # Expected: only out-of-range values clamped
        expected = torch.tensor(
            [
                valid_1,  # Unchanged
                HSR_JOINT_LIMITS["arm_flex_joint"][0],  # Clamped to min
                HSR_JOINT_LIMITS["arm_roll_joint"][1],  # Clamped to max
                valid_2,  # Unchanged
                HSR_JOINT_LIMITS["wrist_roll_joint"][0],  # Clamped to min
                valid_3,  # Unchanged
                HSR_JOINT_LIMITS["head_pan_joint"][1],  # Clamped to max
                HSR_JOINT_LIMITS["head_tilt_joint"][0],  # Clamped to min
            ],
            dtype=torch.float32,
        )

        assert torch.allclose(clamped, expected, atol=1e-6)


class TestCalcAbsoluteActionWithLimits:
    def test_enforce_limits_true(self):
        """Test that enforce_limits=True clamps values outside valid ranges"""
        frame = {
            "action.arm": torch.tensor(
                [
                    HSR_JOINT_LIMITS["arm_lift_joint"][1] + 0.5,
                    HSR_JOINT_LIMITS["arm_flex_joint"][1] + 0.5,
                    HSR_JOINT_LIMITS["arm_roll_joint"][1] + 0.5,
                    HSR_JOINT_LIMITS["wrist_flex_joint"][1] + 0.5,
                    HSR_JOINT_LIMITS["wrist_roll_joint"][1] + 0.5,
                ],
                dtype=torch.float32,
            ),  # All above max
            "action.arm.is_fresh": torch.tensor([True, True, True, True, True]),
            "action.gripper": torch.tensor(
                [HSR_JOINT_LIMITS["hand_motor_joint"][1] + 0.5], dtype=torch.float32
            ),  # Above max
            "action.gripper.is_fresh": torch.tensor([True]),
            "action.head": torch.tensor(
                [
                    HSR_JOINT_LIMITS["head_pan_joint"][1] + 0.5,
                    HSR_JOINT_LIMITS["head_tilt_joint"][1] + 0.5,
                ],
                dtype=torch.float32,
            ),  # Above max
            "action.head.is_fresh": torch.tensor([True, True]),
        }

        abs_action, abs_is_fresh = calc_absolute_action(frame, enforce_limits=True)

        # Expected: all values clamped to max limits
        expected_action = torch.tensor(
            [
                HSR_JOINT_LIMITS["arm_lift_joint"][1],
                HSR_JOINT_LIMITS["arm_flex_joint"][1],
                HSR_JOINT_LIMITS["arm_roll_joint"][1],
                HSR_JOINT_LIMITS["wrist_flex_joint"][1],
                HSR_JOINT_LIMITS["wrist_roll_joint"][1],
                HSR_JOINT_LIMITS["hand_motor_joint"][1],
                HSR_JOINT_LIMITS["head_pan_joint"][1],
                HSR_JOINT_LIMITS["head_tilt_joint"][1],
            ],
            dtype=torch.float32,
        )

        assert torch.allclose(abs_action, expected_action, atol=1e-6)
        assert torch.all(
            abs_is_fresh
            == torch.tensor([True, True, True, True, True, True, True, True])
        )

    def test_enforce_limits_false(self):
        """Test that enforce_limits=False does not clamp values"""
        # Use values above max limits
        arm_vals = [
            HSR_JOINT_LIMITS["arm_lift_joint"][1] + 0.5,
            HSR_JOINT_LIMITS["arm_flex_joint"][1] + 0.5,
            HSR_JOINT_LIMITS["arm_roll_joint"][1] + 0.5,
            HSR_JOINT_LIMITS["wrist_flex_joint"][1] + 0.5,
            HSR_JOINT_LIMITS["wrist_roll_joint"][1] + 0.5,
        ]
        gripper_val = HSR_JOINT_LIMITS["hand_motor_joint"][1] + 0.5
        head_vals = [
            HSR_JOINT_LIMITS["head_pan_joint"][1] + 0.5,
            HSR_JOINT_LIMITS["head_tilt_joint"][1] + 0.5,
        ]

        frame = {
            "action.arm": torch.tensor(arm_vals, dtype=torch.float32),
            "action.arm.is_fresh": torch.tensor([True, True, True, True, True]),
            "action.gripper": torch.tensor([gripper_val], dtype=torch.float32),
            "action.gripper.is_fresh": torch.tensor([True]),
            "action.head": torch.tensor(head_vals, dtype=torch.float32),
            "action.head.is_fresh": torch.tensor([True, True]),
        }

        abs_action, abs_is_fresh = calc_absolute_action(frame, enforce_limits=False)

        # Expected: no clamping, values remain as-is
        expected_action = torch.tensor(
            arm_vals + [gripper_val] + head_vals, dtype=torch.float32
        )

        assert torch.equal(abs_action, expected_action)
        assert torch.all(
            abs_is_fresh
            == torch.tensor([True, True, True, True, True, True, True, True])
        )

    def test_is_fresh_preserved(self):
        """Test that is_fresh flags are preserved when clamping"""
        frame = {
            "action.arm": torch.tensor(
                [
                    HSR_JOINT_LIMITS["arm_lift_joint"][1] + 0.5,
                    HSR_JOINT_LIMITS["arm_flex_joint"][1] + 0.5,
                    HSR_JOINT_LIMITS["arm_roll_joint"][1] + 0.5,
                    HSR_JOINT_LIMITS["wrist_flex_joint"][1] + 0.5,
                    HSR_JOINT_LIMITS["wrist_roll_joint"][1] + 0.5,
                ],
                dtype=torch.float32,
            ),
            "action.arm.is_fresh": torch.tensor([True, False, True, False, True]),
            "action.gripper": torch.tensor(
                [HSR_JOINT_LIMITS["hand_motor_joint"][1] + 0.5], dtype=torch.float32
            ),
            "action.gripper.is_fresh": torch.tensor([False]),
            "action.head": torch.tensor(
                [
                    HSR_JOINT_LIMITS["head_pan_joint"][1] + 0.5,
                    HSR_JOINT_LIMITS["head_tilt_joint"][1] + 0.5,
                ],
                dtype=torch.float32,
            ),
            "action.head.is_fresh": torch.tensor([True, False]),
        }

        abs_action, abs_is_fresh = calc_absolute_action(frame, enforce_limits=True)

        # Expected is_fresh flags unchanged
        expected_is_fresh = torch.tensor(
            [True, False, True, False, True, False, True, False]
        )

        assert torch.equal(abs_is_fresh, expected_is_fresh)


class TestCalcDeltaActionWithLimits:
    def test_delta_with_enforce_limits(self):
        """Test that delta action is calculated from clamped absolute action"""
        # Use midpoint values for observation state (within limits)
        obs_state = torch.tensor(
            [
                (
                    HSR_JOINT_LIMITS["arm_lift_joint"][0]
                    + HSR_JOINT_LIMITS["arm_lift_joint"][1]
                )
                / 2,
                (
                    HSR_JOINT_LIMITS["arm_flex_joint"][0]
                    + HSR_JOINT_LIMITS["arm_flex_joint"][1]
                )
                / 2,
                (
                    HSR_JOINT_LIMITS["arm_roll_joint"][0]
                    + HSR_JOINT_LIMITS["arm_roll_joint"][1]
                )
                / 2,
                (
                    HSR_JOINT_LIMITS["wrist_flex_joint"][0]
                    + HSR_JOINT_LIMITS["wrist_flex_joint"][1]
                )
                / 2,
                (
                    HSR_JOINT_LIMITS["wrist_roll_joint"][0]
                    + HSR_JOINT_LIMITS["wrist_roll_joint"][1]
                )
                / 2,
                (
                    HSR_JOINT_LIMITS["hand_motor_joint"][0]
                    + HSR_JOINT_LIMITS["hand_motor_joint"][1]
                )
                / 2,
                (
                    HSR_JOINT_LIMITS["head_pan_joint"][0]
                    + HSR_JOINT_LIMITS["head_pan_joint"][1]
                )
                / 2,
                (
                    HSR_JOINT_LIMITS["head_tilt_joint"][0]
                    + HSR_JOINT_LIMITS["head_tilt_joint"][1]
                )
                / 2,
            ],
            dtype=torch.float32,
        )

        frame = {
            "observation.state": obs_state,
            "action.arm": torch.tensor(
                [
                    HSR_JOINT_LIMITS["arm_lift_joint"][1] + 0.5,
                    HSR_JOINT_LIMITS["arm_flex_joint"][1] + 0.5,
                    HSR_JOINT_LIMITS["arm_roll_joint"][1] + 0.5,
                    HSR_JOINT_LIMITS["wrist_flex_joint"][1] + 0.5,
                    HSR_JOINT_LIMITS["wrist_roll_joint"][1] + 0.5,
                ],
                dtype=torch.float32,
            ),  # Will be clamped to max
            "action.arm.is_fresh": torch.tensor([True, True, True, True, True]),
            "action.gripper": torch.tensor(
                [HSR_JOINT_LIMITS["hand_motor_joint"][1] + 0.5], dtype=torch.float32
            ),  # Will be clamped to max
            "action.gripper.is_fresh": torch.tensor([True]),
            "action.head": torch.tensor(
                [
                    HSR_JOINT_LIMITS["head_pan_joint"][1] + 0.5,
                    HSR_JOINT_LIMITS["head_tilt_joint"][1] + 0.5,
                ],
                dtype=torch.float32,
            ),  # Will be clamped to max
            "action.head.is_fresh": torch.tensor([True, True]),
            "action.base": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            "action.base.is_fresh": torch.tensor([True, True, True]),
            "observation.state.is_fresh": torch.tensor(
                [True, True, True, True, True, True, True, True], dtype=torch.bool
            ),
        }

        delta_action, delta_is_fresh = calc_delta_action(frame, enforce_limits=True)

        # Calculate expected delta: clamped_max - obs_state, except gripper stays absolute
        expected_delta = torch.tensor(
            [
                HSR_JOINT_LIMITS["arm_lift_joint"][1] - obs_state[0].item(),
                HSR_JOINT_LIMITS["arm_flex_joint"][1] - obs_state[1].item(),
                HSR_JOINT_LIMITS["arm_roll_joint"][1] - obs_state[2].item(),
                HSR_JOINT_LIMITS["wrist_flex_joint"][1] - obs_state[3].item(),
                HSR_JOINT_LIMITS["wrist_roll_joint"][1] - obs_state[4].item(),
                HSR_JOINT_LIMITS["hand_motor_joint"][1],  # Gripper stays absolute
                HSR_JOINT_LIMITS["head_pan_joint"][1] - obs_state[6].item(),
                HSR_JOINT_LIMITS["head_tilt_joint"][1] - obs_state[7].item(),
                0.1,  # base actions
                0.2,
                0.3,
            ],
            dtype=torch.float32,
        )

        assert torch.allclose(delta_action, expected_delta, atol=1e-5)

    def test_delta_without_enforce_limits(self):
        """Test delta action calculation without enforcing limits"""
        # Use midpoint values for observation state
        obs_state = torch.tensor(
            [
                (
                    HSR_JOINT_LIMITS["arm_lift_joint"][0]
                    + HSR_JOINT_LIMITS["arm_lift_joint"][1]
                )
                / 2,
                (
                    HSR_JOINT_LIMITS["arm_flex_joint"][0]
                    + HSR_JOINT_LIMITS["arm_flex_joint"][1]
                )
                / 2,
                (
                    HSR_JOINT_LIMITS["arm_roll_joint"][0]
                    + HSR_JOINT_LIMITS["arm_roll_joint"][1]
                )
                / 2,
                (
                    HSR_JOINT_LIMITS["wrist_flex_joint"][0]
                    + HSR_JOINT_LIMITS["wrist_flex_joint"][1]
                )
                / 2,
                (
                    HSR_JOINT_LIMITS["wrist_roll_joint"][0]
                    + HSR_JOINT_LIMITS["wrist_roll_joint"][1]
                )
                / 2,
                (
                    HSR_JOINT_LIMITS["hand_motor_joint"][0]
                    + HSR_JOINT_LIMITS["hand_motor_joint"][1]
                )
                / 2,
                (
                    HSR_JOINT_LIMITS["head_pan_joint"][0]
                    + HSR_JOINT_LIMITS["head_pan_joint"][1]
                )
                / 2,
                (
                    HSR_JOINT_LIMITS["head_tilt_joint"][0]
                    + HSR_JOINT_LIMITS["head_tilt_joint"][1]
                )
                / 2,
            ],
            dtype=torch.float32,
        )

        # Use values above max (will NOT be clamped in this test)
        arm_actions = [
            HSR_JOINT_LIMITS["arm_lift_joint"][1] + 0.5,
            HSR_JOINT_LIMITS["arm_flex_joint"][1] + 0.5,
            HSR_JOINT_LIMITS["arm_roll_joint"][1] + 0.5,
            HSR_JOINT_LIMITS["wrist_flex_joint"][1] + 0.5,
            HSR_JOINT_LIMITS["wrist_roll_joint"][1] + 0.5,
        ]
        gripper_action = HSR_JOINT_LIMITS["hand_motor_joint"][1] + 0.5
        head_actions = [
            HSR_JOINT_LIMITS["head_pan_joint"][1] + 0.5,
            HSR_JOINT_LIMITS["head_tilt_joint"][1] + 0.5,
        ]

        frame = {
            "observation.state": obs_state,
            "action.arm": torch.tensor(arm_actions, dtype=torch.float32),
            "action.arm.is_fresh": torch.tensor([True, True, True, True, True]),
            "action.gripper": torch.tensor([gripper_action], dtype=torch.float32),
            "action.gripper.is_fresh": torch.tensor([True]),
            "action.head": torch.tensor(head_actions, dtype=torch.float32),
            "action.head.is_fresh": torch.tensor([True, True]),
            "action.base": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            "action.base.is_fresh": torch.tensor([True, True, True]),
            "observation.state.is_fresh": torch.tensor(
                [True, True, True, True, True, True, True, True], dtype=torch.bool
            ),
        }

        delta_action, delta_is_fresh = calc_delta_action(frame, enforce_limits=False)

        # Without clamping, use raw action values - obs_state
        # Gripper stays absolute (not delta)
        expected_delta = torch.tensor(
            [
                arm_actions[0] - obs_state[0].item(),
                arm_actions[1] - obs_state[1].item(),
                arm_actions[2] - obs_state[2].item(),
                arm_actions[3] - obs_state[3].item(),
                arm_actions[4] - obs_state[4].item(),
                gripper_action,  # Gripper stays absolute
                head_actions[0] - obs_state[6].item(),
                head_actions[1] - obs_state[7].item(),
                0.1,  # base actions
                0.2,
                0.3,
            ],
            dtype=torch.float32,
        )

        assert torch.allclose(delta_action, expected_delta, atol=1e-5)
