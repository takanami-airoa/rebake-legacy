"""
Episode loading functions for rosbag to lerobot conversion
"""

from typing import Any, Dict, List, Tuple, cast

import numpy as np
import torch
from moma_metadata import Metadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rosbags.highlevel import AnyReader
from rosbags.typesys.store import Typestore
from rosbags.typesys.stores.ros2_dashing import tf2_msgs__msg__TFMessage

from hsr_data_converter.convert_config import ConvertConfig
from hsr_data_converter.hsr_utils import hsr_features
from hsr_data_converter.recorded_topics import RecordedTopics
from hsr_data_converter.tf_buffer import TFBuffer, matrix_to_xyz_rpy

from .action_calculator import (
    calc_absolute_action,
    calc_delta_action,
    calc_initial_subaction,
)


def get_updated_features_from_topics(
    received_topics: RecordedTopics, last_frame_timestamp: float
) -> Dict[str, Any]:
    """
    Get updated features from the received topics

    Parameters
    ----------
    received_topics : RecordedTopics
        received topics from rosbag
    last_frame_timestamp : float
        last frame timestamp in nanoseconds

    Returns
    -------
    Dict[str, Any]
        Updated features extracted from the received topics
    """
    updated_features: dict[str, Any] = {}

    # update observation features
    updated_features["observation.image.hand"] = received_topics.hand_rgb[1]
    updated_features["observation.image.head"] = received_topics.head_rgb[1]
    updated_features["observation.image.hand.is_fresh"] = torch.full(
        [3, 1, 1],
        received_topics.hand_rgb[0] > last_frame_timestamp,
        dtype=torch.bool,
    )
    updated_features["observation.image.head.is_fresh"] = torch.full(
        [3, 1, 1],
        received_topics.head_rgb[0] > last_frame_timestamp,
        dtype=torch.bool,
    )

    # update state features
    joint_positions_time, joint_positions = received_topics.joint_positions
    joint_names = received_topics.joint_names
    state_indices = [
        joint_names.index(name) for name in hsr_features["observation.state"]["names"]
    ]
    state = joint_positions[state_indices]
    updated_features["observation.state"] = np.array(state, dtype=np.float32).reshape(8)
    updated_features["observation.state.is_fresh"] = torch.full(
        [8], joint_positions_time > last_frame_timestamp, dtype=torch.bool
    )
    updated_features["observation.wrist.wrench"] = torch.tensor(
        received_topics.wrist_wrench, dtype=torch.float32
    )

    # update available action features
    if received_topics.abs_arm_action[1] is not None:
        action_indices = [
            hsr_features["action.arm"]["names"].index(name)
            for name in received_topics.abs_arm_action_names
        ]
        abs_arm_action_time, abs_arm_action = received_topics.abs_arm_action
        if abs_arm_action is not None:
            updated_features["action.arm"] = torch.tensor(
                abs_arm_action[action_indices], dtype=torch.float32
            ).flatten()
        else:
            updated_features["action.arm"] = torch.zeros(
                [len(action_indices)], dtype=torch.float32
            )
        updated_features["action.arm.is_fresh"] = torch.full(
            [len(action_indices)],
            abs_arm_action_time is not None
            and abs_arm_action_time > last_frame_timestamp,
            dtype=torch.bool,
        )

    if received_topics.abs_gripper_action[1] is not None:
        if received_topics.abs_gripper_name is None:
            raise ValueError("abs_gripper_name is None.")

        action_indices = [
            hsr_features["action.gripper"]["names"].index(name)
            for name in received_topics.abs_gripper_name
        ]
        abs_head_action_time, abs_gripper_action = received_topics.abs_gripper_action

        if abs_gripper_action is not None:
            updated_features["action.gripper"] = torch.tensor(
                abs_gripper_action[action_indices], dtype=torch.float32
            ).flatten()
        else:
            updated_features["action.gripper"] = torch.zeros(
                [len(action_indices)], dtype=torch.float32
            )
        updated_features["action.gripper.is_fresh"] = torch.full(
            [len(action_indices)],
            abs_head_action_time is not None
            and abs_head_action_time > last_frame_timestamp,
            dtype=torch.bool,
        )

    if received_topics.abs_head_action[1] is not None:
        action_indices = [
            hsr_features["action.head"]["names"].index(name)
            for name in received_topics.abs_head_action_names
        ]
        abs_head_action_time, abs_head_action = received_topics.abs_head_action

        if abs_head_action is not None:
            updated_features["action.head"] = torch.tensor(
                abs_head_action[action_indices], dtype=torch.float32
            ).flatten()
        else:
            updated_features["action.head"] = torch.zeros(
                [len(action_indices)], dtype=torch.float32
            )
        updated_features["action.head.is_fresh"] = torch.full(
            [len(action_indices)],
            abs_head_action_time is not None
            and abs_head_action_time > last_frame_timestamp,
            dtype=torch.bool,
        )

    if received_topics.delta_base_action[1] is not None:
        action_indices = [
            hsr_features["action.base"]["names"].index(name)
            for name in received_topics.delta_base_action_names
        ]
        abs_base_action_time, delta_base_action = received_topics.delta_base_action
        if delta_base_action is not None:
            updated_features["action.base"] = torch.tensor(
                delta_base_action[action_indices], dtype=torch.float32
            ).flatten()
        else:
            updated_features["action.base"] = torch.zeros(
                [len(action_indices)], dtype=torch.float32
            )
        updated_features["action.base.is_fresh"] = torch.full(
            [len(action_indices)],
            abs_base_action_time is not None
            and abs_base_action_time > last_frame_timestamp,
            dtype=torch.bool,
        )

    return updated_features


def check_skip_condition(received_topics: RecordedTopics) -> bool:
    """
    Check if the frame should be skipped

    Parameters
    ----------
    received_topics : RecordedTopics
        received topics from rosbag

    Returns
    -------
    bool
        True if the frame should be skipped
    """
    # skip if joint is not moving
    joints_moving = np.any(
        np.abs(received_topics.joint_velocities[1]) > 1e-6
    )  # TODO: implement a more sophisticated check.
    if not joints_moving:
        return True

    return False


def load_hsr_episodes(
    cfg: ConvertConfig,
    metadata: Metadata,
    meta_dir,
    typestore: Typestore,
    dataset: LeRobotDataset,
) -> Tuple[List[List[Dict[str, Any]]], List[List[str]], List[List[float]], int]:
    """
    Load an episode from a rosbag file and convert it to a list of frames

    Parameters
    ----------
    cfg : ConvertConfig
        configuration for the conversion
    metadata : Metadata
        meta data (= meta.json) for the rosbag file
    meta_dir : Path
        path to the meta.json dir
    typestore : Typestore
        typestore for the rosbag file

    Returns
    -------
    Tuple[List[List[Dict[str, Any]]], List[List[str]], List[List[float]], int]
        list of frames, tasks, timestamps and the last timestamp of the episode
    """
    # add task to LerobotDataset meta
    for task in metadata.run.instructions:
        task_index = dataset.meta.get_task_index(task.text[0])
        if task_index is None:
            dataset.meta.add_task(task.text[0])

    # get load segments
    load_segments = [s for s in metadata.run.segments if not s.is_composite]
    if not load_segments:
        raise ValueError("No segments found")
    load_composite_segments = [s for s in metadata.run.segments if s.is_composite]
    if not load_composite_segments:
        raise ValueError("No composite segments found")
    if len(load_composite_segments) > 1:
        raise ValueError("Multiple composite segments found")

    load_composite_instruction = next(
        (
            i
            for i in metadata.run.instructions
            if i.idx == load_composite_segments[0].instruction_idx
        ),
        None,
    )
    if load_composite_instruction is None:
        raise ValueError(
            f"Instruction idx {load_composite_segments[0].instruction_idx} not found"
        )

    episode_task = load_composite_instruction.text[0]

    # FIXME: need support for multiple files?
    bag_files = [f for f in metadata.files if f.type == "rosbag"]
    bag_file = bag_files[0]
    bag_path = meta_dir / bag_file.name

    if not bag_path.exists():
        raise FileNotFoundError(f"rosbag not found in {bag_path}")

    print(f"[INFO] Processing rosbag {bag_file.name}.")
    print(f"[INFO] episode task: {episode_task}")
    print(f"[INFO] rosbag path: {bag_path}")

    episodes: List[List[Dict[str, Any]]] = []
    all_tasks: List[List[str]] = []
    all_timestamps: List[List[float]] = []

    received_topics = RecordedTopics()

    start_time = None  # [ns]
    operation_started = False
    not_initialized = True
    next_frame_time = -np.inf
    last_frame_time = -np.inf
    delta_time = 1_000_000_000 // cfg.fps

    last_timestamp = 0  # [ns]
    tf_buffer = TFBuffer()

    # extract topics from imports since it's been moved to main.py
    extract_topics = [
        "/hsrb/arm_trajectory_controller/command",
        "/hsrb/head_trajectory_controller/command",
        "/hsrb/gripper_controller/command",
        "/hsrb/gripper_controller/grasp/goal",
        "/hsrb/command_velocity",
        "/control_mode",
        "/hsrb/joint_states",
        "/hsrb/hand_camera/image_raw/compressed",
        "/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed",
        "/tf",
        "/tf_static",
        "/hsrb/wrist_wrench/raw",
    ]

    # Pre-populate static transforms
    with AnyReader([bag_path], default_typestore=typestore) as bag_reader:
        static_connections = [
            x for x in bag_reader.connections if x.topic == "/tf_static"
        ]
        for connection, _, rawdata in bag_reader.messages(
            connections=static_connections
        ):
            msg = bag_reader.deserialize(rawdata, connection.msgtype)
            tf_buffer.add_transform_msg(
                cast(tf2_msgs__msg__TFMessage, msg), is_static=True
            )

    with AnyReader([bag_path], default_typestore=typestore) as bag_reader:
        connections = [x for x in bag_reader.connections if x.topic in extract_topics]
        # iterate over each segment
        timestamp_in_lerobot_episode_w_start_time = 0

        for segment in load_segments:
            frames: list[dict[str, Any]] = []
            tasks: list[str] = []
            timestamps: list[float] = []
            last_frame: dict[str, Any] = {
                "action.arm": None,
                "action.gripper": None,
                "action.head": None,
                "action.base": None,
            }
            not_initialized = True
            # get instruction corresponding to the segment
            instruction = next(
                (
                    i
                    for i in metadata.run.instructions
                    if i.idx == segment.instruction_idx
                ),
                None,
            )
            if instruction is None:
                raise ValueError(f"Instruction idx {segment.instruction_idx} not found")
            subtask = instruction.text[0]
            success_primitive_action = segment.success
            seg_start = (
                int(1e9 * segment.start_time) if segment.start_time >= 0 else None
            )
            seg_end = int(1e9 * segment.end_time) if segment.end_time >= 0 else None
            UPDATE_NEXT_FRAME_TIME = True

            for connection, last_timestamp, rawdata in bag_reader.messages(
                connections=connections, start=seg_start, stop=seg_end
            ):
                new_topic = bag_reader.deserialize(rawdata, connection.msgtype)
                # If it's a dynamic TF message, add it to the buffer immediately
                if connection.topic == "/tf":
                    tf_buffer.add_transform_msg(
                        cast(tf2_msgs__msg__TFMessage, new_topic), is_static=False
                    )
                    continue  # Don't need to update received_topics with it

                received_topics.update_topics(
                    connection=connection,
                    last_timestamp=last_timestamp,
                    topic=new_topic,
                )

                operation_started |= received_topics.is_operating

                if (
                    start_time is None
                    and received_topics.is_valid
                    and operation_started
                ):
                    start_time = (
                        last_timestamp  # start_time is the first valid timestamp
                    )
                    timestamp_in_lerobot_episode_w_start_time = last_timestamp
                    next_frame_time = last_timestamp

                if UPDATE_NEXT_FRAME_TIME:
                    next_frame_time = last_timestamp
                    UPDATE_NEXT_FRAME_TIME = False

                # create a frame if the next frame time is reached
                if (
                    next_frame_time <= last_timestamp
                    and received_topics.is_valid
                    and operation_started
                ):
                    # check skip condition
                    if check_skip_condition(received_topics):
                        continue

                    # Search for the eef transformation matrix
                    eef_pose_mat = tf_buffer.lookup_transform(
                        target_frame="base_link",
                        source_frame="hand_palm_link",
                        timestamp=last_timestamp,
                    )
                    # If lookup fails, skip this frame
                    if eef_pose_mat is None:
                        continue

                    # update features
                    updated_features = get_updated_features_from_topics(
                        received_topics, last_frame_timestamp=last_frame_time
                    )
                    cur_frame = last_frame.copy()
                    cur_frame.update(updated_features)

                    # calculate initial subaction when first frame is created
                    if not_initialized:
                        initial_subaction = calc_initial_subaction(cur_frame)
                        cur_frame.update(initial_subaction)
                        not_initialized = False

                    # update eef features
                    # Convert matrix to 6D pose
                    ee_pose_absolute = matrix_to_xyz_rpy(eef_pose_mat)

                    # Calculate relative pose
                    if (
                        "observation.end_effector_pose.absolute" in last_frame
                        and last_frame["observation.end_effector_pose.absolute"]
                        is not None
                    ):
                        prev_ee_pose = last_frame[
                            "observation.end_effector_pose.absolute"
                        ]
                        diff = ee_pose_absolute - prev_ee_pose
                        for i in range(3, 6):
                            if diff[i] > np.pi:
                                ee_pose_absolute[i] -= 2 * np.pi
                            elif diff[i] < -np.pi:
                                ee_pose_absolute[i] += 2 * np.pi
                        ee_pose_relative = ee_pose_absolute - prev_ee_pose
                    else:
                        # First frame has zero relative pose
                        ee_pose_relative = np.zeros(6, dtype=np.float32)

                    cur_frame["observation.end_effector_pose.absolute"] = (
                        ee_pose_absolute
                    )
                    cur_frame["observation.end_effector_pose.relative"] = (
                        ee_pose_relative
                    )

                    # update action features
                    (
                        cur_frame["action.absolute"],
                        cur_frame["action.absolute.is_fresh"],
                    ) = calc_absolute_action(cur_frame)
                    (
                        cur_frame["action.relative"],
                        cur_frame["action.relative.is_fresh"],
                    ) = calc_delta_action(cur_frame)

                    # update suplementary features
                    cur_frame["next.done"] = torch.tensor(
                        False, dtype=torch.bool
                    ).reshape(1)
                    if start_time is not None:
                        timestamp_sec = (
                            timestamp_in_lerobot_episode_w_start_time - start_time
                        ) / 1_000_000_000
                    else:
                        timestamp_sec = 0.0

                    if len(metadata.run.instructions) > 1:
                        cur_frame["short_horizon_task_index"] = torch.tensor(
                            dataset.meta.get_task_index(
                                load_composite_instruction.text[0]
                            ),
                            dtype=torch.int64,
                        ).reshape(1)
                        cur_frame["primitive_action_index"] = torch.tensor(
                            dataset.meta.get_task_index(instruction.text[0]),
                            dtype=torch.int64,
                        ).reshape(1)
                        cur_frame["success_primitive_action"] = torch.tensor(
                            success_primitive_action, dtype=torch.bool
                        ).reshape(1)
                    else:
                        cur_frame["short_horizon_task_index"] = torch.tensor(
                            -1, dtype=torch.int64
                        ).reshape(1)  # we define -1 as None
                        cur_frame["primitive_action_index"] = torch.tensor(
                            dataset.meta.get_task_index(instruction.text[0]),
                            dtype=torch.int64,
                        ).reshape(1)
                        cur_frame["success_primitive_action"] = torch.tensor(
                            success_primitive_action, dtype=torch.bool
                        ).reshape(1)

                    tasks.append(subtask)
                    timestamps.append(timestamp_sec)

                    frames.append(cur_frame)
                    last_frame = cur_frame
                    last_frame_time = next_frame_time  # update last frame time
                    next_frame_time += delta_time
                    timestamp_in_lerobot_episode_w_start_time += delta_time

            if cfg.separate_per_primitive:
                if len(frames) > 1:
                    episodes.append(frames)
                    all_tasks.append(tasks)
                    all_timestamps.append(timestamps)
                start_time = timestamp_in_lerobot_episode_w_start_time
            else:
                if len(episodes) > 0 and len(frames) > 1:
                    episodes[-1].extend(frames)
                    all_tasks[-1].extend(tasks)
                    all_timestamps[-1].extend(timestamps)
                elif len(episodes) == 0:
                    episodes.append(frames)
                    all_tasks.append(tasks)
                    all_timestamps.append(timestamps)
    for frames in episodes:
        if len(frames) > 0:
            frames[-1]["next.done"] = torch.tensor(True, dtype=torch.bool).reshape(
                1
            )  # set done flag to the last frame

    return episodes, all_tasks, all_timestamps, last_timestamp
