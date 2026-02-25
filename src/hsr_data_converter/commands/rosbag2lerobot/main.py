"""
Main entry point for HSR rosbag to lerobot conversion
"""

from pathlib import Path

from lerobot.configs import parser

from hsr_data_converter.commands.rosbag2lerobot.convert_config import ConvertConfig

from .aws_handler import handle_aws_upload, setup_aws_environment
from .core_converter import perform_conversion
from .utils import validate_git_information, validate_lineage_enabled


@parser.wrap()
def convert_hsr_rosbag_to_lerobot_format(cfg: ConvertConfig):
    """
    Convert HSR rosbag datasets to LeRobotDatasetV2 format (action as delta joint except gripper)
    specified by the configuration cfg

    Directory structure specified by cfg.raw_dir:

    ├ raw_dir
    │ ├ 20250308_rosbag
    │ │ ├ ...bag
    │ │ └ meta.json
    │ ├ 20250309_rosbag
    │ │ ├ ...bag
    │ │ └ meta.json
    │ └ 20250310_rosbag
    │   ├ ...bag
    │   └ meta.json
    ...

    Parameters
    ----------
    cfg : ConvertConfig
        configuration for the conversion

    Returns
    -------
    LeRobotDataset
        the converted dataset

    """
    # Validate environment variables
    try:
        validate_git_information()
        validate_lineage_enabled()
    except ValueError as e:
        print(f"[ERROR] {e}")
        raise

    # Handle AWS preprocessing if needed
    if cfg.use_aws:
        source_dir, raw_dir, out_dir, cleanup_func = setup_aws_environment(cfg)
    else:
        raw_dir = Path(cfg.raw_dir)
        out_dir = Path(cfg.out_dir)
        source_dir = f"local://{raw_dir}"
        cleanup_func = None

    try:
        # Perform the actual conversion
        result = perform_conversion(cfg, raw_dir, out_dir, source_dir)

        # Handle AWS postprocessing if needed
        if cfg.use_aws:
            handle_aws_upload(cfg, out_dir)

        return result

    finally:
        # Cleanup if needed
        if cleanup_func:
            cleanup_func()


if __name__ == "__main__":
    convert_hsr_rosbag_to_lerobot_format()
