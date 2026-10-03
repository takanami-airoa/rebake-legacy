import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import draccus  # draccus is used for generating the CLI
from moma_metadata import MetadataV0_0
from tqdm import tqdm


@dataclass
class CLIArgs:
    """
    Command-line arguments for the dataset conversion script.
    """

    raw_dir: str  # The root directory containing the original dataset
    out_dir: str  # The output directory for the converted dataset


def convert_dataset(args: CLIArgs) -> None:
    """
    Convert the dataset directory structure with progress logging.

    Parameters:
        args (CLIArgs): The command-line arguments containing the raw_dir and out_dir.
    """
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read the global meta.json from the root directory
    meta_path = raw_dir / "meta.json"
    with open(meta_path, "r") as f:
        meta_list: List[Dict] = json.load(f)

    # Process each entry in the meta.json file
    for metaV0_dict in tqdm(meta_list, desc="Processing meta entries"):
        metaV0 = MetadataV0_0.from_dict(metaV0_dict)
        bags_dir = raw_dir / metaV0.bag_path

        bag_paths = list(bags_dir.glob("*.bag"))
        for bag_path in tqdm(
            bag_paths, desc=f"Copying bags in {bags_dir.name}", leave=False
        ):
            # Prepare V1 meta
            metaV1_dict = metaV0_dict.copy()
            metaV1_dict["version"] = "0.1"
            metaV1_dict["bag_path"] = bag_path.name

            # Create output subdirectory
            out_sub_dir = out_dir / bag_path.stem
            out_sub_dir.mkdir(parents=True, exist_ok=True)

            # Write meta.json in V1 format
            meta_path = out_sub_dir / "meta.json"
            with open(meta_path, "w") as f:
                json.dump([metaV1_dict], f, indent=2)

            # Copy the bag file
            shutil.copy(bag_path, out_sub_dir)


# uv run src/hsr_data_converter/utils/convertV0_to_V1.py --raw_dir /path/to/formatV0 --out_dir /path/to/formatV1
if __name__ == "__main__":
    args = draccus.parse(config_class=CLIArgs)
    convert_dataset(args)
