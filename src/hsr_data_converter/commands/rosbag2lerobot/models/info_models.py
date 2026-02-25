import json
from hsr_data_converter.commands.rosbag2lerobot.types.info_types import (
    InfoInterface,
    InfoFeaturesDetailInterface,
    InfoFeaturesDetailDtype,
)


class Info:
    """
    Class to handle and retrieve metadata from info.json for video processing.

    Attributes:
        info_data (InfoInterface): Parsed JSON data from info.json containing metadata related to video processing.
    """

    def __init__(self, out_dir: str) -> None:
        """
        Initialize the Info class and load metadata from json file.

        Args:
            out_dir (str): Directory path where the info.json is located.
        """
        with open(f"{out_dir}/meta/info.json") as f:
            self.info_data: InfoInterface = json.load(f)

    def get_video_path(self) -> str:
        """
        Retrieve the video path template.

        Returns:
            str: Video path template from the info.json.
        """
        return self.info_data["video_path"]

    def get_total_chunks(self) -> int:
        """
        Retrieve the total number of chunks.

        Returns:
            int: Total number of chunks from the info.json.
        """
        return self.info_data["total_chunks"]

    def get_total_videos(self) -> int:
        """
        Retrieve the total number of videos.

        Returns:
            int: Total number of videos from the info.json.
        """
        return self.info_data["total_videos"]

    def get_total_episodes(self) -> int:
        """
        Retrieve the total number of episodes.

        Returns:
            int: Total number of episodes from the info.json.
        """
        return self.info_data["total_episodes"]

    def get_features(self) -> dict[str, InfoFeaturesDetailInterface]:
        """
        Retrieve the feature details.

        Returns:
            dict[str, InfoFeaturesDetailInterface]: Dictionary of features and their details from the info.json.
        """
        return self.info_data["features"]

    def get_video_keys(self) -> list[str]:
        """
        Retrieve keys of features that are of dtype 'VIDEO'.

        Returns:
            list[str]: List of keys for video features.
        """
        return [
            key
            for key, info_features_detail in self.info_data["features"].items()
            if info_features_detail["dtype"] == InfoFeaturesDetailDtype.VIDEO.value
        ]

    def get_video_path_list(
        self,
    ) -> list[tuple[int, str]]:
        """
        Generate a list of tuples containing episode indices and corresponding formatted video paths.

        Returns:
            list[tuple[int, str]]: List of tuples with episode index and formatted video path.
        """
        result: list[tuple[int, str]] = []
        for episode_chunk in range(self.get_total_chunks()):
            for episode_index in range(self.get_total_episodes()):
                for video_key in self.get_video_keys():
                    result.append(
                        (
                            episode_index,
                            (
                                self.get_video_path().format(
                                    episode_chunk=episode_chunk,
                                    video_key=video_key,
                                    episode_index=episode_index,
                                )
                            ),
                        )
                    )

        return result
