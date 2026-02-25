import os
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from hsr_data_converter.commands.rosbag2lerobot.aws_handler import (
    setup_aws_environment,
    handle_aws_upload,
    generate_upload_path,
)


class TestGenerateUploadPath:
    @patch.dict(os.environ, {"GIT_HASH": "abc123def456"})
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.aws_handler.time.time",
        return_value=1755230456,
    )
    def test_generate_upload_path_success(self, mock_time):
        result = generate_upload_path("aist/batch_01/template-001")
        assert result == "aist/batch_01/template-001/1755230456_abc123def456"

    @patch.dict(os.environ, {"GIT_HASH": "abc123def456"})
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.aws_handler.time.time",
        return_value=1755230456,
    )
    def test_generate_upload_path_with_trailing_slash(self, mock_time):
        result = generate_upload_path("aist/batch_01/template-001/")
        assert result == "aist/batch_01/template-001/1755230456_abc123def456"

    @patch.dict(os.environ, {}, clear=True)
    def test_generate_upload_path_missing_git_hash(self):
        # Clear GIT_HASH if it exists
        with patch.dict(os.environ, {"GIT_HASH": ""}, clear=False):
            os.environ.pop("GIT_HASH", None)
            with pytest.raises(
                ValueError, match="GIT_HASH environment variable is not set"
            ):
                generate_upload_path("aist/batch_01/template-001")

    @patch.dict(os.environ, {"GIT_HASH": ""})
    def test_generate_upload_path_empty_git_hash(self):
        with pytest.raises(
            ValueError, match="GIT_HASH environment variable is not set"
        ):
            generate_upload_path("aist/batch_01/template-001")


class TestSetupAwsEnvironment:
    @patch("hsr_data_converter.commands.rosbag2lerobot.aws_handler.AWSHelper")
    @patch("hsr_data_converter.commands.rosbag2lerobot.aws_handler.Path.mkdir")
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.aws_handler.time.strftime",
        return_value="20230101_123456",
    )
    def test_standard_aws_setup(self, mock_strftime, mock_mkdir, MockAWSHelper):
        mock_cfg = MagicMock()
        mock_cfg.secret_name = "test_secret"
        mock_cfg.rosbags_bucket_name = "test_rosbags_bucket"
        mock_cfg.lerobot_bucket_name = "test_lerobot_bucket"
        mock_cfg.template_dir = "template"

        mock_aws_helper = MockAWSHelper.return_value
        mock_aws_helper.check_meta_file_exists.return_value = True

        template_path, raw_dir, out_dir, cleanup = setup_aws_environment(mock_cfg)

        assert template_path == "s3://test_rosbags_bucket/template"
        assert raw_dir == Path("tmp/hsr_conversion_20230101_123456/rosbags")
        assert out_dir == Path("tmp/hsr_conversion_20230101_123456/lerobots")

        MockAWSHelper.assert_called_once_with("test_secret")
        mock_mkdir.assert_any_call(parents=True, exist_ok=True)
        mock_aws_helper.check_meta_file_exists.assert_called_once_with(
            "test_rosbags_bucket", "template"
        )
        mock_aws_helper.download_s3_directory.assert_called_once_with(
            "test_rosbags_bucket", "template", raw_dir / "template"
        )

    def test_error_missing_parameters(self):
        mock_cfg = MagicMock()
        mock_cfg.secret_name = ""
        mock_cfg.rosbags_bucket_name = ""
        mock_cfg.lerobot_bucket_name = ""
        mock_cfg.template_dir = ""

        with pytest.raises(
            ValueError,
            match="AWS mode requires: secret_name, rosbags_bucket_name, lerobot_bucket_name, template_dir",
        ):
            setup_aws_environment(mock_cfg)

    @patch("hsr_data_converter.commands.rosbag2lerobot.aws_handler.AWSHelper")
    def test_error_meta_json_not_found(self, MockAWSHelper):
        mock_cfg = MagicMock()
        mock_cfg.secret_name = "test_secret"
        mock_cfg.rosbags_bucket_name = "test_rosbags_bucket"
        mock_cfg.lerobot_bucket_name = "test_lerobot_bucket"
        mock_cfg.template_dir = "template"

        mock_aws_helper = MockAWSHelper.return_value
        mock_aws_helper.check_meta_file_exists.return_value = False

        with pytest.raises(FileNotFoundError, match="meta.json not found in template"):
            setup_aws_environment(mock_cfg)

        mock_aws_helper.check_meta_file_exists.assert_called_once_with(
            "test_rosbags_bucket", "template"
        )


class TestHandleAwsUpload:
    @patch("hsr_data_converter.commands.rosbag2lerobot.aws_handler.AWSHelper")
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.aws_handler.Path.exists",
        return_value=False,
    )
    def test_no_output_directory(self, mock_exists, MockAWSHelper):
        mock_cfg = MagicMock()
        mock_cfg.secret_name = "test_secret"
        mock_cfg.template_dir = "template"
        mock_cfg.lerobot_bucket_name = "test_lerobot_bucket"

        with patch("builtins.print") as mock_print:
            handle_aws_upload(mock_cfg, Path("output_directory"))

            mock_exists.assert_called_once()
            mock_print.assert_any_call(
                "[WARN] No output directory or files found to upload"
            )

    @patch("hsr_data_converter.commands.rosbag2lerobot.aws_handler.AWSHelper")
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.aws_handler.Path.exists",
        return_value=True,
    )
    @patch("hsr_data_converter.commands.rosbag2lerobot.aws_handler.Path.iterdir")
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.aws_handler.generate_upload_path",
        return_value="template/1755230456_abc123def456",
    )
    def test_upload_successful_with_datasets(
        self, mock_generate_path, mock_iterdir, mock_exists, MockAWSHelper
    ):
        mock_cfg = MagicMock()
        mock_cfg.secret_name = "test_secret"
        mock_cfg.template_dir = "template"
        mock_cfg.lerobot_bucket_name = "test_lerobot_bucket"

        mock_dataset_dir = MagicMock()
        mock_dataset_dir.is_dir.return_value = True

        mock_iterdir.return_value = [mock_dataset_dir]

        with patch("builtins.print") as mock_print:
            handle_aws_upload(mock_cfg, Path("output_directory"))

            mock_exists.assert_called_once()
            mock_iterdir.assert_called()
            mock_generate_path.assert_called_once_with("template")
            mock_print.assert_any_call(
                "[INFO] Upload completed successfully to s3://test_lerobot_bucket/template/1755230456_abc123def456"
            )
            MockAWSHelper().upload_directory_to_s3.assert_any_call(
                mock_dataset_dir,
                "test_lerobot_bucket",
                "template/1755230456_abc123def456",
            )

    @patch("hsr_data_converter.commands.rosbag2lerobot.aws_handler.AWSHelper")
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.aws_handler.Path.exists",
        return_value=True,
    )
    @patch("hsr_data_converter.commands.rosbag2lerobot.aws_handler.Path.iterdir")
    @patch(
        "hsr_data_converter.commands.rosbag2lerobot.aws_handler.generate_upload_path",
        return_value="template/1755230456_abc123def456",
    )
    def test_multiple_dataset_directories(
        self, mock_generate_path, mock_iterdir, mock_exists, MockAWSHelper
    ):
        mock_cfg = MagicMock()
        mock_cfg.secret_name = "test_secret"
        mock_cfg.template_dir = "template"
        mock_cfg.lerobot_bucket_name = "test_lerobot_bucket"

        mock_dataset_dir1 = MagicMock()
        mock_dataset_dir1.is_dir.return_value = True
        mock_dataset_dir2 = MagicMock()
        mock_dataset_dir2.is_dir.return_value = True

        mock_iterdir.return_value = [mock_dataset_dir1, mock_dataset_dir2]

        with patch("builtins.print") as mock_print:
            handle_aws_upload(mock_cfg, Path("output_directory"))

            mock_exists.assert_called_once()
            mock_iterdir.assert_called()
            mock_generate_path.assert_called_once_with("template")
            mock_print.assert_any_call(
                "[INFO] Upload completed successfully to s3://test_lerobot_bucket/template/1755230456_abc123def456"
            )
            MockAWSHelper().upload_directory_to_s3.assert_any_call(
                mock_dataset_dir1,
                "test_lerobot_bucket",
                "template/1755230456_abc123def456_0",
            )
            MockAWSHelper().upload_directory_to_s3.assert_any_call(
                mock_dataset_dir2,
                "test_lerobot_bucket",
                "template/1755230456_abc123def456_1",
            )
