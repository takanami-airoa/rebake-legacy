import unittest
import pytest
from unittest.mock import MagicMock, patch, call
import botocore
from hsr_data_converter.utils.aws_helper import AWSHelper
from botocore.exceptions import ClientError
from pathlib import Path


class TestConstructor(unittest.TestCase):
    def test_full(self):
        secret_name = "test_secret_name"
        region_name = "test_region_name"
        endpoint_url = "test_endpoint_url"

        actual = AWSHelper(
            secret_name=secret_name, region_name=region_name, endpoint_url=endpoint_url
        )

        assert actual.secret_name == secret_name
        assert actual.region_name == region_name
        assert actual.endpoint_url == endpoint_url
        self.assertIsNone(actual._credentials)
        self.assertIsNone(actual._s3_client)

    def test_default(self):
        secret_name = "test_secret_name"

        actual = AWSHelper(secret_name=secret_name)

        assert actual.secret_name == secret_name
        assert actual.region_name == "ap-northeast-1"
        assert actual.endpoint_url == "https://s3.ap-northeast-1.wasabisys.com"
        self.assertIsNone(actual._credentials)
        self.assertIsNone(actual._s3_client)


class TestPrivateGetCredentials:
    @patch("boto3.client")
    @patch("json.loads")
    def test_credentials_is_none(self, mock_json_loads, mock_boto_client):
        secret_name = "test_secret_name"
        region_name = "test_region_name"
        endpoint_url = "test_endpoint_url"

        mock_client_instance = MagicMock()
        mock_boto_client.return_value = mock_client_instance
        mock_client_instance.get_secret_value.return_value = {
            "SecretString": '{"WASABI_ACCESS_KEY_ID": "AKIA123", "WASABI_SECRET_ACCESS_KEY": "SECRET456"}'
        }
        mock_json_loads.return_value = {
            "WASABI_ACCESS_KEY_ID": "AKIA123",
            "WASABI_SECRET_ACCESS_KEY": "SECRET456",
        }

        helper = AWSHelper(
            secret_name=secret_name, region_name=region_name, endpoint_url=endpoint_url
        )
        creds = helper._get_credentials()

        assert creds == {"access_key": "AKIA123", "secret_key": "SECRET456"}
        mock_boto_client.assert_called_once_with(
            "secretsmanager", region_name=region_name
        )
        mock_client_instance.get_secret_value.assert_called_once_with(
            SecretId=secret_name
        )
        mock_json_loads.assert_called_once()

    @patch("boto3.client")
    @patch("json.loads")
    def test_credentials_is_not_none(self, mock_json_loads, mock_boto_client):
        secret_name = "test_secret_name"
        region_name = "test_region_name"
        endpoint_url = "test_endpoint_url"

        mock_client_instance = MagicMock()
        mock_boto_client.return_value = mock_client_instance

        helper = AWSHelper(
            secret_name=secret_name, region_name=region_name, endpoint_url=endpoint_url
        )
        helper._credentials = {"access_key": "AKIA123", "secret_key": "SECRET456"}
        creds = helper._get_credentials()

        assert creds == {"access_key": "AKIA123", "secret_key": "SECRET456"}
        mock_boto_client.assert_not_called()
        mock_client_instance.assert_not_called()
        mock_json_loads.assert_not_called()

    @patch("boto3.client")
    @patch("json.loads")
    def test_error_boto_core_error(self, mock_json_loads, mock_boto_client):
        secret_name = "test_secret_name"
        region_name = "test_region_name"
        endpoint_url = "test_endpoint_url"

        mock_client_instance = MagicMock()
        mock_boto_client.return_value = mock_client_instance
        mock_client_instance.get_secret_value.side_effect = (
            botocore.exceptions.BotoCoreError()
        )

        helper = AWSHelper(
            secret_name=secret_name, region_name=region_name, endpoint_url=endpoint_url
        )

        with pytest.raises(botocore.exceptions.BotoCoreError):
            helper._get_credentials()

        mock_boto_client.assert_called_once_with(
            "secretsmanager", region_name=region_name
        )
        mock_client_instance.get_secret_value.assert_called_once_with(
            SecretId=secret_name
        )
        mock_json_loads.assert_not_called()


class TestPrivateGetS3Client:
    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_credentials")
    @patch("boto3.client")
    def test_s3_client_is_none(self, mock_boto_client, mock_get_credentials):
        secret_name = "test_secret_name"
        region_name = "test_region_name"
        endpoint_url = "test_endpoint_url"

        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        mock_get_credentials.return_value = {
            "access_key": "AKIA123",
            "secret_key": "SECRET456",
        }

        helper = AWSHelper(
            secret_name=secret_name, region_name=region_name, endpoint_url=endpoint_url
        )
        client = helper._get_s3_client()

        assert client == mock_s3
        mock_get_credentials.assert_called_once_with()
        mock_boto_client.assert_called_once_with(
            "s3",
            aws_access_key_id="AKIA123",
            aws_secret_access_key="SECRET456",
            endpoint_url=endpoint_url,
            region_name=region_name,
        )

    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_credentials")
    @patch("boto3.client")
    def test_s3_client_is_not_none(self, mock_boto_client, mock_get_credentials):
        secret_name = "test_secret_name"
        region_name = "test_region_name"
        endpoint_url = "test_endpoint_url"

        helper = AWSHelper(
            secret_name=secret_name, region_name=region_name, endpoint_url=endpoint_url
        )
        mock_s3 = MagicMock()
        helper._s3_client = mock_s3
        client = helper._get_s3_client()

        assert client == mock_s3
        mock_get_credentials.assert_not_called()
        mock_boto_client.assert_not_called()

    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_credentials")
    @patch("boto3.client")
    def test_error_get_credentials(self, mock_boto_client, mock_get_credentials):
        secret_name = "test_secret_name"
        region_name = "test_region_name"
        endpoint_url = "test_endpoint_url"

        mock_get_credentials.side_effect = Exception("test error.")

        helper = AWSHelper(
            secret_name=secret_name, region_name=region_name, endpoint_url=endpoint_url
        )
        with pytest.raises(Exception) as e:
            helper._get_s3_client()

        assert str(e.value) == "test error."
        mock_get_credentials.assert_called_once_with()
        mock_boto_client.assert_not_called()

    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_credentials")
    @patch("boto3.client")
    def test_error_boto3(self, mock_boto_client, mock_get_credentials):
        secret_name = "test_secret_name"
        region_name = "test_region_name"
        endpoint_url = "test_endpoint_url"

        mock_boto_client.side_effect = Exception("test error.")

        mock_get_credentials.return_value = {
            "access_key": "AKIA123",
            "secret_key": "SECRET456",
        }

        helper = AWSHelper(
            secret_name=secret_name, region_name=region_name, endpoint_url=endpoint_url
        )
        with pytest.raises(Exception) as e:
            helper._get_s3_client()

        assert str(e.value) == "test error."
        mock_get_credentials.assert_called_once_with()
        mock_boto_client.assert_called_once_with(
            "s3",
            aws_access_key_id="AKIA123",
            aws_secret_access_key="SECRET456",
            endpoint_url=endpoint_url,
            region_name=region_name,
        )


class TestCheckMetaFileExists(unittest.TestCase):
    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    def test_meta_file_exists(self, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        helper = AWSHelper(secret_name="test_secret")

        mock_s3_client.head_object.return_value = {}

        result = helper.check_meta_file_exists("my_bucket", "some/prefix/")
        assert result is True
        mock_s3_client.head_object.assert_called_once_with(
            Bucket="my_bucket", Key="some/prefix/meta.json"
        )

    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    def test_meta_file_not_exists(self, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        helper = AWSHelper(secret_name="test_secret")

        error_response = {"Error": {"Code": "404"}}
        mock_s3_client.head_object.side_effect = ClientError(
            error_response, "HeadObject"
        )

        result = helper.check_meta_file_exists("my_bucket", "some/prefix/")
        assert result is False
        mock_s3_client.head_object.assert_called_once_with(
            Bucket="my_bucket", Key="some/prefix/meta.json"
        )

    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    def test_error_meta_file_other_error(self, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        helper = AWSHelper(secret_name="test_secret")

        error_response = {"Error": {"Code": "500"}}
        mock_s3_client.head_object.side_effect = ClientError(
            error_response, "HeadObject"
        )

        with self.assertRaises(ClientError):
            helper.check_meta_file_exists("my_bucket", "some/prefix/")
        mock_s3_client.head_object.assert_called_once_with(
            Bucket="my_bucket", Key="some/prefix/meta.json"
        )


class TestDownloadS3Directory(unittest.TestCase):
    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    @patch("pathlib.Path.mkdir")
    def test_download_s3_directory(self, mock_mkdir, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_s3_client.download_file.return_value = None

        mock_pages = [
            {
                "Contents": [
                    {"Key": "some/prefix/file1.txt"},
                    {"Key": "some/prefix/subdir/file2.txt"},
                ]
            },
            {
                "Contents": [
                    {"Key": "some/prefix/file3.txt"},
                ]
            },
        ]
        mock_paginator.paginate.return_value = mock_pages

        helper = AWSHelper(secret_name="test_secret")
        local_dir = Path("/mock/local/dir")

        helper.download_s3_directory("test_bucket", "some/prefix/", local_dir)

        mock_s3_client.get_paginator.assert_called_once_with("list_objects_v2")
        assert mock_mkdir.call_count == 4
        assert mock_s3_client.download_file.call_count == 3
        mock_s3_client.download_file.assert_has_calls(
            [
                call(
                    "test_bucket", "some/prefix/file1.txt", "/mock/local/dir/file1.txt"
                ),
                call(
                    "test_bucket",
                    "some/prefix/subdir/file2.txt",
                    "/mock/local/dir/subdir/file2.txt",
                ),
                call(
                    "test_bucket", "some/prefix/file3.txt", "/mock/local/dir/file3.txt"
                ),
            ]
        )

    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    @patch("pathlib.Path.mkdir")
    def test_empty_pages(self, mock_mkdir, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = []

        helper = AWSHelper(secret_name="test_secret")
        local_dir = Path("/mock/local/dir")

        helper.download_s3_directory("test_bucket", "empty_prefix/", local_dir)

        mock_s3_client.get_paginator.assert_called_once_with("list_objects_v2")
        assert mock_mkdir.call_count == 1
        mock_s3_client.download_file.assert_not_called()

    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    @patch("pathlib.Path.mkdir")
    def test_page_without_contents(self, mock_mkdir, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{}]

        helper = AWSHelper(secret_name="test_secret")
        local_dir = Path("/mock/local/dir")

        helper.download_s3_directory("test_bucket", "no_contents_prefix/", local_dir)

        mock_s3_client.get_paginator.assert_called_once_with("list_objects_v2")
        assert mock_mkdir.call_count == 1
        mock_s3_client.download_file.assert_not_called()

    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    @patch("pathlib.Path.mkdir")
    def test_empty_contents_in_page(self, mock_mkdir, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{"Contents": []}]

        helper = AWSHelper(secret_name="test_secret")
        local_dir = Path("/mock/local/dir")

        helper.download_s3_directory("test_bucket", "empty_contents_prefix/", local_dir)

        mock_s3_client.get_paginator.assert_called_once_with("list_objects_v2")
        assert mock_mkdir.call_count == 1
        mock_s3_client.download_file.assert_not_called()

    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    @patch("pathlib.Path.mkdir")
    def test_empty_relative_path(self, mock_mkdir, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{"Contents": [{"Key": "some/prefix/"}]}]

        helper = AWSHelper(secret_name="test_secret")
        local_dir = Path("/mock/local/dir")

        helper.download_s3_directory("test_bucket", "some/prefix/", local_dir)

        mock_s3_client.get_paginator.assert_called_once_with("list_objects_v2")
        assert mock_mkdir.call_count == 1
        mock_s3_client.download_file.assert_not_called()

    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    @patch("pathlib.Path.mkdir")
    def test_error_in_get_paginator(self, mock_mkdir, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        mock_s3_client.get_paginator.side_effect = Exception("Paginator error")

        helper = AWSHelper(secret_name="test_secret")
        local_dir = Path("/mock/local/dir")

        with self.assertRaises(Exception) as context:
            helper.download_s3_directory("test_bucket", "some/prefix/", local_dir)

        assert str(context.exception) == "Paginator error"
        mock_s3_client.get_paginator.assert_called_once_with("list_objects_v2")
        assert mock_mkdir.call_count == 1
        mock_s3_client.download_file.assert_not_called()

    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    @patch("pathlib.Path.mkdir")
    def test_error_in_download_file(self, mock_mkdir, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "some/prefix/file1.txt"}]}
        ]

        mock_s3_client.download_file.side_effect = Exception("Download error")

        helper = AWSHelper(secret_name="test_secret")
        local_dir = Path("/mock/local/dir")

        with self.assertRaises(Exception) as context:
            helper.download_s3_directory("test_bucket", "some/prefix/", local_dir)

        assert str(context.exception) == "Download error"
        assert mock_mkdir.call_count == 2
        mock_s3_client.download_file.assert_called_once_with(
            "test_bucket", "some/prefix/file1.txt", "/mock/local/dir/file1.txt"
        )


class TestUploadDirectoryToS3(unittest.TestCase):
    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    @patch("os.walk")
    def test_upload_directory_to_s3(self, mock_os_walk, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        local_dir = Path("/mock/local/dir")
        mock_os_walk.return_value = [
            ("/mock/local/dir", ("subdir",), ("file1.txt", "file2.txt")),
            ("/mock/local/dir/subdir", (), ("file3.txt",)),
        ]

        helper = AWSHelper(secret_name="test_secret")

        helper.upload_directory_to_s3(local_dir, "test_bucket", "some/prefix/")

        expected_calls = [
            call(str(local_dir / "file1.txt"), "test_bucket", "some/prefix/file1.txt"),
            call(str(local_dir / "file2.txt"), "test_bucket", "some/prefix/file2.txt"),
            call(
                str(local_dir / "subdir/file3.txt"),
                "test_bucket",
                "some/prefix/subdir/file3.txt",
            ),
        ]
        mock_s3_client.upload_file.assert_has_calls(expected_calls, any_order=True)
        assert mock_s3_client.upload_file.call_count == 3

    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    @patch("os.walk")
    def test_empty_directory(self, mock_os_walk, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        local_dir = Path("/mock/local/empty_dir")
        mock_os_walk.return_value = [
            ("/mock/local/empty_dir", (), ()),
        ]

        helper = AWSHelper(secret_name="test_secret")

        helper.upload_directory_to_s3(local_dir, "test_bucket", "some/prefix/")

        mock_s3_client.upload_file.assert_not_called()

    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    @patch("os.walk")
    def test_error_upload_file(self, mock_os_walk, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        local_dir = Path("/mock/local/dir")
        mock_os_walk.return_value = [
            ("/mock/local/dir", (), ("file1.txt",)),
        ]

        mock_s3_client.upload_file.side_effect = Exception("Upload error")

        helper = AWSHelper(secret_name="test_secret")

        with self.assertRaises(Exception) as context:
            helper.upload_directory_to_s3(local_dir, "test_bucket", "some/prefix/")

        assert str(context.exception) == "Upload error"
        mock_s3_client.upload_file.assert_called_once_with(
            str(local_dir / "file1.txt"), "test_bucket", "some/prefix/file1.txt"
        )


class TestUploadProcessingComplete(unittest.TestCase):
    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    def test_upload_processing_complete_success(self, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        helper = AWSHelper(secret_name="test_secret")

        helper.upload_processing_complete("test_bucket", "some/prefix/")

        expected_key = "some/prefix/.processing_complete"
        mock_s3_client.put_object.assert_called_once_with(
            Bucket="test_bucket", Key=expected_key, Body=b""
        )

    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    def test_error_upload_processing_complete_client_error(self, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        mock_s3_client.put_object.side_effect = ClientError(
            {"Error": {"Code": "500", "Message": "Internal Server Error"}}, "PutObject"
        )

        helper = AWSHelper(secret_name="test_secret")

        with self.assertRaises(ClientError) as context:
            helper.upload_processing_complete("test_bucket", "some/prefix/")

        assert context.exception.response["Error"]["Code"] == "500"
        mock_s3_client.put_object.assert_called_once()


class TestListS3Directory(unittest.TestCase):
    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    def test_list_s3_directory_with_common_prefixes(self, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator

        mock_pages = [
            {
                "CommonPrefixes": [
                    {"Prefix": "some/prefix/dir1/"},
                    {"Prefix": "some/prefix/dir2/"},
                ]
            }
        ]
        mock_paginator.paginate.return_value = mock_pages

        helper = AWSHelper(secret_name="test_secret")

        result = helper.list_s3_directory("test_bucket", "some/prefix/")

        expected_directories = ["some/prefix/dir1/", "some/prefix/dir2/"]
        assert result == expected_directories

    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    def test_list_s3_directory_without_common_prefixes(self, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator

        mock_pages = [{}]
        mock_paginator.paginate.return_value = mock_pages

        helper = AWSHelper(secret_name="test_secret")

        result = helper.list_s3_directory("test_bucket", "empty_prefix/")

        assert result == []

    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    def test_error_list_s3_directory_paginator_exception(self, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        mock_s3_client.get_paginator.side_effect = Exception("Paginator error")

        helper = AWSHelper(secret_name="test_secret")

        with self.assertRaises(Exception) as context:
            helper.list_s3_directory("test_bucket", "some/prefix/")

        assert str(context.exception) == "Paginator error"


class TestListTemplateDirectories(unittest.TestCase):
    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    def test_list_template_directories_with_templates(self, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator

        mock_pages = [
            {
                "CommonPrefixes": [
                    {"Prefix": "batch/template-abc/"},
                    {"Prefix": "batch/template-def/"},
                    {"Prefix": "batch/non-template/"},
                ]
            }
        ]
        mock_paginator.paginate.return_value = mock_pages

        helper = AWSHelper(secret_name="test_secret")

        result = helper.list_template_directories("test_bucket", "batch/")

        expected_directories = ["batch/template-abc/", "batch/template-def/"]
        assert result == expected_directories

    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    def test_list_template_directories_no_templates(self, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator

        mock_pages = [
            {
                "CommonPrefixes": [
                    {"Prefix": "batch/some-dir/"},
                    {"Prefix": "batch/another-dir/"},
                ]
            }
        ]
        mock_paginator.paginate.return_value = mock_pages

        helper = AWSHelper(secret_name="test_secret")

        result = helper.list_template_directories("test_bucket", "batch/")

        assert result == []

    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    def test_list_template_directories_empty_page(self, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator

        mock_pages = [{}]
        mock_paginator.paginate.return_value = mock_pages

        helper = AWSHelper(secret_name="test_secret")

        result = helper.list_template_directories("test_bucket", "batch/")

        assert result == []

    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    def test_error_list_template_directories_paginator_exception(
        self, mock_get_s3_client
    ):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        mock_s3_client.get_paginator.side_effect = Exception("Paginator error")

        helper = AWSHelper(secret_name="test_secret")

        with self.assertRaises(Exception) as context:
            helper.list_template_directories("test_bucket", "batch/")

        assert str(context.exception) == "Paginator error"


class TestFindRosbagDirectories(unittest.TestCase):
    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    def test_find_rosbag_directories_with_results(self, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator

        mock_pages = [
            {
                "Contents": [
                    {"Key": "rosbags/dir1/meta.json"},
                    {"Key": "rosbags/dir1/file1.bag"},
                    {"Key": "rosbags/dir1/file2.bag"},
                    {"Key": "rosbags/dir2/meta.json"},
                    {"Key": "rosbags/dir2/file3.bag"},
                    {"Key": "rosbags/dir3/file4.bag"},
                ]
            }
        ]
        mock_paginator.paginate.return_value = mock_pages

        helper = AWSHelper(secret_name="test_secret")

        result = helper.find_rosbag_directories("test_bucket", "rosbags/")

        expected_directories = [
            {
                "directory": "rosbags/dir1/",
                "meta_file": "meta.json",
                "bag_files": ["file1.bag", "file2.bag"],
            },
            {
                "directory": "rosbags/dir2/",
                "meta_file": "meta.json",
                "bag_files": ["file3.bag"],
            },
        ]
        assert result == expected_directories

    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    def test_find_rosbag_directories_no_results(self, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator

        mock_pages = [
            {
                "Contents": [
                    {"Key": "rosbags/dir1/file1.txt"},
                    {"Key": "rosbags/dir2/file2.txt"},
                ]
            }
        ]
        mock_paginator.paginate.return_value = mock_pages

        helper = AWSHelper(secret_name="test_secret")

        result = helper.find_rosbag_directories("test_bucket", "rosbags/")

        assert result == []

    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    def test_find_rosbag_directories_empty(self, mock_get_s3_client):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator

        mock_pages = [{}]
        mock_paginator.paginate.return_value = mock_pages

        helper = AWSHelper(secret_name="test_secret")

        result = helper.find_rosbag_directories("test_bucket", "rosbags/")

        assert result == []

    @patch("hsr_data_converter.utils.aws_helper.AWSHelper._get_s3_client")
    def test_error_find_rosbag_directories_paginator_exception(
        self, mock_get_s3_client
    ):
        mock_s3_client = MagicMock()
        mock_get_s3_client.return_value = mock_s3_client

        mock_s3_client.get_paginator.side_effect = Exception("Paginator error")

        helper = AWSHelper(secret_name="test_secret")

        with self.assertRaises(Exception) as context:
            helper.find_rosbag_directories("test_bucket", "rosbags/")

        assert str(context.exception) == "Paginator error"
