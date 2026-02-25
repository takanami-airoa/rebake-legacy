from unittest.mock import MagicMock, patch

from hsr_data_converter.commands.rosbag2lerobot.lineage import (
    create_conversion_session,
)


class TestCreateConversionSession:
    @patch("hsr_data_converter.commands.rosbag2lerobot.lineage.get_marquez_url")
    def test_marquez_url_none_returns_none(self, mock_get_marquez_url):
        mock_get_marquez_url.return_value = None

        result = create_conversion_session("robot123", "weblab")

        assert result is None
        mock_get_marquez_url.assert_called_once()

    @patch("hsr_data_converter.commands.rosbag2lerobot.lineage.get_marquez_url")
    @patch("hsr_data_converter.commands.rosbag2lerobot.lineage.get_aws_job_information")
    def test_aws_job_information_none_returns_none(
        self, mock_get_aws_job_information, mock_get_marquez_url
    ):
        mock_get_marquez_url.return_value = "http://marquez.example.com:9000"
        mock_get_aws_job_information.return_value = None

        result = create_conversion_session("robot123", "weblab")

        assert result is None
        mock_get_marquez_url.assert_called_once()
        mock_get_aws_job_information.assert_called_once()

    @patch("hsr_data_converter.commands.rosbag2lerobot.lineage.get_marquez_url")
    @patch("hsr_data_converter.commands.rosbag2lerobot.lineage.get_aws_job_information")
    @patch("hsr_data_converter.commands.rosbag2lerobot.lineage.get_git_information")
    @patch("hsr_data_converter.commands.rosbag2lerobot.lineage.ConversionSession")
    def test_successful_session_creation(
        self,
        mock_conversion_session,
        mock_get_git_information,
        mock_get_aws_job_information,
        mock_get_marquez_url,
    ):
        # Setup mock return values
        mock_get_marquez_url.return_value = "http://marquez.example.com:9000"
        mock_get_aws_job_information.return_value = {
            "job_name": "test-job",
            "job_id": "job-12345",
        }
        mock_get_git_information.return_value = {
            "git_url": "https://github.com/example/repo.git",
            "git_hash": "abc123def456",
            "git_tag": "v1.0.0",
            "git_branch": "main",
        }

        # Mock ConversionSession instance
        mock_session_instance = MagicMock()
        mock_conversion_session.return_value = mock_session_instance

        # Call the function
        result = create_conversion_session("robot123", "weblab")

        # Verify all functions were called
        mock_get_marquez_url.assert_called_once()
        mock_get_aws_job_information.assert_called_once()
        mock_get_git_information.assert_called_once()

        # Verify ConversionSession was called with correct parameters
        mock_conversion_session.assert_called_once()
        call_args = mock_conversion_session.call_args

        # Check positional and keyword arguments
        assert call_args.kwargs["namespace"] == "airoa"
        assert call_args.kwargs["marquez_url"] == "http://marquez.example.com:9000"
        assert call_args.kwargs["facet_prefix"] == "airoa"

        # Check that common_facet and aws_job_facet are present
        assert "common_facet" in call_args.kwargs
        assert "aws_job_facet" in call_args.kwargs

        # Verify return value
        assert result == mock_session_instance

    @patch("hsr_data_converter.commands.rosbag2lerobot.lineage.get_marquez_url")
    @patch("hsr_data_converter.commands.rosbag2lerobot.lineage.get_aws_job_information")
    @patch("hsr_data_converter.commands.rosbag2lerobot.lineage.get_git_information")
    @patch("hsr_data_converter.commands.rosbag2lerobot.lineage.CommonRunFacet")
    @patch("hsr_data_converter.commands.rosbag2lerobot.lineage.AWSJobRunFacet")
    @patch("hsr_data_converter.commands.rosbag2lerobot.lineage.ConversionSession")
    def test_facets_created_with_correct_parameters(
        self,
        mock_conversion_session,
        mock_aws_job_facet,
        mock_common_facet,
        mock_get_git_information,
        mock_get_aws_job_information,
        mock_get_marquez_url,
    ):
        # Setup mock return values
        mock_get_marquez_url.return_value = "http://marquez.example.com:9000"
        mock_get_aws_job_information.return_value = {
            "job_name": "test-job",
            "job_id": "job-12345",
        }
        mock_get_git_information.return_value = {
            "git_url": "https://github.com/example/repo.git",
            "git_hash": "abc123def456",
            "git_tag": "v1.0.0",
            "git_branch": "main",
        }

        # Mock facet instances
        mock_common_facet_instance = MagicMock()
        mock_aws_job_facet_instance = MagicMock()
        mock_common_facet.return_value = mock_common_facet_instance
        mock_aws_job_facet.return_value = mock_aws_job_facet_instance

        # Mock ConversionSession instance
        mock_session_instance = MagicMock()
        mock_conversion_session.return_value = mock_session_instance

        # Call the function
        result = create_conversion_session("robot123", "weblab")

        # Verify CommonRunFacet was called with correct parameters
        mock_common_facet.assert_called_once_with(
            robotId="robot123",
            location="weblab",
            repositoryUri="https://github.com/example/repo.git",
            repositoryHash="abc123def456",
            repositoryTag="v1.0.0",
            repositoryBranch="main",
        )

        # Verify AWSJobRunFacet was called with correct parameters
        mock_aws_job_facet.assert_called_once_with(
            name="test-job",
            id="job-12345",
        )

        # Verify ConversionSession was called with the facet instances
        mock_conversion_session.assert_called_once_with(
            namespace="airoa",
            common_facet=mock_common_facet_instance,
            aws_job_facet=mock_aws_job_facet_instance,
            marquez_url="http://marquez.example.com:9000",
            facet_prefix="airoa",
        )

        # Verify return value
        assert result == mock_session_instance
