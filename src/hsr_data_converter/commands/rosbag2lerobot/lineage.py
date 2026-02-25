from airoa_lineage.conversion import ConversionSession
from airoa_lineage.facets import AWSJobRunFacet, CommonRunFacet

from .utils import get_aws_job_information, get_git_information, get_marquez_url


def create_conversion_session(robot_id: str, location: str) -> ConversionSession | None:
    marquez_url = get_marquez_url()
    aws_job_information = get_aws_job_information()
    if marquez_url is None or aws_job_information is None:
        # Skip lineage tracking if Marquez URL or AWS job info is not available.
        return None

    git_information = get_git_information()

    common_facet = CommonRunFacet(
        robotId=robot_id,
        location=location,
        repositoryUri=git_information["git_url"],
        repositoryHash=git_information["git_hash"],
        repositoryTag=git_information["git_tag"],
        repositoryBranch=git_information["git_branch"],
    )

    aws_job_facet = AWSJobRunFacet(
        name=aws_job_information["job_name"],
        id=aws_job_information["job_id"],
    )

    session = ConversionSession(
        namespace="airoa",
        common_facet=common_facet,
        aws_job_facet=aws_job_facet,
        marquez_url=marquez_url,
        facet_prefix="airoa",
    )

    return session
