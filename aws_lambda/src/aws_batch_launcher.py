import json
import os
import boto3
from src.hsr_data_converter.utils.aws_helper import AWSHelper


def lambda_handler(event, context):
    lerobot_bucket_name = os.environ.get("LEROBOT_BUCKET_NAME")
    secret_name = os.environ.get("SECRET_NAME")
    job_queue = os.environ.get("BATCH_JOB_QUEUE")
    job_definition = os.environ.get("BATCH_JOB_DEFINITION")

    fps = os.environ.get("FPS", "10")
    region_name = os.environ.get("AWS_REGION", "ap-northeast-1")

    # 必須環境変数のチェック
    required_vars = {
        "LEROBOT_BUCKET_NAME": lerobot_bucket_name,
        "SECRET_NAME": secret_name,
        "BATCH_JOB_QUEUE": job_queue,
        "BATCH_JOB_DEFINITION": job_definition,
    }

    missing_vars = [name for name, value in required_vars.items() if not value]
    if missing_vars:
        error_msg = f"Required environment variables not set: {', '.join(missing_vars)}"
        print(error_msg)
        return {"statusCode": 400, "body": json.dumps({"error": error_msg})}
    message_str = event["Records"][0]["Sns"]["Message"]
    message_json = json.loads(message_str)

    # S3 イベント情報を取り出す
    wasabi_event = message_json["Records"][0]
    bucket_name = wasabi_event["s3"]["bucket"]["name"]
    object_key = wasabi_event["s3"]["object"]["key"]
    batch_dir = os.path.dirname(object_key) + "/"

    print("Bucket Name:", bucket_name)
    print("Object Key:", object_key)
    print("Batch dir:", batch_dir)

    # AWSヘルパーを使用してmeta.json + *.bagペアを持つディレクトリを検索
    aws_helper = AWSHelper(secret_name)
    try:
        rosbag_directories = aws_helper.find_rosbag_directories(bucket_name, batch_dir)
        print(
            f"Found {len(rosbag_directories)} template directories: {[d['directory'] for d in rosbag_directories]}"
        )
    except Exception as e:
        print(f"Error listing template directories: {e}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

    if not rosbag_directories:
        print("No template directories found")
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "No template directories found"}),
        }

    batch = boto3.client("batch", region_name=region_name)
    submitted_jobs = []

    # templateディレクトリごとにジョブを送信
    for template_dir_info in rosbag_directories:
        template_dir = template_dir_info["directory"]
        job_name = template_dir.replace("/", "_") + "job"
        print(f"Submitting job for template directory: {template_dir}")

        try:
            response = batch.submit_job(
                jobName=job_name,
                jobQueue=job_queue,
                jobDefinition=job_definition,
                containerOverrides={
                    "command": [
                        ".venv/bin/python",
                        "-m",
                        "hsr_data_converter.commands.rosbag2lerobot.main",
                        "--secret_name",
                        "Ref::secret_name",
                        "--rosbags_bucket_name",
                        "Ref::rosbags_bucket_name",
                        "--lerobot_bucket_name",
                        "Ref::lerobot_bucket_name",
                        "--template_dir",
                        "Ref::template_dir",
                        "--fps",
                        "Ref::fps",
                        "--use_aws",
                        "true",
                    ],
                },
                parameters={
                    "secret_name": secret_name,
                    "rosbags_bucket_name": bucket_name,
                    "fps": fps,
                    "template_dir": template_dir,
                    "lerobot_bucket_name": lerobot_bucket_name,
                },
            )

            submitted_jobs.append(
                {"template_dir": template_dir, "job_id": response["jobId"]}
            )
            print(f"Job submitted for {template_dir}. Job ID: {response['jobId']}")

        except Exception as e:
            print(f"Error submitting job for {template_dir}: {e}")
            submitted_jobs.append({"template_dir": template_dir, "error": str(e)})

    return {"statusCode": 200, "body": json.dumps({"jobs": submitted_jobs})}
