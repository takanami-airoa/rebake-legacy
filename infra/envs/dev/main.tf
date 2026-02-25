terraform {
  required_version = "~>1.9.6"
  backend "s3" {
    # パラメータはterraform.tfbackendにて指定する
  }
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Get current AWS account ID
data "aws_caller_identity" "current" {}


locals {
  # Naming convention: {environment}-{system}-{resource}
  resource_names = {
    # VPC resources
    vpc_name = "${var.environment}-${var.system_name}-vpc"

    # SNS resources
    sns_topic = "${var.environment}-${var.system_name}-trigger"

    # AWS Batch resources
    batch_compute_environment = "${var.environment}-${var.system_name}-compute-env"
    batch_job_queue           = "${var.environment}-${var.system_name}-job-queue"
    batch_job_definition      = "${var.environment}-${var.system_name}-job-definition"
  }

  common_tags = {
    Environment = var.environment
    System      = var.system_name
  }
}



# ECR Repository
module "ecr" {
  source = "../../modules/ecr"

  environment = var.environment
  system_name = var.system_name

  common_tags = local.common_tags
}


# Secrets Manager
module "secrets_manager" {
  source = "../../modules/secrets_manager"

  environment = var.environment
  system_name = var.system_name
  secret_name = var.secret_name # FIXME: Plan to manage secrets with Terraform in the future

  common_tags = local.common_tags
}


# IAM Roles (all roles managed in single module)
module "iam" {
  source = "../../modules/iam"

  environment = var.environment
  system_name = var.system_name
  region      = var.aws_region
  account     = data.aws_caller_identity.current.account_id

  secrets_manager_arn = module.secrets_manager.secret_arn

  common_tags = local.common_tags
}



# VPC Module
module "vpc" {
  source = "../../modules/vpc"

  environment = var.environment
  system_name = var.system_name

  vpc_cidr             = "10.0.0.0/16"
  public_subnet_cidrs  = ["10.0.1.0/24", "10.0.2.0/24"]
  private_subnet_cidrs = ["10.0.11.0/24", "10.0.12.0/24"]
  availability_zones   = ["ap-northeast-1a", "ap-northeast-1c"]
  aws_region           = var.aws_region

  common_tags = local.common_tags
}


# Security Group
module "security_group" {
  source      = "../../modules/security_group"
  environment = var.environment
  system_name = var.system_name
  vpc_id      = module.vpc.vpc_id
  vpc_cidr    = module.vpc.vpc_cidr
  common_tags = local.common_tags
}


# AWS Batch (Fargate)
module "batch" {
  source = "../../modules/batch"

  environment         = var.environment
  system_name         = var.system_name
  container_image_uri = "${module.ecr.repository_url}:latest"

  vpc_id            = module.vpc.vpc_id
  subnet_ids        = module.vpc.public_subnet_ids
  security_group_id = module.security_group.batch_security_group_id

  job_queue_priority = 1

  max_vcpus  = var.max_vcpus
  job_vcpus  = var.job_vcpus
  job_memory = var.job_memory
  fps        = var.fps

  log_retention_days = 7

  aws_region = var.aws_region

  # IAM role ARNs from iam module
  batch_service_role_arn   = module.iam.batch_computing_role_arn
  batch_execution_role_arn = module.iam.batch_exe_role_arn
  batch_job_role_arn       = module.iam.batch_job_role_arn

  secret_name          = var.secret_name
  ros_bags_bucket_name = var.ros_bags_bucket_name
  lerobot_bucket_name  = var.lerobot_bucket_name

  git_hash   = var.git_hash
  git_branch = var.git_branch
  git_url    = var.git_url
  git_tag    = var.git_tag

  lineage_enabled = var.lineage_enabled
  marquez_url     = var.marquez_url

  common_tags = local.common_tags
}


# Lambda Function
module "lambda" {
  source = "../../modules/lambda"

  environment = var.environment
  system_name = var.system_name

  execution_role_arn = module.iam.lambda_execution_role_arn

  lambda_sources       = var.lambda_sources
  handler              = var.lambda_handler
  lerobot_bucket_name  = var.lerobot_bucket_name
  secret_name          = module.secrets_manager.secret_name
  batch_job_queue      = module.batch.job_queue_name
  batch_job_definition = module.batch.job_definition_name

  common_tags = local.common_tags
}


# SNS Topic
module "sns" {
  source = "../../modules/sns"

  topic_name          = local.resource_names.sns_topic
  lambda_function_arn = module.lambda.function_arn
}
