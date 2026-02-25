# General variables
variable "environment" {
  description = "Environment name (e.g., dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "system_name" {
  description = "System name for naming convention"
  type        = string
  default     = "datapipeline"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-northeast-1"
}

# Secrets Manager variables
variable "secret_name" {
  description = "Name of the secret in AWS Secrets Manager"
  type        = string
}

# Lambda variables
variable "lambda_sources" {
  description = "List of Lambda source code objects"
  type = list(object({
    path     = string
    filename = string
  }))
  default = [
    {
      path     = "../../../aws_lambda/src/aws_batch_launcher.py"
      filename = "aws_batch_launcher.py"
    },
    {
      path     = "../../../src/hsr_data_converter/utils/aws_helper.py"
      filename = "src/hsr_data_converter/utils/aws_helper.py"
    }
  ]
}

# Lambda variables
variable "lambda_handler" {
  description = "Lambda function handler"
  type        = string
  default     = "blank_lambda_function.lambda_handler"
}

# WASABI bucket variables (LeRobot output bucket)
variable "lerobot_bucket_name" {
  description = "Name of the LeRobot output bucket"
  type        = string
}

# WASABI bucket variables
variable "ros_bags_bucket_name" {
  description = "Name of the rosbags bucket"
  type        = string
}

variable "max_vcpus" {
  description = "Maximum number of vCPUs for the job"
  type        = number
  default     = 80
}

variable "job_vcpus" {
  description = "Number of vCPUs for the job"
  type        = number
  default     = 4
}

variable "job_memory" {
  description = "Amount of memory (in MB) for the job"
  type        = number
  default     = 24576
}

variable "fps" {
  description = "Frames per second for the job"
  type        = number
  default     = 30
}

# Git metadata variables
variable "git_hash" {
  description = "Git commit hash"
  type        = string
  default     = ""
}

variable "git_branch" {
  description = "Git branch name"
  type        = string
  default     = ""
}

variable "git_url" {
  description = "Git repository URL"
  type        = string
  default     = ""
}

variable "git_tag" {
  description = "Git tag"
  type        = string
  default     = ""
}

variable "lineage_enabled" {
  description = "True if lineage is used, false otherwise."
  type        = bool
  default     = false
}

variable "marquez_url" {
  description = "marquez url"
  type        = string
  default     = "https://url-for-marquez"
}
