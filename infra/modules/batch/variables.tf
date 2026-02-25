variable "environment" {
  description = "Environment name"
  type        = string
}

variable "system_name" {
  description = "System name"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID for the Batch compute environment"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs for the Batch compute environment"
  type        = list(string)
}

variable "security_group_id" {
  description = "Security group ID for Batch compute environment"
  type        = string
}

variable "max_vcpus" {
  description = "Number of vCPUs for the job (Fargate: 0.25, 0.5, 1, 2, 4, 8, 16)"
  type        = number
  default     = 1
}

variable "job_vcpus" {
  description = "Number of vCPUs for the job"
  type        = number
  default     = 1
}

variable "job_memory" {
  description = "Amount of memory (in MB) for the job"
  type        = number
  default     = 2048
}

variable "job_queue_priority" {
  description = "Priority of the job queue"
  type        = number
  default     = 1
}


variable "container_image_uri" {
  description = "URI of the container image in ECR"
  type        = string
}


# IAM role ARNs

variable "batch_service_role_arn" {
  description = "ARN of the IAM role for Batch service"
  type        = string
}

variable "batch_execution_role_arn" {
  description = "ARN of the IAM role for Batch job execution"
  type        = string
}

variable "batch_job_role_arn" {
  description = "ARN of the IAM role for Batch job execution"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-northeast-1"
}

variable "log_retention_days" {
  description = "Number of days to retain CloudWatch logs"
  type        = number
  default     = 7
}

variable "common_tags" {
  description = "A map of tags to assign to the resource"
  type        = map(string)
  default     = {}
}

variable "template_dir" {
  description = "Template directory"
  type        = string
  default     = ""
}


variable "fps" {
  description = "FPS"
  type        = number
}

variable "lerobot_bucket_name" {
  description = "Lerobot bucket name"
  type        = string
}

variable "ros_bags_bucket_name" {
  description = "Rosbags bucket name"
  type        = string
}

variable "secret_name" {
  description = "Secret name"
  type        = string
}

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
  default     = ""
}
