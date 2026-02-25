# Local values for naming
locals {
  lambda_role_name          = "${var.environment}-${var.system_name}-lambda-role"
  batch_exe_role_name       = "${var.environment}-${var.system_name}-batch-exe-role"
  batch_job_role_name       = "${var.environment}-${var.system_name}-batch-job-role"
  batch_computing_role_name = "${var.environment}-${var.system_name}-batch-computing-role"
}

# Lambda(ロール)
resource "aws_iam_role" "lambda_role" {
  name = local.lambda_role_name

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })

  tags = var.common_tags
}

# Lambda(ポリシー)
resource "aws_iam_role_policy" "lambda_basic_execution_role_policy" {
  name = "${var.environment}-${var.system_name}-lambda-basic-execution-role-policy"
  role = aws_iam_role.lambda_role.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "logs:CreateLogGroup"
        Resource = "arn:aws:logs:${var.region}:${var.account}:*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:${var.region}:${var.account}:log-group:/aws/lambda/${var.environment}-${var.system_name}*"
      }
    ]
  })
}

resource "aws_iam_role_policy" "lambda_batch_policy" {
  name = "${var.environment}-${var.system_name}-lambda-batch-policy"
  role = aws_iam_role.lambda_role.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "batch:SubmitJob",
          "batch:DescribeJobs",
          "batch:TerminateJob"
        ]
        Resource = "*"
      }
    ]
  })
}

# VPC policy (optional)
resource "aws_iam_role_policy" "lambda_vpc_policy" {
  name = "${var.environment}-${var.system_name}-lambda-vpc-policy"
  role = aws_iam_role.lambda_role.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ec2:CreateNetworkInterface",
          "ec2:DeleteNetworkInterface",
          "ec2:DescribeNetworkInterfaces"
        ]
        Resource = "*"
      }
    ]
  })
}

# Secrets Manager permissions for Lambda
resource "aws_iam_role_policy" "lambda_secrets_manager_policy" {
  name = "${var.environment}-${var.system_name}-lambda-secrets-manager-policy"
  role = aws_iam_role.lambda_role.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = var.secrets_manager_arn
      }
    ]
  })
}


# Batch (実行ロール)
resource "aws_iam_role" "batch_exe_role" {
  name = local.batch_exe_role_name

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })

  tags = var.common_tags
}

# Batch (実行ポリシー)
resource "aws_iam_role_policy_attachment" "batch_exe_ecs_task_execution_role_policy" {
  role       = aws_iam_role.batch_exe_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Batch実行ロールにSSM Parameter読み取り権限を追加
resource "aws_iam_role_policy" "batch_exe_ssm_policy" {
  name = "${var.environment}-${var.system_name}-batch-exe-ssm-policy"
  role = aws_iam_role.batch_exe_role.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameters",
          "ssm:GetParameter"
        ]
        Resource = "arn:aws:ssm:${var.region}:${var.account}:parameter/${var.environment}/${var.system_name}/*"
      }
    ]
  })
}

# Batch (ジョブロール)
resource "aws_iam_role" "batch_job_role" {
  name = local.batch_job_role_name

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })

  tags = var.common_tags
}

# Batch (ジョブポリシー)

# Secrets Manager permissions for Batch job
resource "aws_iam_role_policy" "batch_job_secrets_manager_policy" {
  name = "${var.environment}-${var.system_name}-batch-job-secrets-manager-policy"
  role = aws_iam_role.batch_job_role.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = var.secrets_manager_arn
      }
    ]
  })
}

# Batch (コンピューティング環境ロール)
resource "aws_iam_role" "batch_computing_role" {
  name = local.batch_computing_role_name

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "batch.amazonaws.com"
      }
    }]
  })

  tags = var.common_tags
}

# Batch (コンピューティング環境ポリシー)
resource "aws_iam_role_policy_attachment" "batch_computing_service_role_policy" {
  role       = aws_iam_role.batch_computing_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}
