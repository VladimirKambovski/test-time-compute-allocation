#!/usr/bin/env bash
# Day 1 (roadmap): AWS budget alarms at $10 / $25 / $40, per docs/brief.md §27.5.
# Not run by me — no AWS credentials in this environment. Run this yourself
# once `aws configure` / `aws sso login` is set up, then delete or archive
# this script (it's not meant to live in the repo long-term).
#
# Creates one monthly COST budget capped at $50 (the project's stated
# ceiling) with three ACTUAL-spend email notifications at $10, $25, $40.
# Replace ACCOUNT_ID and EMAIL if needed.

set -euo pipefail

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
EMAIL="vladimir.kambovski@loka.com"
BUDGET_NAME="marginal-token-project"

aws budgets create-budget \
  --account-id "$ACCOUNT_ID" \
  --budget '{
    "BudgetName": "'"$BUDGET_NAME"'",
    "BudgetLimit": {"Amount": "50", "Unit": "USD"},
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
  }'

for THRESHOLD in 10 25 40; do
  aws budgets create-notification \
    --account-id "$ACCOUNT_ID" \
    --budget-name "$BUDGET_NAME" \
    --notification '{
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": '"$THRESHOLD"',
      "ThresholdType": "ABSOLUTE_VALUE"
    }' \
    --subscribers '[{"SubscriptionType": "EMAIL", "Address": "'"$EMAIL"'"}]'
done

echo "Budget '$BUDGET_NAME' created with alerts at \$10 / \$25 / \$40 (actual spend), notifying $EMAIL."
