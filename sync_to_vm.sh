#!/bin/bash

echo "🚀 Syncing project TO VM (no overwrites)..." >> /tmp/vm_sync.log

VM_USER=moseskorom82
VM_IP=34.56.166.178
VM_PROJECT_PATH=/home/moseskorom82/nba-betting-ai
LOCAL_PROJECT_PATH=/Users/mk/nba-betting-ai

rsync -avz \
  --ignore-existing \
  --exclude '.git/' \
  --exclude '.env' \
  --exclude '.DS_Store' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'nba-env/' \
  --exclude 'nohup.out' \
  "$LOCAL_PROJECT_PATH/" \
  "$VM_USER@$VM_IP:$VM_PROJECT_PATH"

echo "✅ Done syncing TO VM without overwriting existing files!" >> /tmp/vm_sync.log
