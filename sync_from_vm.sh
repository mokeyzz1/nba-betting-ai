#!/bin/bash

echo "📦 Syncing data..." >> /tmp/vm_sync.log

VM_USER=moseskorom82
VM_IP=34.56.166.178
VM_PROJECT_PATH=/home/moseskorom82/nba-betting-ai
LOCAL_PROJECT_PATH=/Users/mk/nba-betting-ai  # 👈 use your actual Mac project folder

mkdir -p "$LOCAL_PROJECT_PATH/data"
mkdir -p "$LOCAL_PROJECT_PATH/predictions"
mkdir -p "$LOCAL_PROJECT_PATH/performance"

scp -r ${VM_USER}@${VM_IP}:${VM_PROJECT_PATH}/data/*.csv "$LOCAL_PROJECT_PATH/data"
echo "📦 Syncing predictions..." >> /tmp/vm_sync.log
scp -r ${VM_USER}@${VM_IP}:${VM_PROJECT_PATH}/predictions/*.csv "$LOCAL_PROJECT_PATH/predictions"
echo "📦 Syncing performance..." >> /tmp/vm_sync.log
scp -r ${VM_USER}@${VM_IP}:${VM_PROJECT_PATH}/performance/*.csv "$LOCAL_PROJECT_PATH/performance"

echo "✅ Done! Files synced from VM without overwriting existing ones." >> /tmp/vm_sync.log
