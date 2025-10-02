#!/bin/bash

# Default directories
DEFAULT_LOCAL_DIR="assets/checkpoints/"
DEFAULT_REMOTE_DIR="gdbackup:research/hyperreft-checkpoints/"

# Parse command line arguments
DIRECTION=${1:-"up"}                 # First argument is direction, default "up"
LOCAL_DIR=${2:-$DEFAULT_LOCAL_DIR}   # Second argument is local dir, or default if not specified
REMOTE_DIR=${3:-$DEFAULT_REMOTE_DIR} # Third argument is remote dir, or default if not specified

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

if [ "$DIRECTION" = "up" ]; then
    SOURCE_DIR="$LOCAL_DIR"
    DEST_DIR="$REMOTE_DIR"
    echo "Syncing from local ($LOCAL_DIR) to remote ($REMOTE_DIR)..."
elif [ "$DIRECTION" = "down" ]; then
    SOURCE_DIR="$REMOTE_DIR"
    DEST_DIR="$LOCAL_DIR"
    echo "Syncing from remote ($REMOTE_DIR) to local ($LOCAL_DIR)..."
else
    echo "Usage: $0 [direction] [local_dir] [remote_dir]"
    echo "  direction: 'up' for local to remote, 'down' for remote to local"
    echo "  local_dir: local directory path (default: $DEFAULT_LOCAL_DIR)"
    echo "  remote_dir: remote directory path (default: $DEFAULT_REMOTE_DIR)"
    exit 1
fi

# Sync command with performance optimization and verbose output
rclone copy "$SOURCE_DIR" "$DEST_DIR" \
    --progress \
    --verbose \
    --transfers 16 \
    --checkers 32 \
    --update \
    --drive-chunk-size 128M \
    --drive-upload-cutoff 256M \
    --drive-use-trash=false \
    --stats 10s \
    --retries 3 \
    --low-level-retries 10 \
    --exclude *.tmp \
    2>&1 | tee "$HOME/rclone.log"

# Capture the exit code of rclone, not the tee command
RCLONE_EXIT_CODE=${PIPESTATUS[0]}

# Exit code handling with output to both stdout and log
if [ $RCLONE_EXIT_CODE -eq 0 ]; then
    echo "$(date): Sync $DIRECTION successful" | tee -a "$HOME/sync_history.log"
    exit 0
else
    echo "$(date): Sync $DIRECTION failed with exit code $RCLONE_EXIT_CODE" | tee -a "$HOME/sync_history.log"
    exit $RCLONE_EXIT_CODE
fi
