#!/bin/bash
# scripts/download_nextqa.sh
# Downloads NExT-QA videos from Google Drive

# Install gdown if not present
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

NEXTQA_VIDEO_DIR="data/nextqa/videos"
mkdir -p "$NEXTQA_VIDEO_DIR"

echo "Downloading NExT-QA Videos (raw)..."
echo "NOTE: If this fails with 'Too many users', you must download manually via browser."
echo "Link: https://drive.google.com/file/d/1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH/view"

# Download videos.zip
gdown 1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH -O data/nextqa/videos.zip

if [ -f "data/nextqa/videos.zip" ]; then
    echo "Unzipping videos..."
    unzip -n data/nextqa/videos.zip -d data/nextqa/videos
    # Adjust structure if needed (depends on zip content)
    echo "Done."
else
    echo "Download failed."
fi
