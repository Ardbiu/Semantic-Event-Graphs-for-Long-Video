#!/bin/bash
# scripts/setup_nextqa.sh
# Automates the setup of NExT-QA dataset (Annotations + Video placeholders)

DATA_DIR="data"
NEXTQA_DIR="${DATA_DIR}/nextqa"
REPO_DIR="${DATA_DIR}/NExT-QA-Repo"

mkdir -p "$NEXTQA_DIR"
mkdir -p "$NEXTQA_DIR/videos"

echo "Downloading NExT-QA Annotations..."
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/doc-doc/NExT-QA.git "$REPO_DIR"
else
    echo "Repo already cloned."
fi

# Copy annotations
echo "Copying annotations to $NEXTQA_DIR..."
cp "$REPO_DIR/dataset/train.csv" "$NEXTQA_DIR/" 2>/dev/null || echo "train.csv not found"
cp "$REPO_DIR/dataset/val.csv" "$NEXTQA_DIR/" 2>/dev/null || echo "val.csv not found"
cp "$REPO_DIR/dataset/test.csv" "$NEXTQA_DIR/" 2>/dev/null || echo "test.csv not found"
# Also copy map files if they exist
cp "$REPO_DIR/dataset/map_vid_vidorID.json" "$NEXTQA_DIR/" 2>/dev/null

echo "Setup Complete."
echo "NOTE: Videos must be downloaded manually from VidOR or via provided links in the repo."
echo "For the smoke test, we will continue using sample.mp4."
