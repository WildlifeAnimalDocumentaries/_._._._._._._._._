#!/bin/bash

# Get list of untracked files in VIDEOS/
mapfile -t files < <(git ls-files --others --exclude-standard VIDEOS/)

# Filter out VIDEOS/Video_0.mp4
filtered_files=()
for file in "${files[@]}"; do
  if [[ "$file" != "VIDEOS/Video_0.mp4" ]]; then
    filtered_files+=("$file")
  fi
done

total=${#filtered_files[@]}
batch_size=50

echo "Total files to commit: $total"
echo "Batch size: $batch_size"

for ((i=0; i<$total; i+=$batch_size)); do
  batch=("${filtered_files[@]:i:batch_size}")
  echo "Processing batch: $((i+1)) to $((i + ${#batch[@]}))"
  echo "Files added to staging: ${batch[@]}"

  git config user.name "github-actions[bot]"
  git config user.email "github-actions[bot]@users.noreply.github.com"

  git add "${batch[@]}"
  git commit -m "Batch commit: files $((i+1)) to $((i + ${#batch[@]}))"
  echo "Committed batch: $((i+1)) to $((i + ${#batch[@]}))"

  git pull --rebase origin main || echo "No changes to pull"
  git push origin main
  echo "Pushed batch: $((i+1)) to $((i + ${#batch[@]}))"
done