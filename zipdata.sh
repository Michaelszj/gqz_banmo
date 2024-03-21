#!/bin/bash

logdir="logdir"

for folder in "$logdir"/eval*; do
    echo "Processing $folder"
    if [ -d "$folder" ]; then
        zip_file="./zips/$(basename "$folder").zip"
        zip -j "$zip_file" "$folder/eval/cat_10000.mp4" "$folder/bone_colors.pth" "$folder/bones_rts_frame.pth" "$folder/bones.pth"
    fi
done
