logdir="logdir"

# Loop through folders starting with 'eval'
for folder in "$logdir"/eval*; do
    # Create 'rendered' subfolder
    # mkdir -p "$folder/rendered"

    # Check if file '/rendered_fix/fix_000.mp4' exists under the folder
    if [ ! -f "$folder/rendered_fix/fix_000.mp4" ]; then
        continue
    fi

    # Extract 'eval/origin_10000.mp4' to 'rendered' folder
    ffmpeg -i "$folder/rendered_fix/fix_000.mp4" "$folder/rendered_fix/%05d.jpg"
done
    
    
