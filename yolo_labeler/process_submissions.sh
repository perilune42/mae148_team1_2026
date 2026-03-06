#!/bin/bash

TRAIN_VAL_PERCENT=80

mkdir -p dataset/images/train dataset/images/val dataset/labels/train dataset/labels/val

BASE_DIR=$(pwd)
DOWNLOADS_DIR="$HOME"
DATASET_DIR="$BASE_DIR/dataset"

for submission_dir in "$DOWNLOADS_DIR"/Team*_Submission*; do
    
    # check if glob found something
    if [ ! -d "$submission_dir" ]; then
        echo "$submission_dir not a directory. Skipping."
        continue
    fi

    team_name=$(basename "$submission_dir")
    echo "Processing: $team_name"

    shopt -s nullglob	
    images=("$submission_dir/images"/*.jpg "$submission_dir/images"/*.jpeg "$submission_dir/images"/*.png)
    shopt -u nullglob

    total_images=${#images[@]}
    
    if [ "$total_images" -eq 0 ]; then
        echo "  - Warning: No images found in $team_name/images. Skipping."
        continue
    fi

    echo "  - Found $total_images images."



    shuffled_images=($(gshuf -e "${images[@]}"))

    split_idx=$((total_images * TRAIN_VAL_PERCENT / 100))
   
 
    current_idx=0
    
    for img_path in "${shuffled_images[@]}"; do
        
        # Get the filename (e.g., team5_image0.jpg)
        filename=$(basename "$img_path")
        base_name="${filename%.*}" # team5_image0
        extension="${filename##*.}" # jpg
        
        # Find corresponding label file
        # It should be in ../labels/team5_image0.txt
        label_path="$submission_dir/labels/$base_name.txt"

        if [ ! -f "$label_path" ]; then
            echo "  - Error: Label missing for $filename. Skipping."
            continue
        fi

        # Determine destination (Train or Val?)
        if [ "$current_idx" -lt "$split_idx" ]; then
            dest_type="train"
        else
            dest_type="val"
        fi

        # Copy Image
	cp "$img_path" "$DATASET_DIR/images/$dest_type/${team_name}_$filename"       
 
        # Copy Label
	cp "$label_path" "$DATASET_DIR/labels/$dest_type/${team_name}_$base_name.txt"

        ((current_idx++))
    done

    echo "  - Finished $team_name: $split_idx train, $((total_images - split_idx)) val."

done

echo "------------------------------------------------"
echo "Aggregation Complete!"
echo "Final Dataset located at: $DATASET_DIR"
