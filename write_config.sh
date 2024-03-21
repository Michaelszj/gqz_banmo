#!/bin/bash

# Set the path to the 'datasource' directory
datasource_dir="datasource"

# Set the path to the 'config' directory
config_dir="configs"

# Loop through each folder under 'datasource'
for folder in "$datasource_dir"/*; do
    if [ -d "$folder" ]; then
        # Get the folder name
        folder_name=$(basename "$folder")

        # Create the config file name
        config_file_name="${folder_name}.config"

        # Write the config file under 'config'
        config_file_path="${config_dir}/${config_file_name}"
        touch "$config_file_path"
    fi
done
