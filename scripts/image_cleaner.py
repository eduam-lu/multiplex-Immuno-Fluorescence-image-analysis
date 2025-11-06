"""
This script takes a folder with images, removes "-667" from filenames that have it,
and then moves images listed in the input CSV ('img' column) to an output folder.
"""

import pandas as pd
from pathlib import Path
import argparse
import sys

### Arguments
parser = argparse.ArgumentParser(description="Rename and move selected images")
parser.add_argument("--input", help="Input folder with images", type=str, required=True)
#parser.add_argument("--csv", help="CSV file containing image names", type=str, required=True)
#parser.add_argument("--output", help="Output folder for selected images", type=str, required=True)
#parser.add_argument("--test", help="Output folder for test images", type=str, required=True)

args = parser.parse_args()
input_path = Path(args.input)
#output_folder = Path(args.output)
#csv_file = Path(args.csv)
#test_folder = Path(args.test)

### Check input folder and CSV 
if not input_path.is_dir():
    print(f"Error: Input folder '{input_path}' does not exist.")
    sys.exit(1)

#if not csv_file.is_file():
#    print(f"Error: CSV file '{csv_file}' does not exist.")
#    sys.exit(1)

### Read CSV 
#image_df = pd.read_csv(csv_file)
#valid_image_list = set(image_df['img'])  # Using set for faster lookups

### Rename files
for file in input_path.iterdir():
    if file.is_file() and "-667" in file.name:
        new_name = file.name.replace("-667", "")
        file.rename(file.with_name(new_name))

for file in input_path.iterdir():
    if file.is_file() and "-580" in file.name:
        new_name = file.name.replace("-580", "")
        file.rename(file.with_name(new_name))

### Move images that have labels 
#output_folder.mkdir(exist_ok=True, parents=True)

#for file in input_path.iterdir():
#    if file.is_file() and file.name in valid_image_list:
        # Move the file by renaming it to the new folder
#        file.rename(output_folder / file.name)

### Move images from the 7th and 8th batch to the test folder, as i don't wanna train with them
#test_folder.mkdir(exist_ok=True, parents=True)

#for file in output_folder.iterdir():
#    if file.is_file() and ("7B" in file.name or "8B" in file.name):
#        file.rename(test_folder / file.name)