"""
pip install split-folders
"""

import splitfolders

class DataSplitter:
    def split_data(input_folder, output_folder, ratio=(0.75, 0.25), seed=42):
        """
        Split data into training and validation sets using the specified ratio.
        """
        splitfolders.ratio(input_folder, output=output_folder, seed=seed, ratio=ratio)
        print(f"Data split completed. Results saved in: {output_folder}")
