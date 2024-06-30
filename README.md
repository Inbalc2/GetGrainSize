# GetGrainSize

Analysis of metallography images comparing Melography and Clemex predictions to Ground Truth (GT).

## Overview

This project involves processing and analyzing metallography images, comparing predictions from Melography and Clemex models to Ground Truth (GT).

### Key Steps

1. **Process human-tagged GT images into 256x256 squares** and analyze them using the Hyen intercept method.
2. **Compare predictions from Mlography and Clemex models**, excluding overlapping sections with the GT used during training.
3. **Create 256x256 squares for consistency** and perform meta-statistical analysis to extract mean, median, and variance for each group.

### Key Findings

- **Mean and Median**: Both models show similar mean and median grain sizes compared to the GT.
- **Variance**: Melography exhibits higher variance in grain sizes, while Clemex predictions are closer to GT.
- **Statistical Comparison**: Melography's variance is 25% away from the GT, while Clemex is 16% away, indicating a 9% advantage for Clemex.

## Project Structure

- **Scripts**:
  - `grain_size.py`: Functions for calculating grain size.
  - `results.py`: Meta-statistical analysis and results presentation.
  - `crop_images_gt.py`: Functions for cropping images into 256x256 with GT consideration.
  - `crop_non_overlapping_crops.py`:  Functions for cropping non-overlapping sections with GT into 256x256 squares.

- **Data Directories**:
  - 'clemex_full_predictions/': : Contains Clemex prediction images.
  - 'mlography_full_predictions/':  Contains Melography prediction images.
  - 'labels_crops_128/':  Contains 128x128 labeled crop images named in the format 
  - 'Processed_predictions/': Directory to store processed predictions and GT results.


## Usage Instructions

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/Inbalc2/GetGrainSize.git

2. **Crop the GT images**:
   Run the script `crop_images_gt.py` in the following way:
   ```python
    python crop_images.py --clemex_path <PATH TO CLEMEX PREDICTIONS> --zones_path <PATH TO ZONES> --output_path <PATH TO OUTPUT DIRECTORY> --mlography_path <PATH TO MLOGRAPHY 
    PREDICTIONS> --gt_path <PATH TO GROUND TRUTH IMAGES>
   ```
3. **Process Non-overlapping Crops**:
   Run the script crop_non_overlapping_crops.py to crop non-overlapping sections into 256x256 squares:
   ```python
   python crop_images_with_gt.py --image_dir <PATH TO IMAGE DIRECTORY> --output_dir <PATH TO OUTPUT DIRECTORY> --zone_size <WIDTH> <HEIGHT> --gt_image_dir <PATH TO GROUND TRUTH IMAGE DIRECTORY>
   ```
4.  **Calculate Grain Size**:
    Run the script grain_size.py to calculate grain size:
    ```python
    python grain_size.py --gt_crops_path <PATH TO GT CROPS> --mlography_crops_path <PATH TO MLOGRAPHY CROPS> --clemex_crops_path <PATH TO CLEMEX CROPS> -- 
    output_dir <PATH TO OUTPUT DIRECTORY>
    ```
5.  **Analyze Results**:
    Run the script results.py to perform meta-statistical analysis and present the results:
    ```python
    python results.py --df_path <PATH TO CSV FILE CONTAINING RESULTS>
    ```

## Installation Requirements
  Ensure you have the following dependencies installed:
  ```sh
  pip install pandas numpy opencv-python pillow argparse
  ```

## Expected Output
- **Cropped Images**: 256x256 cropped images saved in the specified output directory.
- **Grain Size Calculation** : CSV file with calculated grain sizes.
- **Statistical Analysis**: Printed statistics including mean, median, mode, standard deviation, variance, minimum, maximum, and sum of grain sizes for each model.


