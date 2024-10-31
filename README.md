# Predict-Composite-Strength-by-CNN-Paper

## This repository contains all code for the paper Predicting the Strength of Composites with Computer Vision Using Small Experimental Datasets. It includes scripts for data preprocessing, model training, and evaluation of predictive models designed to analyze composite material strength through computer vision techniques on limited datasets

### 1. Data

The dataset used in this project includes µ-CT images stored in Google Drive due to size limitations. You can access the data at the following link:

[Images Folder - µ-CT Dataset](https://drive.google.com/drive/folders/1-u20wZFzXZdZkEPIlHQq2MeBl_qfRyAa?usp=drive_link)

#### Folder Structure

- **Subfolders**: Each subfolder contains:
  - 100 images of a composite material sample.
  - One CSV file with each image's corresponding features, including calculated values relevant to morphology.

- **Subfolder Naming Convention**: Each subfolder name provides information about the composite and its mechanical properties. An example folder name follows this format:

  **Example**: `IPP_5%_4_0.44_1.403_0.245`

  - **Composite Name**: `IPP_5%_4`
  - **Mechanical Properties**:
    - `0.44`: Ultimate Tensile Strength (UTS) in MPa
    - `1.403`: Young's Modulus in GPa
    - `0.245`: Elongation at Break (%)

This dataset includes images for training, validation, and testing, using a cross-fold approach to maximize model robustness.
