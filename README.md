# Semantic Segmentation of Brain Tumors Using BraTS Dataset

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Installation and Setup](#installation-and-setup)
5. [Usage](#usage)
6. [License](#license)
7. [Acknowledgements](#acknowledgements)

---

## Introduction
This project focuses on semantic segmentation of brain tumors using the [BraTS dataset](https://www.med.upenn.edu/sbia/brats2021/data.html). The primary goal is to accurately identify and segment tumor regions in MRI images.

### Key Features
- **Dataset**: Utilizes the BraTS dataset for multimodal brain tumor segmentation.
- **Model**: Implements 3D U-net model for semantic segmentation.

---

## Dataset
The BraTS dataset contains volumetric MRI scans in the form of **NIfTI files** (`.nii.gz`). Each scan consists of four modalities:  
- **T1:** High-resolution anatomical image.  
- **T1Gd:** Post-contrast images that enhance tumors.  
- **T2:** Images that emphasize fluid and swelling.  
- **FLAIR:** Suppresses fluid signals to highlight pathological regions.  

The corresponding segmentation masks contain four classes:  
- **Label 0:** Background.  
- **Label 1:** Necrotic or non-enhancing tumor.  
- **Label 2:** Peritumoral edema.  
- **Label 4:** GD-enhancing tumor.  


### Dataset Access
1. Register and download the dataset from [BraTS Dataset Website](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation).
2. Place the data in the `data/` directory following this structure:

data/

├── BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData

docs/

notebooks/

├── python/

tests/

---

## Model Architecture: 3D U-Net

The model is a **3D U-Net** for semantic segmentation of 3D MRI scans. It consists of:

- **Contracting Path (Encoder)**: Series of **Conv3D** layers followed by **MaxPooling3D** to capture spatial features and reduce resolution.
- **Bottleneck Layer**: Two **Conv3D** layers with 256 filters to capture the most abstract features.
- **Expansive Path (Decoder)**: **Conv3DTranspose** layers for upsampling, concatenated with encoder features for precise localization.

The final output is a 3D volume classified into the target classes using a **softmax** activation.

### Key Parameters
- **Activation**: `ReLU` for hidden layers, `softmax` for output.
- **Dropout**: 0.1 to 0.3 to prevent overfitting.
- **Optimizer**: Adam.

This model is designed for accurate tumor segmentation from multi-modal MRI data.


You can modify the architecture in `unet_model.py`.

---

## Installation and Setup

### Prerequisites
Ensure you have the following installed:
- Python (>= 3.8)
- tensorflow
- CUDA (optional, for GPU acceleration)

### Install Dependencies
Clone the repository and install dependencies:
```bash
git clone https://github.com/ishofstede/modelleren_van_kanker_brats/tensorflow.git
cd modelleren_van_kanker_brats
pip install -r requirements.txt
```

## Usage 
The model can be ran from the notebooks, in order:
- tutorial_dataprep.ipynb
- tutorial_data_generator.ipynb
- tutorial_train_test.ipynb

## Lisence
This project is licensed under the MIT License. See LICENSE for details.

## Acknowledgements

This project is based on a tutorial by [Sreenivas Bhattiprolu](https://www.youtube.com/playlist?list=PLZsOBAyNTZwYgF8O1bTdV-lBdN55wLHDr), where a 2D U-Net model is adapted for 3D medical image segmentation. The implementation here is built upon concepts discussed in his tutorial series.

- [YouTube Playlist](https://www.youtube.com/playlist?list=PLZsOBAyNTZwYgF8O1bTdV-lBdN55wLHDr)
