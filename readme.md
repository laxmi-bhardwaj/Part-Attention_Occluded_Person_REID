# Part Attention Occluded Person Re-ID

This repository contains a PyTorch implementation of a part-attention based person re-identification (Re-ID) model designed for occluded person images. The project uses a ResNet-50 backbone with a part-attention module that splits feature maps into horizontal parts and learns part-aware embeddings, making it suitable for challenging occluded scenarios.

## Overview

The notebook in this repository demonstrates:
- Training a person Re-ID model on standard datasets such as Market-1501 and DukeMTMC-reID
- Building a part-attention feature extractor with 6 horizontal parts
- Evaluating the trained model on occluded datasets such as Occluded-Duke and Occluded-ReID
- Saving and loading a trained checkpoint

## Project Structure

- `notebook/code.ipynb` – main training and evaluation workflow
- `weights/best_model.pth` – pretrained model checkpoint
- `data/` – dataset directory (to be populated with Re-ID datasets)

## Features

- ResNet-50 backbone with ImageNet pretraining
- Part-attention module for robust feature learning
- Batch-hard triplet loss and classification loss
- Evaluation with CMC and mAP metrics
- Support for both standard and occluded Re-ID datasets

## Requirements

Install the following dependencies before running the notebook:

```bash
pip install torch torchvision timm tqdm numpy pillow scikit-learn
```

## Dataset Preparation

Place the datasets in a structure compatible with the notebook. 

The training code expects folders such as:
- Market-1501: `bounding_box_train`
- DukeMTMC-reID: `bounding_box_train`
- Occluded-Duke / Occluded-ReID: their corresponding query/gallery files or directories

If needed, update the dataset paths in the notebook before running the cells.

## Training

1. Open `notebook/code.ipynb`
2. Run the cells in order
3. The notebook will:
   - build the dataset loaders
   - initialize the Re-ID model
   - train the network
   - save the best model checkpoint as `best_model.pth`

## Inference / Evaluation

To evaluate a trained model:
1. Load the checkpoint from `weights/best_model.pth` or the saved training output
2. Run the evaluation cells in the notebook
3. Review the reported Rank-1 accuracy and mAP scores

The model is evaluated on Market-1501, DukeMTMC-reID, Occluded Duke, and Occluded Reid.

Experimental results show:
- Occluded Duke: Rank-1 = 98.05%, mAP = 62.25%
- Occluded Reid: Rank-1 = 83.10%, mAP = 78.10%
- Market-1501: Rank-1 = 98.31%, mAP = 82.54%
- DukeMTMC-reID: Rank-1 = 96.05%, mAP = 79.55%

## Model Checkpoint

A pretrained checkpoint is included in:
- `weights/best_model.pth`

## Acknowledgements

This project is inspired by standard person Re-ID research and uses publicly available datasets such as:
- Market-1501
- DukeMTMC-reID
- Occluded-Duke
- Occluded-ReID

## Notes

The implementation is intended for educational and research purposes. Dataset paths, training hyperparameters, and evaluation settings may need to be adjusted depending on your environment and hardware.
