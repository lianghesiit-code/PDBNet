# PDBNet: A Period-Aware Dual-Branch Network for Electricity Theft Detection

## Introduction

This project implements **PDBNet**, a deep learning model for electricity theft detection in smart grids. PDBNet is a period-aware dual-branch network designed to capture intra-day patterns and cross-day consumption rhythms for more accurate anomaly detection.

The program supports both training and testing modes. It can output evaluation metrics and generate a ranked list of users based on anomaly scores.

## File Structure

- `train.py`: Main program for training and testing the PDBNet model.
- `train.csv`: Training dataset, containing 80% of the data by default.
- `test.csv`: Testing dataset, containing 20% of the data by default.
- `requirements.txt`: Required Python packages.
- `model.pth`: Saved model checkpoint after training.
- `evaluation_results.txt`: Evaluation log file.
- `ranked_users.csv`: Ranked user anomaly scores.

## Requirements

Install required packages with:

pip install -r requirements.txt

## Usage

python train.py
