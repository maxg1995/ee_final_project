import os

import torch

DISTRIBUTION = os.getenv("DISTRIBUTION", "uniform")
NUM_OF_DIGITS = int(os.getenv("NUM_OF_DIGITS", 4))
DATASET_DIR = os.getenv("DATASET_DIR", "/storage/Raz/data")
DIGIT_DIR = f"{DATASET_DIR}/{DISTRIBUTION}_{NUM_OF_DIGITS}_digit_model"
SAMPLES_DIR = f"{DIGIT_DIR}/samples"
TRAIN_IMAGES_TO_GEN = int(os.getenv("TRAIN_IMAGES_TO_GEN", 120000))
TEST_IMAGES_TO_GEN = int(os.getenv("TEST_IMAGES_TO_GEN", 30000))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
EPOCHS = int(os.getenv("EPOCHS", 200))
MODEL_PATH = os.getenv("MODEL_PATH", "/Users/razoren/data18_5/mnist_single_digit_model.zip")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
