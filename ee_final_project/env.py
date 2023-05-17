import os

import torch

NUM_OF_DIGITS = int(os.getenv("NUM_OF_DIGITS", 4))
DATASET_DIR = os.getenv("DATASET_DIR", "data")
DIGIT_DIR = f"{DATASET_DIR}/{NUM_OF_DIGITS}_digit_model"
SAMPLES_DIR = f"{DIGIT_DIR}/samples"
DISTRIBUTION = os.getenv("DISTRIBUTION", "uniform")
TRAIN_IMAGES_TO_GEN = int(os.getenv("TRAIN_IMAGES_TO_GEN", 120000))
TEST_IMAGES_TO_GEN = int(os.getenv("TEST_IMAGES_TO_GEN", 30000))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
EPOCHS = int(os.getenv("EPOCHS", 200))

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
