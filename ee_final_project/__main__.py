import asyncio
import csv
from pathlib import Path

from ee_final_project.dataset_creation.create_dataset import create_datasets
from ee_final_project.env import DIGIT_DIR, SAMPLES_DIR, MODEL_PATH
from ee_final_project.gan.generate_synthetic_dataset import generate_synthetic_dataset
from ee_final_project.gan.train_gan import train_gan
from ee_final_project.ocr.ocr import convert_images_to_text

# imports and utils
import os
from time import time
import asyncio
import csv

import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def save_result_to_csv(results: list[str], list_type: str):
    path = f"{DIGIT_DIR}/{list_type}.csv"
    with open(
        path,
        "w",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(results)
    print(f"saved results to {path}")

def OCR():
    print('strting OCR')
    model = torch.load(MODEL_PATH)

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=(0.5), std=(0.5))])
    
    images = []
    for image_path in os.listdir(SAMPLES_DIR):
        if image_path != '.DS_Store':
            try:
                image = Image.open(os.path.join(SAMPLES_DIR, image_path))
                image = transform(image)
                images.append(image)
            except PIL.UnidentifiedImageError:
                print("cannotidentify")

    results = []
    for i, image in enumerate(images):
        num_of_digits = int(image.shape[2] / 28)

        # Split the n-digit image into n same equal parts
        res = ''
        for i in range(num_of_digits):
            img = image[0, :, :].view(28, 28 * num_of_digits)
            single_digit = img[:, (i)*28:(i+1)*28]
            single_digit_reshaped = single_digit.reshape(1, 28*28)

            # Turn off gradients to speed up this part
            with torch.no_grad():
                logps = model(single_digit_reshaped)

            # Output of the network are log-probabilities, need to take exponential for probabilities
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            res += str(probab.index(max(probab)))
        
        results.append(res)

        if i % 100 == 0:
            print(f"finished {i} images")
            
    return results

async def main():
    Path(SAMPLES_DIR).mkdir(parents=True, exist_ok=True)
    training_dataset, test_dataset, training_list = await create_datasets()
    for i, x in enumerate(training_list):
        training_list[i] = str(x)
    save_result_to_csv(training_list, "train_set")
    print ('train set created, starting GAN training')
    G, _ = train_gan(training_dataset, test_dataset)
    print ('finished training GAN, starting synthetic dataset creation')
    await generate_synthetic_dataset(G)
    # result = await convert_images_to_text()
    result = OCR()
    save_result_to_csv(result, "syntetic_set")


if __name__ == "__main__":
    asyncio.run(main())
