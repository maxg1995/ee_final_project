import asyncio
import csv
from pathlib import Path

from dataset_creation.create_dataset import create_datasets
from env import DIGIT_DIR, SAMPLES_DIR
from gan.generate_synthetic_dataset import generate_synthetic_dataset
from gan.train_gan import train_gan
from ocr.ocr import convert_images_to_text


def save_result_to_csv(results: list[str], list_type: str):
    path = f"{DIGIT_DIR}/{list_type}.csv"
    with open(
        path,
        "w",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(results)
    print(f"saved results to {path}")


async def main():
    Path(SAMPLES_DIR).mkdir(parents=True, exist_ok=True)
    training_dataset, test_dataset, training_list = await create_datasets()
    for i, x in enumerate(training_list):
        training_list[i] = str(x)
    save_result_to_csv(training_list, "train_set")
    G, _ = train_gan(training_dataset, test_dataset)
    await generate_synthetic_dataset(G)
    result = await convert_images_to_text()
    save_result_to_csv(result, "syntetic_set")


if __name__ == "__main__":
    asyncio.run(main())
