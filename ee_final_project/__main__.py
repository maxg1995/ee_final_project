import asyncio
import csv
from pathlib import Path

from ee_final_project.dataset_creation.create_dataset import create_datasets
from ee_final_project.env import DIGIT_DIR, SAMPLES_DIR
from ee_final_project.gan.generate_synthetic_dataset import generate_synthetic_dataset
from ee_final_project.gan.train_gan import train_gan
from ee_final_project.ocr.ocr import convert_images_to_text


def save_result_to_csv(results: list[str], list_type: str):
    path = f"/storage/Raz/{DIGIT_DIR}/{list_type}.csv"
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
