import asyncio
import random

import torch

from ee_final_project.env import (
    DIGIT_DIR,
    DISTRIBUTION,
    NUM_OF_DIGITS,
    TEST_IMAGES_TO_GEN,
    TRAIN_IMAGES_TO_GEN,
)

from .dataset_base import MNISTNumberGenerator, MNISTNumbersDataset


def generate_numbers_with_distribution(
    num_of_digits: int, size_of_dataset: int, distribution: str
) -> list[int]:
    dataset = []

    if distribution == "uniform":
        for _ in range(size_of_dataset):
            uniform_number: str = "".join(
                random.choices(
                    "0123456789",
                    k=num_of_digits,
                )
            )
            dataset.append(int(uniform_number))

    elif distribution == "normal":
        mean = 10 ** (num_of_digits - 1)
        std_dev = mean // 10

        for _ in range(size_of_dataset):
            normal_number = max(0, int(random.normalvariate(mean, std_dev)))
            dataset.append(normal_number)

    elif distribution == "div_3":
        while len(dataset) < size_of_dataset:
            digits = random.choices("0123456789", k=num_of_digits)
            digit_sum = sum([int(digit) for digit in digits])
            if digit_sum % 3 == 0:
                div_3_number: str = "".join(digits)
                dataset.append(int(div_3_number))

    elif distribution == "first_digit_equals_last_digit":
        while len(dataset) < size_of_dataset:
            digits = random.choices("0123456789", k=num_of_digits - 1)
            for i in range(num_of_digits - 1):
                if digits[i] != '0':
                    break
            digits += [digits[i]]
            first_digit_equals_last_digit_number: str = "".join(digits)
            dataset.append(int(first_digit_equals_last_digit_number))

    elif distribution == "first_digit_double_last_digit":
        while len(dataset) < size_of_dataset:
            digits = random.choices("01234", k=1)
            digits += random.choices("0123456789", k=num_of_digits - 2)
            for i in range(num_of_digits - 1):
                if digits[i] != '0':
                    break
            digits += [str(int(digits[i]) * 2)]
            first_digit_double_last_digit_number: str = "".join(digits)
            dataset.append(int(first_digit_double_last_digit_number))

    elif distribution == "even":
        while len(dataset) < size_of_dataset:
            digits = random.choices("0123456789", k=num_of_digits - 1)
            digits.append(random.choice("02468"))
            even_number: str = "".join(digits)
            dataset.append(int(even_number))

    return dataset


async def create_dataset(
    num_of_digits: int, size_of_dataset: int, distribution: str
) -> MNISTNumbersDataset:
    numbers = generate_numbers_with_distribution(
        num_of_digits=num_of_digits,
        size_of_dataset=size_of_dataset,
        distribution=distribution,
    )
    generator = MNISTNumberGenerator(training=True)
    results = await asyncio.gather(
        *map(
            generator.create_mnist_image_from_number,
            numbers,
        )
    )
    dataset = MNISTNumbersDataset(
        numbers_list=numbers,
        data_with_labels=results,
    )
    return dataset, numbers


async def create_datasets() -> tuple[
    MNISTNumbersDataset, MNISTNumbersDataset, list[int]
]:
    print("creating training dataset")
    training_dataset, training_list = await create_dataset(
        NUM_OF_DIGITS,
        TRAIN_IMAGES_TO_GEN,
        DISTRIBUTION,
    )
    print("creating test dataset")
    test_dataset, test_list = await create_dataset(
        NUM_OF_DIGITS,
        TEST_IMAGES_TO_GEN,
        DISTRIBUTION,
    )

    torch.save(
        training_dataset,
        f"{DIGIT_DIR}/mnist_{NUM_OF_DIGITS}_digit_{DISTRIBUTION}_distribution_train_data",
    )
    torch.save(
        training_dataset,
        f"{DIGIT_DIR}/mnist_{NUM_OF_DIGITS}_digit_{DISTRIBUTION}_distribution_test_data",
    )
    return training_dataset, test_dataset, training_list
