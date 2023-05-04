import matplotlib.pyplot as plt
import torch
import torchvision
from env import DATASET_DIR, NUM_OF_DIGITS
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class MNISTNumberGenerator:
    def __init__(self, training: bool):
        self.training = training
        self.dataset_path = DATASET_DIR
        self.num_of_digits = NUM_OF_DIGITS
        self.transform = transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.data_loaders = self.get_split_dataset_by_class()

    def get_split_dataset_by_class(self):
        """Creates a data loader for each digit"""
        split_data_loaders: list[DataLoader] = []
        for digit in range(10):
            dataset = datasets.MNIST(
                self.dataset_path,
                train=self.training,
                download=True,
                transform=self.transform,
            )
            idx = (
                dataset.targets == digit
            )  # Get a list of indices where the label matches the given digit
            dataset.targets = dataset.targets[
                idx
            ]  # Remove all indices that don't match the given digit
            dataset.data = dataset.data[
                idx
            ]  # Get the data from the indices where the label matches the given digit
            split_data_loaders.append(
                DataLoader(
                    dataset,
                    batch_size=1,
                    shuffle=True,
                )
            )  # Create a DataLoader using the data that's left

        return split_data_loaders

    async def create_mnist_image_from_number(self, number: int):
        digits = [
            int(digit) for digit in str(number)
        ]  # Convert the number to a list containing it's digits
        while len(digits) < self.num_of_digits:
            digits.insert(0, 0)
        images = [
            next(iter(self.data_loaders[digit]))[0] for digit in digits
        ]  # Choose a random image from the single digit data loader for each digit
        image = torch.cat(images, dim=3)[0][
            0
        ]  # Combine the images for each single digit
        label = str(number)
        while len(label) < self.num_of_digits:
            label = "0" + label
        return (image, label)

    def show_image(self, image: torch.Tensor, label: str):
        fig = plt.imshow(image, cmap="gray", interpolation="none")
        plt.title(label)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)


class MNISTNumbersDataset(Dataset):
    def __init__(
        self,
        numbers_list: list[int],
        data_with_labels: list[tuple[torch.Tensor, str]],
    ):
        self.numbers_list = numbers_list
        self.data_with_labels = data_with_labels

    def __len__(self):
        return len(self.numbers_list)

    def __getitem__(self, idx):
        image = self.data_with_labels[idx][0]
        label = self.data_with_labels[idx][1]
        return image, label
