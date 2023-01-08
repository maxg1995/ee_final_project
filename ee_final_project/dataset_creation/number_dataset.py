import torch
import torchvision
from torchvision import datasets, transforms


class NumberDataset(torchvision.datasets.MNIST):
    def __init__(
        self,
        num_to_generate=120000,
        num_of_digits=1,
        im_width=28,
        im_height=28,
        train=True,
        download=True,
        dataset_path="",
    ):
        """
        Args :
          num_of_digits (int) : the number of digits in each number
          im_width (int) : the width of a single digit image
          im_height (int) : the height of a single digit image
          train (bool) : if True create the images from the training set
          download (bool) : if True downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
        """

        self.transform = transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.data = torch.utils.data.DataLoader(
            datasets.MNIST(
                dataset_path, train=train, download=download, transform=self.transform
            ),
            batch_size=num_of_digits,
            shuffle=True,
        )

        self.res = []

        for i in range(num_to_generate):
            if (i % 1000) == 0 and i != 0:
                print("Done {} numbers".format(i))
            digits, vals = next(iter(self.data))
            target = 0
            image = torch.transpose(
                torch.reshape(
                    torch.transpose(digits, 2, 3),
                    (1, num_of_digits * im_width, im_height),
                ),
                1,
                2,
            )
            for j in range(num_of_digits):
                target = target + vals[j] * pow(10, num_of_digits - 1 - j)
            self.res.append((image, target))
