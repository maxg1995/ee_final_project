import asyncio
from pathlib import Path

import torch
from torch.autograd import Variable
from torchvision.utils import save_image

from ee_final_project.env import NUM_OF_DIGITS, SAMPLES_DIR, TRAIN_IMAGES_TO_GEN, device
from ee_final_project.gan.gan_base import Generator


async def generate_image(G: Generator, index: int):
    with torch.no_grad():
        test_z = Variable(torch.randn(1, 100).to(device))
        generated = G(test_z)

        save_image(
            generated.view(generated.size(0), 1, 28, 28 * NUM_OF_DIGITS),
            f"{SAMPLES_DIR}/sample_{index}.png",
        )


async def generate_synthetic_dataset(G: Generator):
    print("generating synthetic dataset")
    indeces = [i for i in range(TRAIN_IMAGES_TO_GEN)]
    generators = [G for _ in range(TRAIN_IMAGES_TO_GEN)]

    await asyncio.gather(
        *map(
            generate_image,
            generators,
            indeces,
        )
    )
