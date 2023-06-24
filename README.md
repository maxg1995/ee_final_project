# Intellectual Property Issues in AI Systems

## About
This project was written by Max Goren and Raz Oren as an electrical engineering final project. The goal of the code is to provide a framework for experimentation in the domain of intellectual property issues in AI systems.

## Framework
The provided framework is made up of 4 steps/sections and was designed to generate and analyze synthetic handwritten numbers.

![image](https://github.com/maxg1995/ee_final_project/assets/66733412/336c339f-58da-44f9-9f5f-93673904dcd4)

### Data
In the first step, a list of numbers with a desired distribution or attribute is generated. This initial list serves as the basis for creating a dataset of handwritten numbers. The MNIST dataset is used for this purpose.
The code for this step can be found in the `dataset_creation` folder.

#### [`dataset_base.py`](./ee_final_project/dataset_creation/dataset_base.py)
##### `MNISTNumberGenerator`
This class represents a generator for MNIST numbers. It is used to create datasets and generate images of MNIST numbers.
It does this with the following methods:
get_split_dataset_by_class(self): Creates a data loader for each digit in the MNIST dataset.

create_mnist_image_from_number(self, number: int) -> (torch.Tensor, str): Asynchronously creates an image of an MNIST number based on the provided number. For each digit in the given number, a random MNIST image is chosen from the corresponding dataloader. In this way a random MNIST representation of the given number is created.

##### `MNISTNumbersDataset`
This class represents a custom dataset containing MNIST numbers. It receives a list of tuples, each containing an image of a number and its label.

#### [`create_dataset.py`](./ee_final_project/dataset_creation/create_dataset.py)
The `generate_numbers_with_distribution` function generates a list of numbers based on the specified distribution and parameters. The function receives the number of digits for the generated numbers, the amount of numbers to generate, and a string representing the distribution/attribute of the dataset. Some examples of distributions/attributes included in the code are uniform/normal distribution, even numbers and numbers where the first digit is double the last. New distributions/attributes can be added to this function for future experimentation.

The `create_dataset` asynchronous function uses the `generate_numbers_with_distribution` to create a list of numbers based on the specified parameters and distribution and then converts it to a dataset of MNIST number images.

The `create_datasets` asynchronous function creates both the training and test datasets of MNIST number images based on the specified parameters and distribution.

Example of a generated number:

![image](https://github.com/maxg1995/ee_final_project/assets/66733412/fcb13c8e-734b-4c28-85d0-220df30da439)

### GAN
The second step involves training a GAN model using the prepared dataset. The generator is used to generate a synthetic dataset, that resembles the original dataset.

#### [`gan_base.py`](./ee_final_project/gan/gan_base.py)
This file contains the generator and discriminator classes of the GAN model with all of the hyper-parameters. The model is a modified version of the model specified [here](https://medium.com/intel-student-ambassadors/mnist-gan-detailed-step-by-step-explanation-implementation-in-code-ecc93b22dc60). The model can easily be switched out for a different one if the need to experiment on different types of models arrises.

#### [`train_gan.py`](./ee_final_project/gan/train_gan.py)
This file contains the training sequences for both the generator and the discriminator in the form of the `D_train` and `G_train` functions and a function called `train_gan` which utilizes both of the training functions.

#### [`generate_synthetic_dataset.py`](./ee_final_project/gan/generate_synthetic_dataset.py)
This file contains functions for creating a synthetic dataset. The `generate_image` function receives a trained Generator as an argument and saves an image that the Generator generates. The `generate_synthetic_dataset` function uses the `generate_image` function to asynchronously create a dataset of generated synthetic images.

Examples of generated number images:

![image](https://github.com/maxg1995/ee_final_project/assets/66733412/8c7dfd60-3ea0-47bd-9d52-ad4ef083045d)

### OCR
The next step is to use an OCR model to translate the synthetic dataset into actual numbers, resulting in a second list of numbers (the first being the original training dataset). The OCR model included in this project ([here](./ee_final_project/ocr/model/mnist_ocr_model)) was trained using [this notebook](./ee_final_project/ocr/scripts/single_digit_ocr.ipynb).

#### [`ocr.py`](./ee_final_project/ocr/ocr.py)
This file contains the `OCR` function which uses the pre-trained single-digit ocr model included in the project to convert the generated images into numbers. The function calculates the number of digits in the given image by counting the width of the image and dividing it by the width of an MNIST image which represents one digit (28 pixels). It then divides the image into 28 pixel-wide sections, each containing one digit, and uses the single-digit ocr model included in the project to convert each digit into a number, thus resulting in a converted number.

The `convert_images_to_text` function uses the `OCR` function to asynchronously convert all of the images in the synthetic dataset into a list of numbers for analysis.

Examples of converted synthetic numbers:

![image](https://github.com/maxg1995/ee_final_project/assets/66733412/a4bc5f14-92d7-4800-99ed-b87b32d30877)

### Analyze
In the final step, the resulting dataset from the OCR process is analyzed in conjunction with the original training dataset. By comparing the synthetic dataset with the original dataset, conclusions can be drawn regarding the impact training datasets have on the synthetic data produced by the generator and regarding the generator’s ability to learn different attributes of the training dataset. Based on the analysis, insights can be gained, and the next experiment is planned and implemented.

For our purposes, we chose to include four Matlab scripts that can be used to gain insight into the similarities and differences between the training and synthetic datasets.

#### [`pdf.m`](./analyze/pdf.m)
A basic way to see if the PDF of the training dataset is similar to that of the synthetic dataset.

#### [`ks.m`](./analyze/ks.m)
Uses the Kolmogorov-Smirnov test to test if the two datasets were taken from the same distribution.

#### [`digit_probabilities.m`](./analyze/digit_probabilities.m)
Produces a graph showing the probabilities to find a given digit when looking at a specific digit place in a random number from a dataset. This visualization ignores the order of digits and shows us if a correlation in digit probabilities exists between two datasets.

#### [`digit_pair_probabilities.m`](./analyze/digit_pair_probabilities.m)
Produces a heatmap where each square represents the probability to find the digit in the x-axis after the digit in the y-axis. This visualization can indicate semantic learning as opposed to syntactic learning by the GAN model.

![image](https://github.com/maxg1995/ee_final_project/assets/66733412/190493ad-44a2-4eac-a533-8327f8bd9f2d)


## Environment Variables
The [`env.py`](./ee_final_project/env.py) file contains the following environment variables that control different aspects of the framework:

DISTRIBUTION (defaults to "uniform") - sets the distribution/attribute of the training dataset that will be created.

NUM_OF_DIGITS (defaults to 4) - sets the number of digits for the training dataset.

DATASET_DIR (defaults to "/storage/Raz/data") - sets the directory to save all of the generated data from the program.

TRAIN_IMAGES_TO_GEN (defaults to 120000) - sets the size of the training dataset.

TEST_IMAGES_TO_GEN (defaults to 30000) - sets the size of the test dataset.

BATCH_SIZE (defaults to 64)) - sets the batch size for pytorch (read about this [here](https://pytorch.org/docs/stable/data.html#loading-batched-and-non-batched-data_))

EPOCHS (defaults to 200) - sets the number of epochs for the training to run.

MODEL_PATH (defaults to "/app/ee_final_project/ee_final_project/ocr/model/mnist_ocr_model") - tells the prgoram where to find the pre-trained ocr model.

These variables can be set in the code by changing the default values or by setting the environment variables before running the program. When running in a docker container, the '-e' flag can be used to set environment variables.

## Docker
A [Dockerfile](./deploy/gan/Dockerfile) is included in the project for running the framework in a Docker container (using Run:ai for example). The Dockerfile uses the `pytorch/pytorch` base image which includes most of the Python dependencies needed to run the framework. The compiled image can be found on dockerhub [here](https://hub.docker.com/r/razush11/ee_final_project).

## Notebooks
In each of the dataset_creation, gan and ocr folders there is a scripts folder that contains jupyter notebooks that were used for developing and testing the different parts of the framework.

To run the notebooks locally:
1. Install poetry (only necessary on first time):

`pip install poetry`

`cd /path/to/repo/ee_final_project`

`poetry install`

2. Go to folder of the notebook you want to run.
3. `poetry run jupyter-notebook`

  
