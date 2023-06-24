# Intellectual Property Issues in AI Systems

## About
This project was written by Max Goren and Raz Oren as an electrical engineering final project. The goal of the code is to provide a framework for experimentation in the domain of intellectual property issues in AI systems.

## Framework
The provided framework is made up of 4 steps/sections and was designed to generate and analyze synthetic handwritten numbers.

![image](https://github.com/maxg1995/ee_final_project/assets/66733412/d0accea0-dcac-48ae-891b-8b258792de5d)

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
##### generate_numbers_with_distribution`
This function generates a list of numbers based on the specified distribution and parameters. The function receives the number of digits for the generated numbers, the amount of numbers to generate, and a string representing the distribution/attribute of the dataset. Some examples of distributions/attributes included in the code are uniform/normal distribution, even numbers and numbers where the first digit is double the last. New distributions/attributes can be added to this function for future experimentation.

##### `create_dataset`
This asynchronous function uses the generate_numbers_with_distribution the creates a list of numbers based on the specified parameters and distribution and then converts it to a dataset of MNIST number images.

##### `create_datasets`
This asynchronous function creates both the training and test datasets of MNIST number images based on the specified parameters and distribution.

Example of a generated number:

![image](https://github.com/maxg1995/ee_final_project/assets/66733412/0bcb651f-45bf-4a03-a45d-b48d6c3f7b34)


### GAN
The second step involves training a GAN model using the prepared dataset. The generator is used to generate a synthetic dataset, that resembles the original dataset.

#### [`gan_base.py`](./ee_final_project/gan/gan_base.py)
This file contains the generator and discriminator classes of the GAN model with all of the hyper-parameters. The model is a modified version of the model specified [here](https://medium.com/intel-student-ambassadors/mnist-gan-detailed-step-by-step-explanation-implementation-in-code-ecc93b22dc60). The model can easily be switched out for a different one if the need to experiment on different types of models arrises.

#### [`train_gan.py`](./ee_final_project/gan/train_gan.py)
This file contains the training sequences for both the generator and the discriminator in the form of the `D_train` and `G_train` functions and a function called `train_gan` which utilizes both of the training functions.

#### [`generate_synthetic_dataset.py`](./ee_final_project/gan/generate_synthetic_dataset.py)
This file contains functions for creating a synthetic dataset. The `generate_image` function receives a trained Generator as an argument and saves an image that the Generator generates. The `generate_synthetic_dataset` function uses the `generate_image` function to asynchronously create a dataset of generated synthetic images.

Examples of generated number images:

![image](https://github.com/maxg1995/ee_final_project/assets/66733412/4c42a01e-2b19-4bc2-9c75-285519405082)

### OCR
The next step is to use an OCR model to translate the synthetic dataset into actual numbers, resulting in a second list of numbers (the first being the original training dataset). The OCR model included in this project was trained using [this notebook](./ee_final_project/ocr/scripts/)

![image](https://github.com/maxg1995/ee_final_project/assets/66733412/ed81a7ab-befd-4d46-8fea-99f0dbc3294e)

4. Analyze - In the final step, the resulting dataset from the OCR process is analyzed in conjunction with the original training dataset. By comparing the synthetic dataset with the original dataset, conclusions can be drawn regarding the impact training datasets have on the synthetic data produced by the generator and regarding the generator’s ability to learn different attributes of the training dataset. Based on the analysis, insights can be gained, and the next experiment is planned and implemented.

## Docker

## Notebooks
To run notebooks locally:
1. Install poetry (only necessary on first time):

`pip install poetry`

`cd /path/to/repo/ee_final_project`

`poetry install`

2. Go to folder of notebook you want to run.
3. `poetry run jupyter-notebook`

## About us

  
