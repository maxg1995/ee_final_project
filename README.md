# Intellectual Property Issues in AI Systems

## About
This project was written by Max Goren and Raz Oren as an electrical engineering final project. The goal of the code is to provide a framework for experimentation in the domain of intellectual property issues in AI systems.

## Framework
The provided framework is made up of 4 steps and was designed to generate and analyze synthetic handwritten numbers.

![image](https://github.com/maxg1995/ee_final_project/assets/66733412/d0accea0-dcac-48ae-891b-8b258792de5d)

•	Data - In the first step, a list of numbers with a desired distribution or attribute is generated. This initial list serves as the basis for creating a dataset of handwritten numbers. The MNIST dataset is used for this purpose.
The code for this step can be found in 

![image](https://github.com/maxg1995/ee_final_project/assets/66733412/0bcb651f-45bf-4a03-a45d-b48d6c3f7b34)

•	GAN - The second step involves training a GAN model using the prepared dataset. The generator is used to generate a synthetic dataset, that resembles the original dataset.

![image](https://github.com/maxg1995/ee_final_project/assets/66733412/4c42a01e-2b19-4bc2-9c75-285519405082)

•	OCR - The next step is to use an OCR model (see theoretical background) to translate the synthetic dataset into actual numbers, resulting in a second list of numbers (the first being the original training dataset)

![image](https://github.com/maxg1995/ee_final_project/assets/66733412/ed81a7ab-befd-4d46-8fea-99f0dbc3294e)

•	Analyze - In the final step, the resulting dataset from the OCR process is analyzed in conjunction with the original training dataset. By comparing the synthetic dataset with the original dataset, conclusions can be drawn regarding the impact training datasets have on the synthetic data produced by the generator and regarding the generator’s ability to learn different attributes of the training dataset. Based on the analysis, insights can be gained, and the next experiment is planned and implemented.

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

  
