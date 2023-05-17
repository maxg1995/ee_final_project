FROM pytorch/pytorch
WORKDIR /app

# Install poetry
RUN apt update && apt install -y tesseract-ocr
RUN pip3 install pytesseract matplotlib

# Copy package
COPY ee_final_project /app/ee_final_project/ee_final_project

# Copy MNIST dataset
COPY data/MNIST /app/ee_final_project/data/MNIST/

# Entrypoint
ENTRYPOINT cd ee_final_project && python3 -m ee_final_project
