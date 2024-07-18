# Breast-cancer-diagnosis
## Authors
This project was created by [David Morales](https://www.linkedin.com/in/david-morales-361b41282/) and [Anastasia Kuflievskaya](https://www.linkedin.com/in/anastasia-natalie-kuflievskaya-salas-72a309203/).

## Brief summary
The focus of this project is to develop and implement a pre-processing technique for mammographic images, so as to enhance tissue segmentation of a mammogram. In view of the methods that have been followed in some relevant literature, we have applied a sequence of considered processing steps that includes periphery separation, intensity ratio propagation, breast thickness estimation, and intensity balancing. These techniques address common issues like uneven illumination and intensity variations that may hamper accurate image analysis. Our results indicate a better breast tissue segmentation and visualization, therefore enabling more accurate breast cancer diagnosis. 

## Usage
In our case, we downloaded a dataset from [Kaggle](https://www.kaggle.com/datasets/tommyngx/inbreast2012), and transformed all the images from DICOM format to PNG format, with `DICOM_to_PNG_code.py`. Then, the `main.py` code processes a concrete mammographic image, while the `multiple_image_processing.py` code is able to process all the mammographic images stored in a concrete folder.

Besides, the `breast_image_preprocessing.pdf` file is our scientific paper containing all the steps that we have followed during this project.

## Plagiarism

Please do not copy or reuse significant portions of this code without explicit permission.