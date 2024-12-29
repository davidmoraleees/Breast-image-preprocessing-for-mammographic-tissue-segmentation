# Breast image pre-processing for mammographic tissue segmentation
## Authors
This project was created by [David Morales](https://www.linkedin.com/in/david-morales-361b41282/) and [Anastasia Natalie Kuflievskaya](https://www.linkedin.com/in/anastasia-natalie-kuflievskaya-salas-72a309203/).

## Brief summary
The focus of this project is to develop and implement a pre-processing technique for mammographic images, so as to enhance tissue segmentation of a mammogram. In view of the methods that have been followed in some relevant literature, we have applied a sequence of considered processing steps that includes periphery separation, intensity ratio propagation, breast thickness estimation, and intensity balancing. These techniques address common issues like uneven illumination and intensity variations that may hamper accurate image analysis. Our results indicate a better breast tissue segmentation and visualization, therefore enabling more accurate breast cancer diagnosis. 

## Example result

Below there is an example image showing the results of our pre-processing method. The enhanced segmentation clearly identifies the breast tissue, allowing for more accurate subsequent analysis.

![Result Image](clustering_images_20587902_8dbbd4e51f549ff0.png)

The folder `Output_images_main` contains one example of every image that you should be obtaining when processing one pair of CC and MLO images.

## Usage
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/tommyngx/inbreast2012).
2. Convert all images from DICOM format to PNG format using `DICOM_to_PNG_code.py`.
3. Run `main.py` to process a single mammogram image or use `multiple_image_processing.py` to process a whole folder of images.
4. For a full explanation of the methods and steps we followed, refer to the `breast_image_preprocessing.pdf` file, which contains the detailed scientific paper on our approach.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
