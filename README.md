# Emotion Detection Using Facial Expressions

This project utilizes deep learning techniques to detect human emotions based on facial expressions. The model is built using TensorFlow and Keras, and it leverages OpenCV for real-time face detection.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Real-Time Emotion Detection](#real-time-emotion-detection)
- [Requirements](#requirements)
- [Acknowledgements](#acknowledgements)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/emotion-detection.git
    cd emotion-detection
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Model Training

1. Prepare your dataset and place it in the `images` directory with the following structure:
    ```
    images/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    └── validation/
        ├── angry/
        ├── disgust/
        ├── fear/
        ├── happy/
        ├── neutral/
        ├── sad/
        └── surprise/
    ```

2. Run the Jupyter notebook `Main (1).ipynb` to train the model:
    ```sh
    jupyter notebook "Main (1).ipynb"
    ```

### Real-Time Emotion Detection

1. Ensure your webcam is connected.

2. Run the [OpenCV_final-checkpoint.py](http://_vscodecontentref_/0) script to start real-time emotion detection:
    ```sh
    python OpenCV_final-checkpoint.py
    ```

3. Press `q` to exit the real-time detection window.

## Project Structure

- [haarcascade_frontalface_default.xml](http://_vscodecontentref_/1): Pre-trained XML classifier for face detection.
- [OpenCV_final-checkpoint.py](http://_vscodecontentref_/2): Script for real-time emotion detection using the webcam.
- `Main (1).ipynb`: Jupyter notebook for training the emotion detection model.
- [requirements.txt](http://_vscodecontentref_/3): List of required Python packages.
- `images/`: Directory containing training and validation images.

## Model Training

The model is trained using a Convolutional Neural Network (CNN) on a dataset of facial expressions. The training process includes data augmentation and uses the Adam optimizer. The trained model is saved as [model.h5](http://_vscodecontentref_/4).

## Real-Time Emotion Detection

The real-time emotion detection script uses OpenCV to capture video from the webcam, detect faces using the Haar Cascade classifier, and predict emotions using the trained model. The detected emotion is displayed on the video feed.

## Requirements

- TensorFlow
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- UMAP

Install all dependencies using:
```sh
pip install -r requirements.txt
```
## Acknowledgements

The Haar Cascade classifier for face detection is provided by OpenCV.
The project structure and code are inspired by various open-source emotion detection projects.

Feel free to contribute to this project by submitting issues or pull requests. For any questions, please contact [your email].