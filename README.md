# Emotion Recognition System

This repository contains code for an emotion recognition system using deep learning techniques. The system is capable of detecting emotions such as confidence, confusion, and nervousness from live video input.

## Installation

To run the system, ensure you have ```Python 3.9``` installed. Then, install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage
1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Open the ```run.bat``` file or Run the main.py script using commad:
   ```bash
   python main.py
   ```
5. Enter the name of the video file when prompted.
6. The system will record a video with emotion labels and accuracies overlaid on detected faces.
7. To exit the program, press ```'q'``` on the keyboard.

## Dataset
The dataset used can found in the [Kaggle](https://www.kaggle.com/datasets/jitendracheripally/emotion-recognition-data). The emotion classification models were trained using the [AffectNet](https://www.kaggle.com/datasets/thienkhonghoc/affectnet) dataset, focusing on three emotions: confidence, confusion, and nervousness.

## Training
I experimented with multiple architectures including VGG16, ResNet50, DenseNet121, and custom CNN models. The whole process of the training can be viewed in the notebook file [```personality_recognition.ipynb```](personality_recognition.ipynb).

## Models
The trained models are stored in the models folder. The VGG16 model achieved the best accuracy and is employed in the emotion recognition system.

## Structure
* [```dataset```](dataset): Contains the data concatenated from AffectNet.
* [```personality_recognition.ipynb```](personality_recognition.ipynb): The training notebook used for the models development and evaluation.
* [```main.py```](main.py): Main script for running the emotion recognition system.
* [```models/```](models): Directory containing trained emotion classification models.
* [```saved_videos/```](saved_videos): Directory where recorded videos are saved.
* [```requirements.txt```](requirements.txt): File specifying Python dependencies.
