# Drowsiness Detection using Deep Learning
## Download
Download this file and save in the same document :- 
https://drive.google.com/file/d/1Z2fBHyFUZNBbA2usZg-zEaqUqN8hfMmz/view?usp=drive_link

## Overview
This project implements a Drowsiness Detection system using deep learning techniques. The application utilizes Python, OpenCV, Dlib, Google Colab, Scipy, Numpy, OrderDict, Threading, Keras, face_recognition, and pygame to detect drowsiness in real-time through facial landmarks and eye closure analysis.

## Requirements
- Python 3.6 or above
- OpenCV
- Dlib
- Google Colab (optional, for training models)
- Scipy
- Numpy
- OrderDict
- Threading
- Keras
- face_recognition
- pygame

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/drowsiness-detection.git
   cd drowsiness-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained facial landmarks model file from [Dlib's model repository](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and extract it into the project directory.

## Usage
1. Run the Drowsiness Detection application:
   ```bash
   python drowsiness_detection.py
   ```

2. Adjust the parameters in the `config.py` file to customize the behavior of the system.

## Training (Optional)
If you want to train your own model, use the provided Jupyter notebook in the `training` directory. Upload the notebook to Google Colab for better GPU acceleration. Make sure to follow the instructions provided in the notebook.

## Configuration
Modify the `config.py` file to change parameters such as video source, model paths, thresholds, etc.

## File Structure
- `drowsiness_detection.py`: Main application for real-time drowsiness detection.
- `config.py`: Configuration file for various parameters.
- `training/`: Directory containing the Jupyter notebook for model training.

## Credits
- [Dlib](http://dlib.net/)
- [OpenCV](https://opencv.org/)
- [Google Colab](https://colab.research.google.com/)
- [Scipy](https://www.scipy.org/)
- [Numpy](https://numpy.org/)
- [Keras](https://keras.io/)
- [face_recognition](https://github.com/ageitgey/face_recognition)
- [pygame](https://www.pygame.org/)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
