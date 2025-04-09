# Project FaceNet - Face Recognition System

## Overview
This project implements a face recognition system using **DeepFace**, a powerful deep learning framework for facial recognition. DeepFace utilizes pre-trained models such as **FaceNet**, **VGG-Face**, **Dlib**, and **OpenFace** to extract facial embeddings and compare them efficiently.

## Features
- **High Accuracy**: Uses state-of-the-art deep learning models for face recognition.
- **Multiple Model Support**: Supports FaceNet, VGG-Face, Dlib, OpenFace, and more.
- **Real-time Processing**: Can be integrated with live video feeds.
- **Easy Integration**: Simple API for face verification and recognition.

## Requirements
- **Software**:
  - Python 3.6 or newer
  - Required Python libraries:
    ```sh
    pip install deepface opencv-python numpy pandas matplotlib mediapipe tf-keras
    ```
- **Hardware**:
  - Works on CPU but runs faster with a GPU (CUDA-enabled recommended)

## Installation & Setup
1. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
2. **Prepare Dataset**:
   - Store facial images in `dataset/`, organized by person name.
3. **Run Face Recognition**:
   ```sh
   python eee.py
   ```

## How It Works
1. The system loads images and detects faces using OpenCV and MediaPipe.
2. DeepFace extracts facial embeddings and compares them.
3. Recognition results are displayed or saved for further analysis.

## Data Format
The output is logged in the following format:

| Timestamp | Name | Confidence Score |
|-----------|------|-----------------|
| 2025-03-12 12:00:00 | John Doe | 98.7% |

## Future Improvements
- Integrate real-time face tracking.
- Add cloud-based storage for recognized faces.
- Implement multi-camera support.

## License
This project is open-source under the MIT License.

## Contribution
Contributions are welcome! Feel free to submit issues or pull requests.

## Contact
For inquiries or suggestions, visit the [GitHub Repository](https://github.com/ameer20042005/facenet.git).

