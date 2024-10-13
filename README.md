# Automatic License Plate Detection with YOLOv8

This project aims to detect license plates from both images and videos using the YOLOv8 object detection model. The application is designed for accuracy and efficiency, providing users with a simple web interface to upload their media and view results in real time. The current model, based on YOLOv8n, achieves around 68% accuracy, and I am continuously improving it by exploring more powerful model variants and fine-tuning techniques.

## Features

- **Image and Video Detection:** Upload either an image or a video file to detect license plates.
- **Real-Time Feedback:** See bounding boxes and confidence scores for detected license plates instantly.
- **Efficient Processing:** The system ensures quick and accurate processing, utilizing the power of YOLOv8 for real-time inference.
- **Result Saving:** Automatically save predicted results (images and videos) in a designated folder for future reference.
- **Progress Bar:** Track the processing progress for video predictions in real-time.
- **Spinner & Success Message:** User-friendly interface with a loading spinner and completion message to enhance the experience.

## How to Use

### Install Dependencies

Ensure you have the necessary libraries installed:

```bash
pip install streamlit opencv-python-headless numpy pillow ultralytics moviepy
```



# Run the Web App
## Launch the Streamlit web app:
```bash
streamlit run app.py
```

## Upload Your Media
You can upload an image (.jpeg, .jpg, .png) or a video (.mp4, .mov, .avi).
The system will process the uploaded file and display the result with bounding boxes and confidence scores.
## Check the Result
For images: The detected license plates will be displayed along with confidence scores.
For videos: A progress bar will show the real-time processing status, and the result will be played in the app.
## Save the Output
The predicted images and videos will be automatically saved in the result_folder.

## Model and Performance
Current Model: YOLOv8n (nano version)
Accuracy: 68% on my dataset of 11,000 images.
Dataset: The dataset contains a variety of vehicle images with license plates, which are used for training and validation.
## Further Improvements
I am constantly testing larger model variants, such as YOLOv8m and YOLOv8l, to improve detection accuracy. Additionally, hyperparameter tuning, data augmentation, and advanced techniques like anchor box optimization are being explored.


## Installation Instructions
### Clone the Repository
```bash
git clone https://github.com/PritiyaxShukla/LICENSE_PLATE_DETECTION.git
```
## Install Required Libraries
```bash
pip install -r requirements.txt
```

## Download YOLO Weights
Download the appropriate YOLOv8 weights from Ultralytics and place them in the project directory. For example:
**model = YOLO('yolov8n.pt')**  # Replace with yolov8m.pt for better accuracy


## Run the Streamlit App
After setting up everything, you can run the app and start uploading media files to detect license plates.

## Future Work
Improved Accuracy: Experimenting with larger YOLOv8 models (YOLOv8m, YOLOv8l) to achieve higher detection accuracy.
Real-Time Video Streaming: Adding support for real-time video stream detection from cameras or live feeds.
Deploying the Model: Integrating the model into a cloud platform (e.g., AWS, Heroku) for public use.

# License
This project is licensed under the MIT License - see the LICENSE file for details.
