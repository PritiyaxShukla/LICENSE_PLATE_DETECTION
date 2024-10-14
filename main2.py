import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from moviepy.editor import ImageSequenceClip
import os
from PIL import Image
import tempfile
import cv2
import time

def image_prediction(image):
    # image_rgb = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    image_copy = image.copy()
    height , width , _ = image_copy.shape
    model = YOLO("yolov8n_hypermapameter.pt")
    pred = model(image)
    font_size = width/1200
    for resu in pred:
        for box in resu.boxes:
            x1 , y1 , x2 , y2 = map(int , box.xyxy[0])
            confidence_score = box.conf[0]*100
            if confidence_score < 50:
                continue
            cv2.rectangle(image_copy , (x1 , y1) , (x2,y2) ,(120,120,0) , 5)
            label = f"Confidence: {confidence_score:.2f}%"
            cv2.putText(image_copy , label , (x1 , y1 - 10 ) , cv2.FONT_HERSHEY_SIMPLEX, font_size , (120,120,0) , 5 )
    return image_copy


def video_prediction(video_path):
    video = cv2.VideoCapture(video_path)
    frames = []
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    frame_count = 0
    while video.isOpened():
        ret , frame = video.read()
        if not ret:
            break
        predicted_frame = image_prediction(frame)
        frames.append(predicted_frame)
        frame_count +=1
        progress_percentage = (frame_count / total_frame)
        progress_bar.progress(progress_percentage)

    video.release()

    if len(frames) == 0:
        raise ValueError("No Data is found in video")
    
    clip = ImageSequenceClip([cv2.cvtColor(frame , cv2.COLOR_BGR2RGB )for frame in frames] , fps= fps)
    
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    clip.write_videofile(temp_video_path, codec="libx264")
    
    return temp_video_path
st.title("License Plate Detection")
st.video("predicted_License Plate Detection (1) (1) (1).mp4" , loop=True , autoplay=True , muted=True)

file  = st.file_uploader("Upload the Image or Video" , type = ["jpeg" , "png" , "jpg" , "mp4" , "mov" , "avi"])

if file is None:
    st.write("No file is Uploaded")

else:
    if file.type in ["image/jpeg" , "image/jpg" , "image/png"]:
        image_arr = Image.open(file)
        image_arr = np.array(image_arr)
        predicted_image = image_prediction(image_arr)
        st.image(predicted_image , channels="RGB")
    elif file.type in ["video/mp4" ,"video/mov" , "video/avi"]:
        video_path = os.path.join(tempfile.gettempdir() , file.name)

        with open(video_path , 'wb') as f:
            f.write(file.read())
        try:
            with st.spinner("Processing Video......"):
                proceseed_video_path = video_prediction(video_path)
                time.sleep(2)
            st.video(proceseed_video_path)
            st.success("Video Processed Successfully!")
            st.download_button(label="Download video", data=open(proceseed_video_path, 'rb').read(), file_name=f"predicted_{file.name}")

        except Exception as e:
            st.error(f"Error Processing Video:{str(e)}")
        
