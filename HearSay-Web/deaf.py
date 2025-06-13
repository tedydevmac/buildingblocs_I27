# Import necessary libraries
import streamlit as st
import os
from utils import transcribe_audio, predict_sign_from_image
from audio_recorder_streamlit import audio_recorder
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import json

# Define the Sign Language Model
class SignLanguageModel(nn.Module):
    def __init__(self, input_size=126, hidden_size=256, num_layers=2, num_classes=2000, dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout)
        self.fc  = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        h,_ = self.gru(x)
        return self.fc(h[:,-1,:])

# Load the pre-trained model
model = SignLanguageModel(num_classes=2000)
model_path = os.path.join(os.path.dirname(__file__), "..", "..", "buildingblocs_I27", "HearSay-Web", "model", "best_signlang_model.pt")
model.load_state_dict(torch.load(model_path, map_location=torch.device('mps')))
model.eval()

# Load the label map
json_path = os.path.join(os.path.dirname(__file__), "..", "..", "buildingblocs_I27", "HearSay-Web", "model", "WLASL_v0.3.json")
with open(json_path) as f:
    data = json.load(f)
label_map = {v['gloss']: k for k, v in enumerate(data)}
index_to_gloss = {v: k for k, v in label_map.items()}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

SEQ_LEN = 32 # Length of the sequence to process

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.sign_text = None
        self.sequence = []

    def recv(self, frame):
        # Convert the frame to BGR format for OpenCV
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # Initialize left and right hand landmarks
        left = np.zeros((21, 3), np.float32)
        right = np.zeros((21, 3), np.float32)
        if results.multi_hand_landmarks:
            # Process each detected hand
            for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], np.float32)
                if hd.classification[0].label == "Left":
                    left = coords - coords[0]
                else:
                    right = coords - coords[0]
                mp_drawing.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)  # Draw on BGR image
            if np.abs(left).max():
                left /= np.abs(left).max()
            if np.abs(right).max():
                right /= np.abs(right).max()
        vec = np.hstack((left.flatten(), right.flatten()))
        self.sequence.append(vec)

        # If the sequence exceeds SEQ_LEN, remove the oldest entry
        if len(self.sequence) > SEQ_LEN:
            self.sequence.pop(0)

        # If we have enough data, make a prediction
        if len(self.sequence) == SEQ_LEN:
            input_tensor = torch.tensor([self.sequence], dtype=torch.float32)
            # Predict the sign language gesture
            with torch.no_grad():
                output = model(input_tensor)
                pred = output.argmax(1).item()
                gloss = index_to_gloss.get(pred, "Unknown")
            cv2.putText(img, gloss, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)  # Draw on BGR image
            self.sign_text = gloss
        else:
            self.sign_text = None

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Global variable to store conversation history
conversation = []

# Streamlit app
def main():
    st.set_page_config(page_title="HearSay", layout="centered")
    st.title("HearSay")

    # Custom CSS for chat bubbles
    st.markdown("""
    <style>
    .chat-bubble {
        display: flex;
        align-items: flex-end;
        margin-bottom: 10px;
    }
    .bubble-sign {
        background: #e0e0e0;
        color: #222;
        border-radius: 20px 20px 20px 0px;
        padding: 12px 18px;
        margin-right: auto;
        max-width: 70%;
    }
    .bubble-speech {
        background: #f5f5f5;
        color: #222;
        border-radius: 20px 20px 0px 20px;
        padding: 12px 18px;
        margin-left: auto;
        max-width: 70%;
    }
    .icon-sign {
        margin-right: 8px;
    }
    .icon-speech {
        margin-left: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Webcam video input
    st.subheader("Live Video (Webcam)")
    video_ctx = webrtc_streamer(key="sign-video", video_processor_factory=VideoProcessor)

    # Speech-to-text input (recording)
    speech_text = None  # Ensure variable is always defined
    audio_bytes = audio_recorder()
    if audio_bytes:
        os.makedirs("temp", exist_ok=True)
        with open("temp/mic_input.wav", "wb") as f:
            f.write(audio_bytes)
        speech_text = transcribe_audio("temp/mic_input.wav")

    # Use the sign_text from the video processor if available
    sign_text = None
    if video_ctx and video_ctx.video_processor:
        sign_text = video_ctx.video_processor.sign_text

    # Display the conversation history
    if sign_text:
        conversation.append({"type": "sign", "text": sign_text})
    if speech_text:
        conversation.append({"type": "speech", "text": speech_text})

    # Display the latest sign and speech text
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Conversation History")
    for msg in conversation:
        if msg["type"] == "sign":
            st.markdown(f'<div class="chat-bubble"><span class="icon-sign">🧑‍🦲</span><div class="bubble-sign">{msg["text"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble" style="justify-content: flex-end;"><div class="bubble-speech">{msg["text"]}</div><span class="icon-speech">🗣️</span></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()