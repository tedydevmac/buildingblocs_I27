import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pickle

df = pd.read_csv('wlasl_with_splits.csv')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def extract_features(video_path, max_frames=60):
    hand_keypoints_sequence = []
    pose_keypoints_sequence = []
    original_frames = []
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands, \
        mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return None, None, None
        
        frame_idx = 0
        
        while cap.isOpened() and frame_idx < max_frames:
            success, image = cap.read()
            if not success:
                break
            
            resized_frame = cv2.resize(image, (128, 128))
            original_frames.append(resized_frame)
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            hand_results = hands.process(image_rgb)
            pose_results = pose.process(image_rgb)
            
            frame_hand_keypoints = []
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        frame_hand_keypoints.extend([landmark.x, landmark.y, landmark.z])
            else:
                # Fill with zeros if no hands detected
                frame_hand_keypoints = [0.0] * (21 * 3)
            
            frame_pose_keypoints = []
            if pose_results.pose_landmarks:
                for landmark in pose_results.pose_landmarks.landmark:
                    frame_pose_keypoints.extend([landmark.x, landmark.y, landmark.z])
            else:
                frame_pose_keypoints = [0.0] * (33 * 3)
            
            hand_keypoints_sequence.append(frame_hand_keypoints)
            pose_keypoints_sequence.append(frame_pose_keypoints)
            frame_idx += 1
        
        cap.release()
    
    if len(hand_keypoints_sequence) < max_frames:
        hand_keypoints_sequence.extend([hand_keypoints_sequence[-1] if hand_keypoints_sequence else [0.0] * (21 * 3)] * (max_frames - len(hand_keypoints_sequence)))
        pose_keypoints_sequence.extend([pose_keypoints_sequence[-1] if pose_keypoints_sequence else [0.0] * (33 * 3)] * (max_frames - len(pose_keypoints_sequence)))
        original_frames.extend([np.zeros((128, 128, 3), dtype=np.uint8)] * (max_frames - len(original_frames)))
    
    return np.array(hand_keypoints_sequence), np.array(pose_keypoints_sequence), np.array(original_frames)

os.makedirs('processed_data', exist_ok=True)

num_videos_to_process = len(df)  
skipped = 0

for i, row in tqdm(df.iloc[:num_videos_to_process].iterrows(), total=num_videos_to_process):
    video_id = row['video_id']
    gloss = row['gloss']
    split = row['split']
    video_path = row['video_path']
    
    output_dir = os.path.join('processed_data', split)
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{video_id}.pkl")
    
    if os.path.exists(output_file):
        continue
    
    try:
        hand_keypoints, pose_keypoints, frames = extract_features(video_path)
        
        if hand_keypoints is None or pose_keypoints is None or frames is None:
            skipped += 1
            continue
        
        with open(output_file, 'wb') as f:
            pickle.dump({
                'video_id': video_id,
                'gloss': gloss,
                'hand_keypoints': hand_keypoints,
                'pose_keypoints': pose_keypoints,
                'frames': frames
            }, f)
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        skipped += 1

print(f"Processed {num_videos_to_process - skipped} videos. Skipped {skipped} videos.")