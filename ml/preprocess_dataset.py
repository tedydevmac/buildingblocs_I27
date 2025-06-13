"""
use mediapipe to extract the hand landmarks from each video --> save as npz file
got the dataset from https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed
"""

import os, json, cv2, numpy as np
from multiprocessing import Pool
import mediapipe as mp

mp_hands = mp.solutions.hands
HANDS = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

SEQ_LEN = 32
VIDEOS_DIR = "/Users/tedgoh/buildingblocs_I27_ml/archive/videos"
OUT_DIR = "landmarks_cache"
JSON_PATH = "/Users/tedgoh/buildingblocs_I27_ml/archive/WLASL_v0.3.json"
os.makedirs(OUT_DIR, exist_ok=True)

def extract_and_save(args):
    video_id, label_idx, relpath = args
    video_path = os.path.join(VIDEOS_DIR, relpath)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return
        
    seq = []  # will store landmark sequences
    while True:
        ret, frame = cap.read()
        if not ret: break  # end of video
        
        # convert to RGB (mediapipe uses rgb, cv uses bgr)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = HANDS.process(frame)
        
        # 21 landmarks per hand, 3 coordinates x,y,z per landmark
        left = np.zeros((21,3), np.float32)
        right = np.zeros((21,3), np.float32)
        
        if results.multi_hand_landmarks:
            for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                # extract coordinates from landmarks
                coords = np.array([[p.x,p.y,p.z] for p in lm.landmark], np.float32)
                
                # it sometimes flips hands, so we use their labels
                if hd.classification[0].label=="Left":
                    # center around the wrist point (first landmark)
                    left = coords - coords[0]
                else:
                    right = coords - coords[0]

            if left.max():  left /= np.abs(left).max()
            if right.max(): right /= np.abs(right).max()
            
        # combine both hands into a single feature vector
        vec = np.hstack((left.flatten(), right.flatten()))  # shape: (126,)
        seq.append(vec)

    cap.release()

    # all sequences need to be the same length for batching
    lenght = len(seq)
    if lenght >= SEQ_LEN:
        # if we have more frames than needed, sample evenly across the video
        # i tried random sampling first but this preserves temporal structure better
        idxs = np.linspace(0, lenght-1, SEQ_LEN, dtype=int)
        seq = [seq[i] for i in idxs]
    else:
        # if have fewer frames, pad with zeros
        seq += [np.zeros(126, np.float32)]*(SEQ_LEN - lenght)
    
    seq = np.stack(seq)
    
    out_path = os.path.join(OUT_DIR, f"{video_id}.npz")
    np.savez_compressed(out_path, seq=seq, label=label_idx)
    
    # uncomment to see progress
    # if int(video_id) % 100 == 0:
    #    print(f"Processed video {video_id}")
    
    return

print(f"Loading dataset from {JSON_PATH}")
try:
    with open(JSON_PATH) as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: JSON file not found at {JSON_PATH}")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: Invalid JSON format in {JSON_PATH}")
    exit(1)
    
# map each sign (gloss) to index - tried using a defaultdict first but this is cleaner
gloss_to_idx = {entry['gloss']:i for i,entry in enumerate(data)}
print(f"Found {len(gloss_to_idx)} unique signs")

tasks = []
for entry in data:
    label = gloss_to_idx[entry['gloss']]
    for inst in entry.get('instances', []):
        vid = inst['video_id']
        tasks.append((vid, label, f"{vid}.mp4"))
        
print(f"Prepared {len(tasks)} videos for processing")

if __name__ == "__main__":
    import time
    start_time = time.time()

    NUM_PROCESSES = 8
    with Pool(processes=NUM_PROCESSES) as p:
        print(f"Starting preprocessing with {NUM_PROCESSES} parallel processes...")
        p.map(extract_and_save, tasks)
    
    elapsed = time.time() - start_time
    print(f"Preprocessing completed in {elapsed:.1f} seconds")
    print(f"Results saved to: {OUT_DIR}")
    print(f"Processed {len(tasks)} videos ({len(tasks)/elapsed:.1f} videos/second)")
    
    # make sure things look okay
    # import random
    # sample = os.path.join(OUT_DIR, f"{random.choice(tasks)[0]}.npz")
    # print(f"Random sample: {sample}")
    # print(np.load(sample)['seq'].shape)