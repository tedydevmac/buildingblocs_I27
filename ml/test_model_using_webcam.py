import cv2
import torch
import numpy as np
import mediapipe as mp
from pleasework import SignLanguageModel  

print("Loading sign language recognition model...")
model = SignLanguageModel(num_classes=2000)

MODEL_PATH = "/Users/tedgoh/buildingblocs_I27_ml/best_signlang_model.pt"

device = torch.device('mps')
if not torch.backends.mps.is_available():
    print("MPS not available, falling back to CPU")
    device = torch.device('cpu')
    
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure the model path is correct and the file exists")
    exit(1)
    
model.eval()
model = model.to(device)

import json
JSON_PATH = "/Users/tedgoh/buildingblocs_I27_ml/archive/WLASL_v0.3.json"
try:
    with open(JSON_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} sign labels")
except FileNotFoundError:
    print(f"Error: JSON file not found at {JSON_PATH}")
    exit(1)
    
label_map = {v['gloss']: k for k, v in enumerate(data)}  # sign name -> index
index_to_gloss = {v: k for k, v in label_map.items()}    # index -> sign name

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,      
    max_num_hands=2,              
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

SEQ_LEN = 32  
sequence = [] 
prediction_cooldown = 0  
current_prediction = "Ready..."

# 0 = obs cam
# 1 = my iphone
# 2 = webcam
CAMERA_ID = 2
print(f"Opening camera {CAMERA_ID}...")
cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("Error: Could not open webcam. Try changing CAMERA_ID.")
    exit(1)
    
# set resolution for better performance
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Webcam initialized. Press 'q' to quit.")
print("Starting real-time sign recognition...")

frame_count = 0  
last_prediction = ""  
confidence_threshold = 0

try:
    print("Starting recognition loop...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame - camera disconnected?")
            break
            
        frame_count += 1
        
        # for testing cause mirror view easier to see
        # frame = cv2.flip(frame, 1)
        
        # fps counter
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(img_rgb)

        # initialize the empty hand landmarks
        left = np.zeros((21,3), np.float32)  # 21 landmarks per hand
        right = np.zeros((21,3), np.float32)
        
        if results.multi_hand_landmarks:
            for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], np.float32)
                
                # which hand
                hand_label = hd.classification[0].label
                
                # draw the hand skeleton
                # use different colors for left&right hands
                color = (0, 255, 0) if hand_label == "Left" else (255, 0, 0)  
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS,
                                         mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2))
                
                if hand_label == "Left":
                    left = coords - coords[0]
                else:
                    right = coords - coords[0]

            # got better results after adding normalization
            if np.abs(left).max():
                left /= np.abs(left).max()
            if np.abs(right).max():
                right /= np.abs(right).max()
                
        # combine both hands into feature vector
        # 21 landmarks × 3 coords × 2 hands = 126 features
        vec = np.hstack((left.flatten(), right.flatten()))
        
        sequence.append(vec)
        
        # keep sequence at fixed length by removing oldest frame
        if len(sequence) > SEQ_LEN:
            sequence.pop(0)  
        
        # run prediction when we have enough frames
        if len(sequence) == SEQ_LEN:
            # Convert to torch tensor
            input_tensor = torch.tensor([sequence], dtype=torch.float32).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                
                # get the predicted class
                pred = output.argmax(1).item()
                
                # get top 3 predictions and their probabilities (for debugging)
                # probs = torch.nn.functional.softmax(output, dim=1)[0]
                # top3_prob, top3_indices = torch.topk(probs, 3)
                # top3_labels = [index_to_gloss.get(idx.item(), "Unknown") for idx in top3_indices]
                
                # get the clas name
                gloss = index_to_gloss.get(pred, "Unknown")
                
                if gloss != last_prediction:
                    last_prediction = gloss
                
            # show prediction on screen
            cv2.putText(frame, gloss, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 0), 4, cv2.LINE_AA)  # Outline
            cv2.putText(frame, gloss, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 2, cv2.LINE_AA)  # Text
                        
            cv2.putText(frame, "Sign Language Detector Active", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # show how many frames collected so far
            cv2.putText(frame, f"Collecting frames: {len(sequence)}/{SEQ_LEN}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Sign Language Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User pressed 'q', exiting...")
            break

except KeyboardInterrupt:
    print("Interrupted by user")
except Exception as e:
    print(f"Error: {e}")
finally:
    # clean up
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")