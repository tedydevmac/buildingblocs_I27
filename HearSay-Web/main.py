import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time
import subprocess
import threading
import queue

st.title("Surrounding Awareness")

@st.cache_resource
def init_tts_queue():
    return queue.Queue()

@st.cache_resource
def fetch_model():
    return YOLO("best.pt")

detection_model = fetch_model()
speech_queue = init_tts_queue()

if 'last_inference_time' not in st.session_state:
    st.session_state.last_inference_time = 0

if 'what_detected' not in st.session_state:
    st.session_state.what_detected = []

if 'tts_thread_started' not in st.session_state:
    st.session_state.tts_thread_started = False


def tts_background_worker(q_obj, engine=None):
    while True:
        try:
            phrase = q_obj.get(timeout=1)
            if phrase is None:
                break  
            
            try:
                subprocess.run(['say', phrase], check=True, timeout=30)
            except subprocess.TimeoutExpired:
                print(f"[TTS Timeout] Skipped: {phrase}")
            except subprocess.CalledProcessError as err:
                print(f"[TTS Error] Command failed: {err}")
            except Exception as e:
                print(f"[TTS Unexpected Error] {e}")
            
            q_obj.task_done()
        
        except queue.Empty:
            continue  
        except Exception as thread_ex:
            print(f"[TTS Thread] Error: {thread_ex}")


if not st.session_state.tts_thread_started:
    threading.Thread(target=tts_background_worker, args=(speech_queue,), daemon=True).start()
    st.session_state.tts_thread_started = True


def locate_in_grid(x1, y1, x2, y2, frame_w, frame_h):
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    col = min(int(mid_x / (frame_w / 3)), 2)
    row = min(int(mid_y / (frame_h / 3)), 2)

    grid_labels = [
        ["top left", "top center", "top right"],
        ["center left", "center", "center right"],
        ["bottom left", "bottom center", "bottom right"]
    ]
    return grid_labels[row][col]


def announce_detections(found_items, w, h):
    if not found_items:
        phrase = "No objects detected"
    else:
        described = []
        for label, x1, y1, x2, y2, conf in found_items:
            pos = locate_in_grid(x1, y1, x2, y2, w, h)
            described.append(f"{label} in {pos}")

        if len(described) == 1:
            phrase = f"I see {described[0]}"
        else:
            phrase = f"I see {', '.join(described[:-1])}, and {described[-1]}"
    
    try:
        speech_queue.put_nowait(phrase)
    except queue.Full:
        print("[Speech Queue] Full - skipping speech")

    return phrase


@st.cache_resource
def start_camera():
    return cv2.VideoCapture(0)


camera = start_camera()
grabbed, frame = camera.read()

if grabbed:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    current = time.time()
    elapsed = current - st.session_state.last_inference_time


    if elapsed >= 20 or st.session_state.last_inference_time == 0:
        st.session_state.last_inference_time = current

        preds = detection_model(frame_rgb)
        results = preds[0]

        st.write(f"Number of detections: {len(results.boxes)}")

        bbox_xyxy = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_names = results.names if hasattr(results, 'names') else detection_model.names

        final_detections = []
        overlay = frame_rgb.copy()

        for idx, bbox in enumerate(bbox_xyxy):
            x1, y1, x2, y2 = map(int, bbox)
            cls_id = int(class_ids[idx])
            score = confidences[idx]
            label_text = f"{class_names[cls_id]} {score:.2f}"

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(overlay, label_text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            final_detections.append((class_names[cls_id], x1, y1, x2, y2, score))

        st.session_state.what_detected = final_detections

        grid_overlay = overlay.copy()
        h, w = grid_overlay.shape[:2]

        cv2.line(grid_overlay, (w // 3, 0), (w // 3, h), (255, 255, 255), 2)
        cv2.line(grid_overlay, (2 * w // 3, 0), (2 * w // 3, h), (255, 255, 255), 2)
        cv2.line(grid_overlay, (0, h // 3), (w, h // 3), (255, 255, 255), 2)
        cv2.line(grid_overlay, (0, 2 * h // 3), (w, 2 * h // 3), (255, 255, 255), 2)

        st.image(grid_overlay, caption=f"Detections", channels="RGB")

        spoken = announce_detections(final_detections, w, h)

        st.subheader(f"Detections")
        st.info(f" Speaking: {spoken}")

        if not final_detections:
            st.write("Nothing caught on camera.")
        else:
            for thing in final_detections:
                name, x1, y1, x2, y2, conf = thing
                pos = locate_in_grid(x1, y1, x2, y2, w, h)
                st.write(f"{name} spotted in {pos}")

        time.sleep(1)
        st.rerun()

    else:
        wait_for = int(20 - elapsed)
        st.caption(f"Time - {wait_for}.")
        time.sleep(1)
        st.rerun()

else:
    st.error("Got issue when accessing camera.")
