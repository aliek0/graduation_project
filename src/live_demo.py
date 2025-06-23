import cv2
import numpy as np
import torch
from collections import deque, Counter
from ultralytics import YOLO
from model import STGCN_v2
# We import ACTION_NAMES from train.py to ensure consistency
from train import ACTION_NAMES

# --- SETTINGS (UPDATED FOR 5 CLASSES) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'kth_best_model_5_class.pth'
NUM_CLASSES = 5
SEQUENCE_LENGTH = 30
PREDICTION_BUFFER_SIZE = 15
TARGET_FRAME_COUNT = 150 

def map_yolo_to_ntu_skeleton(keypoints):
    skeleton = np.zeros((25, 3))
    if keypoints.shape[0] > 0:
        skeleton[0] = [keypoints[0, 0], keypoints[0, 1], 0]
        skeleton[4] = [keypoints[6, 0], keypoints[6, 1], 0]
        skeleton[8] = [keypoints[5, 0], keypoints[5, 1], 0]
        skeleton[5] = [keypoints[8, 0], keypoints[8, 1], 0]
        skeleton[6] = [keypoints[10, 0], keypoints[10, 1], 0]
        skeleton[9] = [keypoints[7, 0], keypoints[7, 1], 0]
        skeleton[10] = [keypoints[9, 0], keypoints[9, 1], 0]
        skeleton[1] = ((skeleton[4] + skeleton[8]) / 2)
        skeleton[20] = skeleton[1]
    return skeleton

def main():
    print("Loading models...")
    stgcn_model = STGCN_v2(num_classes=NUM_CLASSES).to(DEVICE)
    stgcn_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    stgcn_model.eval()

    pose_model = YOLO('yolov8n-pose.pt')
    print("Models loaded successfully. Starting live demo...")

    cap = cv2.VideoCapture(0)
    sequence_data = deque(maxlen=SEQUENCE_LENGTH)
    predictions_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
    display_action = "..."

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        results = pose_model(frame, stream=False, verbose=False)
        annotated_frame = results[0].plot()

        if results and len(results[0].keypoints.xy) > 0 and len(results[0].keypoints.xyn[0]) > 0:
            keypoints = results[0].keypoints.xyn[0].cpu().numpy()
            skeleton = map_yolo_to_ntu_skeleton(keypoints)
            sequence_data.append(skeleton)

            if len(sequence_data) == SEQUENCE_LENGTH:
                live_data = np.array(list(sequence_data))
                
                padded_data = np.zeros((TARGET_FRAME_COUNT, 2, 25, 3))
                padded_data[:SEQUENCE_LENGTH, :, :, :] = np.expand_dims(live_data, axis=1)

                center_joint = padded_data[:, :, 1, :][:, :, np.newaxis, :]
                normalized_data = padded_data - center_joint
                
                data_tensor = torch.from_numpy(np.transpose(normalized_data, (3, 0, 2, 1)).astype(np.float32)).to(DEVICE)
                input_tensor = data_tensor.unsqueeze(0)

                with torch.no_grad():
                    outputs = stgcn_model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    
                    top_prob, top_idx = torch.topk(probabilities, 1)
                    if top_prob[0].item() > 0.30:
                        predictions_buffer.append(top_idx.item())

                    if len(predictions_buffer) > 0:
                        most_common_prediction = Counter(predictions_buffer).most_common(1)[0][0]
                        display_action = ACTION_NAMES[most_common_prediction]
        else:
            if len(sequence_data) > 0:
                sequence_data.popleft()

        cv2.rectangle(annotated_frame, (0, 0), (400, 40), (245, 117, 16), -1)
        cv2.putText(annotated_frame, display_action, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Live Action Recognition (5 Classes)', annotated_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Demo terminated.")

if __name__ == '__main__':
    main()
