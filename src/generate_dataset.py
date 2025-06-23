import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import re

# --- SETTINGS ---
SEQUENCES_FILE_PATH = 'KTH_sequences.txt'
AVI_VIDEO_DIR = 'KTH_avi_videolari/'
OUTPUT_NPY_DIR = 'KTH_yolo_npy/'

# --- Dataset Split Configuration ---
TRAIN_SUBJECTS = [11, 12, 13, 14, 15, 16, 17, 18]
VAL_SUBJECTS = [19, 20, 21, 23, 24, 25, 1, 4]
TEST_SUBJECTS = [22, 2, 3, 5, 6, 7, 8, 9, 10]

def parse_sequences_file(filepath):
    """
    Parses the KTH_sequences.txt file and returns a dictionary with video
    information, excluding the 'jogging' class.
    """
    sequences = {}
    with open(filepath, 'r') as f:
        for line in f:
            # --- CHANGE: Skip lines containing 'jogging' ---
            if '_jogging_' in line:
                continue
            # ----------------------------------------------
            line = line.strip()
            if not line or 'frames' not in line:
                continue
            
            parts = line.split('\t')
            video_name = parts[0].strip()
            
            frame_ranges_str = parts[-1].replace('frames', '').strip()
            frame_ranges = [tuple(map(int, r.split('-'))) for r in frame_ranges_str.split(', ')]
            
            sequences[video_name] = frame_ranges
            
    return sequences

def map_yolo_to_ntu_skeleton(keypoints):
    """Maps YOLO's 17 keypoints to the 25-joint structure expected by ST-GCN."""
    skeleton = np.zeros((25, 3))
    if keypoints.shape[0] > 0:
        skeleton[0] = [keypoints[0, 0], keypoints[0, 1], 0]   # Nose
        skeleton[4] = [keypoints[6, 0], keypoints[6, 1], 0]   # Right Shoulder
        skeleton[8] = [keypoints[5, 0], keypoints[5, 1], 0]   # Left Shoulder
        skeleton[5] = [keypoints[8, 0], keypoints[8, 1], 0]   # Right Elbow
        skeleton[6] = [keypoints[10, 0], keypoints[10, 1], 0] # Right Wrist
        skeleton[9] = [keypoints[7, 0], keypoints[7, 1], 0]   # Left Elbow
        skeleton[10] = [keypoints[9, 0], keypoints[9, 1], 0]  # Left Wrist
        skeleton[1] = ((skeleton[4] + skeleton[8]) / 2)      # Spine (Mid-shoulder)
        skeleton[20] = skeleton[1]                           # Center of mass (approximated)
    return skeleton

def process_and_save_kth_dataset():
    """
    Main function: Processes videos, splits them into segments, extracts skeletons,
    and saves them as .npy files sorted into train/val/test sets.
    """
    sequences_info = parse_sequences_file(SEQUENCES_FILE_PATH)
    print(f"Successfully read frame info for {len(sequences_info)} videos.")

    for folder in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_NPY_DIR, folder), exist_ok=True)

    pose_model = YOLO('yolov8n-pose.pt')
    
    for video_name, segments in tqdm(sequences_info.items(), desc="Processing Videos"):
        video_path = os.path.join(AVI_VIDEO_DIR, video_name + "_uncomp.avi")
        
        if not os.path.exists(video_path):
            print(f"Warning: Video not found, skipping: {video_path}")
            continue

        person_id = int(re.search(r'person(\d+)', video_name).group(1))
        
        if person_id in TRAIN_SUBJECTS:
            split_folder = 'train'
        elif person_id in VAL_SUBJECTS:
            split_folder = 'val'
        elif person_id in TEST_SUBJECTS:
            split_folder = 'test'
        else:
            continue

        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        for i, (start_frame, end_frame) in enumerate(segments):
            clip_skeletons = []
            
            if start_frame > len(frames) or end_frame > len(frames):
                print(f"Warning: Invalid frame range for {video_name} ({start_frame}-{end_frame}), skipping segment.")
                continue

            for frame_idx in range(start_frame - 1, end_frame):
                frame = frames[frame_idx]
                results = pose_model(frame, stream=False, verbose=False)
                
                if results and len(results[0].keypoints.xy) > 0 and len(results[0].keypoints.xyn[0]) > 0:
                    keypoints_xy = results[0].keypoints.xyn[0].cpu().numpy()
                    skeleton = map_yolo_to_ntu_skeleton(keypoints_xy)
                    clip_skeletons.append(skeleton)
                else:
                    clip_skeletons.append(np.zeros((25, 3)))
            
            if len(clip_skeletons) > 0:
                final_data = np.array(clip_skeletons)
                final_data_expanded = np.expand_dims(final_data, axis=1)
                output_array = np.zeros((final_data.shape[0], 2, 25, 3), dtype=np.float32)
                output_array[:, :1, :, :] = final_data_expanded

                output_filename = f"{video_name}_seg{i+1}.npy"
                output_path = os.path.join(OUTPUT_NPY_DIR, split_folder, output_filename)
                np.save(output_path, output_array)

    print("All videos processed successfully and .npy files for 5 classes have been created!")

if __name__ == '__main__':
    process_and_save_kth_dataset()
