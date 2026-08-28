import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

def scene_detection_SAD (video_path, threshold):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        return []
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
    
    scene_changes = []
    sad_val = []
    mean_val = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.GaussianBlur(curr_gray, (21, 21), 0)
        
        diff = cv2.absdiff(prev_gray, curr_gray)
        
        sad = np.sum(diff)
        mean_sad = np.mean(diff)
        
        '''sad_val.append(sad)
        mean_val.append(mean_sad)'''
        
        if mean_sad > threshold:
            scene_changes.append((frame_idx, mean_sad))
            
            #print(f"frejm {frame_idx}, {mean_sad}")
        
        prev_gray = curr_gray
        
    cap.release()
    return scene_changes #, sad_val, mean_val

def adaptive_threshold1(video_path, window_size = 30, threshold_factor = 3.0, min_sad = 5.0, peak_multiplier = 2.6):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        return [], [], [], []   
    
    prev_gray = cv2.resize(prev_frame, (640, 360))
    prev_gray = cv2.cvtColor(prev_gray, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
    
    frame_idx = 0
    window_sads = []
    all_mean_sads = []
    thresholds = []
    scene_changes = []
    frame_indices = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        curr_gray = cv2.resize(frame, (640, 360))
        curr_gray = cv2.cvtColor(curr_gray, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.GaussianBlur(curr_gray, (21, 21), 0)
        
        diff = cv2.absdiff(prev_gray, curr_gray) #diff za obicna dva frejma
        mean_sad = np.mean(diff) #diff objedinjen u jedan prosecan broj
        
        if len(window_sads) >= window_size:
            
            mean_sad_window = np.mean(window_sads) #mean broj za sve vrednosti u trenutnom windowu
            std_sad = np.std(window_sads) 
            
            adaptive_threshold = mean_sad_window + (threshold_factor * std_sad)
            adaptive_threshold = max(adaptive_threshold, min_sad)
            #print(f"frame idx: {frame_idx}, mean sad wind: {mean_sad_window}, thresh: {adaptive_threshold}")
            
            all_mean_sads.append(mean_sad)
            frame_indices.append(frame_idx)
            thresholds.append(adaptive_threshold)
            
            '''if mean_sad > adaptive_threshold and mean_sad > previous_sad * 2.6:
                
                time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                print(f"scene changed at frame: {frame_idx}, mean_sad: {mean_sad}, vreme: {time_sec}")
                scene_changes.append(frame_idx)'''

            
        else:
            # Nema dovoljno podataka
            #thresholds.append(np.nan)
            pass
        
        
        window_sads.append(mean_sad)  
        
        if len(window_sads) > window_size:
            window_sads.pop(0)
          
        prev_gray = curr_gray
        frame_idx += 1
        
    cap.release()
    
    for i in range(1, len(all_mean_sads) - 1):
    
            current = all_mean_sads[i]
            previous = all_mean_sads[i - 1]
            next_value = all_mean_sads[i + 1]
    
            threshold = thresholds[i]
            
            if np.isnan(threshold):
                continue
    
            if (current > threshold and current > previous * peak_multiplier and current > next_value):
    
                scene_changes.append((frame_indices[i], current))
                time_sec = frame_indices[i] / fps
                #print(f"scene changed at frame: {frame_indices[i]}, vreme: {time_sec}")
    
    return (scene_changes, all_mean_sads, thresholds, frame_indices)

def adaptive_histogram(video_path, window_size = 45, threshold_factor = 4.0, min_hist_diff = 0.05, peak_multiplier = 1.5):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        return [], [], [], []

    prev_frame = cv2.resize(prev_frame, (640, 360))
    prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
    prev_hsv = cv2.GaussianBlur(prev_hsv, (21, 21), 0)
    prev_hist = cv2.calcHist([prev_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256]) #histogram prethodnog frejma
    prev_hist = cv2.normalize(prev_hist, prev_hist, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX).flatten()

    frame_idx = 0
    window_diffs = []
    all_hist_diffs = []
    thresholds = []
    frame_indices = []
    scene_changes = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        curr_frame = cv2.resize(frame, (640, 360))
        curr_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
        curr_hist = cv2.calcHist([curr_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        curr_hist = cv2.normalize(curr_hist, curr_hist).flatten()

        # --------------------------------
        # Razlika između histograma
        # --------------------------------

        hist_diff = cv2.compareHist(
            prev_hist.astype(np.float32),
            curr_hist.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA)

        window_diffs.append(hist_diff)

        # --------------------------------
        # Adaptivni threshold
        # --------------------------------

        if len(window_diffs) >= window_size:

            mean_hist_window = np.mean(window_diffs)
            std_hist = np.std(window_diffs)

            adaptive_threshold = (mean_hist_window + threshold_factor * std_hist)
            adaptive_threshold = max(adaptive_threshold, min_hist_diff)

            all_hist_diffs.append(hist_diff)
            frame_indices.append(frame_idx)
            thresholds.append(adaptive_threshold)

        if len(window_diffs) > window_size:
            window_diffs.pop(0)

        prev_hist = curr_hist
        frame_idx += 1

    cap.release()

    # --------------------------------
    # Detekcija scene change-a
    # --------------------------------

    for i in range(1, len(all_hist_diffs) - 1):

        current = all_hist_diffs[i]
        previous = all_hist_diffs[i - 1]
        next_value = all_hist_diffs[i + 1]

        threshold = thresholds[i]

        if np.isnan(threshold):
            continue

        if (current > threshold and current > previous * peak_multiplier and current > next_value):
            frame = frame_indices[i]
            scene_changes.append((frame, current))

    return (scene_changes, all_hist_diffs, thresholds, frame_indices)
 
def scene_dynamics(all_mean_sads, frame_indices, scene_boundaries):
    scores = []

    #all_mean_sads lista svih sadova za sve frejmove (znaci koliko se svaki frejm promenio)
    #frame_indices uklapaju se uz ovo gore kao koji broj frejma odgovara kojoj promeni
    #scene_boundaries gde se desava cut scene
    #segment - sve SAD vrednosti koje pripadaju jednoj sceni

    for start, end in zip(scene_boundaries[:-1], scene_boundaries[1:]):

        segment = [s for f, s in zip(frame_indices, all_mean_sads) if start <= f < end]
        
        # Ako nema SAD vrednosti za ovu scenu, preskoči je
        '''if len(segment) == 0:
            print(f"prazna scena {start} - {end}")
            continue'''
        
        scores.append({
            "start": start,
            "end": end,
            "mean": np.mean(segment),
            "max": np.max(segment),
            "std": np.std(segment)
        })

    return scores

def interval_for_score(mean_sad, low_thresh, high_thresh, min_interval = 60, max_interval = 90
):
    if mean_sad < low_thresh: return max_interval

    elif mean_sad > high_thresh: return min_interval

    else:
        frac = ((mean_sad - low_thresh) / (high_thresh - low_thresh))
        interval = int(max_interval - frac * (max_interval - min_interval))
        return interval
    
def generate_keyframes(scores, total_frames, min_interval):
    keyframes = [0]

    for i, scene in enumerate(scores):
        start = scene["start"]
        end = scene["end"]
        interval = scene["interval"]
        
        
        print(f"\nSCENA {i}")
        print(f"Start: {start}")
        print(f"End: {end}")
        print(f"Interval: {interval}")
        
        # granica scene je keyframe
        if start not in keyframes:
            keyframes.append(start)

        # dodatni keyframeovi unutar scene
        frame = start + interval

        while frame < end:
            if end - frame >= min_interval:
                keyframes.append(frame)
            frame += interval

    # uklanja duplikate i sortira
    keyframes = sorted(set(keyframes))

    return keyframes
