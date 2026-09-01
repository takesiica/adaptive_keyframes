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

def adaptive_histogram(video_path, window_size=45, threshold_factor=4.0, min_hist_diff=0.05, peak_multiplier=1.5, min_peak_diff=0.5):
    
    #threshold_factor Određuje koliko standardnih devijacija iznad proseka mora da bude histogram difference da bi bio sumnjiv kao promena scene.
    #std je brojka koja odredjuje koliko je daleko neka vrednost od prosecne vrednosti za te podatke
    #min_hist_diff - minimalni threshold, sluzi da algoritam ne bude preosetljiv, postavlja sta sme da bude najniza vrednost 
    #peak_multiplier - da trenutns vrednost treba da bude dosta veca od prethodne
    
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()

    if not ret:
        return [], [], [], []

    prev_frame = cv2.resize(prev_frame, (640, 360))
    prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
    prev_hsv = cv2.GaussianBlur(prev_hsv, (21, 21), 0)

    prev_hist = cv2.calcHist(
        [prev_hsv], [0, 1], None,
        [32, 32],
        [0, 180, 0, 256]
    )

    prev_hist = cv2.normalize(prev_hist, prev_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()

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
        curr_hist = cv2.normalize( curr_hist, curr_hist).flatten()
        hist_diff = cv2.compareHist(prev_hist.astype(np.float32), curr_hist.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)

        window_diffs.append(hist_diff)

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

    # Detekcija scene change-a

    for i in range(1, len(all_hist_diffs) - 1):

        current = all_hist_diffs[i]
        previous = all_hist_diffs[i - 1]
        next_value = all_hist_diffs[i + 1]

        threshold = thresholds[i]

        if np.isnan(threshold):
            continue

        if (
            current > threshold
            and current > previous * peak_multiplier
            and current > next_value
            and current > min_peak_diff
        ):
            frame = frame_indices[i]
            scene_changes.append((frame, current))

    return (scene_changes, all_hist_diffs, thresholds, frame_indices)

def combine_sad_histogram(
    video_path,
    window_size=30,
    sad_threshold_factor=3.0,
    min_sad=5.0,
    sad_peak_multiplier=2.6,
    hist_threshold_factor=4.0,
    min_hist_diff=0.05,
    hist_peak_multiplier=1.5,
    min_peak_diff=0.5,
    sad_low_threshold=10.0,
    sad_high_threshold=30.0
):

    # 1. Pokreni postojeći SAD algoritam

    sad_changes, sad_values, sad_thresholds, sad_frames = adaptive_threshold1(
        video_path,
        window_size=window_size,
        threshold_factor=sad_threshold_factor,
        min_sad=min_sad,
        peak_multiplier=sad_peak_multiplier
    )

    # 2. Pokreni postojeći histogram algoritam

    hist_changes, hist_values, hist_thresholds, hist_frames = adaptive_histogram(
        video_path,
        window_size=window_size,
        threshold_factor=hist_threshold_factor,
        min_hist_diff=min_hist_diff,
        peak_multiplier=hist_peak_multiplier,
        min_peak_diff=min_peak_diff
    )

    # 3. Poravnanje frejmova

    # Oba algoritma počinju da vraćaju podatke tek kada
    # imaju dovoljno podataka za window.

    # Zato uzimamo samo frejmove koji postoje kod oba.

    sad_dict = dict(zip(sad_frames, sad_values))
    hist_dict = dict(zip(hist_frames, hist_values))

    common_frames = sorted(set(sad_dict.keys()) & set(hist_dict.keys()))

    if not common_frames:
        return [], [], [], [], [], []

    sad = np.array([sad_dict[f] for f in common_frames], dtype=np.float32)
    hist = np.array([hist_dict[f] for f in common_frames], dtype=np.float32 )
    
    # 4. Normalizacija
    
    # SAD i histogram nisu na istoj skali.
    # Zato ih prvo svodimo na 0-1.

    sad_min = np.min(sad)
    sad_max = np.max(sad)

    if sad_max > sad_min:
        sad_norm = (sad - sad_min) / (sad_max - sad_min)
    else:
        sad_norm = np.zeros_like(sad)

    hist_min = np.min(hist)
    hist_max = np.max(hist)

    if hist_max > hist_min:
        hist_norm = (hist - hist_min) / (hist_max - hist_min)
    else:
        hist_norm = np.zeros_like(hist)

    # 5. Računamo SAD pouzdanost

    weights_sad = []
    weights_hist = []

    for i in range(len(common_frames)):

        # Uzimamo prethodnih window_size SAD vrednosti
        start = max(0, i - window_size + 1)
        sad_window = sad[start:i + 1]

        mean_sad = np.mean(sad_window)

        if i % 100 == 0:
            print(
                f"frame={common_frames[i]}, "
                f"mean_sad={mean_sad:.2f}"
            )

        # Računamo koliko je SAD blizu granicama
        # 0 = mirna scena, 1 = dinamična scena
        sad_ratio = (
            (mean_sad - sad_low_threshold) /
            (sad_high_threshold - sad_low_threshold)
        )

        # Ograničavamo vrednost na [0, 1]
        sad_ratio = np.clip(sad_ratio, 0.0, 1.0)

        # Adaptivno određivanje težina
        w_sad = 0.2 + 0.6 * sad_ratio
        w_hist = 1.0 - w_sad

        weights_sad.append(w_sad)
        weights_hist.append(w_hist)

    weights_sad = np.array(weights_sad)
    weights_hist = np.array(weights_hist)

    # 6. Kombinovani signal

    combined = ( weights_sad * sad_norm + weights_hist * hist_norm)

    # 7. Adaptive threshold za kombinovani signal

    combined_thresholds = []

    for i in range(len(combined)):
        start = max(0, i - window_size + 1)
        combined_window = combined[start:i + 1]

        mean_combined = np.mean(combined_window)
        std_combined = np.std(combined_window)

        threshold = (mean_combined + sad_threshold_factor * std_combined)
        combined_thresholds.append(threshold)

    combined_thresholds = np.array(combined_thresholds)

    # 8. Detekcija peakova

    scene_changes = []

    for i in range(1, len(combined) - 1):

        current = combined[i]
        previous = combined[i - 1]
        next_value = combined[i + 1]

        threshold = combined_thresholds[i]

        if (
            current > threshold
            and current > previous
            and current > next_value
            and current > min_peak_diff
        ):
            
            scene_changes.append((common_frames[i], current))
            
    return (scene_changes, sad, hist, combined, combined_thresholds, common_frames, weights_sad, weights_hist)

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

def generate_keyframes_by_scene(scene_changes, total_frames):
    """
    Postavlja keyframe samo na početak svake scene.

    scene_changes: lista tuple-ova (frame, score)
    total_frames: ukupan broj frejmova

    Vraća listu keyframeova.
    """

    keyframes = [0]

    for frame, _ in scene_changes:
        if frame > 0 and frame < total_frames:
            keyframes.append(frame)

    keyframes = sorted(set(keyframes))

    return keyframes