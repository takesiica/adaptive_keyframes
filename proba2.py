import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import adaptive_histogram

input_path = ".\\static_videos\\static4.mp4"

print('kakkakaak')

# FPS videa
cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

scene_changes, all_hist_diffs, thresholds, frame_indices = adaptive_histogram(input_path, window_size=30, threshold_factor=5.0, min_hist_diff=0.1, peak_multiplier=1.5)
time_indices = [frame / fps for frame in frame_indices]

plt.figure(figsize=(12, 6))
plt.plot(time_indices, all_hist_diffs, label="Histogram difference")
plt.plot(time_indices, thresholds, label="Adaptive threshold")

# Scene changes
for i, (frame, diff) in enumerate(scene_changes):

    time = frame / fps
    plt.scatter(time, diff, s=50, zorder=5, label="Detected scene change" if i == 0 else None)
    plt.axvline( x=time, linestyle="--", alpha=0.5)
    
    print(
        f"Scene change {i+1}: "
        f"frame = {frame}, "
        f"time = {time:.2f} s"
    )
    
plt.title("Adaptive Histogram Scene Detection")
plt.xlabel("Time (seconds)")
plt.ylabel("Histogram Difference")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


#ako je dinamicnija scena da window bude manji 
#da li window treba da bude fixan ili moze da se menja