import cv2
from functions import adaptive_histogram, combine_sad_histogram
from evaluate import evaluatef
import numpy as np


videos = [
    {
        "name": "Video 1",
        "path": ".\\evaluation\\test1.mp4",
        "true_times": [9, 18]
    },
    {
        "name": "Video 2",
        "path": ".\\evaluation\\test2.mp4",
        "true_times": [5, 9, 14, 19, 24, 27, 31, 36, 40, 44]
    },
    {
        "name": "Video 3",
        "path": ".\\evaluation\\test3.mp4",
        "true_times": [4, 9, 13, 18]
    },
    {
        "name": "Video 4",
        "path": ".\\evaluation\\test4.mp4",
        "true_times": [3, 6, 10, 16, 20, 23]
    },
    {
        "name": "Video 5",
        "path": ".\\evaluation\\test5.mp4",
        "true_times": [3,5,11]
    },
    {
        "name": "Video 6",
        "path": ".\\evaluation\\test6.mp4",
        "true_times": [5, 10, 15]
    }
]

WINDOW_SIZE = 30

SAD_THRESHOLD_FACTOR = 3.0
MIN_SAD = 5.0
SAD_PEAK_MULTIPLIER = 2.6

HIST_THRESHOLD_FACTOR = 4.0
MIN_HIST_DIFF = 0.05
HIST_PEAK_MULTIPLIER = 1.5
MIN_PEAK_DIFF = 0.5

TOLERANCE = 1.5

SAD_LOW_THRESHOLD = 2.0
SAD_HIGH_THRESHOLD = 6.0

all_precisions = []
all_recalls = []
all_f1_scores = []


print("\n")
print("=" * 90)
print(" KOMBINOVANI SAD + HISTOGRAM")
print("=" * 90)

print(f"Window size:          {WINDOW_SIZE}")
print(f"SAD low threshold:    {SAD_LOW_THRESHOLD}")
print(f"SAD high threshold:   {SAD_HIGH_THRESHOLD}")
print(f"Tolerance:             {TOLERANCE}")

'''
print("\n")
print("Težine:")
print("Mirna scena:       SAD = 0.2 | Histogram = 0.8")
print("Srednja scena:     SAD = 0.5 | Histogram = 0.5")
print("Dinamična scena:   SAD = 0.8 | Histogram = 0.2")
'''

for video in videos:

    input_path = video["path"]
    true_times = video["true_times"]

    print("\n")
    print("-" * 90)
    print(video["name"])
    print("-" * 90)

    (scene_changes, sad, hist, combined, combined_thresholds, common_frames, weights_sad, weights_hist) = combine_sad_histogram(

        input_path,
        window_size=WINDOW_SIZE,

        sad_threshold_factor=SAD_THRESHOLD_FACTOR,
        min_sad=MIN_SAD,
        sad_peak_multiplier=SAD_PEAK_MULTIPLIER,

        hist_threshold_factor=HIST_THRESHOLD_FACTOR,
        min_hist_diff=MIN_HIST_DIFF,
        hist_peak_multiplier=HIST_PEAK_MULTIPLIER,
        min_peak_diff=MIN_PEAK_DIFF,

        sad_low_threshold=SAD_LOW_THRESHOLD,
        sad_high_threshold=SAD_HIGH_THRESHOLD
    )

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    detected_times = [frame / fps for frame, _ in scene_changes]
    
    precision, recall, f1 = evaluatef(detected_times, true_times, tolerance=TOLERANCE)

    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1_scores.append(f1)

    mean_w_sad = np.mean(weights_sad)
    mean_w_hist = np.mean(weights_hist)

    print( f"Detektovano: {len(detected_times)}")
    print(f"Detektovana vremena: " f"{[round(t, 2) for t in detected_times]}")
    print(f"Tačna vremena:       {true_times}")
    print()
    print(f"Precision = {precision:.3f}")
    print(f"Recall    = {recall:.3f}")
    print(f"F1        = {f1:.3f}")
    print()
    print(f"Prosečna wSAD  = {mean_w_sad:.3f}")
    print(f"Prosečna wHist = {mean_w_hist:.3f}")

mean_precision = np.mean(all_precisions)
mean_recall = np.mean(all_recalls)
mean_f1 = np.mean(all_f1_scores)


print("\n\n")
print("=" * 90)
print(" UKUPNI REZULTAT")
print("=" * 90)

print(f"Prosečan Precision: {mean_precision:.3f}")
print(f"Prosečan Recall:    {mean_recall:.3f}")
print(f"Prosečan F1:        {mean_f1:.3f}")

print("=" * 90)