import cv2
from functions import adaptive_threshold1, adaptive_histogram
from evaluate import evaluatef


# ==========================================
# PODACI ZA SVIH 6 VIDEO ZAPISA
# ==========================================

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


# ==========================================
# PARAMETRI DETEKCIJE
# ==========================================

WINDOW_SIZE = 45
THRESHOLD_FACTOR = 4.0
TOLERANCE = 1.0

MIN_HIST_VALUES = [ 0.02, 0.03, 0.05, 0.07, 0.10]

PEAK_MULTIPLIER_VALUES = [1.1, 1.3, 1.5, 1.7, 2.0]

results = []

print("\n========================================")
print(" GRID SEARCH - HISTOGRAM")
print("========================================")

print(f"Window size:       {WINDOW_SIZE}")
print(f"Threshold factor:  {THRESHOLD_FACTOR}")
print(f"Tolerance:         {TOLERANCE}")

print("\nTestirani min_hist_diff:")
print(MIN_HIST_VALUES)

print("\nTestirani peak_multiplier:")
print(PEAK_MULTIPLIER_VALUES)


for min_hist_diff in MIN_HIST_VALUES:

    for peak_multiplier in PEAK_MULTIPLIER_VALUES:

        all_precisions = []
        all_recalls = []
        all_f1_scores = []

        print("\n----------------------------------------")
        print(
            f"min_hist_diff = {min_hist_diff}, "
            f"peak_multiplier = {peak_multiplier}"
        )
        print("----------------------------------------")

        for video in videos:

            input_path = video["path"]
            true_times = video["true_times"]

            # --------------------------------------
            # 1. Histogram scene detection
            # --------------------------------------

            scene_changes, all_hist_diffs, thresholds, frame_indices = adaptive_histogram(
                input_path,
                window_size=WINDOW_SIZE,
                threshold_factor=THRESHOLD_FACTOR,
                min_hist_diff=min_hist_diff,
                peak_multiplier=peak_multiplier
            )

            # --------------------------------------
            # 2. FPS
            # --------------------------------------

            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            # --------------------------------------
            # 3. Frejmovi -> sekunde
            # --------------------------------------

            detected_times = [
                frame / fps
                for frame, _ in scene_changes
            ]

            # --------------------------------------
            # 4. Evaluacija
            # --------------------------------------

            precision, recall, f1 = evaluatef(
                detected_times,
                true_times,
                tolerance=TOLERANCE
            )

            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1_scores.append(f1)

            print(
                f"{video['name']}: "
                f"P={precision:.3f}, "
                f"R={recall:.3f}, "
                f"F1={f1:.3f}"
            )

        # --------------------------------------
        # 5. Prosek za svih 6 videa
        # --------------------------------------

        mean_precision = sum(all_precisions) / len(all_precisions)
        mean_recall = sum(all_recalls) / len(all_recalls)
        mean_f1 = sum(all_f1_scores) / len(all_f1_scores)

        results.append({
            "min_hist_diff": min_hist_diff,
            "peak_multiplier": peak_multiplier,
            "precision": mean_precision,
            "recall": mean_recall,
            "f1": mean_f1
        })

        print(
            f"PROSEK: "
            f"P={mean_precision:.3f}, "
            f"R={mean_recall:.3f}, "
            f"F1={mean_f1:.3f}"
        )


# ==========================================
# SORTIRANJE PO F1
# ==========================================

results.sort(
    key=lambda x: x["f1"],
    reverse=True
)


# ==========================================
# SVI REZULTATI
# ==========================================

print("\n\n========================================")
print(" SVI REZULTATI - SORTIRANO PO F1")
print("========================================")

print(
    f"{'min_hist':<12}"
    f"{'peak':<10}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<10}"
)

print("-" * 56)

for result in results:

    print(
        f"{result['min_hist_diff']:<12.2f}"
        f"{result['peak_multiplier']:<10.1f}"
        f"{result['precision']:<12.3f}"
        f"{result['recall']:<12.3f}"
        f"{result['f1']:<10.3f}"
    )


# ==========================================
# NAJBOLJA KOMBINACIJA
# ==========================================

best = results[0]

print("\n\n========================================")
print(" NAJBOLJA KOMBINACIJA")
print("========================================")

print(f"min_hist_diff:   {best['min_hist_diff']}")
print(f"peak_multiplier: {best['peak_multiplier']}")
print(f"Precision:       {best['precision']:.3f}")
print(f"Recall:          {best['recall']:.3f}")
print(f"F1-score:        {best['f1']:.3f}")