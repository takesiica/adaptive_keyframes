import cv2
from functions import get_ground_truth_times, adaptive_threshold1
from analysis_cache import get_video_analysis
from evaluate import evaluatef
import numpy as np
import os


# ============================================================
# VIDEO + BBC ANOTACIJE
# ============================================================

videos = [
    {
        "name": "Video 1",
        "path": "bbc_01.mp4",
        "annotation_path": "01_From_Pole_to_Pole.txt"
    }
]


# ============================================================
# PARAMETRI SAD
# ============================================================

WINDOW_SIZE = 30

SAD_THRESHOLD_FACTOR = 3.0
MIN_SAD = 5.0
SAD_PEAK_MULTIPLIER = 2.6

TOLERANCE = 1.5


# ============================================================
# REZULTATI
# ============================================================

all_precisions = []
all_recalls = []
all_f1_scores = []


# ============================================================
# NASLOV
# ============================================================

print("\n")
print("=" * 90)
print(" SAD - BBC ANOTACIJE")
print("=" * 90)

print(f"Window size:          {WINDOW_SIZE}")
print(f"SAD threshold factor: {SAD_THRESHOLD_FACTOR}")
print(f"Min SAD:              {MIN_SAD}")
print(f"SAD peak multiplier:  {SAD_PEAK_MULTIPLIER}")
print(f"Tolerance:            {TOLERANCE} s")


# ============================================================
# OBRADA SVAKOG VIDEA
# ============================================================

for video in videos:

    input_path = video["path"]
    annotation_path = video["annotation_path"]

    print("\n")
    print("-" * 90)
    print(video["name"])
    print("-" * 90)

    # --------------------------------------------------------
    # PROVERA VIDEO FAJLA
    # --------------------------------------------------------

    if not os.path.exists(input_path):
        print("GREŠKA: Video ne postoji:")
        print(f"        {input_path}")
        continue

    # --------------------------------------------------------
    # PROVERA ANOTACIJA
    # --------------------------------------------------------

    if not os.path.exists(annotation_path):
        print("GREŠKA: Anotacije ne postoje:")
        print(f"        {annotation_path}")
        continue

    # --------------------------------------------------------
    # INFORMACIJE O VIDEU
    # --------------------------------------------------------

    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    if fps <= 0:
        print("GREŠKA: Nije moguće dobiti FPS.")
        continue

    print(f"FPS:                  {fps:.3f}")
    print(f"Ukupno frejmova:      {frame_count}")

    # --------------------------------------------------------
    # GROUND TRUTH IZ BBC ANOTACIJA
    # --------------------------------------------------------

    true_frames, true_times = get_ground_truth_times(
        annotation_path,
        fps
    )

    print(
        f"Broj anotiranih shotova: "
        f"{len(true_frames) + 1}"
    )

    print(
        f"Broj ground-truth promena: "
        f"{len(true_frames)}"
    )

    # --------------------------------------------------------
    # SAD DETEKCIJA
    # --------------------------------------------------------

    (
        scene_changes,
        all_mean_sads,
        thresholds,
        frame_indices
    ) = adaptive_threshold1(

        input_path,

        window_size=WINDOW_SIZE,
        threshold_factor=SAD_THRESHOLD_FACTOR,
        min_sad=MIN_SAD,
        peak_multiplier=SAD_PEAK_MULTIPLIER
    )

    # --------------------------------------------------------
    # DETEKTOVANI FREJMOVI
    # --------------------------------------------------------

    detected_frames = [
        frame
        for frame, _ in scene_changes
    ]

    # --------------------------------------------------------
    # KONVERZIJA FREJMOVA U SEKUNDE
    # --------------------------------------------------------

    detected_times = [
        frame / fps
        for frame in detected_frames
    ]

    # --------------------------------------------------------
    # EVALUACIJA
    # --------------------------------------------------------

    precision, recall, f1 = evaluatef(
        detected_times,
        true_times,
        tolerance=TOLERANCE
    )

    # --------------------------------------------------------
    # ČUVANJE REZULTATA
    # --------------------------------------------------------

    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1_scores.append(f1)

    # --------------------------------------------------------
    # ISPIS REZULTATA
    # --------------------------------------------------------

    print()

    print(
        f"Detektovano promena: "
        f"{len(detected_frames)}"
    )

    print(
        f"Detektovani frejmovi: "
        f"{detected_frames}"
    )

    print()

    print(
        f"Ground-truth frejmovi: "
        f"{true_frames}"
    )

    print()

    print(
        f"Detektovana vremena: "
        f"{[round(t, 2) for t in detected_times]}"
    )

    print(
        f"Tačna vremena:       "
        f"{[round(t, 2) for t in true_times]}"
    )

    print()

    print(f"Precision = {precision:.3f}")
    print(f"Recall    = {recall:.3f}")
    print(f"F1        = {f1:.3f}")


# ============================================================
# UKUPNI REZULTAT
# ============================================================

if len(all_precisions) > 0:

    mean_precision = np.mean(all_precisions)
    mean_recall = np.mean(all_recalls)
    mean_f1 = np.mean(all_f1_scores)

    print("\n\n")
    print("=" * 90)
    print(" UKUPNI REZULTAT - SAD")
    print("=" * 90)

    print(
        f"Prosečan Precision: "
        f"{mean_precision:.3f}"
    )

    print(
        f"Prosečan Recall:    "
        f"{mean_recall:.3f}"
    )

    print(
        f"Prosečan F1:        "
        f"{mean_f1:.3f}"
    )

    print("=" * 90)

else:

    print("\nNema rezultata za prikaz.")