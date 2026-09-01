import os
import csv
import json
import cv2
import numpy as np

from functions import (
    scene_dynamics,
    interval_for_score,
    generate_keyframes_by_scene,
    combine_sad_histogram
)

from ffmpeg_utils import (
    create_720p,
    encode_adaptive_GOP,
    encode_fixed_GOP,
    encode_default,
    get_video_metrics,
    get_psnr,
    get_ssim,
    get_decoding_time
)


# ============================================================
# VIDEOS
# ============================================================

videos = [
    {
        "name": "Video 1",
        "path": ".\\evaluation\\test1.mp4"
    },
    {
        "name": "Video 2",
        "path": ".\\evaluation\\test2.mp4"
    },
    {
        "name": "Video 3",
        "path": ".\\evaluation\\test3.mp4"
    },
    {
        "name": "Video 4",
        "path": ".\\evaluation\\test4.mp4"
    },
    {
        "name": "Video 5",
        "path": ".\\evaluation\\test5.mp4"
    },
    {
        "name": "Video 6",
        "path": ".\\evaluation\\test6.mp4"
    }
]


# ============================================================
# PARAMETRI COMBINOVANOG SAD + HISTOGRAM ALGORITMA
# ============================================================

WINDOW_SIZE = 30

# SAD
SAD_THRESHOLD_FACTOR = 3.0
MIN_SAD = 5.0
SAD_PEAK_MULTIPLIER = 2.6

# Histogram
HIST_THRESHOLD_FACTOR = 4.0
MIN_HIST_DIFF = 0.05
HIST_PEAK_MULTIPLIER = 1.5
MIN_PEAK_DIFF = 0.5

# Adaptivna težina SAD-a
SAD_LOW_THRESHOLD = 2.0
SAD_HIGH_THRESHOLD = 6.0


# ============================================================
# OUTPUT FOLDER
# ============================================================

RESULTS_FOLDER = ".\\results"

os.makedirs(RESULTS_FOLDER, exist_ok=True)


# ============================================================
# REZULTATI
# ============================================================

all_results = []


# ============================================================
# FUNKCIJA ZA DODAVANJE REZULTATA
# ============================================================

def save_result(
    video_name,
    method,
    metrics,
    psnr,
    ssim,
    decode_time,
    keyframes=None
):

    result = {
        "video": video_name,
        "method": method,
        "i_frames": metrics["i_frame_count"],
        "bitrate_kbps": metrics["bitrate"],
        "size_mb": metrics["size"],
        "average_gop_seconds": metrics["average_interval"],
        "psnr_db": psnr,
        "ssim": ssim,
        "decoding_time_seconds": decode_time
    }

    if keyframes is not None:
        result["keyframes"] = keyframes

    all_results.append(result)


# ============================================================
# GLAVNA EVALUACIJA
# ============================================================

print("\n")
print("=" * 90)
print(" EVALUACIJA SVA TRI METODA")
print("=" * 90)


for video in videos:

    video_name = video["name"]
    original_path = video["path"]

    print("\n")
    print("=" * 90)
    print(video_name)
    print("=" * 90)


    # ========================================================
    # 0. 720p VIDEO
    # ========================================================

    evaluation_path = os.path.join(
        RESULTS_FOLDER,
        f"{video_name.replace(' ', '_')}_720p.mp4"
    )

    if not os.path.exists(evaluation_path):

        print("\nPravljenje 720p verzije...")

        create_720p(
            original_path,
            evaluation_path
        )

    else:

        print("\n720p verzija već postoji - preskačem.")


    # ========================================================
    # VIDEO INFO
    # ========================================================

    cap = cv2.VideoCapture(evaluation_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    print(f"Ukupno frejmova: {total_frames}")
    print(f"FPS: {fps:.2f}")


    # ========================================================
    # 1. COMBINOVANA SCENE DETECTION
    # ========================================================

    print("\n[1/3] Računanje adaptive keyframeova...")
    print("       SAD + Histogram")


    (
        scene_changes,
        sad,
        hist,
        combined,
        combined_thresholds,
        frame_indices,
        weights_sad,
        weights_hist
    ) = combine_sad_histogram(

        evaluation_path,

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


    print("\nDetektovane promene scene:")
    print(scene_changes)


    # ========================================================
    # 2. SCENE BOUNDARIES
    # ========================================================

    scene_boundaries = [0]

    scene_boundaries.extend(
        frame for frame, _ in scene_changes
    )

    scene_boundaries.append(total_frames)

    scene_boundaries = sorted(
        set(scene_boundaries)
    )


    # ========================================================
    # 3. SCENE DYNAMICS
    # ========================================================

    scores = scene_dynamics(
        sad,
        frame_indices,
        scene_boundaries
    )


    # ========================================================
    # 4. ADAPTIVNI INTERVALI
    # ========================================================

    if scores:

        scene_means = [
            scene["mean"]
            for scene in scores
        ]

        low_thresh = np.percentile(
            scene_means,
            25
        )

        high_thresh = np.percentile(
            scene_means,
            75
        )

        for scene in scores:

            scene["interval"] = interval_for_score(
                scene["mean"],
                low_thresh,
                high_thresh,
                min_interval=60,
                max_interval=180
            )


    # ========================================================
    # 5. KEYFRAME NA POČETKU SVАКЕ SCENE
    # ========================================================

    adaptive_keyframes = generate_keyframes_by_scene(
        scene_changes,
        total_frames
    )

    print("\nAdaptive keyframeovi:")
    print(adaptive_keyframes)


    if len(adaptive_keyframes) > 1:

        differences = [
            adaptive_keyframes[i]
            - adaptive_keyframes[i - 1]
            for i in range(1, len(adaptive_keyframes))
        ]

        print("Razmaci:", differences)
        print("Minimalni razmak:", min(differences))
        print(
            "Prosečan razmak:",
            sum(differences) / len(differences)
        )


    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    fixed_output = os.path.join(
        RESULTS_FOLDER,
        f"{video_name.replace(' ', '_')}_fixed.mp4"
    )

    default_output = os.path.join(
        RESULTS_FOLDER,
        f"{video_name.replace(' ', '_')}_default.mp4"
    )

    adaptive_output = os.path.join(
        RESULTS_FOLDER,
        f"{video_name.replace(' ', '_')}_adaptive.mp4"
    )


    # ========================================================
    # 6. FIXED GOP
    # ========================================================

    print("\n[2/3] Enkodovanje FIXED GOP...")

    encode_fixed_GOP(
        evaluation_path,
        fixed_output
    )

    fixed_metrics = get_video_metrics(
        fixed_output
    )

    fixed_psnr = get_psnr(
        evaluation_path,
        fixed_output
    )

    fixed_ssim = get_ssim(
        evaluation_path,
        fixed_output
    )

    fixed_decode = get_decoding_time(
        fixed_output,
        repetitions=3
    )

    save_result(
        video_name,
        "Fixed GOP",
        fixed_metrics,
        fixed_psnr,
        fixed_ssim,
        fixed_decode
    )


    # ========================================================
    # 7. DEFAULT X264
    # ========================================================

    print("\n[3/3] Enkodovanje DEFAULT x264...")

    encode_default(
        evaluation_path,
        default_output
    )

    default_metrics = get_video_metrics(
        default_output
    )

    default_psnr = get_psnr(
        evaluation_path,
        default_output
    )

    default_ssim = get_ssim(
        evaluation_path,
        default_output
    )

    default_decode = get_decoding_time(
        default_output,
        repetitions=3
    )

    save_result(
        video_name,
        "Default x264",
        default_metrics,
        default_psnr,
        default_ssim,
        default_decode
    )


    # ========================================================
    # 8. ADAPTIVE GOP
    # ========================================================

    print("\n[4/3] Enkodovanje ADAPTIVE GOP...")

    encode_adaptive_GOP(
        evaluation_path,
        adaptive_output,
        adaptive_keyframes
    )

    adaptive_metrics = get_video_metrics(
        adaptive_output
    )

    adaptive_psnr = get_psnr(
        evaluation_path,
        adaptive_output
    )

    adaptive_ssim = get_ssim(
        evaluation_path,
        adaptive_output
    )

    adaptive_decode = get_decoding_time(
        adaptive_output,
        repetitions=3
    )

    save_result(
        video_name,
        "Adaptive GOP",
        adaptive_metrics,
        adaptive_psnr,
        adaptive_ssim,
        adaptive_decode,
        adaptive_keyframes
    )


# ============================================================
# ISPIS SVIH REZULTATA
# ============================================================

print("\n\n")
print("=" * 125)
print(" SVI REZULTATI")
print("=" * 125)

print(
    f"{'Video':<10}"
    f"{'Metod':<18}"
    f"{'I-frame':<10}"
    f"{'Bitrate':<15}"
    f"{'Size':<12}"
    f"{'GOP':<12}"
    f"{'PSNR':<10}"
    f"{'SSIM':<12}"
    f"{'Decode':<12}"
)

print("-" * 125)


for result in all_results:

    print(
        f"{result['video']:<10}"
        f"{result['method']:<18}"
        f"{result['i_frames']:<10}"
        f"{result['bitrate_kbps']:<15.2f}"
        f"{result['size_mb']:<12.2f}"
        f"{result['average_gop_seconds']:<12.3f}"
        f"{result['psnr_db']:<10.3f}"
        f"{result['ssim']:<12.6f}"
        f"{result['decoding_time_seconds']:<12.3f}"
    )


# ============================================================
# ČUVANJE CSV
# ============================================================

csv_path = os.path.join(
    RESULTS_FOLDER,
    "results.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "video",
            "method",
            "i_frames",
            "bitrate_kbps",
            "size_mb",
            "average_gop_seconds",
            "psnr_db",
            "ssim",
            "decoding_time_seconds",
            "keyframes"
        ]
    )

    writer.writeheader()

    for result in all_results:

        row = result.copy()

        if "keyframes" in row:

            row["keyframes"] = str(
                row["keyframes"]
            )

        else:

            row["keyframes"] = ""

        writer.writerow(row)


# ============================================================
# ČUVANJE JSON
# ============================================================

json_path = os.path.join(
    RESULTS_FOLDER,
    "results.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        all_results,
        file,
        indent=4
    )


# ============================================================
# PROSECI PO METODI
# ============================================================

methods = [
    "Fixed GOP",
    "Default x264",
    "Adaptive GOP"
]


print("\n\n")
print("=" * 100)
print(" PROSEČNE VREDNOSTI PO METODI")
print("=" * 100)

print(
    f"{'Metod':<18}"
    f"{'I-frameovi':<12}"
    f"{'Bitrate':<15}"
    f"{'Size':<12}"
    f"{'GOP':<12}"
    f"{'PSNR':<10}"
    f"{'SSIM':<12}"
    f"{'Decode':<12}"
)

print("-" * 100)


for method in methods:

    method_results = [
        r for r in all_results
        if r["method"] == method
    ]

    if not method_results:
        continue

    mean_i_frames = sum(
        r["i_frames"]
        for r in method_results
    ) / len(method_results)

    mean_bitrate = sum(
        r["bitrate_kbps"]
        for r in method_results
    ) / len(method_results)

    mean_size = sum(
        r["size_mb"]
        for r in method_results
    ) / len(method_results)

    mean_gop = sum(
        r["average_gop_seconds"]
        for r in method_results
    ) / len(method_results)

    mean_psnr = sum(
        r["psnr_db"]
        for r in method_results
    ) / len(method_results)

    mean_ssim = sum(
        r["ssim"]
        for r in method_results
    ) / len(method_results)

    mean_decode = sum(
        r["decoding_time_seconds"]
        for r in method_results
    ) / len(method_results)

    print(
        f"{method:<18}"
        f"{mean_i_frames:<12.2f}"
        f"{mean_bitrate:<15.2f}"
        f"{mean_size:<12.2f}"
        f"{mean_gop:<12.3f}"
        f"{mean_psnr:<10.3f}"
        f"{mean_ssim:<12.6f}"
        f"{mean_decode:<12.3f}"
    )


# ============================================================
# KRAJ
# ============================================================

print("\n")
print("=" * 90)
print(" GOTOVO")
print("=" * 90)

print(f"CSV rezultati:  {csv_path}")
print(f"JSON rezultati: {json_path}")
print()