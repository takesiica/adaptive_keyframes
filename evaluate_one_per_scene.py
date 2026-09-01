import os
import csv
import json
import cv2

from functions import (
    adaptive_threshold1,
    generate_keyframes_by_scene
)

from ffmpeg_utils import (
    encode_one_I_frame_per_scene,
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
# PARAMETRI SCENE DETEKCIJE
# ============================================================

WINDOW_SIZE = 45
THRESHOLD_FACTOR = 4.0
MIN_SAD = 5.0
PEAK_MULTIPLIER = 1.5


# ============================================================
# FOLDERI
# ============================================================

RESULTS_FOLDER = ".\\results"

os.makedirs(RESULTS_FOLDER, exist_ok=True)


# ============================================================
# POSTOJEĆI REZULTATI
# ============================================================

csv_path = os.path.join(
    RESULTS_FOLDER,
    "results.csv"
)

json_path = os.path.join(
    RESULTS_FOLDER,
    "results.json"
)


# ============================================================
# UČITAVANJE POSTOJEĆEG JSON-A
# ============================================================

if os.path.exists(json_path):

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as file:

        all_results = json.load(file)

else:

    all_results = []


# ============================================================
# PROVERA DA LI VEĆ POSTOJI METOD
# ============================================================

existing_one_per_scene = {
    (result["video"], result["method"])
    for result in all_results
}


# ============================================================
# POČETAK
# ============================================================

print("\n")
print("=" * 100)
print(" EVALUACIJA - ONE I-FRAME PER SCENE")
print("=" * 100)


# ============================================================
# EVALUACIJA SVIH 6 VIDEA
# ============================================================

new_results = []


for video in videos:

    video_name = video["name"]

    print("\n")
    print("=" * 90)
    print(video_name)
    print("=" * 90)

    # --------------------------------------------------------
    # Ako već postoji rezultat, preskoči
    # --------------------------------------------------------

    if (
        video_name,
        "One I-frame per Scene"
    ) in existing_one_per_scene:

        print(
            "Rezultat za ovaj video već postoji - preskačem."
        )

        continue


    # --------------------------------------------------------
    # 720p INPUT
    # --------------------------------------------------------

    input_path = os.path.join(
        RESULTS_FOLDER,
        f"{video_name.replace(' ', '_')}_720p.mp4"
    )


    if not os.path.exists(input_path):

        print(
            "GREŠKA: 720p verzija ne postoji:"
        )

        print(input_path)

        continue


    # --------------------------------------------------------
    # VIDEO INFO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(input_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    print(
        f"Ukupno frejmova: {total_frames}"
    )

    print(
        f"FPS: {fps:.2f}"
    )


    # ========================================================
    # 1. SCENE DETECTION
    # ========================================================

    print(
        "\n[1/3] Računanje scene boundaries..."
    )

    scene_changes, _, _, _ = adaptive_threshold1(
        input_path,
        window_size=WINDOW_SIZE,
        threshold_factor=THRESHOLD_FACTOR,
        min_sad=MIN_SAD,
        peak_multiplier=PEAK_MULTIPLIER
    )


    # ========================================================
    # 2. KEYFRAME NA POČETKU SVАKE SCENE
    # ========================================================

    scene_keyframes = generate_keyframes_by_scene(
        scene_changes,
        total_frames
    )

    print("\nScene boundaries:")

    print([
        frame
        for frame, _ in scene_changes
    ])


    print("\nKeyframeovi:")

    print(scene_keyframes)


    # ========================================================
    # 3. ENKODOVANJE
    # ========================================================

    output_path = os.path.join(
        RESULTS_FOLDER,
        f"{video_name.replace(' ', '_')}_one_per_scene.mp4"
    )

    print(
        "\n[2/3] Enkodovanje ONE I-FRAME PER SCENE..."
    )

    encode_one_I_frame_per_scene(
        input_path,
        output_path,
        scene_keyframes
    )


    # ========================================================
    # 4. METRIKE
    # ========================================================

    print(
        "\n[3/3] Računanje metrika..."
    )

    metrics = get_video_metrics(
        output_path
    )


    # --------------------------------------------------------
    # PSNR
    # --------------------------------------------------------

    psnr = get_psnr(
        input_path,
        output_path
    )


    # --------------------------------------------------------
    # SSIM
    # --------------------------------------------------------

    ssim = get_ssim(
        input_path,
        output_path
    )


    # --------------------------------------------------------
    # DECODING TIME
    # --------------------------------------------------------

    decode_time = get_decoding_time(
        output_path,
        repetitions=5
    )


    # ========================================================
    # ISPIS
    # ========================================================

    print("\n")
    print("--- REZULTATI ---")

    print(
        f"I-frameovi: "
        f"{metrics['i_frame_count']}"
    )

    print(
        f"Bitrate: "
        f"{metrics['bitrate']:.2f} kb/s"
    )

    print(
        f"Veličina: "
        f"{metrics['size']:.2f} MB"
    )

    print(
        f"Prosečan GOP: "
        f"{metrics['average_interval']:.3f} s"
    )

    print(
        f"PSNR: "
        f"{psnr:.3f} dB"
    )

    print(
        f"SSIM: "
        f"{ssim:.6f}"
    )

    print(
        f"Decoding time: "
        f"{decode_time:.3f} s"
    )


    # ========================================================
    # ČUVANJE REZULTATA
    # ========================================================

    result = {
        "video": video_name,
        "method": "One I-frame per Scene",
        "i_frames": metrics["i_frame_count"],
        "bitrate_kbps": metrics["bitrate"],
        "size_mb": metrics["size"],
        "average_gop_seconds": metrics["average_interval"],
        "psnr_db": psnr,
        "ssim": ssim,
        "decoding_time_seconds": decode_time,
        "keyframes": scene_keyframes
    }

    new_results.append(result)


# ============================================================
# AKO NEMA NOVIH REZULTATA
# ============================================================

if not new_results:

    print("\n")
    print("=" * 90)
    print("Nema novih rezultata za dodavanje.")
    print("=" * 90)

    exit()


# ============================================================
# DODAVANJE U JSON
# ============================================================

print("\n")
print("Čuvanje rezultata u JSON...")

all_results.extend(new_results)


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
# DODAVANJE U CSV
# ============================================================

print("Čuvanje rezultata u CSV...")


fieldnames = [
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


# Ako CSV ne postoji, napravi ga sa headerom
csv_exists = os.path.exists(csv_path)


with open(
    csv_path,
    "a",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    if not csv_exists:

        writer.writeheader()


    for result in new_results:

        row = result.copy()

        row["keyframes"] = str(
            row["keyframes"]
        )

        writer.writerow(row)


# ============================================================
# ISPIS NOVIH REZULTATA
# ============================================================

print("\n")
print("=" * 120)
print(" ONE I-FRAME PER SCENE - REZULTATI")
print("=" * 120)

print(
    f"{'Video':<10}"
    f"{'Metod':<25}"
    f"{'I-frame':<10}"
    f"{'Bitrate':<15}"
    f"{'Size':<12}"
    f"{'GOP':<12}"
    f"{'PSNR':<10}"
    f"{'SSIM':<12}"
    f"{'Decode':<12}"
)

print("-" * 120)


for result in new_results:

    print(
        f"{result['video']:<10}"
        f"{result['method']:<25}"
        f"{result['i_frames']:<10}"
        f"{result['bitrate_kbps']:<15.2f}"
        f"{result['size_mb']:<12.2f}"
        f"{result['average_gop_seconds']:<12.3f}"
        f"{result['psnr_db']:<10.3f}"
        f"{result['ssim']:<12.6f}"
        f"{result['decoding_time_seconds']:<12.3f}"
    )


# ============================================================
# KRAJ
# ============================================================

print("\n")
print("=" * 100)
print(" GOTOVO")
print("=" * 100)

print(
    f"CSV:  {csv_path}"
)

print(
    f"JSON: {json_path}"
)

print(
    f"Dodato novih rezultata: {len(new_results)}"
)

print()