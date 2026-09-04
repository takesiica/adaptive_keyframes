import os
import csv
import json
import cv2
import numpy as np

from functions import (
    scene_dynamics,
    interval_for_score,
    generate_keyframes,
    combine_sad_histogram,
    generate_adaptive_gop_structure
)

from ffmpeg_utils import (
    create_720p,
    encode_adaptive_GOP,
    encode_adaptive_pb_GOP,
    encode_adaptive_fixed_pb_GOP,
    encode_fixed_GOP,
    encode_default,
    get_video_metrics,
    get_psnr,
    get_ssim,
    get_decoding_time,
    get_frame_type_counts
)


# ============================================================
# VIDEOS
# ============================================================

videos = [
    {"name": "Video 1", "path": ".\\evaluation\\test1.mp4"},
    {"name": "Video 2", "path": ".\\evaluation\\test2.mp4"},
    {"name": "Video 3", "path": ".\\evaluation\\test3.mp4"},
    {"name": "Video 4", "path": ".\\evaluation\\test4.mp4"},
    {"name": "Video 5", "path": ".\\evaluation\\test5.mp4"},
    {"name": "Video 6", "path": ".\\evaluation\\test6.mp4"}
]


# ============================================================
# IZBOR METODE
# ============================================================

# Promeni SAMO ovu vrednost kada želiš drugu metodu.
#
# "fixed"
# "default"
# "adaptive_i"
# "adaptive_fixed_pb"
# "adaptive_pb"

METHOD = "adaptive_pb"


# ============================================================
# PARAMETRI
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

# Adaptive GOP
MIN_INTERVAL = 60
MAX_INTERVAL = 180

# Adaptive P/B
B_MAX = 3
B_MIN = 1


# ============================================================
# OUTPUT
# ============================================================

RESULTS_FOLDER = ".\\results"

os.makedirs(
    RESULTS_FOLDER,
    exist_ok=True
)

all_results = []


# ============================================================
# NAZIV METODE
# ============================================================

method_names = {
    "fixed": "Fixed GOP",
    "default": "Default x264",
    "adaptive_i": "Adaptive I",
    "adaptive_fixed_pb": "Adaptive I + Fixed P/B",
    "adaptive_pb": "Adaptive I + Adaptive P/B"
}

if METHOD not in method_names:
    raise ValueError(
        f"Nepoznata metoda: {METHOD}"
    )

METHOD_NAME = method_names[METHOD]


# ============================================================
# ČUVANJE REZULTATA
# ============================================================

def save_result(
    video_name,
    method,
    metrics,
    psnr,
    ssim,
    decode_time,
    keyframes=None,
    frame_types=None,
    frame_count=None
):

    result = {
        "video": video_name,
        "method": method,

        "i_frames": metrics["i_frame_count"],

        "p_frames":
            frame_types["P"]
            if frame_types is not None
            else None,

        "b_frames":
            frame_types["B"]
            if frame_types is not None
            else None,

        "bitrate_kbps": metrics["bitrate"],
        "size_mb": metrics["size"],
        "average_gop_seconds":
            metrics["average_interval"],

        "psnr_db": psnr,
        "ssim": ssim,

        "decoding_time_seconds":
            decode_time,

        "frame_count":
            frame_count
    }

    if keyframes is not None:
        result["keyframes"] = keyframes

    all_results.append(result)


# ============================================================
# GLAVNA EVALUACIJA
# ============================================================

print("\n" + "=" * 100)
print(f" EVALUACIJA: {METHOD_NAME}")
print("=" * 100)

for video in videos:

    video_name = video["name"]
    original_path = video["path"]

    print("\n" + "=" * 100)
    print(video_name)
    print("=" * 100)

    # ========================================================
    # 0. 720p
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

        print(
            "\n720p verzija već postoji - preskačem."
        )

    cap = cv2.VideoCapture(
        evaluation_path
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    cap.release()

    print(
        f"Ukupno frejmova: {total_frames}"
    )

    print(
        f"FPS: {fps:.2f}"
    )


    # ========================================================
    # PROMENLJIVE ZA ADAPTIVNE METODE
    # ========================================================

    adaptive_keyframes = None
    scores = None
    gop_scenes = None

    low_thresh = 0
    high_thresh = 1


    # ========================================================
    # 1. SCENE DETECTION
    # Samo za adaptive metode
    # ========================================================

    if METHOD in [
        "adaptive_i",
        "adaptive_fixed_pb",
        "adaptive_pb"
    ]:

        print(
            "\nRačunanje adaptive keyframeova..."
        )

        print(
            "SAD + Histogram"
        )

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

            sad_threshold_factor=
                SAD_THRESHOLD_FACTOR,

            min_sad=
                MIN_SAD,

            sad_peak_multiplier=
                SAD_PEAK_MULTIPLIER,

            hist_threshold_factor=
                HIST_THRESHOLD_FACTOR,

            min_hist_diff=
                MIN_HIST_DIFF,

            hist_peak_multiplier=
                HIST_PEAK_MULTIPLIER,

            min_peak_diff=
                MIN_PEAK_DIFF,

            sad_low_threshold=
                SAD_LOW_THRESHOLD,

            sad_high_threshold=
                SAD_HIGH_THRESHOLD
        )


        # ----------------------------------------------------
        # Scene changes
        # ----------------------------------------------------

        print(
            "\nDetektovane promene scene:"
        )

        print(
            scene_changes
        )


        # ====================================================
        # 2. SCENE BOUNDARIES + DYNAMICS
        # ====================================================

        scene_boundaries = sorted(
            set(
                [0]
                +
                [
                    frame
                    for frame, _ in scene_changes
                ]
                +
                [total_frames]
            )
        )

        scores = scene_dynamics(
            sad,
            frame_indices,
            scene_boundaries
        )


        # ====================================================
        # 3. ADAPTIVNI INTERVALI
        # ====================================================

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

                scene["interval"] = (
                    interval_for_score(
                        scene["mean"],
                        low_thresh,
                        high_thresh,
                        min_interval=
                            MIN_INTERVAL,
                        max_interval=
                            MAX_INTERVAL
                    )
                )


        # ====================================================
        # 4. ADAPTIVE I KEYFRAMEOVI
        # ====================================================

        adaptive_keyframes = generate_keyframes(
            scores,
            total_frames,
            min_interval=MIN_INTERVAL
        )

        print(
            "\nAdaptive keyframeovi:"
        )

        print(
            adaptive_keyframes
        )


        if len(adaptive_keyframes) > 1:

            differences = [
                adaptive_keyframes[i]
                -
                adaptive_keyframes[i - 1]

                for i in range(
                    1,
                    len(adaptive_keyframes)
                )
            ]

            print(
                "Razmaci:",
                differences
            )

            print(
                "Minimalni razmak:",
                min(differences)
            )

            print(
                "Prosečan razmak:",
                sum(differences)
                /
                len(differences)
            )


        # ====================================================
        # 5. ADAPTIVE P/B STRUKTURA
        # ====================================================

        if METHOD == "adaptive_pb":

            gop_scenes = [
                generate_adaptive_gop_structure(
                    scene,
                    low_thresh,
                    high_thresh,
                    b_max=B_MAX,
                    b_min=B_MIN
                )

                for scene in scores
            ]

            print(
                "\nAdaptive P/B struktura:"
            )

            for i, g in enumerate(
                gop_scenes
            ):

                print(
                    f"Scena {i}: "
                    f"start={g['start']} "
                    f"end={g['end']} "
                    f"mean={g['mean']:.4f} "
                    f"complexity={g['complexity']:.3f} "
                    f"type={g['type']} "
                    f"B={g['bframes']}"
                )


        # ====================================================
        # ADAPTIVE I + FIXED P/B
        # ====================================================

        elif METHOD == "adaptive_fixed_pb":

            gop_scenes = [
                {
                    "start":
                        scene["start"],

                    "end":
                        scene["end"]
                }

                for scene in scores
            ]

            print(
                "\nAdaptive I + Fixed P/B"
            )

            print(
                "Sve scene koriste:"
            )

            print(
                "B=3, "
                "b-adapt=0, "
                "b-pyramid=none"
            )


    # ========================================================
    # OUTPUT PATH
    # ========================================================

    prefix = video_name.replace(
        " ",
        "_"
    )


    if METHOD == "fixed":

        output_path = os.path.join(
            RESULTS_FOLDER,
            f"{prefix}_fixed.mp4"
        )

    elif METHOD == "default":

        output_path = os.path.join(
            RESULTS_FOLDER,
            f"{prefix}_default.mp4"
        )

    elif METHOD == "adaptive_i":

        output_path = os.path.join(
            RESULTS_FOLDER,
            f"{prefix}_adaptive.mp4"
        )

    elif METHOD == "adaptive_fixed_pb":

        output_path = os.path.join(
            RESULTS_FOLDER,
            f"{prefix}_adaptive_fixed_pb.mp4"
        )

    elif METHOD == "adaptive_pb":

        output_path = os.path.join(
            RESULTS_FOLDER,
            f"{prefix}_adaptive_pb.mp4"
        )


    # ========================================================
    # 6. ENKODOVANJE
    # ========================================================

    print(
        f"\nEnkodovanje: {METHOD_NAME}"
    )


    # --------------------------------------------------------
    # FIXED GOP
    # --------------------------------------------------------

    if METHOD == "fixed":

        encode_fixed_GOP(
            evaluation_path,
            output_path
        )


    # --------------------------------------------------------
    # DEFAULT X264
    # --------------------------------------------------------

    elif METHOD == "default":

        encode_default(
            evaluation_path,
            output_path
        )


    # --------------------------------------------------------
    # ADAPTIVE I
    # --------------------------------------------------------

    elif METHOD == "adaptive_i":

        encode_adaptive_GOP(
            evaluation_path,
            output_path,
            adaptive_keyframes
        )


    # --------------------------------------------------------
    # ADAPTIVE I + FIXED P/B
    # --------------------------------------------------------

    elif METHOD == "adaptive_fixed_pb":

        encode_adaptive_fixed_pb_GOP(
            evaluation_path,
            output_path,
            gop_scenes,
            adaptive_keyframes
        )


    # --------------------------------------------------------
    # ADAPTIVE I + ADAPTIVE P/B
    # --------------------------------------------------------

    elif METHOD == "adaptive_pb":

        encode_adaptive_pb_GOP(
            evaluation_path,
            output_path,
            gop_scenes,
            adaptive_keyframes
        )


    # ========================================================
    # 7. METRIKE
    # ========================================================

    print(
        "\nRačunanje metrika..."
    )


    metrics = get_video_metrics(
        output_path
    )


    psnr = get_psnr(
        evaluation_path,
        output_path
    )


    ssim = get_ssim(
        evaluation_path,
        output_path
    )


    decode_time = get_decoding_time(
        output_path,
        repetitions=3
    )


    # ========================================================
    # 8. FRAME TYPES
    # ========================================================

    frame_types = None
    frame_count = None


    if METHOD in [
        "adaptive_fixed_pb",
        "adaptive_pb"
    ]:

        frame_types = get_frame_type_counts(
            output_path
        )


        cap = cv2.VideoCapture(
            output_path
        )

        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        cap.release()


        print(
            "\nProvera broja frejmova:"
        )

        print(
            f"Original: "
            f"{total_frames}"
        )

        print(
            f"Izlaz:    "
            f"{frame_count}"
        )

        print(
            f"Razlika:  "
            f"{frame_count - total_frames}"
        )


        print(
            "\nStvarna struktura:"
        )

        print(
            f"I={frame_types['I']} "
            f"P={frame_types['P']} "
            f"B={frame_types['B']}"
        )


        # ----------------------------------------------------
        # OBAVEZNA PROVERA
        # ----------------------------------------------------

        if frame_count != total_frames:

            print(
                "\nUPOZORENJE:"
            )

            print(
                "Broj frejmova izlaznog videa "
                "nije isti kao originalni!"
            )


    # ========================================================
    # 9. ČUVANJE REZULTATA
    # ========================================================

    save_result(

        video_name,

        METHOD_NAME,

        metrics,

        psnr,

        ssim,

        decode_time,

        adaptive_keyframes,

        frame_types,

        frame_count
    )


# ============================================================
# ISPIS REZULTATA
# ============================================================

print(
    "\n\n"
    + "=" * 145
)

print(
    f" REZULTATI: {METHOD_NAME}"
)

print(
    "=" * 145
)


print(
    f"{'Video':<10}"
    f"{'Metod':<28}"
    f"{'I':<7}"
    f"{'P':<7}"
    f"{'B':<7}"
    f"{'Bitrate':<13}"
    f"{'Size':<10}"
    f"{'GOP':<10}"
    f"{'PSNR':<9}"
    f"{'SSIM':<11}"
    f"{'Decode':<10}"
)


print(
    "-" * 145
)


for r in all_results:

    print(

        f"{r['video']:<10}"

        f"{r['method']:<28}"

        f"{str(r['i_frames']):<7}"

        f"{str(r['p_frames']) if r['p_frames'] is not None else '-':<7}"

        f"{str(r['b_frames']) if r['b_frames'] is not None else '-':<7}"

        f"{r['bitrate_kbps']:<13.2f}"

        f"{r['size_mb']:<10.2f}"

        f"{r['average_gop_seconds']:<10.3f}"

        f"{r['psnr_db']:<9.3f}"

        f"{r['ssim']:<11.6f}"

        f"{r['decoding_time_seconds']:<10.3f}"
    )


# ============================================================
# CSV
# ============================================================

csv_path = os.path.join(
    RESULTS_FOLDER,
    f"{METHOD}_results.csv"
)


with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [

        "video",
        "method",

        "i_frames",
        "p_frames",
        "b_frames",

        "bitrate_kbps",
        "size_mb",
        "average_gop_seconds",

        "psnr_db",
        "ssim",

        "decoding_time_seconds",

        "frame_count",
        "keyframes"
    ]


    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )


    writer.writeheader()


    for result in all_results:

        row = result.copy()

        row["keyframes"] = str(
            row.get(
                "keyframes",
                ""
            )
        )

        writer.writerow(row)


# ============================================================
# JSON
# ============================================================

json_path = os.path.join(
    RESULTS_FOLDER,
    f"{METHOD}_results.json"
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
# PROSEK TRENUTNE METODE
# ============================================================

if all_results:

    def avg(key):

        values = [
            r[key]

            for r in all_results

            if r[key] is not None
        ]

        return (
            sum(values) / len(values)
            if values
            else 0
        )


    print(
        "\n\n"
        + "=" * 120
    )

    print(
        f" PROSEK: {METHOD_NAME}"
    )

    print(
        "=" * 120
    )


    print(
        f"{'Metod':<28}"
        f"{'I':<9}"
        f"{'P':<9}"
        f"{'B':<9}"
        f"{'Bitrate':<14}"
        f"{'Size':<11}"
        f"{'GOP':<11}"
        f"{'PSNR':<10}"
        f"{'SSIM':<12}"
        f"{'Decode':<11}"
    )


    print(
        "-" * 120
    )


    print(

        f"{METHOD_NAME:<28}"

        f"{avg('i_frames'):<9.2f}"

        f"{avg('p_frames'):<9.2f}"

        f"{avg('b_frames'):<9.2f}"

        f"{avg('bitrate_kbps'):<14.2f}"

        f"{avg('size_mb'):<11.2f}"

        f"{avg('average_gop_seconds'):<11.3f}"

        f"{avg('psnr_db'):<10.3f}"

        f"{avg('ssim'):<12.6f}"

        f"{avg('decoding_time_seconds'):<11.3f}"
    )


# ============================================================
# KRAJ
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    " GOTOVO"
)

print(
    "=" * 100
)

print(
    f"Metoda:          {METHOD_NAME}"
)

print(
    f"CSV rezultati:   {csv_path}"
)

print(
    f"JSON rezultati:  {json_path}"
)