import os
import csv
import json
import cv2
import numpy as np

from functions import (
    scene_dynamics,
    interval_for_score,
    generate_keyframes,
    generate_adaptive_gop_structure
)

from save_analysis import get_video_analysis

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
# VIDEO
# ============================================================

VIDEOS = [
    {
        "name": "video6.2",
        "path": "/home/takesi/adaptive_keyframes/evaluation/test6.mp4"
    }
]


# ============================================================
# KOJU METODU ŽELIŠ DA TESTIRAŠ
# ============================================================

# Pokrećeš JEDNU po jednu:
#
# METHODS = ["fixed"]
# METHODS = ["default"]
# METHODS = ["adaptive_i"]
# METHODS = ["adaptive_fixed_pb"]
# METHODS = ["adaptive_pb"]
#
# Ako želiš svih 5 odjednom:
#
# METHODS = [
#     "fixed",
#     "default",
#     "adaptive_i",
#     "adaptive_fixed_pb",
#     "adaptive_pb"
# ]

METHODS = [
    "adaptive_fixed_pb",
    "adaptive_pb"
]


# ============================================================
# NAZIVI METODA
# ============================================================

METHOD_NAMES = {

    "fixed":
        "Fixed GOP",

    "default":
        "Default x264",

    "adaptive_i":
        "Adaptive I - SAD",

    "adaptive_fixed_pb":
        "Adaptive I + Fixed P/B - SAD",

    "adaptive_pb":
        "Adaptive I + Adaptive P/B - SAD"
}


# ============================================================
# ADAPTIVNI PARAMETRI
# ============================================================

MIN_INTERVAL = 60
MAX_INTERVAL = 180

B_MAX = 3
B_MIN = 1


# ============================================================
# OUTPUT
# ============================================================

RESULTS_FOLDER = "results"

os.makedirs(
    RESULTS_FOLDER,
    exist_ok=True
)


# ============================================================
# INFORMACIJE O VIDEU
# ============================================================

def get_video_info(video_path):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    cap.release()

    return total_frames, fps


# ============================================================
# OUTPUT PUTANJA
# ============================================================

def get_output_path(video_name, method):

    prefix = video_name.replace(
        " ",
        "_"
    )

    paths = {

        "fixed":
            f"{prefix}_fixed.mp4",

        "default":
            f"{prefix}_default.mp4",

        "adaptive_i":
            f"{prefix}_adaptive_sad.mp4",

        "adaptive_fixed_pb":
            f"{prefix}_adaptive_fixed_pb_sad.mp4",

        "adaptive_pb":
            f"{prefix}_adaptive_pb_sad.mp4"
    }

    return os.path.join(
        RESULTS_FOLDER,
        paths[method]
    )


# ============================================================
# ČUVANJE JEDNOG REZULTATA
# ============================================================

def save_result(result):

    csv_path = os.path.join(
        RESULTS_FOLDER,
        "all_methods_results.csv"
    )

    json_path = os.path.join(
        RESULTS_FOLDER,
        "all_methods_results.json"
    )


    # ========================================================
    # POSTOJEĆI REZULTATI
    # ========================================================

    existing_results = []


    if os.path.exists(csv_path):

        try:

            with open(
                csv_path,
                "r",
                newline="",
                encoding="utf-8"
            ) as file:

                reader = csv.DictReader(file)

                for row in reader:

                    # Pretvaranje brojeva
                    for key in [
                        "i_frames",
                        "p_frames",
                        "b_frames",
                        "bitrate_kbps",
                        "size_mb",
                        "average_gop_seconds",
                        "psnr_db",
                        "ssim",
                        "decoding_time_seconds",
                        "frame_count"
                    ]:

                        if key in row and row[key] != "":
                            try:
                                row[key] = float(row[key])
                            except:
                                pass

                    existing_results.append(row)

        except Exception as e:

            print(
                f"Greška pri čitanju postojećeg CSV-a: {e}"
            )


    # ========================================================
    # UKLANJANJE STAROG REZULTATA ISTOG VIDEA + METODE
    # ========================================================

    existing_results = [

        r

        for r in existing_results

        if not (
            r.get("video") == result["video"]
            and
            r.get("method") == result["method"]
        )
    ]


    # ========================================================
    # DODAJ NOVI REZULTAT
    # ========================================================

    existing_results.append(
        result
    )


    # ========================================================
    # CSV
    # ========================================================

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


    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for r in existing_results:

            row = r.copy()

            if "keyframes" in row:
                row["keyframes"] = str(
                    row["keyframes"]
                )

            writer.writerow(
                row
            )


    # ========================================================
    # JSON
    # ========================================================

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            existing_results,
            file,
            indent=4
        )


    print()
    print(
        f"Rezultat sačuvan:"
    )

    print(
        f"  {csv_path}"
    )

    print(
        f"  {json_path}"
    )


# ============================================================
# GLAVNI PROGRAM
# ============================================================

print()
print("=" * 110)
print(" EVALUACIJA METODA")
print("=" * 110)

print()
print("Metode koje će biti testirane:")

for method in METHODS:

    print(
        f"  - {METHOD_NAMES[method]}"
    )


# ============================================================
# VIDEO
# ============================================================

for video in VIDEOS:

    video_name = video["name"]
    original_path = video["path"]

    analysis_path = original_path


    print()
    print()
    print("=" * 110)
    print(
        f" VIDEO: {video_name}"
    )
    print("=" * 110)

# ========================================================
# 720p
# ========================================================

    evaluation_path = os.path.join(
        RESULTS_FOLDER,
        f"{video_name.replace(' ', '_')}_720p.mp4"
    )


    # Proveri da li 720p fajl postoji
    # i da li je stvarno ispravan
    if os.path.exists(evaluation_path):

        print()
        print(
            "720p verzija već postoji - proveravam fajl..."
        )

        existing_frames, existing_fps = get_video_info(
            evaluation_path
        )

        if existing_frames > 0 and existing_fps > 0:

            print(
                f"720p fajl je ispravan: "
                f"{existing_frames} frejmova, "
                f"{existing_fps:.2f} FPS"
            )

        else:

            print()
            print(
                "UPOZORENJE: 720p fajl je neispravan."
            )

            print(
                "Brišem ga i pravim ponovo..."
            )

            os.remove(evaluation_path)

            create_720p(
                original_path,
                evaluation_path
            )

    else:

        print()
        print(
            "Pravljenje 720p verzije..."
        )

        create_720p(
            original_path,
            evaluation_path
        )


# ========================================================
# VIDEO INFO
# ========================================================

    total_frames, fps = get_video_info(
        evaluation_path
    )

    if total_frames <= 0 or fps <= 0:

        raise RuntimeError(
            "720p video nije ispravan. "
            "Proveri create_720p() i FFmpeg output."
        )

    print()
    print(
        f"Ukupno frejmova: {total_frames}"
    )

    print(
        f"FPS: {fps:.2f}"
    )


    # ========================================================
    # CACHE
    # ========================================================

    sad_results = None
    hist_results = None

    adaptive_analysis_loaded = False


    # ========================================================
    # METODE
    # ========================================================

    for method_index, METHOD in enumerate(
        METHODS,
        start=1
    ):

        METHOD_NAME = METHOD_NAMES[
            METHOD
        ]


        print()
        print()
        print("#" * 110)

        print(
            f" METODA {method_index}/{len(METHODS)}: "
            f"{METHOD_NAME}"
        )

        print(
            "#" * 110
        )


        # ====================================================
        # PROMENLJIVE
        # ====================================================

        adaptive_keyframes = None
        gop_scenes = None


        # ====================================================
        # ADAPTIVNA ANALIZA
        # ====================================================

        if METHOD in [
            "adaptive_i",
            "adaptive_fixed_pb",
            "adaptive_pb"
        ]:


            if not adaptive_analysis_loaded:

                print()
                print(
                    "Učitavanje SAD + histogram cache-a..."
                )

                (
                    sad_results,
                    hist_results
                ) = get_video_analysis(
                    analysis_path
                )

                adaptive_analysis_loaded = True

            else:

                print()
                print(
                    "SAD + histogram već učitani."
                )


            # =================================================
            # SAD
            # =================================================

            scene_changes = (
                sad_results["scene_changes"]
            )

            sad = (
                sad_results["values"]
            )

            frame_indices = (
                sad_results["frame_indices"]
            )


            print()
            print(
                f"SAD detektovano promena: "
                f"{len(scene_changes)}"
            )


            # =================================================
            # SCENE BOUNDARIES
            # =================================================

            scene_boundaries = sorted(
                set(
                    [0]
                    +
                    [
                        frame
                        for frame, _
                        in scene_changes
                    ]
                    +
                    [total_frames]
                )
            )


            # =================================================
            # SCENE DYNAMICS
            # =================================================

            scores = scene_dynamics(
                sad,
                frame_indices,
                scene_boundaries
            )


            print()
            print(
                f"Broj scena: {len(scores)}"
            )


            # =================================================
            # DINAMIČKI PRAGOVI
            # =================================================

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


                print()
                print(
                    "SAD pragovi za dinamiku:"
                )

                print(
                    f"Low threshold:  "
                    f"{low_thresh:.4f}"
                )

                print(
                    f"High threshold: "
                    f"{high_thresh:.4f}"
                )


                # =================================================
                # GOP INTERVAL
                # =================================================

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


            # =================================================
            # KEYFRAMEOVI
            # =================================================

            adaptive_keyframes = (

                generate_keyframes(

                    scores,

                    total_frames,

                    min_interval=
                        MIN_INTERVAL
                )
            )


            print()
            print(
                "Adaptive keyframeovi:"
            )

            print(
                adaptive_keyframes
            )


            print()
            print(
                f"Ukupno keyframeova: "
                f"{len(adaptive_keyframes)}"
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


                print()
                print(
                    f"Minimalni razmak: "
                    f"{min(differences)}"
                )

                print(
                    f"Prosečan razmak: "
                    f"{sum(differences) / len(differences):.2f}"
                )


            # =================================================
            # ADAPTIVE P/B
            # =================================================

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


                print()
                print(
                    "Adaptive P/B struktura:"
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


            # =================================================
            # ADAPTIVE I + FIXED P/B
            # =================================================

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


                print()
                print(
                    "Adaptive I + Fixed P/B"
                )

                print(
                    "Sve scene koriste:"
                )

                print(
                    "B=3, "
                    "b-adapt=0, "
                    "b-pyramid=none"
                )


        # ====================================================
        # OUTPUT
        # ====================================================

        output_path = get_output_path(
            video_name,
            METHOD
        )


        print()
        print(
            f"Enkodovanje: {METHOD_NAME}"
        )


        # ====================================================
        # ENKODOVANJE
        # ====================================================

        if METHOD == "fixed":

            encode_fixed_GOP(
                evaluation_path,
                output_path
            )


        elif METHOD == "default":

            encode_default(
                evaluation_path,
                output_path
            )


        elif METHOD == "adaptive_i":

            encode_adaptive_GOP(
                evaluation_path,
                output_path,
                adaptive_keyframes
            )


        elif METHOD == "adaptive_fixed_pb":

            encode_adaptive_fixed_pb_GOP(
                evaluation_path,
                output_path,
                adaptive_keyframes
            )


        elif METHOD == "adaptive_pb":

            encode_adaptive_pb_GOP(
                evaluation_path,
                output_path,
                gop_scenes,
                adaptive_keyframes
            )


        # ====================================================
        # METRIKE
        # ====================================================

        print()
        print(
            "Računanje metrika..."
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


        # ====================================================
        # DECODING TIME
        # ====================================================

        print()
        print(
            "Merenje vremena dekodovanja..."
        )


        decode_time = get_decoding_time(
            output_path,
            repetitions=3
        )


        print()
        print(
            f"Decoding time: "
            f"{decode_time:.3f} s"
        )


        # ====================================================
        # FRAME TYPES
        # ====================================================

        frame_types = None
        frame_count = None


        # Za sve metode pokušavamo da dobijemo stvarnu
        # I/P/B strukturu.
        try:

            frame_types = get_frame_type_counts(
                output_path
            )

        except Exception as e:

            print()
            print(
                f"Napomena: nije moguće dobiti I/P/B "
                f"strukturu: {e}"
            )


        # ====================================================
        # BROJ FREJMOVA
        # ====================================================

        try:

            cap = cv2.VideoCapture(
                output_path
            )

            frame_count = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            cap.release()

        except Exception:

            frame_count = None


        # ====================================================
        # PROVERA
        # ====================================================

        if frame_count is not None:

            print()
            print(
                "Provera broja frejmova:"
            )

            print(
                f"Original: {total_frames}"
            )

            print(
                f"Izlaz:    {frame_count}"
            )

            print(
                f"Razlika:  "
                f"{frame_count - total_frames}"
            )


            if frame_count != total_frames:

                print()
                print(
                    "UPOZORENJE: "
                    "broj frejmova nije isti!"
                )


        # ====================================================
        # ISPIS METRIKA
        # ====================================================

        print()
        print("=" * 90)
        print(
            f" REZULTAT: {METHOD_NAME}"
        )
        print("=" * 90)


        print(
            f"Bitrate:       {metrics['bitrate']:.2f} kbps"
        )

        print(
            f"Size:          {metrics['size']:.2f} MB"
        )

        print(
            f"Average GOP:   {metrics['average_interval']:.3f} s"
        )

        print(
            f"PSNR:          {psnr:.3f} dB"
        )

        print(
            f"SSIM:          {ssim:.6f}"
        )

        print(
            f"Decode time:   {decode_time:.3f} s"
        )


        if frame_types is not None:

            print(
                f"I frames:      {frame_types['I']}"
            )

            print(
                f"P frames:      {frame_types['P']}"
            )

            print(
                f"B frames:      {frame_types['B']}"
            )


        # ====================================================
        # REZULTAT
        # ====================================================

        result = {

            "video":
                video_name,

            "method":
                METHOD_NAME,

            "i_frames":
                metrics["i_frame_count"],

            "p_frames":
                frame_types["P"]
                if frame_types is not None
                else None,

            "b_frames":
                frame_types["B"]
                if frame_types is not None
                else None,

            "bitrate_kbps":
                metrics["bitrate"],

            "size_mb":
                metrics["size"],

            "average_gop_seconds":
                metrics["average_interval"],

            "psnr_db":
                psnr,

            "ssim":
                ssim,

            "decoding_time_seconds":
                decode_time,

            "frame_count":
                frame_count,

            "keyframes":
                adaptive_keyframes
                if adaptive_keyframes is not None
                else []
        }


        save_result(
            result
        )


# ============================================================
# KRAJ
# ============================================================

print()
print()
print("=" * 110)
print(" GOTOVO")
print("=" * 110)

print()
print(
    "Rezultati svih do sada testiranih metoda nalaze se u:"
)

print(
    "results/all_methods_results.csv"
)

print(
    "results/all_methods_results.json"
)