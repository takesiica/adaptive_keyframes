import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from feature_cache import get_raw_features

from functions import (
    scene_dynamics,
    interval_for_score,
    generate_keyframes,
    generate_adaptive_gop_structure
)

from save_analysis import get_video_analysis

from ffmpeg_utils import (
    encode_fixed_GOP,
    encode_default,
    encode_adaptive_GOP,
    encode_adaptive_fixed_pb_GOP,
    encode_adaptive_pb_GOP,
    encode_adaptive_fixed_pb_GOP_scene_by_scene,
    probe_video,
    probe_frames,
    validate_video,
    get_quality_metrics,
    measure_decoding
)


# ============================================================
# PARAMETRI
# ============================================================

MIN_INTERVAL = 60
MAX_INTERVAL = 180

B_MIN = 1
B_MAX = 3

METHODS = [
    "fixed",
    "default",
    "adaptive_i",
    "adaptive_fixed_pb",
    "adaptive_pb",
    "fixed_pb_scene_by_scene"
]


# ============================================================
# PLAN ZA ADAPTIVNE METODE
# ============================================================

def make_adaptive_plan(
    input_path,
    raw,
    frame_count,
    cache_dir
):

    sad, _ = get_video_analysis(
        input_path,
        cache_dir,
        raw_features=raw
    )

    # --------------------------------------------------------
    # GRANICE SCENA
    # --------------------------------------------------------

    boundaries = sorted(
        set(
            [0]
            + [
                frame
                for frame, _ in sad["scene_changes"]
            ]
            + [frame_count]
        )
    )

    # --------------------------------------------------------
    # DINAMIKA SCENA
    # --------------------------------------------------------

    scenes = scene_dynamics(
        sad["values"],
        sad["frame_indices"],
        boundaries
    )

    # --------------------------------------------------------
    # PROVERA SCENA
    # --------------------------------------------------------

    expected_scenes = list(
        zip(
            boundaries[:-1],
            boundaries[1:]
        )
    )

    actual_scenes = [
        (
            scene["start"],
            scene["end"]
        )
        for scene in scenes
    ]

    if actual_scenes != expected_scenes:
        raise ValueError(
            "Scene boundaries and SAD features do not match."
        )

    # --------------------------------------------------------
    # PRAGOVI DINAMIKE
    # --------------------------------------------------------

    means = [
        scene["mean"]
        for scene in scenes
    ]

    low, high = np.percentile(
        means,
        [25, 75]
    )

    if (
        not np.isfinite(low + high)
        or high <= low
    ):
        raise ValueError(
            "Invalid scene-dynamics thresholds."
        )

    # --------------------------------------------------------
    # ADAPTIVNI GOP INTERVAL
    # --------------------------------------------------------

    for scene in scenes:

        scene["interval"] = interval_for_score(
            scene["mean"],
            low,
            high,
            min_interval=MIN_INTERVAL,
            max_interval=MAX_INTERVAL
        )

    # --------------------------------------------------------
    # ADAPTIVNI I-FRAMEOVI
    # --------------------------------------------------------

    keyframes = generate_keyframes(
        scenes,
        frame_count,
        min_interval=MIN_INTERVAL
    )

    # --------------------------------------------------------
    # SVAKA SCENA MORA POČETI I-FRAMEOM
    # --------------------------------------------------------

    scene_start_keyframes = [
        scene["start"]
        for scene in scenes
    ]

    keyframes = sorted(
        set(
            keyframes
            + scene_start_keyframes
        )
    )

    # --------------------------------------------------------
    # PROVERA KEYFRAME PLANA
    # --------------------------------------------------------

    if (
        not keyframes
        or keyframes[0] != 0
        or any(
            k < 0 or k >= frame_count
            for k in keyframes
        )
    ):
        raise ValueError(
            "Invalid adaptive keyframe positions."
        )

    # --------------------------------------------------------
    # ADAPTIVNA GOP STRUKTURA
    # --------------------------------------------------------

    gop_scenes = [
        generate_adaptive_gop_structure(
            scene,
            low,
            high,
            b_max=B_MAX,
            b_min=B_MIN
        )
        for scene in scenes
    ]

    # --------------------------------------------------------
    # PROVERA B-FRAME PLANA
    # --------------------------------------------------------

    for scene in gop_scenes:

        if scene["bframes"] not in (1, 2, 3):
            raise ValueError(
                f"Invalid B-frame count: "
                f"{scene['bframes']}"
            )

    # --------------------------------------------------------
    # PROVERA GRANICA GOP SCENA
    # --------------------------------------------------------

    gop_scene_ranges = [
        (
            scene["start"],
            scene["end"]
        )
        for scene in gop_scenes
    ]

    if gop_scene_ranges != expected_scenes:
        raise ValueError(
            "GOP scene boundaries do not match "
            "adaptive scene boundaries."
        )

    return {
        "keyframes": keyframes,
        "scenes": scenes,
        "gop_scenes": gop_scenes,
        "low": float(low),
        "high": float(high)
    }


# ============================================================
# PROVERA KONTROLISANOG PLANA
# ============================================================

def validate_controlled_plan(plan):

    keyframes = plan["keyframes"]
    gop_scenes = plan["gop_scenes"]

    # --------------------------------------------------------
    # I-FRAMEOVI
    # --------------------------------------------------------

    if not keyframes:
        raise ValueError(
            "Adaptive plan contains no keyframes."
        )

    if keyframes[0] != 0:
        raise ValueError(
            "Adaptive keyframe plan must start at frame 0."
        )

    if keyframes != sorted(set(keyframes)):
        raise ValueError(
            "Adaptive keyframes are not sorted/unique."
        )

    # --------------------------------------------------------
    # GOP SCENE
    # --------------------------------------------------------

    if not gop_scenes:
        raise ValueError(
            "Adaptive plan contains no GOP scenes."
        )

    for i, scene in enumerate(gop_scenes):

        if scene["end"] <= scene["start"]:
            raise ValueError(
                f"Invalid GOP scene {i}: "
                f"[{scene['start']}, {scene['end']})"
            )

        if scene["bframes"] not in (1, 2, 3):
            raise ValueError(
                f"Invalid B-frame count in scene {i}: "
                f"{scene['bframes']}"
            )

    # --------------------------------------------------------
    # SCENE GRANICE MORAJU BITI KONTINUIRANE
    # --------------------------------------------------------

    for previous, current in zip(
        gop_scenes,
        gop_scenes[1:]
    ):

        if previous["end"] != current["start"]:
            raise ValueError(
                "GOP scenes are not contiguous."
            )

    # --------------------------------------------------------
    # PRVA SCENA POČINJE OD 0
    # --------------------------------------------------------

    if gop_scenes[0]["start"] != 0:
        raise ValueError(
            "First GOP scene must start at frame 0."
        )

    # --------------------------------------------------------
    # SVAKI POČETAK SCENE MORA BITI I-FRAME
    # --------------------------------------------------------

    scene_starts = [
        scene["start"]
        for scene in gop_scenes
    ]

    missing_scene_starts = sorted(
        set(scene_starts)
        - set(keyframes)
    )

    if missing_scene_starts:
        raise ValueError(
            "Some scene starts are missing from "
            f"the keyframe plan: {missing_scene_starts}"
        )


# ============================================================
# ISPIS ADAPTIVNOG PLANA
# ============================================================

def print_adaptive_plan(plan):

    print()
    print("=" * 100)
    print("ADAPTIVE PLAN")
    print("=" * 100)

    print(
        f"Number of I-frames: "
        f"{len(plan['keyframes'])}"
    )

    print(
        f"Scene thresholds: "
        f"{plan['low']:.3f} / "
        f"{plan['high']:.3f}"
    )

    print()

    print("Keyframes:")
    print(plan["keyframes"])

    print()

    print("B-FRAME PLAN:")

    print(
        f"{'Scene':<10}"
        f"{'Frames':<25}"
        f"{'Mean SAD':<15}"
        f"{'GOP interval':<18}"
        f"{'B frames':<10}"
    )

    print("-" * 100)

    for i, scene in enumerate(
        plan["gop_scenes"]
    ):

        print(
            f"{i + 1:<10}"
            f"[{scene['start']}, "
            f"{scene['end']})"
            f"{'':<8}"
            f"{scene['mean']:<15.3f}"
            f"{scene['interval']:<18}"
            f"{scene['bframes']:<10}"
        )


# ============================================================
# ENCODING
# ============================================================

def encode_video(
    method,
    input_path,
    output_path,
    plan
):

    if method == "fixed":

        encode_fixed_GOP(
            input_path,
            output_path
        )

    elif method == "default":

        encode_default(
            input_path,
            output_path
        )

    elif method == "adaptive_i":

        encode_adaptive_GOP(
            input_path,
            output_path,
            plan["keyframes"]
        )

    elif method == "adaptive_fixed_pb":

        encode_adaptive_fixed_pb_GOP(
            input_path,
            output_path,
            plan["keyframes"]
        )

    elif method == "adaptive_pb":

        encode_adaptive_pb_GOP(
            input_path,
            output_path,
            plan["gop_scenes"],
            plan["keyframes"]
        )

    elif method == "fixed_pb_scene_by_scene":

        encode_adaptive_fixed_pb_GOP_scene_by_scene(
            input_path,
            output_path,
            plan["gop_scenes"],
            plan["keyframes"]
        )

    else:

        raise ValueError(
            f"Unknown method: {method}"
        )


# ============================================================
# GLAVNA EVALUACIJA
# ============================================================

def evaluate_video(
    input_path,
    results_dir,
    cache_dir,
    methods,
    repetitions,
    cpu=None,
    overwrite=False
):

    input_path = str(
        Path(input_path).resolve()
    )

    video_name = Path(
        input_path
    ).stem

    output_dir = (
        Path(results_dir)
        / video_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # OSNOVNE INFORMACIJE O VIDEU
    # --------------------------------------------------------

    info = probe_video(
        input_path
    )

    frames = probe_frames(
        input_path
    )

    frame_count = frames[
        "frame_count"
    ]

    print()
    print("=" * 80)
    print(
        f"VIDEO: {video_name}"
    )
    print("=" * 80)

    print(
        f"Resolution: "
        f"{info['video']['width']}x"
        f"{info['video']['height']}"
    )

    print(
        f"Frames:     {frame_count}"
    )

    print(
        f"FPS:        "
        f"{info['video'].get('r_frame_rate', 'unknown')}"
    )

    # --------------------------------------------------------
    # RAW SAD FEATURES
    # --------------------------------------------------------

    raw = get_raw_features(
        input_path,
        cache_dir
    )

    if raw["frame_count"] != frame_count:
        raise ValueError(
            "OpenCV and FFprobe frame counts differ."
        )

    # --------------------------------------------------------
    # JEDAN ZAJEDNIČKI ADAPTIVNI PLAN
    # --------------------------------------------------------

    adaptive_plan = make_adaptive_plan(
        input_path,
        raw,
        frame_count,
        cache_dir
    )

    # --------------------------------------------------------
    # VALIDACIJA PLANA
    # --------------------------------------------------------

    validate_controlled_plan(
        adaptive_plan
    )

    # --------------------------------------------------------
    # ISPIS PLANA
    # --------------------------------------------------------

    print_adaptive_plan(
        adaptive_plan
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    results = []

    csv_path = (
        Path(results_dir)
        / "results.csv"
    )

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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
        "decode_times_seconds",
        "decode_mean_seconds",
        "decode_median_seconds",
        "decode_std_seconds",
        "encode_time_seconds",
        "frame_count",
        "keyframes"
    ]

    # --------------------------------------------------------
    # OBRADA SVAKOG METODA
    # --------------------------------------------------------

    for method in methods:

        print()
        print("-" * 80)
        print(
            f"METHOD: {method}"
        )
        print("-" * 80)

        # ----------------------------------------------------
        # VAŽNO:
        # RESETUJEMO encode_time ZA SVAKI METOD
        # ----------------------------------------------------

        encode_time = None

        output_path = (
            output_dir
            / f"{video_name}_{method}.mp4"
        )

        # ----------------------------------------------------
        # PROVERA POSTOJEĆEG OUTPUTA
        # ----------------------------------------------------

        if output_path.exists():

            if overwrite:

                print(
                    f"Existing output found. "
                    f"Deleting: {output_path}"
                )

                output_path.unlink()

            else:

                raise FileExistsError(
                    "\n"
                    f"Output already exists:\n"
                    f"  {output_path}\n\n"
                    "The script will NOT use the old output "
                    "because it may have been generated with "
                    "different parameters.\n\n"
                    "Either delete the file manually or run "
                    "the script with --overwrite."
                )

        # ----------------------------------------------------
        # PLAN
        # ----------------------------------------------------

        if method in (
            "fixed",
            "default"
        ):

            plan = {
                "keyframes": None,
                "scenes": [],
                "gop_scenes": []
            }

        else:

            plan = adaptive_plan

        # ----------------------------------------------------
        # ENCODING
        # ----------------------------------------------------

        print(
            "Encoding..."
        )

        encode_start = (
            time.perf_counter()
        )

        encode_video(
            method,
            input_path,
            str(output_path),
            plan
        )

        encode_time = (
            time.perf_counter()
            - encode_start
        )

        print(
            f"Encoding time: "
            f"{encode_time:.2f} s"
        )

        # ----------------------------------------------------
        # PROVERA OUTPUTA
        # ----------------------------------------------------

        if not output_path.exists():

            raise FileNotFoundError(
                f"Encoding finished but output does not exist: "
                f"{output_path}"
            )

        # ----------------------------------------------------
        # VALIDACIJA
        # ----------------------------------------------------

        print(
            "Validating..."
        )

        out_info, out_frames = validate_video(
            info,
            frames,
            str(output_path),
            plan["keyframes"]
        )

        # ----------------------------------------------------
        # BITRATE
        # ----------------------------------------------------

        bitrate = (
            float(
                out_info["video"]["bit_rate"]
            )
            / 1000
        )

        # ----------------------------------------------------
        # VELIČINA
        # ----------------------------------------------------

        size_mb = (
            output_path.stat().st_size
            / (1024 * 1024)
        )

        # ----------------------------------------------------
        # FRAME TYPES
        # ----------------------------------------------------

        frame_types = out_frames[
            "frame_types"
        ]

        i_frames = frame_types["I"]
        p_frames = frame_types["P"]
        b_frames = frame_types["B"]

        actual_keyframes = (
            out_frames["keyframes"]
        )

        # ----------------------------------------------------
        # PROSEČAN GOP
        # ----------------------------------------------------

        if len(actual_keyframes) > 1:

            gaps = np.diff(
                actual_keyframes
            )

            average_gop_frames = float(
                np.mean(gaps)
            )

            avg_num, avg_den = (
                out_info["video"]
                ["avg_frame_rate"]
                .split("/")
            )

            fps = (
                float(avg_num)
                / float(avg_den)
            )

            if fps > 0:

                average_gop_seconds = (
                    average_gop_frames
                    / fps
                )

            else:

                average_gop_seconds = None

        else:

            average_gop_seconds = None

        # ----------------------------------------------------
        # PSNR + SSIM
        # ----------------------------------------------------

        print(
            "Calculating quality..."
        )

        quality = get_quality_metrics(
            input_path,
            str(output_path),
            frame_count
        )

        # ----------------------------------------------------
        # DECODING TIME
        #
        # Čuvamo SVA pojedinačna merenja.
        # ----------------------------------------------------

        print(
            "Measuring decoding time..."
        )

        timing = measure_decoding(
            str(output_path),
            repetitions,
            cpu
        )

        decode_times = timing[
            "decode_times_seconds"
        ]

        decode_mean = timing[
            "decode_mean_seconds"
        ]

        decode_median = timing[
            "decode_median_seconds"
        ]

        if len(decode_times) > 1:

            decode_std = float(
                np.std(
                    decode_times,
                    ddof=1
                )
            )

        else:

            decode_std = 0.0

        # ----------------------------------------------------
        # REZULTAT
        # ----------------------------------------------------

        result = {

            "video": video_name,

            "method": method,

            "i_frames": i_frames,

            "p_frames": p_frames,

            "b_frames": b_frames,

            "bitrate_kbps": bitrate,

            "size_mb": size_mb,

            "average_gop_seconds": (
                average_gop_seconds
            ),

            "psnr_db": (
                quality["psnr_db"]
            ),

            "ssim": (
                quality["ssim"]
            ),

            "decoding_time_seconds": (
                decode_median
            ),

            "decode_times_seconds": json.dumps(
                decode_times
            ),

            "decode_mean_seconds": (
                decode_mean
            ),

            "decode_median_seconds": (
                decode_median
            ),

            "decode_std_seconds": (
                decode_std
            ),

            "encode_time_seconds": (
                encode_time
            ),

            "frame_count": frame_count,

            "keyframes": json.dumps(
                actual_keyframes
            )
        }

        results.append(
            result
        )

        # ----------------------------------------------------
        # ISPIS REZULTATA
        # ----------------------------------------------------

        print()

        print(
            f"I frames:       "
            f"{i_frames}"
        )

        print(
            f"P frames:       "
            f"{p_frames}"
        )

        print(
            f"B frames:       "
            f"{b_frames}"
        )

        print(
            f"Bitrate:        "
            f"{bitrate:.2f} kb/s"
        )

        print(
            f"Size:           "
            f"{size_mb:.2f} MB"
        )

        if average_gop_seconds is not None:

            print(
                f"Average GOP:    "
                f"{average_gop_seconds:.3f} s"
            )

        print(
            f"PSNR:           "
            f"{quality['psnr_db']:.3f} dB"
        )

        print(
            f"SSIM:           "
            f"{quality['ssim']:.6f}"
        )

        print(
            f"Decode times:   "
            f"{[round(x, 4) for x in decode_times]}"
        )

        print(
            f"Decode mean:    "
            f"{decode_mean:.4f} s"
        )

        print(
            f"Decode median:  "
            f"{decode_median:.4f} s"
        )

        print(
            f"Decode std:     "
            f"{decode_std:.4f} s"
        )

        print(
            f"Encode time:    "
            f"{encode_time:.3f} s"
        )

        print(
            f"Keyframes:      "
            f"{actual_keyframes}"
        )

        # ====================================================
        # ODMAH SAČUVAJ REZULTAT
        # ====================================================

        write_header = not csv_path.exists()

        with open(
            csv_path,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            if write_header:
                writer.writeheader()

            writer.writerow(
                result
            )

            # Odmah upisuje podatke u fajl.
            f.flush()

        print()
        print(
            f"RESULT SAVED: "
            f"{csv_path}"
        )

    # --------------------------------------------------------
    # VIDEO ZAVRŠEN
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("VIDEO FINISHED")
    print("=" * 80)

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fixed and adaptive "
            "keyframe/GOP methods."
        )
    )

    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input video(s)"
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=METHODS
    )

    parser.add_argument(
        "--results-dir",
        default="results"
    )

    parser.add_argument(
        "--cache-dir",
        default="analysis_cache"
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=5
    )

    parser.add_argument(
        "--cpu",
        type=int,
        default=None,
        help="Optional CPU affinity for decoding"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete existing encoded output files "
            "before running the method."
        )
    )

    args = parser.parse_args()

    if args.repetitions < 3:

        raise ValueError(
            "Use at least 3 decoding repetitions."
        )

    all_results = []

    for input_path in args.input:

        results = evaluate_video(
            input_path,
            args.results_dir,
            args.cache_dir,
            args.methods,
            args.repetitions,
            args.cpu,
            args.overwrite
        )

        all_results.extend(
            results
        )

    # --------------------------------------------------------
    # KONAČNI PREGLED
    # --------------------------------------------------------

    print()
    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    for result in all_results:

        print(
            f"{result['video']:20s} "
            f"{result['method']:30s} "
            f"bitrate="
            f"{result['bitrate_kbps']:8.2f} "
            f"PSNR="
            f"{result['psnr_db']:6.3f} "
            f"SSIM="
            f"{result['ssim']:.6f} "
            f"decode_mean="
            f"{result['decode_mean_seconds']:.4f}s "
            f"decode_std="
            f"{result['decode_std_seconds']:.4f}s"
        )


if __name__ == "__main__":

    main()