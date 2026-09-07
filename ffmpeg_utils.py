import subprocess
import os
import time
import tempfile
import cv2
import json
import logging
import statistics
from pathlib import Path

from experiment_io import run_command, COMMAND_LOG, append_jsonl

def encode_adaptive_GOP(input_path, output_path, keyframes):
    
    cap = cv2.VideoCapture(input_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    force_keyframes = "+".join(
        f"eq(n,{frame})" for frame in keyframes
    )

    command = [
        "ffmpeg",
        "-n",
        "-i", input_path,

        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "medium",
        "-bf", "3",
        "-threads", "1",


        "-g", str(frame_count + 1),
        # x264 ne sme sam da dodaje scene-cut I-frameove
        "-sc_threshold", "0",

        # tvoji adaptivno određeni keyframeovi
        "-force_key_frames", f"expr:{force_keyframes}",

        "-map", "0:v:0", "-an", "-sn", "-dn",
        output_path
    ]

    run_command(command, check = True)

def encode_fixed_GOP(input_path, output_path):

    command = [
        "ffmpeg",
        "-n",
        "-i", input_path,
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "medium",
        "-bf", "3",
        "-threads", "1",

        # I-frame na svakih 60 frejmova
        "-g", "60",

        # bez dodatnih scene-cut I-frameova
        "-sc_threshold", "0",

        "-map", "0:v:0", "-an", "-sn", "-dn",
        output_path
    ]

    run_command(command, check = True)

def encode_default(input_path, output_path):

    command = [
        "ffmpeg",
        "-n",
        "-i", input_path,

        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "medium",
        "-bf", "3",
        "-threads", "1",

        "-map", "0:v:0", "-an", "-sn", "-dn",
        output_path
    ]

    run_command(command, check=True)
    
def encode_one_I_frame_per_scene(input_path, output_path, keyframes):

    cap = cv2.VideoCapture(input_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    force_keyframes = "+".join(
        f"eq(n,{frame})" for frame in keyframes
    )

    command = [
        "ffmpeg",
        "-n",
        "-i", input_path,

        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "medium",
        "-bf", "3",
        "-threads", "1",

        "-g", str(frame_count + 1),
        # Ne dozvoljavamo x264-u da sam ubacuje scene-cut I-frameove
        "-sc_threshold", "0",

        # I-frame samo na početku detektovanih scena
        "-force_key_frames",
        f"expr:{force_keyframes}",

        "-map", "0:v:0", "-an", "-sn", "-dn",
        output_path
    ]

    run_command(command, check=True)
    
def encode_adaptive_fixed_pb_GOP(input_path, output_path, keyframes):

    # Broj frejmova u videu
    cap = cv2.VideoCapture(input_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Keyframeovi koje je odredio naš algoritam
    force_keyframes = "+".join(
        f"eq(n,{frame})" for frame in keyframes
    )

    command = [
        "ffmpeg",
        "-n",
        "-i", input_path,

        "-c:v", "libx264",

        # Isti kvalitet i preset kao kod ostalih metoda
        "-crf", "23",
        "-preset", "medium",

        # FIKSNA P/B STRUKTURA
        "-x264-params",
        "bframes=3:b-adapt=0:b-pyramid=none",

        "-threads", "1",
        "-g", str(frame_count + 1),
        "-sc_threshold", "0",

        # Koristimo samo naše adaptivno određene keyframeove
        "-force_key_frames",
        f"expr:{force_keyframes}",

        "-map", "0:v:0", "-an", "-sn", "-dn",

        output_path
    ]

    run_command(command, check=True)
    
def encode_scene_by_scene(
    input_path,
    output_path,
    gop_scenes,
    keyframes,
    adaptive_b=True
):

    tmp_dir = tempfile.mkdtemp(
        prefix="scene_encode_"
    )

    segment_paths = []

    completed = False

    try:

        for i, scene in enumerate(gop_scenes):

            start = scene["start"]
            end = scene["end"]

            if adaptive_b:
                bframes = scene["bframes"]
            else:
                bframes = 3

            logging.info(
                "Encoding scene %d/%d: "
                "frames [%d,%d), B=%d",
                i + 1,
                len(gop_scenes),
                start,
                end,
                bframes
            )

            seg_path = os.path.join(
                tmp_dir,
                f"seg_{i:04d}.mp4"
            )

            # ISTI keyframe plan za obe metode
            local_keyframes = [
                k - start
                for k in keyframes
                if start <= k < end
            ]

            # Svaka scena počinje I-frameom
            if 0 not in local_keyframes:
                local_keyframes.insert(0, 0)

            local_keyframes = sorted(
                set(local_keyframes)
            )

            force_expr = "+".join(
                f"eq(n,{k})"
                for k in local_keyframes
            )

            command = [
                "ffmpeg",
                "-n",
                "-i", input_path,

                "-vf",
                (
                    f"select='between(n,{start},{end - 1})',"
                    "setpts=PTS-STARTPTS"
                ),

                "-map", "0:v:0",
                "-an",
                "-sn",
                "-dn",

                "-c:v", "libx264",
                "-crf", "23",
                "-preset", "medium",

                # JEDINA RAZLIKA JE B BROJ
                "-x264-params",
                (
                    f"bframes={bframes}:"
                    "b-adapt=0:"
                    "b-pyramid=none"
                ),

                "-threads", "1",

                "-g", str(end - start + 1),
                "-sc_threshold", "0",

                "-force_key_frames",
                f"expr:{force_expr}",

                seg_path
            ]

            run_command(
                command,
                check=True
            )

            segment_paths.append(seg_path)

        # ----------------------------------------------------
        # CONCAT
        # ----------------------------------------------------

        concat_list = os.path.join(
            tmp_dir,
            "concat_list.txt"
        )

        with open(
            concat_list,
            "w",
            encoding="utf-8"
        ) as f:

            for path in segment_paths:
                f.write(
                    f"file '{path}'\n"
                )

        concat_video = os.path.join(
            tmp_dir,
            "concat_video.mp4"
        )

        run_command(
            [
                "ffmpeg",
                "-n",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list,
                "-c", "copy",
                concat_video
            ],
            check=True
        )

        run_command(
            [
                "ffmpeg",
                "-n",
                "-i", concat_video,
                "-map", "0:v:0",
                "-an",
                "-sn",
                "-dn",
                "-c", "copy",
                output_path
            ],
            check=True
        )

        completed = True

    finally:

        if not completed:

            logging.error(
                "Preserving failed scene artifacts: %s",
                tmp_dir
            )

        else:

            _cleanup_scene_files(
                tmp_dir,
                segment_paths
            )


def encode_adaptive_pb_GOP(
    input_path,
    output_path,
    gop_scenes,
    keyframes
):

    encode_scene_by_scene(
        input_path,
        output_path,
        gop_scenes,
        keyframes,
        adaptive_b=True
    )

def encode_adaptive_fixed_pb_GOP_scene_by_scene(
    input_path,
    output_path,
    gop_scenes,
    keyframes
):

    encode_scene_by_scene(
        input_path,
        output_path,
        gop_scenes,
        keyframes,
        adaptive_b=False
    )

def _cleanup_scene_files(tmp_dir, segment_paths):
        # Remove successful temporary encodes only; published results are never deleted.
        for path in segment_paths:
            if os.path.exists(path):
                os.remove(path)

        if os.path.exists(os.path.join(tmp_dir, "concat_list.txt")):
            os.remove(os.path.join(tmp_dir, "concat_list.txt"))

        if os.path.exists(os.path.join(tmp_dir, "concat_video.mp4")):
            os.remove(os.path.join(tmp_dir, "concat_video.mp4"))

        if os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)

def check_keyframes(output_path):
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time,key_frame,pict_type",
        "-of", "csv=p=0",
        output_path
    ]

    result = run_command(
        command,
        capture_output=True,
        text=True,
        check=True
    )

    keyframes = []

    for line in result.stdout.splitlines():
        parts = line.strip().split(",")

        if len(parts) < 3:
            continue

        try:
            key_frame = int(parts[0])
            time = float(parts[1])
            pict_type = parts[2].strip()
        except ValueError:
            print("WARNING - nevalidna ffprobe linija:", repr(line))
            continue

        if key_frame == 1:
            keyframes.append(time)

            if pict_type != "I":
                print("WARNING:", time, key_frame, pict_type)

    return keyframes

def get_i_frame_count(video_path):
    keyframes = check_keyframes(video_path)
    return len(keyframes)

def get_frame_type_counts(video_path):

    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=pict_type",
        "-of", "csv=p=0",
        video_path
    ]

    result = run_command(command, capture_output=True, text=True, check=True)

    counts = { "I": 0, "P": 0, "B": 0 }

    for line in result.stdout.splitlines():

        frame_type = (line.strip().rstrip(","))

        if frame_type in counts:
            counts[frame_type] += 1

    return counts

def get_bitrate(video_path):

    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=bit_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]

    result = run_command(command, capture_output = True, text = True, check = True)
    bitrate = result.stdout.strip()

    return float(bitrate) / 1000

def get_file_size(video_path):

    size_bytes = os.path.getsize(video_path)

    size_mb = size_bytes / (1024 * 1024)

    return size_mb

def get_average_i_frame_interval(video_path):

    keyframes = check_keyframes(video_path)

    if len(keyframes) < 2:
        return 0

    intervals = []

    for i in range(1, len(keyframes)):
        interval = keyframes[i] - keyframes[i - 1]
        intervals.append(interval)

    return sum(intervals) / len(intervals)

def get_encode_time(input_path, output_path, encode_function, *args):

    import time
    start = time.perf_counter()
    encode_function(input_path, output_path, *args)
    end = time.perf_counter()
    return end - start

def get_video_metrics(video_path):

    bitrate = get_bitrate(video_path)
    size = get_file_size(video_path)
    i_frame_count = get_i_frame_count(video_path)
    average_interval = get_average_i_frame_interval(video_path)

    return {
        "bitrate": bitrate,
        "size": size,
        "i_frame_count": i_frame_count,
        "average_interval": average_interval
    }
    
def _quality_metric(original_path, encoded_path, metric):
    # Equal decoded frame counts and relative PTS are checked by the benchmark first.
    command = [
        "ffmpeg", "-nostdin", "-threads", "1", "-i", str(encoded_path),
        "-threads", "1", "-i", str(original_path), "-filter_complex_threads", "1",
        "-filter_complex",
        f"[0:v:0]setpts=PTS-STARTPTS[enc];[1:v:0]setpts=PTS-STARTPTS[ref];"
        f"[enc][ref]{metric}=stats_file=-:shortest=1:repeatlast=0[metric]",
        "-map", "[metric]", "-an", "-sn", "-dn", "-threads", "1", "-f", "null", "-"
    ]
    result = run_command(command, capture_output=True, text=True)
    marker = "average:" if metric == "psnr" else "All:"
    values = [line.split(marker)[1].split()[0] for line in result.stderr.splitlines()
              if metric.upper() in line and marker in line]
    frames = sum(line.startswith("n:") for line in result.stdout.splitlines())
    if not values or not frames:
        raise RuntimeError(f"Missing {metric} measurement")
    return float(values[-1]), frames


def get_psnr(original_path, encoded_path):
    return _quality_metric(original_path, encoded_path, "psnr")[0]


def get_ssim(original_path, encoded_path):
    return _quality_metric(original_path, encoded_path, "ssim")[0]


def get_quality_metrics(original_path, encoded_path, expected_frames):
    psnr, psnr_frames = _quality_metric(original_path, encoded_path, "psnr")
    ssim, ssim_frames = _quality_metric(original_path, encoded_path, "ssim")
    if psnr_frames != expected_frames or ssim_frames != expected_frames:
        raise ValueError(f"Quality metric coverage mismatch: PSNR={psnr_frames}, SSIM={ssim_frames}, expected={expected_frames}")
    return {"psnr_db": psnr, "ssim": ssim, "psnr_frames": psnr_frames, "ssim_frames": ssim_frames}

def measure_decoding(video_path, repetitions=3, cpu=None):
    if repetitions < 1:
        raise ValueError("At least one timing repetition is required")
    command = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1",
               "-i", str(video_path), "-map", "0:v:0", "-an", "-sn", "-dn",
               "-threads", "1", "-filter_threads", "1", "-f", "null", "-"]
    if cpu is not None:
        if cpu not in os.sched_getaffinity(0):
            raise ValueError(f"CPU {cpu} is outside the allowed affinity")
        command = ["taskset", "-c", str(cpu)] + command
    times = []
    for i in range(repetitions):
        logging.info("Sequential video decoding: repetition %d/%d, cpu=%s", i + 1, repetitions, cpu)
        start = time.perf_counter()
        # Logging/fsync happens after the stopwatch, not inside the timed operation.
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            elapsed = time.perf_counter() - start
            if result.stderr.strip():
                raise RuntimeError(f"Decode reported errors: {result.stderr}")
        except BaseException as exc:
            if COMMAND_LOG.get():
                append_jsonl(COMMAND_LOG.get(), {"argv": command, "repetition": i + 1,
                             "status": "decode_error", "error": repr(exc)})
            raise
        times.append(elapsed)
        if COMMAND_LOG.get():
            append_jsonl(COMMAND_LOG.get(), {"argv": command, "repetition": i + 1,
                         "returncode": result.returncode, "decode_seconds": elapsed})
    return {"decode_command": command, "decode_times_seconds": times,
            "decode_mean_seconds": statistics.mean(times),
            "decode_median_seconds": statistics.median(times)}

def get_decoding_time(video_path, repetitions=5):
    return measure_decoding(video_path, repetitions)["decode_mean_seconds"]

def probe_video(video_path):
    command = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video_path)]
    result = run_command(command, capture_output=True, text=True)
    info = json.loads(result.stdout)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    return {"video": video, "streams": info["streams"], "format": info["format"]}

def probe_frames(video_path):
    command = ["ffprobe", "-v", "error", "-threads", "1", "-select_streams", "v:0",
               "-show_frames", "-show_entries", "frame=key_frame,pict_type,best_effort_timestamp_time",
               "-of", "json", str(video_path)]
    result = run_command(command, capture_output=True, text=True)
    if result.stderr.strip():
        raise ValueError(f"Frame probe reported decode errors: {result.stderr}")
    frames = json.loads(result.stdout)["frames"]
    if not frames:
        raise ValueError("No decoded video frames")
    return {"frame_count": len(frames),
            "keyframes": [i for i, f in enumerate(frames) if f["key_frame"] == 1],
            "frame_types": {kind: sum(f.get("pict_type") == kind for f in frames) for kind in ("I", "P", "B")},
            "timestamps": [float(f["best_effort_timestamp_time"]) for f in frames]}

def validate_video(reference_info, reference_frames, output_path, requested):
    info = probe_video(output_path)
    frames = probe_frames(output_path)
    if len(info["streams"]) != 1 or info["video"]["codec_name"] != "h264":
        raise ValueError("Benchmark output must contain exactly one H.264 video stream")
    for field in ("width", "height", "pix_fmt"):
        if info["video"][field] != reference_info["video"][field]:
            raise ValueError(f"Reference/output {field} mismatch")
    if frames["frame_count"] != reference_frames["frame_count"]:
        raise ValueError(f"Frame count mismatch: {frames['frame_count']} vs {reference_frames['frame_count']}")
    if requested is not None and frames["keyframes"] != requested:
        missing = sorted(set(requested) - set(frames["keyframes"]))
        extra = sorted(set(frames["keyframes"]) - set(requested))
        raise ValueError(f"Keyframe mismatch: missing={missing}, extra={extra}")
    a, b = frames["timestamps"], reference_frames["timestamps"]
    if any(y <= x for x, y in zip(a, a[1:])):
        raise ValueError("Non-monotonic output presentation timestamps")
    error = max(abs((x - a[0]) - (y - b[0])) for x, y in zip(a, b))
    if error > 0.0001:
        raise ValueError(f"Relative presentation timestamps differ by {error:.6f}s")
    frames["max_relative_timestamp_error_seconds"] = error
    return info, frames

def create_720p(input_path, output_path):

    command = [
        "ffmpeg",
        "-n",
        "-i", input_path,

        "-vf", "scale=-2:720",

        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-threads", "1",

        "-map", "0:v:0", "-an", "-sn", "-dn",

        output_path
    ]

    run_command(command, check=True)
    
