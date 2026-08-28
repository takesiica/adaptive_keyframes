import subprocess
import os
import time

def encode_adaptive_GOP(input_path, output_path, keyframes):

    force_keyframes = "+".join(
        f"eq(n,{frame})" for frame in keyframes
    )

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,

        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "medium",
        "-bf", "3",

        # x264 ne sme sam da dodaje scene-cut I-frameove
        "-sc_threshold", "0",

        # tvoji adaptivno određeni keyframeovi
        "-force_key_frames", f"expr:{force_keyframes}",

        "-c:a", "copy",
        output_path
    ]

    subprocess.run(command, check = True)

def encode_fixed_GOP(input_path, output_path):

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,

        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "medium",
        "-bf", "3",

        # I-frame na svakih 60 frejmova
        "-g", "60",

        # bez dodatnih scene-cut I-frameova
        "-sc_threshold", "0",

        "-c:a", "copy",
        output_path
    ]

    subprocess.run(command, check = True)

def encode_default(input_path, output_path):

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,

        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "medium",
        "-bf", "3",

        "-c:a", "copy",
        output_path
    ]

    subprocess.run(command, check=True)

def check_keyframes(output_path):
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time,pict_type",
        "-of", "csv=p=0",
        output_path
    ]

    result = subprocess.run(command, capture_output = True, text = True,
        check=True
    )

    keyframes = []

    for line in result.stdout.splitlines():
        parts = line.split(",")

        if len(parts) >= 2 and parts[1] == "I":
            time = float(parts[0])
            keyframes.append(time)

    return keyframes

def get_i_frame_count(video_path):

    keyframes = check_keyframes(video_path)

    return len(keyframes)

def get_bitrate(video_path):

    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=bit_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]

    result = subprocess.run(command, capture_output = True, text = True, check = True)
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
    
def get_psnr(original_path, encoded_path):

    command = [
        "ffmpeg",
        "-i", encoded_path,
        "-i", original_path,
        "-lavfi",
        "[0:v][1:v]psnr",
        "-f", "null",
        "-"
    ]

    result = subprocess.run(command, capture_output = True, text = True)

    for line in result.stderr.splitlines():

        if "PSNR" in line and "average:" in line:
            parts = line.split("average:")

            if len(parts) > 1:
                value = parts[1].split()[0]
                return float(value)

    return None

def get_ssim(original_path, encoded_path):

    command = [
        "ffmpeg",
        "-i", encoded_path,
        "-i", original_path,
        "-lavfi",
        "[0:v][1:v]ssim",
        "-f", "null",
        "-"
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    for line in result.stderr.splitlines():

        if "SSIM" in line and "All:" in line:
            parts = line.split("All:")

            if len(parts) > 1:
                value = parts[1].split()[0]
                return float(value)

    return None    

def get_decoding_time(video_path, repetitions = 5):

    times = []

    for i in range(repetitions):

        start = time.perf_counter()

        command = [
            "ffmpeg",
            "-v", "error",
            "-threads", "1",
            "-i", video_path,
            "-f", "null",
            "-"
        ]

        subprocess.run(command, check=True)

        end = time.perf_counter()

        decoding_time = end - start
        times.append(decoding_time)

    average_time = sum(times) / len(times)

    return average_time