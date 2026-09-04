import subprocess
import os
import time
import tempfile
import cv2

def encode_adaptive_GOP(input_path, output_path, keyframes):
    
    cap = cv2.VideoCapture(input_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

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
        "-threads", "1",


        "-g", str(frame_count + 1),
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
        "-threads", "1",

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
        "-threads", "1",

        "-c:a", "copy",
        output_path
    ]

    subprocess.run(command, check=True)
    
def encode_one_I_frame_per_scene(input_path, output_path, keyframes):

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

        # Ne dozvoljavamo x264-u da sam ubacuje scene-cut I-frameove
        "-sc_threshold", "0",

        # I-frame samo na početku detektovanih scena
        "-force_key_frames",
        f"expr:{force_keyframes}",

        "-c:a", "copy",
        output_path
    ]

    subprocess.run(command, check=True)
    
def encode_adaptive_fixed_pb_GOP(   
    input_path,
    output_path,
    gop_scenes,
    keyframes
):
    """
    Scene-by-scene encoding sa FIKSNIM brojem B-frameova.

    Svrha:
    Ablation test za proveru da li degradacija rezultata
    dolazi od scene-by-scene encoding arhitekture, a ne
    od adaptivnog izbora broja B-frameova.

    Sve scene koriste:
        bframes=3
        b-adapt=0
        b-pyramid=none
    """

    tmp_dir = tempfile.mkdtemp(
        prefix="adaptive_fixed_pb_"
    )

    segment_paths = []

    concat_list_path = os.path.join(
        tmp_dir,
        "concat_list.txt"
    )

    concat_video_path = os.path.join(
        tmp_dir,
        "concat_video.mp4"
    )

    try:

        for i, scene in enumerate(gop_scenes):

            start = scene["start"]
            end = scene["end"]

            # --------------------------------------------------
            # KLJUČNA IZMENA:
            # svaka scena koristi isti broj B-frameova
            # --------------------------------------------------
            bframes = 3

            seg_path = os.path.join(
                tmp_dir,
                f"seg_{i:04d}.mp4"
            )

            # Keyframeovi koji pripadaju ovoj sceni.
            # Pretvaramo ih u lokalne indekse scene.
            interior = sorted({
                k - start
                for k in keyframes
                if start <= k < end
            })

            # Svaka scena mora početi od frejma 0.
            if 0 not in interior:
                interior = [0] + interior

            force_expr = "+".join(
                f"eq(n,{k})"
                for k in interior
            )

            command = [
                "ffmpeg",
                "-y",

                "-i",
                input_path,

                # Izvuci samo ovu scenu i resetuj timestampove
                "-vf",
                (
                    f"select='between(n,{start},{end - 1})',"
                    "setpts=PTS-STARTPTS"
                ),

                # Audio ćemo dodati na kraju
                "-an",

                "-c:v",
                "libx264",

                "-crf",
                "23",

                "-preset",
                "medium",

                # FIKSNI B-frame parametri
                "-x264-params",
                (
                    f"bframes={bframes}:"
                    "b-adapt=0:"
                    "b-pyramid=none"
                ),

                "-threads",
                "1",

                # x264 ne sme sam da ubacuje scene-cut I-frameove
                "-sc_threshold",
                "0",

                # Naši keyframeovi
                "-force_key_frames",
                f"expr:{force_expr}",

                seg_path
            ]

            subprocess.run(
                command,
                check=True
            )

            segment_paths.append(seg_path)

        # ------------------------------------------------------
        # Napravi concat listu
        # ------------------------------------------------------

        with open(
            concat_list_path,
            "w",
            encoding="utf-8"
        ) as f:

            for p in segment_paths:
                f.write(
                    f"file '{p}'\n"
                )

        # ------------------------------------------------------
        # Spoji video segmente bez ponovnog enkodovanja
        # ------------------------------------------------------

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list_path,
                "-c", "copy",
                concat_video_path
            ],
            check=True
        )

        # ------------------------------------------------------
        # Dodaj originalni audio
        # ------------------------------------------------------

        subprocess.run(
            [
                "ffmpeg",
                "-y",

                "-i",
                concat_video_path,

                "-i",
                input_path,

                "-map",
                "0:v:0",

                "-map",
                "1:a:0?",

                "-c",
                "copy",

                output_path
            ],
            check=True
        )

    finally:

        # ------------------------------------------------------
        # Čišćenje privremenih fajlova
        # ------------------------------------------------------

        for p in segment_paths:

            if os.path.exists(p):
                os.remove(p)

        if os.path.exists(concat_list_path):
            os.remove(concat_list_path)

        if os.path.exists(concat_video_path):
            os.remove(concat_video_path)

        if os.path.isdir(tmp_dir):
            os.rmdir(tmp_dir)
    
def encode_adaptive_pb_GOP(input_path, output_path, gop_scenes, keyframes):
    tmp_dir = tempfile.mkdtemp(prefix="adaptive_pb_")

    segment_paths = []
    concat_list_path = os.path.join(tmp_dir, "concat_list.txt")
    concat_video_path = os.path.join(tmp_dir, "concat_video.mp4")

    try:
        for i, scene in enumerate(gop_scenes):
            start = scene["start"]
            end = scene["end"]
            bframes = scene["bframes"]

            seg_path = os.path.join(tmp_dir, f"seg_{i:04d}.mp4" )

            # Keyframeovi koji pripadaju ovoj sceni
            interior = sorted({
                k - start
                for k in keyframes
                if start <= k < end
            })

            if 0 not in interior:
                interior = [0] + interior

            force_expr = "+".join(
                f"eq(n,{k})"
                for k in interior
            )

            command = [
                "ffmpeg",
                "-y",

                "-i",
                input_path,

                "-vf",
                (
                    f"select='between(n,{start},{end - 1})',"
                    "setpts=PTS-STARTPTS"
                ),

                "-an",

                "-c:v",
                "libx264",

                "-crf",
                "23",

                "-preset",
                "medium",

                "-x264-params",
                (
                    f"bframes={bframes}:"
                    "b-adapt=0:"
                    "b-pyramid=none"
                ),

                "-threads",
                "1",

                "-sc_threshold",
                "0",

                "-force_key_frames",
                f"expr:{force_expr}",

                seg_path
            ]

            subprocess.run(command, check=True)
            segment_paths.append(seg_path)

        # napravi concat listu
        with open(concat_list_path, "w", encoding="utf-8") as f:

            for p in segment_paths:
                f.write(
                    f"file '{p}'\n"
                )

        # spoji segmente bez ponovnog enkodovanja
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list_path,
                "-c", "copy",
                concat_video_path
            ],
            check=True
        )

        # vrati originalni audio
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", concat_video_path,
                "-i", input_path,

                "-map", "0:v:0",
                "-map", "1:a:0?",

                "-c", "copy",

                output_path
            ],
            check=True
        )

    finally:

        for p in segment_paths:
            if os.path.exists(p):
                os.remove(p)

        if os.path.exists(concat_list_path):
            os.remove(concat_list_path)

        if os.path.exists(concat_video_path):
            os.remove(concat_video_path)

        if os.path.isdir(tmp_dir):
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


    result = subprocess.run(command, capture_output = True, text = True,
        check=True
    )

    keyframes = []

    for line in result.stdout.splitlines():
        parts = line.split(",")

        if len(parts) >= 3:
            time = float(parts[0])
            key_frame = int(parts[1])
            pict_type = parts[2]

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

    result = subprocess.run(command, capture_output=True, text=True, check=True)

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

def create_720p(input_path, output_path):

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,

        "-vf", "scale=-2:720",

        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-threads", "1",

        "-c:a", "copy",

        output_path
    ]

    subprocess.run(command, check=True)
    

