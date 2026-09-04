import cv2
import numpy as np
from functions import (adaptive_threshold1, scene_dynamics, interval_for_score, generate_keyframes, generate_keyframes_by_scene, combine_sad_histogram, generate_adaptive_gop_structure)
from ffmpeg_utils import (encode_adaptive_GOP, encode_fixed_GOP, encode_default, get_video_metrics, get_psnr, get_ssim, get_decoding_time, encode_adaptive_pb_GOP, encode_adaptive_fixed_pb_GOP, get_frame_type_counts)


if __name__ == "__main__":

    input_path = ".\\evaluation\\test3.mp4"
    output_path1 = ".\\output1.mp4"
    output_path2 = ".\\output2.mp4"
    output_path3 = ".\\output3.mp4"
    output_path4 = ".\\output4.mp4"

    # 1. Scene detection
    
    #scene detection sa adaptive thresholdom
    '''scene_changes, all_mean_sads, thresholds, frame_indices = adaptive_threshold1(
        input_path, window_size = 45, threshold_factor = 4.0, min_sad = 5.0, peak_multiplier = 1.5) '''
        
    #scene detection hist thresh combined
    (scene_changes, sad, hist, combined, combined_thresholds, frame_indices, weights_sad, weights_hist) = combine_sad_histogram(
    
            input_path,
            window_size=30,
    
            sad_threshold_factor=3.0,
            min_sad=5.0,
            sad_peak_multiplier=2.6,
    
            hist_threshold_factor=4.0,
            min_hist_diff=0.05,
            hist_peak_multiplier=1.5,
            min_peak_diff=0.5,
    
            sad_low_threshold=2.0,
            sad_high_threshold=6.0
        )

    # 2. Video info
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # 3. Scene boundaries
    scene_boundaries = [0] + [frame for frame, _ in scene_changes]
    scene_boundaries.append(total_frames)
    scene_boundaries = sorted(set(scene_boundaries))

    # 4. Scene dynamics
    scores = scene_dynamics(sad, frame_indices, scene_boundaries)

    # 5. Thresholds za dinamiku
    scene_means = [scene["mean"] for scene in scores]
    low_thresh = np.percentile(scene_means, 25)
    high_thresh = np.percentile(scene_means, 75)

    # 6. Interval za svaku scenu
    for scene in scores:
        scene["interval"] = interval_for_score(scene["mean"], low_thresh, high_thresh, min_interval = 60, max_interval = 180)

    #generisanje keyframeova samo na pocetak scene
    
    #keyframes = generate_keyframes_by_scene(scene_changes, total_frames)


    # 7. Generisanje konkretnih keyframeova
    keyframes = generate_keyframes(scores, total_frames, min_interval = 60)


    print("Keyframeovi:")
    print(keyframes)
    
    gop_scenes = [
        generate_adaptive_gop_structure(scene, low_thresh, high_thresh, b_max=3, b_min=1)
        for scene in scores
    ]

    print("\nAdaptive P/B struktura po sceni:")

    for i, g in enumerate(gop_scenes):
        print(
            f"Scena {i}: "
            f"start={g['start']} "
            f"end={g['end']} "
            f"mean={g['mean']:.4f} "
            f"complexity={g['complexity']:.3f} "
            f"type={g['type']} "
            f"B={g['bframes']}"
        )
    
    differences = [
    keyframes[i] - keyframes[i-1]
    for i in range(1, len(keyframes))]

    print("Razmaci:", differences)
    print("Minimalni razmak:", min(differences))
    print("Prosečan razmak:", sum(differences) / len(differences))

    # 8. Enkodovanje i provera
    #encode_fixed_GOP(input_path, output_path1)
    #encode_default(input_path, output_path2)
    encode_adaptive_GOP(input_path, output_path3, keyframes)
    #encode_adaptive_fixed_pb_GOP(input_path, output_path4, gop_scenes, keyframes)
    
    # 9. Merenje metrika
    #fixed_metrics = get_video_metrics(output_path1)
    #default_metrics = get_video_metrics(output_path2)
    adaptive_metrics = get_video_metrics(output_path3)
    #adaptive_pb_metrics = get_video_metrics(output_path4)
    
    #fixed_psnr = get_psnr(input_path, output_path1)
    #default_psnr = get_psnr(input_path, output_path2)
    adaptive_psnr = get_psnr(input_path, output_path3)
    #adaptive_pb_psnr = get_psnr(input_path, output_path4)

    #fixed_ssim = get_ssim(input_path, output_path1)
    #default_ssim = get_ssim(input_path, output_path2)
    adaptive_ssim = get_ssim(input_path, output_path3)
    #adaptive_pb_ssim = get_ssim(input_path, output_path4)
    
    #fixed_decode = get_decoding_time(".\\output1.mp4", repetitions=5)
    #default_decode = get_decoding_time(".\\output2.mp4", repetitions=5)
    adaptive_decode = get_decoding_time(".\\output3.mp4", repetitions=5)
    #adaptive_pb_decode = get_decoding_time(".\\output4.mp4", repetitions=5)
    
    #adaptive_pb_frame_types = get_frame_type_counts(output_path4)
    
    print("\nADAPTIVE GOP")
    print(f"Bitrate: {adaptive_metrics['bitrate']:.2f} kb/s")
    print(f"Veličina: {adaptive_metrics['size']:.2f} MB")
    print(f"I-frameovi: {adaptive_metrics['i_frame_count']}")
    print(f"Prosečan razmak I-frameova: {adaptive_metrics['average_interval']:.3f} s")
    print(f"PSNR: {adaptive_psnr:.3f} dB")
    print(f"SSIM: {adaptive_ssim:.6f}")
    print(f"Decoding time: {adaptive_decode:.3f} s")
    
    
'''
    # 10. Prikaz rezultata
    print("\n=== REZULTATI ===")

    print("\nFIXED GOP")
    print(f"Bitrate: {fixed_metrics['bitrate']:.2f} kb/s")
    print(f"Veličina: {fixed_metrics['size']:.2f} MB")
    print(f"I-frameovi: {fixed_metrics['i_frame_count']}")
    print(f"Prosečan razmak I-frameova: {fixed_metrics['average_interval']:.3f} s")
    print(f"PSNR: {fixed_psnr:.3f} dB")
    print(f"SSIM: {fixed_ssim:.6f}")
    print(f"Decoding time: {fixed_decode:.3f} s")

    print("\nDEFAULT x264")
    print(f"Bitrate: {default_metrics['bitrate']:.2f} kb/s")
    print(f"Veličina: {default_metrics['size']:.2f} MB")
    print(f"I-frameovi: {default_metrics['i_frame_count']}")
    print(f"Prosečan razmak I-frameova: {default_metrics['average_interval']:.3f} s")
    print(f"PSNR: {default_psnr:.3f} dB")
    print(f"SSIM: {default_ssim:.6f}")
    print(f"Decoding time: {default_decode:.3f} s")

    print("\nADAPTIVE GOP")
    print(f"Bitrate: {adaptive_metrics['bitrate']:.2f} kb/s")
    print(f"Veličina: {adaptive_metrics['size']:.2f} MB")
    print(f"I-frameovi: {adaptive_metrics['i_frame_count']}")
    print(f"Prosečan razmak I-frameova: {adaptive_metrics['average_interval']:.3f} s")
    print(f"PSNR: {adaptive_psnr:.3f} dB")
    print(f"SSIM: {adaptive_ssim:.6f}")
    print(f"Decoding time: {adaptive_decode:.3f} s")
    
    print("\nADAPTIVE I + ADAPTIVE P/B")
    print(f"Bitrate: {adaptive_pb_metrics['bitrate']:.2f} kb/s")
    print(f"Veličina: {adaptive_pb_metrics['size']:.2f} MB")
    print(f"I-frameovi: {adaptive_pb_metrics['i_frame_count']}")
    print(f"P-frameovi: {adaptive_pb_frame_types['P']}")
    print(f"B-frameovi: {adaptive_pb_frame_types['B']}")
    print(f"Prosečan razmak I-frameova: {adaptive_pb_metrics['average_interval']:.3f} s")
    print(f"PSNR: {adaptive_pb_psnr:.3f} dB")
    print(f"SSIM: {adaptive_pb_ssim:.6f}")
    print(f"Decoding time: {adaptive_pb_decode:.3f} s")

    print(
        f"Stvarna struktura iz ffprobe: "
        f"I={adaptive_pb_frame_types['I']} "
        f"P={adaptive_pb_frame_types['P']} "
        f"B={adaptive_pb_frame_types['B']}"
    )
    
'''