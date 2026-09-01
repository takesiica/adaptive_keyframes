import cv2
import numpy as np
from functions import (adaptive_threshold1, scene_dynamics, interval_for_score, generate_keyframes, generate_keyframes_by_scene, combine_sad_histogram)
from ffmpeg_utils import (encode_adaptive_GOP, encode_fixed_GOP, encode_default, get_video_metrics, get_psnr, get_ssim, get_decoding_time)


if __name__ == "__main__":

    input_path = ".\\evaluation\\test6.mp4"
    output_path1 = ".\\output1.mp4"
    output_path2 = ".\\output2.mp4"
    output_path3 = ".\\output3.mp4"

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
    
    keyframes = generate_keyframes_by_scene(scene_changes, total_frames)


    # 7. Generisanje konkretnih keyframeova
    #keyframes = generate_keyframes(scores, total_frames, min_interval = 60)


    print("Keyframeovi:")
    print(keyframes)
    
    differences = [
    keyframes[i] - keyframes[i-1]
    for i in range(1, len(keyframes))]

    print("Razmaci:", differences)
    print("Minimalni razmak:", min(differences))
    print("Prosečan razmak:", sum(differences) / len(differences))

    # 8. Enkodovanje i provera
    encode_fixed_GOP(input_path, output_path1)
    encode_default(input_path, output_path2)
    encode_adaptive_GOP(input_path, output_path3, keyframes)
    
    # 9. Merenje metrika
    fixed_metrics = get_video_metrics(output_path1)
    default_metrics = get_video_metrics(output_path2)
    adaptive_metrics = get_video_metrics(output_path3)
    
    fixed_psnr = get_psnr(input_path, output_path1)
    default_psnr = get_psnr(input_path, output_path2)
    adaptive_psnr = get_psnr(input_path, output_path3)

    fixed_ssim = get_ssim(input_path, output_path1)
    default_ssim = get_ssim(input_path, output_path2)
    adaptive_ssim = get_ssim(input_path, output_path3)
    
    fixed_decode = get_decoding_time(".\\output1.mp4", repetitions=5)
    default_decode = get_decoding_time(".\\output2.mp4", repetitions=5)
    adaptive_decode = get_decoding_time(".\\output3.mp4", repetitions=5)

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
    