import os
import json
import argparse
import inspect
import logging
import uuid
from pathlib import Path

from experiment_io import atomic_json, digest
from feature_cache import get_raw_features

from functions import adaptive_threshold1, adaptive_histogram


# ============================================================
# FOLDER ZA ČUVANJE ANALIZE
# ============================================================

CACHE_FOLDER = "analysis_cache"

# ============================================================
# PARAMETRI SAD
# ============================================================

SAD_PARAMETERS = {
    "window_size": 30,
    "threshold_factor": 3.0,
    "min_sad": 5.0,
    "peak_multiplier": 2.6
}


# ============================================================
# PARAMETRI HISTOGRAMA
# ============================================================

HIST_PARAMETERS = {
    "window_size": 45,
    "threshold_factor": 4.0,
    "min_hist_diff": 0.05,
    "peak_multiplier": 1.5,
    "min_peak_diff": 0.5
}


# ============================================================
# VIDEO
# ============================================================



# ============================================================
# PUTANJA CACHE FAJLA
# ============================================================

def get_cache_path(video_path, cache_dir=CACHE_FOLDER, raw=None):
    raw = raw if raw is not None else get_raw_features(video_path, cache_dir)
    identity = {"raw": raw["metadata"], "sad": SAD_PARAMETERS, "hist": HIST_PARAMETERS,
                "detectors": digest(inspect.getsource(adaptive_threshold1) + inspect.getsource(adaptive_histogram))}
    return str(Path(cache_dir) / f"{Path(video_path).stem}-{digest(identity)[:20]}-analysis.json")


# ============================================================
# UČITAVANJE POSTOJEĆEG CACHE-A
# ============================================================

def load_analysis_cache(cache_path):

    if not os.path.exists(cache_path):
        return None

    try:

        with open(
            cache_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        for feature in ("sad", "histogram"):
            values = data[feature]
            if not (len(values["values"]) == len(values["thresholds"]) == len(values["frame_indices"])):
                raise ValueError("Incomplete detector cache")
            if not isinstance(values["scene_changes"], list):
                raise ValueError("Invalid detector cache")

    except Exception as e:

        print(
            f"Greška pri učitavanju cache-a: {e}"
        )
        Path(cache_path).rename(str(cache_path) + ".invalid-" + uuid.uuid4().hex)

        return None


    # --------------------------------------------------------
    # Provera SAD parametara
    # --------------------------------------------------------

    if data.get(
        "sad_parameters"
    ) != SAD_PARAMETERS:

        print()
        print(
            "SAD parametri su promenjeni."
        )

        print(
            "Analiza mora ponovo da se izračuna."
        )

        return None


    # --------------------------------------------------------
    # Provera histogram parametara
    # --------------------------------------------------------

    if data.get(
        "hist_parameters"
    ) != HIST_PARAMETERS:

        print()
        print(
            "Histogram parametri su promenjeni."
        )

        print(
            "Analiza mora ponovo da se izračuna."
        )

        return None


    # --------------------------------------------------------
    # Cache postoji i parametri su isti
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        " UČITAVAM POSTOJEĆI ANALYSIS CACHE"
    )

    print(
        "=" * 70
    )

    print(
        f"Cache: {cache_path}"
    )

    return data


# ============================================================
# ČUVANJE ANALIZE
# ============================================================

def save_analysis_cache(
    cache_path,
    video_path,
    sad_results,
    hist_results
):

    data = {

        "video":
            os.path.basename(
                video_path
            ),

        "sad_parameters":
            SAD_PARAMETERS,

        "hist_parameters":
            HIST_PARAMETERS,

        "sad": {

            "scene_changes":
                sad_results[0],

            "values":
                sad_results[1],

            "thresholds":
                sad_results[2],

            "frame_indices":
                sad_results[3]
        },

        "histogram": {

            "scene_changes":
                hist_results[0],

            "values":
                hist_results[1],

            "thresholds":
                hist_results[2],

            "frame_indices":
                hist_results[3]
        }
    }


    atomic_json(cache_path, data)

    print()
    print(
        "=" * 70
    )

    print(
        " ANALIZA SAČUVANA"
    )

    print(
        "=" * 70
    )

    print(
        f"Fajl: {cache_path}"
    )


# ============================================================
# GLAVNA FUNKCIJA
# ============================================================

def get_video_analysis(video_path, cache_dir=CACHE_FOLDER, *, raw_features=None):

    raw = raw_features if raw_features is not None else get_raw_features(video_path, cache_dir)
    cache_path = get_cache_path(video_path, cache_dir, raw)


    # --------------------------------------------------------
    # Pokušaj učitavanja postojećeg cache-a
    # --------------------------------------------------------

    cached = load_analysis_cache(
        cache_path
    )


    if cached is not None:

        return (
            cached["sad"],
            cached["histogram"]
        )


    # ========================================================
    # SAD ANALIZA
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        " RAČUNANJE SAD ANALIZE"
    )

    print(
        "=" * 70
    )

    print(
        "Parametri:"
    )

    print(
        SAD_PARAMETERS
    )


    sad_results = adaptive_threshold1(

        video_path,
        raw_features=raw,

        window_size=
            SAD_PARAMETERS["window_size"],

        threshold_factor=
            SAD_PARAMETERS["threshold_factor"],

        min_sad=
            SAD_PARAMETERS["min_sad"],

        peak_multiplier=
            SAD_PARAMETERS["peak_multiplier"]
    )


    print()

    print(
        f"SAD detektovano promena: "
        f"{len(sad_results[0])}"
    )

    print(
        f"SAD vrednosti: "
        f"{len(sad_results[1])}"
    )


    # ========================================================
    # HISTOGRAM ANALIZA
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        " RAČUNANJE HISTOGRAM ANALIZE"
    )

    print(
        "=" * 70
    )

    print(
        "Parametri:"
    )

    print(
        HIST_PARAMETERS
    )


    hist_results = adaptive_histogram(

        video_path,
        raw_features=raw,

        window_size=
            HIST_PARAMETERS["window_size"],

        threshold_factor=
            HIST_PARAMETERS["threshold_factor"],

        min_hist_diff=
            HIST_PARAMETERS["min_hist_diff"],

        peak_multiplier=
            HIST_PARAMETERS["peak_multiplier"],

        min_peak_diff=
            HIST_PARAMETERS["min_peak_diff"]
    )


    print()

    print(
        f"Histogram detektovano promena: "
        f"{len(hist_results[0])}"
    )

    print(
        f"Histogram vrednosti: "
        f"{len(hist_results[1])}"
    )


    # ========================================================
    # ČUVANJE
    # ========================================================

    save_analysis_cache(

        cache_path,

        video_path,

        sad_results,

        hist_results
    )


    # ========================================================
    # VRAĆANJE REZULTATA
    # ========================================================

    return (

        {
            "scene_changes":
                sad_results[0],

            "values":
                sad_results[1],

            "thresholds":
                sad_results[2],

            "frame_indices":
                sad_results[3]
        },

        {
            "scene_changes":
                hist_results[0],

            "values":
                hist_results[1],

            "thresholds":
                hist_results[2],

            "frame_indices":
                hist_results[3]
        }
    )


# ============================================================
# POKRETANJE ANALIZE
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache raw SAD/histogram features and unchanged detectors; annotations are separate.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--cache-dir", default=CACHE_FOLDER)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import cv2
    cv2.setNumThreads(1)
    for video_path in args.input:
        sad, histogram = get_video_analysis(video_path, args.cache_dir)
        logging.info("Analysis complete: %s; SAD cuts=%d, histogram cuts=%d", video_path,
                     len(sad["scene_changes"]), len(histogram["scene_changes"]))
