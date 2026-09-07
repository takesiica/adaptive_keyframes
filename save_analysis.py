import os
import json

from functions import adaptive_threshold1, adaptive_histogram


# ============================================================
# FOLDER ZA ČUVANJE ANALIZE
# ============================================================

CACHE_FOLDER = "analysis_cache"

os.makedirs(
    CACHE_FOLDER,
    exist_ok=True
)


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

videos = [
    {
        "name": "test6",
        "path": "/home/takesi/adaptive_keyframes/evaluation/test6.mp4"
    }
]


# ============================================================
# PUTANJA CACHE FAJLA
# ============================================================

def get_cache_path(video_path):

    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    return os.path.join(
        CACHE_FOLDER,
        f"{video_name}_analysis.json"
    )


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

    except Exception as e:

        print(
            f"Greška pri učitavanju cache-a: {e}"
        )

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


    with open(
        cache_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )


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

def get_video_analysis(video_path):

    cache_path = get_cache_path(
        video_path
    )


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

    print()
    print(
        "=" * 90
    )

    print(
        " ČUVANJE SAD + HISTOGRAM ANALIZE"
    )

    print(
        "=" * 90
    )


    for video in videos:

        video_path = video["path"]


        print()
        print(
            "-" * 90
        )

        print(
            video["name"]
        )

        print(
            "-" * 90
        )


        # ----------------------------------------------------
        # Provera videa
        # ----------------------------------------------------

        if not os.path.exists(
            video_path
        ):

            print(
                "GREŠKA: Video ne postoji:"
            )

            print(
                f"        {video_path}"
            )

            continue


        # ----------------------------------------------------
        # Analiza ili učitavanje cache-a
        # ----------------------------------------------------

        sad_results, hist_results = (

            get_video_analysis(
                video_path
            )
        )


        # ----------------------------------------------------
        # Rezultati
        # ----------------------------------------------------

        print()
        print(
            "=" * 70
        )

        print(
            " REZULTAT"
        )

        print(
            "=" * 70
        )


        print(
            f"SAD promena:       "
            f"{len(sad_results['scene_changes'])}"
        )

        print(
            f"Histogram promena: "
            f"{len(hist_results['scene_changes'])}"
        )

        print()


        print(
            f"SAD vrednosti: "
            f"{len(sad_results['values'])}"
        )

        print(
            f"Histogram vrednosti: "
            f"{len(hist_results['values'])}"
        )

        print()


        print(
            f"Sačuvano u: "
            f"{get_cache_path(video_path)}"
        )


    print()
    print(
        "=" * 90
    )

    print(
        " GOTOVO"
    )

    print(
        "=" * 90
    )