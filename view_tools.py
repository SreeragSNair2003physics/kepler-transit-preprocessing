import numpy as np
import lightkurve as lk
from pathlib import Path
from lightkurve.config import get_cache_dir

######################################################################## DEFINING NORMALIZATION #########################################################################

def normalize_view(values):
    array = np.array(values)
    usual = np.median(array)
    shifted = array - usual
    size = abs(shifted.min())
    return shifted / size

########################################################################### DEFINING FOLDING ############################################################################

def fold_positions(times, period, reference_time):
    distance = times - reference_time
    return (distance + period / 2) % period - period / 2

##################################################################### DEFINING FLATTENING & MASKING #####################################################################


def flatten_for_signal(curve, signal):
    period = signal["tce_period"]
    reference_time = signal["tce_time0bk"]
    duration_days = signal["tce_duration"] / 24

    mask = curve.create_transit_mask(
        period=period,
        transit_time=reference_time,
        duration=2 * duration_days
    )

    return curve.flatten(window_length=401, mask=mask)

############################################################################ DEFINING BINS ##############################################################################

def bin_view(phase, brightness, edges):
    medians = []

    for i in range(len(edges) - 1):
        inside = (phase >= edges[i]) & (phase < edges[i + 1])
        values = brightness[inside]

        if len(values) > 0:
            medians.append(np.median(values))
        else:
            medians.append(np.nan)

    return np.array(medians)

######################################################################## DEFINING CACHE LOADING ##############################################################################


def load_cached_star(star_id):
    cache = Path(get_cache_dir())
    files = sorted(cache.rglob(f"kplr{star_id:09d}-*_llc.fits"))

    if len(files) == 0:
        raise FileNotFoundError(f"No saved quarter files for KIC {star_id}")

    curves = []
    for file in files:
        curves.append(lk.read(str(file)))

    return lk.LightCurveCollection(curves).stitch().remove_nans()

########################################################################## DEFINING CONNECTOR ###########################################################################

def make_views(curve, signal):
    period = signal["tce_period"]
    reference_time = signal["tce_time0bk"]
    duration_days = signal["tce_duration"] / 24

    flat = flatten_for_signal(curve, signal)
    phase = fold_positions(flat.time.value, period, reference_time)

    local_half_width = min(4 * duration_days, period / 2)
    local_edges = np.linspace(-local_half_width, local_half_width, 202)
    global_edges = np.linspace(-period / 2, period / 2, 2002)

    local_bins = bin_view(phase, flat.flux.value, local_edges)
    global_bins = bin_view(phase, flat.flux.value, global_edges)

    if not np.isfinite(local_bins).all() or not np.isfinite(global_bins).all():
        raise ValueError("A view contains missing or invalid brightness values")

    depth = np.median(local_bins) - np.min(local_bins)
    return normalize_view(local_bins), normalize_view(global_bins), depth
