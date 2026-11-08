"""Compare transit masking on a small, reproducible DR24 training sample.

Run from the project folder containing labeled_catalog.csv and view_tools.py:
    py 07_batch_compare.py
"""

from pathlib import Path
import time
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests
from lightkurve.config import get_cache_dir

from view_tools import load_cached_star


CATALOG = Path("labeled_catalog.csv")
SELECTED = Path("selected_stars.csv")
RESULTS = Path("mask_results.csv")
ATTEMPT = Path("mask_results.csv.part")
STARS_PER_LABEL = 5
SEED = 42

# One star of each label has already been checked by hand.
KNOWN_STARS = {"PC": 757450, "AFP": 1026957, "NTP": 892667}


def select_stars(catalog):
    """Keep one signal per training star, then choose five of each label."""
    train = catalog[catalog["split"] == "train"].copy()
    train = train.drop_duplicates(subset="kepid")

    # This study uses periods <= 50 days and needs room for a nearby region
    # between one and two durations away from the predicted transit.
    duration_days = train["tce_duration"] / 24
    train = train[(train["tce_period"] <= 50) &
                  (4 * duration_days < train["tce_period"])]

    groups = []
    for label, known_id in KNOWN_STARS.items():
        candidates = train[train["av_training_set"] == label]
        known = candidates[candidates["kepid"] == known_id]
        if len(known) != 1:
            raise ValueError(f"Known star {known_id} is missing from {label} training stars")

        others = candidates[candidates["kepid"] != known_id]
        picked = others.sample(n=STARS_PER_LABEL - 1, random_state=SEED)
        groups.append(pd.concat([known, picked]))

    columns = ["kepid", "av_training_set", "tce_period", "tce_time0bk", "tce_duration"]
    return pd.concat(groups, ignore_index=True)[columns]


def get_with_retries(session, url, **kwargs):
    """Retry temporary archive connection failures up to three times."""
    for attempt in range(3):
        try:
            response = session.get(url, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def download_missing_quarters(star_id, session):
    """Get all long-cadence quarters, reusing files already on disk."""
    cache = Path(get_cache_dir())
    star_text = f"{star_id:09d}"
    prefix = f"kepler/public/lightcurves/{star_text[:4]}/{star_text}/"
    folder = cache / "aws_downloads" / f"kic_{star_id}"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "long_cadence_files.txt"
    existing_names = {path.name for path in cache.rglob(f"kplr{star_text}-*_llc.fits")}

    # Once an online listing has been saved, a complete cached star needs no network.
    if manifest.exists():
        expected_names = manifest.read_text(encoding="utf-8").splitlines()
        if expected_names and all(name in existing_names for name in expected_names):
            print(f"  Quarters available: {len(expected_names)}; downloaded now: 0 (verified cache)")
            return

    listing_response = get_with_retries(
        session,
        "https://stpubdata.s3.amazonaws.com/",
        params={"list-type": "2", "prefix": prefix, "max-keys": 1000},
        timeout=30,
    )
    listing = ET.fromstring(listing_response.content)

    if listing.findtext("{*}IsTruncated") == "true":
        raise RuntimeError("The archive has more pages; this star needs pagination")

    keys = [item.text for item in listing.findall(".//{*}Key")
            if item.text and item.text.endswith("_llc.fits")]
    if not keys:
        raise FileNotFoundError(f"No long-cadence files found for KIC {star_id}")

    downloaded = 0
    for key in keys:
        filename = Path(key).name
        if filename in existing_names:
            continue

        destination = folder / filename
        temporary = folder / (filename + ".part")
        try:
            response = get_with_retries(
                session, "https://stpubdata.s3.amazonaws.com/" + key, timeout=60
            )
            temporary.write_bytes(response.content)
            temporary.replace(destination)
            downloaded += 1
            existing_names.add(filename)
        finally:
            temporary.unlink(missing_ok=True)

    # Save the expected filenames only when every quarter is present.
    expected_names = [Path(key).name for key in keys]
    if not all(name in existing_names for name in expected_names):
        raise RuntimeError(f"Some quarter files are missing for KIC {star_id}")
    temporary_manifest = folder / "long_cadence_files.txt.part"
    temporary_manifest.write_text("\n".join(expected_names) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest)

    print(f"  Quarters available: {len(keys)}; downloaded now: {downloaded}")


def measure_signal(row):
    """Measure the same folded dip with and without a flattening mask."""
    star_id = int(row["kepid"])
    period = float(row["tce_period"])
    reference_time = float(row["tce_time0bk"])
    duration_days = float(row["tce_duration"]) / 24

    curve = load_cached_star(star_id)
    mask = curve.create_transit_mask(
        period=period, transit_time=reference_time, duration=2 * duration_days
    )
    with_mask = curve.flatten(window_length=401, mask=mask)
    without_mask = curve.flatten(window_length=401)

    phase = (with_mask.time.value - reference_time + period / 2) % period - period / 2
    distance = np.abs(phase)
    in_dip = distance <= duration_days / 2
    nearby = (distance >= duration_days) & (distance <= 2 * duration_days)

    if not in_dip.any() or not nearby.any():
        raise ValueError("No measurements in the dip or nearby region")

    masked_flux = with_mask.flux.value
    unmasked_flux = without_mask.flux.value
    depth_with_mask = np.median(masked_flux[nearby]) - np.median(masked_flux[in_dip])
    depth_without_mask = np.median(unmasked_flux[nearby]) - np.median(unmasked_flux[in_dip])

    if not np.isfinite([depth_with_mask, depth_without_mask]).all():
        raise ValueError("Depth is not a finite number")

    return {
        "dip_measurements": int(in_dip.sum()),
        "nearby_measurements": int(nearby.sum()),
        "depth_with_mask": depth_with_mask,
        "depth_without_mask": depth_without_mask,
        "depth_difference": depth_with_mask - depth_without_mask,
    }


def main():
    catalog = pd.read_csv(CATALOG)
    chosen = select_stars(catalog)
    chosen.to_csv(SELECTED, index=False)
    print(f"Selected {len(chosen)} training stars; saved {SELECTED}")

    results = []
    with requests.Session() as session:
        for number, (_, row) in enumerate(chosen.iterrows(), start=1):
            star_id = int(row["kepid"])
            print(f"[{number}/{len(chosen)}] KIC {star_id} ({row['av_training_set']})")
            result = row.to_dict()
            try:
                download_missing_quarters(star_id, session)
                result.update(measure_signal(row))
                result["status"] = "ok"
                result["error"] = ""
                print(f"  Depths: {result['depth_with_mask']:.8f} / "
                      f"{result['depth_without_mask']:.8f}")
            except (requests.RequestException, OSError, ValueError, RuntimeError, ET.ParseError) as exc:
                result["status"] = "failed"
                result["error"] = str(exc)
                print(f"  Failed: {exc}")

            results.append(result)
            pd.DataFrame(results).to_csv(ATTEMPT, index=False)

    successful = sum(r["status"] == "ok" for r in results)
    if successful == len(results):
        ATTEMPT.replace(RESULTS)
        print(f"Saved {RESULTS}: {successful} successful stars out of {len(results)}")
    else:
        print(f"Run incomplete: {successful}/{len(results)} stars. Previous {RESULTS} kept; "
              f"inspect {ATTEMPT} for errors.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
