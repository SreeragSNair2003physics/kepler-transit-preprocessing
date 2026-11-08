import numpy as np
import matplotlib.pyplot as plt
import pandas as pd                                     # Loads the table tool
import lightkurve as lk                                 # Loads the light-curve tool
from lightkurve.config import get_cache_dir
from pathlib import Path

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

    local_edges = np.linspace(-4 * duration_days, 4 * duration_days, 202)
    global_edges = np.linspace(-period / 2, period / 2, 2002)

    local_bins = bin_view(phase, flat.flux.value, local_edges)
    global_bins = bin_view(phase, flat.flux.value, global_edges)

    depth = np.median(local_bins) - np.min(local_bins)
    return normalize_view(local_bins), normalize_view(global_bins), depth

############################################################################# READING CSV ##############################################################################

catalog= pd.read_csv("labeled_catalog.csv")            # Reads the labeled CSV file; 'catalog' is now a table of signal descriptions
star_id= 757450                                        # The chosen star for study

star_rows= catalog[catalog["kepid"] == star_id] 

print(f"\n\n Catalog signals for star {star_id}:")
print(star_rows[["tce_period", "tce_time0bk", "tce_duration", "av_training_set", "split"]].to_string(index=False))

''' catalog["kepid"] == star_id Checks each row to see if it fits 757450; catalog[...]	Keep the rows where that check was True; 
    .to_string(index=False)	Print those rows without pandas leftmost row-number column.
    star_rows is still a catalog table, not brightness data. '''

clean_curve = load_cached_star(star_id)

print("\n\n Usable measurements:", len(clean_curve))
print(" First three times:", clean_curve.time.value[:3])
print(" First three brightness values:", clean_curve.flux.value[:3])               

############################################################################### PLOTTING 1 ###############################################################################

plt.figure(figsize=(10, 4))
plt.plot(clean_curve.time.value, clean_curve.flux.value, ".", markersize=2)
plt.xlim(131, 165)
plt.xlabel("Time (BKJD days)")                                                       # Plotting Kepler’s measurements of one star, KIC 757450
plt.ylabel("Relative brightness")
plt.title(f"KIC {star_id}: early observations")
plt.savefig("kic_757450_early_lightcurve.png", dpi=150)
plt.show()


################################################################ CHECKING PREDICTION & MEASUREMENT OVERLAP #############################################################

print("\n\n The first catalog signal row for star 757450 is:")
print('  ' + str(star_rows.iloc[0]).replace('\n', '\n  '))                              # Shifts table to the right

signal= star_rows.iloc[0]                                    # Takes the first catalog signal row for star 757450; and saves it to 'signal'
period= signal["tce_period"]                                 # Reads its proposed period
reference_time= signal["tce_time0bk"]                        # Reads its reference dip time

predicted_times = [reference_time + n * period for n in range(4)]     # Calculating the first 4 predicted dip times

print("\n\n Predicted dip times:", *predicted_times, sep=', ')              # Prints out the predicted dip times with a comma seperation

############################################################################### PLOTTING 2 ###############################################################################

plt.figure(figsize=(10, 4))
plt.plot(clean_curve.time.value, clean_curve.flux.value, ".", markersize=2)
plt.xlim(131, 165)

for time in predicted_times:
    plt.axvline(time, color="red")                                         # Red vertical lines for the predicted dip times

plt.xlabel("Time (BKJD days)")
plt.ylabel("Relative brightness")
plt.title(f"KIC {star_id}: predicted dip times")
plt.savefig("KIC_757450_predicted_dips.png", dpi=150)
plt.show()

############################################################################ WIDENESS OF DIP ###########################################################################

duration_hours= signal["tce_duration"]         

''' For this signal, tce_duration is 2.078 hours;
    For the dip at 134.452 BKJD, that means about 1.039 hours before and 1.039 hours after the red line ''' 

duration_days= duration_hours / 24          # Since light-curve times are measured in days, dip width should be in days as well

''' So now, 8.88492 days between dips (period) and 2.078 hours per dip (duration) '''


########################################################################## MASKING & FLATTENING ########################################################################

flat_curve = flatten_for_signal(clean_curve, signal)

flat_without_mask = clean_curve.flatten(window_length=401)

############################################################################### PLOTTING 3 ###############################################################################

plt.figure(figsize=(10, 4))
plt.plot(flat_curve.time.value, flat_curve.flux.value, ".", markersize=2)
plt.xlim(131, 165)
plt.xlabel("Time (BKJD days)")                                                          # Flattened curve graph for better viewing of dips
plt.ylabel("Flattened brightness")
plt.title("KIC 757450: brightness after flattening")
plt.savefig("kic_757450_flattened.png", dpi=150)
plt.show()


plt.figure(figsize=(10, 4))
plt.plot(flat_curve.time.value, flat_curve.flux.value, ".", markersize=3, label="Dips protected during flattening")
plt.plot(flat_without_mask.time.value, flat_without_mask.flux.value, ".", markersize=3, label="Dips used during flattening")
plt.xlim(reference_time - 0.20, reference_time + 0.20)
plt.xlabel("Time (BKJD days)")
plt.ylabel("Flattened brightness")                                                     # Comparison with and w/o masking
plt.title("KIC 757450: does masking change the dip?")
plt.legend()
plt.savefig("mask_comparison_757450.png", dpi=150)
plt.show()

################################################################## DIP RELATIVE TO IT'S OWN BRIGHTNESS #################################################################

distance_from_dip = np.abs(flat_curve.time.value - reference_time)                    # Asks how far each measurement is from the predicted dip at 134.452 days

in_dip = distance_from_dip <= duration_days / 2                                       # Marks measurements within half the dip’s expected width of its center
nearby = (distance_from_dip >= 0.10) & (distance_from_dip <= 0.20)                    # Marks measurements 0.10 to 0.20 days away

blue_nearby = np.median(flat_curve.flux.value[nearby])
blue_dip = np.median(flat_curve.flux.value[in_dip])
blue_depth = blue_nearby - blue_dip

orange_nearby = np.median(flat_without_mask.flux.value[nearby])
orange_dip = np.median(flat_without_mask.flux.value[in_dip])
orange_depth = orange_nearby - orange_dip

print("\n\n Measurements inside dip:", in_dip.sum())
print(" Nearby measurements:", nearby.sum())
print(" Depth with mask:", blue_depth)
print(" Depth without mask:", orange_depth)

########################################################################## FOLDING THE WAVES ###########################################################################

''' Folding lines up the measurements, so that the dips are placed on top of each other; That gives us many measurements 
    of the dip's shape instead of relying on one occurrence. '''

times= flat_curve.time.value                        # Takes the time of every measurement from the flattened light curve
brightness= flat_curve.flux.value                   # Takes its brightness values. These values will not be changed by folding

phase = fold_positions(times, period, reference_time)

print("\n\n Measurements:", len(brightness))
print(" Folded positions:", len(phase))                         # Checks if every brightness measurement has a matching folded position

distance_from_folded_dip = np.abs(phase)

all_dips = distance_from_folded_dip <= duration_days / 2
all_nearby = ((distance_from_folded_dip >= duration_days) & (distance_from_folded_dip <= 2 * duration_days))

blue_nearby = np.median(flat_curve.flux.value[all_nearby])
blue_dip = np.median(flat_curve.flux.value[all_dips])
blue_depth = blue_nearby - blue_dip

orange_nearby = np.median(flat_without_mask.flux.value[all_nearby])
orange_dip = np.median(flat_without_mask.flux.value[all_dips])
orange_depth = orange_nearby - orange_dip

print("\n\n Measurements inside folded dip:", all_dips.sum())
print(" Nearby folded measurements:", all_nearby.sum())
print(" Folded depth with mask:", blue_depth)
print(" Folded depth without mask:", orange_depth)

############################################################################### PLOTTING 4 ###############################################################################

plt.figure(figsize=(10, 4))
plt.plot(phase, brightness, ".", markersize=1, alpha=0.15)
plt.xlabel("Days from predicted dip")
plt.ylabel("Flattened brightness")                                                       # Folded curve graph for studying the dips better
plt.title("KIC 757450: folded light curve")
plt.savefig("KIC_757450_folded.png", dpi=150)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(phase, brightness, ".", markersize=2, alpha=0.2)
plt.xlim(-4 * duration_days, 4 * duration_days)
plt.xlabel("Days from predicted dip")
plt.ylabel("Flattened brightness")                                                       # Close-up view of the folded dip
plt.title("KIC 757450: close view of the folded dip")
plt.savefig("KIC_757450_folded_close.png", dpi=150)
plt.show()

############################################################################## BINNING ##################################################################################

'''We divide the horizontal time range into narrow parts of vertical strips. 
   For each part, we gather the brightness measurements inside it, find their median brightness, and plot one dot at that height '''

left_edge = -4 * duration_days
right_edge = 4 * duration_days                                     # Defining the borders of the vertical strips
edges = np.linspace(left_edge, right_edge, 202)

print("\n\n Left border of the close-up:", edges[0])
print(" Right border of the close-up:", edges[-1])
print(" Number of strips:", len(edges) - 1)

i = 100                                                # The middle strip; center of dip
inside = (phase >= edges[i]) & (phase < edges[i + 1])  # Checks & gives T/F for dots inside the strip; edges[100] is it's left border; edges[101] is its right border
values = brightness[inside]                            # Keeps brightness of only the dots marked True

print("\n\n Strip borders:", edges[i], ", ", edges[i + 1]) 
print(" Measurements in the center strip:", len(values))
print(" Median brightness of dip center:", np.median(values))

local_brightness = bin_view(phase, brightness, edges)

print("\n\n Representative brightness values:", len(local_brightness))    # Counts answers in list; should be 201
print(" Empty strips:", np.isnan(local_brightness).sum())             # Marks every missing number True, and counts them

centers = (edges[:-1] + edges[1:]) / 2                  # Calculating center of each strip

print("\n\n Number of strip centers:", len(centers))
print(" Middle strip center:", centers[100])

############################################################################### PLOTTING 5 ###############################################################################

plt.figure(figsize=(10, 4))
plt.plot(centers, local_brightness, ".-", markersize=3)
plt.xlabel("Days from predicted dip")
plt.ylabel("Median brightness in each strip")                                       # Binned graph with matching 201 strip centers & 201 median brightness values
plt.title("KIC 757450: binned close view")
plt.savefig("KIC_757450_binned_close.png", dpi=150)
plt.show()

######################################################################### NORMALIZING THE DIPS #########################################################################

''' Normalization changes the numbers on the vertical axis, not the dip's shape; Your current graph has ordinary brightness near 1.000 
    and the bottom of the dip near 0.984. We do it in two moves:

    1. Move ordinary brightness to 0. Subtract the median brightness from every one of the 201 values. Roughly, 1.000 - 1.000 = 0 and 0.984 - 1.000 = -0.016.
    2. Make the deepest point -1. Divide every shifted value by the dip's size, roughly 0.016. Then -0.016 / 0.016 = -1.

    Suppose another star has a dip with the same shape but only half as deep.
    Putting both graphs on a scale from ordinary brightness 0 to deepest point -1 lets the model compare their shapes '''

dip_size = np.median(local_brightness) - np.min(local_brightness)
local_normalized = normalize_view(local_brightness)


print("\n\n Original dip size:", dip_size)
print(" New median after normalizing:", np.median(local_normalized))
print(" New minimum after normalizing:", local_normalized.min())

###################################################################### GLOBAL VIEW OF THE DIPS ##########################################################################

'''Local view shows about 8.3 hours on either side of the predicted dip. It is good for examining that dip's shape.
   A global view shows the entire 8.88492-day repeating cycle. That matters because some false positives, 
   such as eclipsing pairs of stars, can show another dip away from the center.'''

global_left = -period / 2
global_right = period / 2
global_edges = np.linspace(global_left, global_right, 2002)

print("\n\n Global strip left border:", global_left)
print(" Global strip right border:", global_right)
print(" Global strips:", len(global_edges) - 1)

print("\n\n Calculating global strip medians...", flush=True)                     # Finding out the median of the global view strips; similarly to above

global_brightness = bin_view(phase, brightness, global_edges)

print("\n Global brightness values:", len(global_brightness))
print(" Empty global strips:", np.isnan(global_brightness).sum())


global_centers = (global_edges[:-1] + global_edges[1:]) / 2                       # Finding out the strip center and normalizing like above

global_normalized = normalize_view(global_brightness)

print("\n\n Global strip centers:", len(global_centers))
print(" Global median after normalizing:", np.median(global_normalized))
print(" Global minimum after normalizing:", global_normalized.min())

############################################################################### PLOTTING 6 ###############################################################################

plt.figure(figsize=(10, 4))
plt.plot(global_centers, global_normalized, linewidth=0.8)
plt.xlabel("Days from predicted dip")
plt.ylabel("Normalized brightness")                                            # Global view of the dip; 201 local brightness values and 2001 global brightness values
plt.title("KIC 757450: full-cycle view")
plt.savefig("KIC_757450_global_view.png", dpi=150)
plt.show()

############################################################################# SAVING SAMPLE #############################################################################

sample_file = f"kic_{star_id}_signal_{int(signal['tce_plnt_num'])}.npz"

np.savez_compressed(
    sample_file,
    local=local_normalized,
    global_view=global_normalized,
    depth=dip_size,
    planet=int(signal["planet"]),
    split=str(signal["split"])
)

print("\n\n\n Saved model example:", sample_file)

######################################################################## EXPANDED DEFINITIONS ##########################################################################

''' 

local_array = np.array(local_brightness)    # Turns your list of 201 median brightness values into a NumPy array
usual_level = np.median(local_array)        # Finds the median of those 201 values: approximately 1

shifted = local_array - usual_level         # Subtracts that usual level from every value. Ordinary brightness becomes about 0; the dip becomes negative
dip_size = abs(shifted.min())               # Finds how far the lowest point is below 0
local_normalized = shifted / dip_size       # Divides all 201 values by that size. The lowest value becomes -1 

'''

'''

global_array = np.array(global_brightness)
global_usual = np.median(global_array)
global_shifted = global_array - global_usual
global_dip_size = abs(global_shifted.min())
global_normalized = global_shifted / global_dip_size

'''

'''

local_brightness = []                                  # For keeping brightness of each strip

for i in range(201):
    inside = (phase >= edges[i]) & (phase < edges[i + 1])             # Puts a True sticker on each measurement whose folded position falls inside the current strip
    values = brightness[inside]

    if len(values) > 0:                                 # Checks to see if there are measurements to find median brightness
        local_brightness.append(np.median(values))      # Finds median value of the brightness and adds it to the list

    else:
        local_brightness.append(np.nan)                 # If the strip is empty, adds a placeholder


'''

'''

global_brightness = []

for i in range(2001):
    inside = (phase >= global_edges[i]) & (phase < global_edges[i + 1])
    values = brightness[inside]

    if len(values) > 0:
        global_brightness.append(np.median(values))

    else:
        global_brightness.append(np.nan)


''' 




"""


cache = Path(get_cache_dir())
quarter_files = sorted(cache.rglob(f"kplr{star_id:09d}-*_llc.fits"))


print("\n\n Saved quarter files found:", len(quarter_files))

''' One for each available quarter for this star. A quarter is a chunk of Kepler observations, usually about 90 days long.
    Kepler rotated roughly every 90 days to keep its solar panels pointed toward the Sun, so its data is organized into quarters 
    BKJD is the clock used for Kepler times '''


curves = []
for file in quarter_files:
    curves.append(lk.read(str(file)))

''' As of now,

    star_rows-->	Catalog rows describing possible dips around star 757450
    curves-->	    Listings for 17 files containing that star's brightness measurements '''    

########################################################################### QUARTER MEASUREMENTS #######################################################################

quarters = lk.LightCurveCollection(curves)                    # Downloads all files found; it is now a collection of light curves

print("\n\n Measurements in first quarter:", len(quarters[0]))   # Counts brightness measurements inside the first file pf 'quarters'

whole_curve = quarters.stitch()                    

''' stitch() puts each quarter brightness on a comparable scale,
    then joins their time and brightness measurements into one light curve. By default, Lightkurve divides each quarter brightness values by that quarter
    median brightness. '''

print("\n\n Measurements after joining:", len(whole_curve))    # Counts the time–brightness measurements inside that joined curve

clean_curve = whole_curve.remove_nans()                   # Removes every missing numerical values

print("\n\n Before cleaning:", len(whole_curve))
print(" After cleaning:", len(clean_curve))
print(" Removed:", len(whole_curve) - len(clean_curve))

print("\n\n First three times:", clean_curve.time.value[:3])                         # [:3] takes the first three entries    
print(" First three brightness values:", clean_curve.flux.value[:3]) 

"""



"""

distance= times - reference_time                    # Calculates for each measurement: “How many days is this from the reference dip at 134.452"

phase= (distance + period / 2) % period - period / 2

''' [phase = ...] takes those distances and wraps them around every 8.88492 days. % period is the wrapping operation,
    like a clock returning to the start after one complete cycle. The + period / 2 and - period / 2 put each predicted dip at 0, 
    with times just before it negative and times just after it positive. '''

"""  




"""

####################################################################### MASKING BRIGHTNESS NEAR DIPS ###################################################################

''' Putting a “near a dip” sticker on the brightness measurements around each predicted dip time; that's one measurement of the transit_mask '''

transit_mask= clean_curve.create_transit_mask( period=period, transit_time=reference_time, duration=2 * duration_days )

''' period= period says how far apart the predicted dips are; transit_time= reference_time says where one predicted dip is;
    duration=2 * duration_days says how wide an area to mark around each predicted dip;
    used twice the width to have a little wiggle room; transit_mask is the resulting list of True and False markers, one for each measurement '''

print("\n\n Total measurements masked:", len(transit_mask))                       # Counts all the stickers; one for each usable brighness measurement
print(" Measurements marked near dips:", transit_mask.sum())           # Counts only True stickers

######################################################################### FLATTENING BROAD WAVES #######################################################################


''' The blue brightness values rise and fall in broad waves; We straighten those broad waves so the short dips are easier to study '''

flat_curve= clean_curve.flatten(window_length=401, mask=transit_mask)

''' To work out usual brightness around a dip, we look at nearby measurements to left and right of dip, including the dip but that measurement is masked
    this is the window_length= 401; 200 before the measurement itself, and 200 after it  '''   

"""