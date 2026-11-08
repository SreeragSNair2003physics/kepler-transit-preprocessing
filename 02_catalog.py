import pandas as pd                                           # Tool for working with tables
from sklearn.model_selection import train_test_split          # Tool for splitting

############################################################################# READING CSV ##############################################################################

table = pd.read_csv("q1_q17_dr24_tce.csv")

print("\nFirst five rows:")
print(table.head())                                           # First 5 rows; each row is a signal
print("\nRows and columns:", table.shape)

''' | Column              | Row 0 says | Meaning 

    | `kepid`             | `1162345`  | Which star was observed 
    | `tce_plnt_num`      | `2`        | The catalog number for this signal around that star; it does not mean a confirmed second planet 
    | `tce_period`        | `0.831850` | Proposed time between dips, in days 
    | `tce_time0bk`       | `132.227`  | A reference dip time, in Kepler BKJD time system 
    | `tce_duration`      | `2.392`    | Proposed length of one dip, in hours 
    | `av_training_set`   | `AFP`      | Labeled an **astrophysical false positive, rather than a planet candidate '''

signal = table.iloc[0]                                 # Takes the row at position 0;  the first row printed above

print("\nStar ID:", signal["kepid"])                     # Get the value under "kepid" from this row
print("Period in days:", signal["tce_period"])         
print("Label:", signal["av_training_set"])

################################################################ SELECTING NEEDED PLANETARY LABELS #####################################################################

known = table["av_training_set"].isin(["PC", "AFP", "NTP"])      # Selects the label column in table; isin checks it against the three defined labels
labeled = table[known].copy()                                    # Takes the true values from above and copies into a table named 'labeled'

print("\nFirst five decisions:")
print(known.head())                                              # Only first 5 taken
print("\nLabeled signals:", len(labeled))

################################################################# CHOOSING PLANETARY CANDIDATES ########################################################################

is_planet = labeled["av_training_set"] == "PC"                   # is_planet contains True for PC and False for AFP or NTP
labeled["planet"] = is_planet.astype(int)                        # Changes True to 1 and False to 0; and stores them in a new column named 'planet'

print("\nOriginal row vs Old label vs New number")
print(labeled[["av_training_set", "planet"]].head())

'''Original row numbers are on the far left; the gaps are expected as the ones labeled UNK are removed'''

print("\nAmount of ones and zeros")
print(labeled["planet"].value_counts())                          # Counts how many 1s and 0s are there in 'planet' label column

########################################################################### SORTING #######################################################################################

star_ids = labeled["kepid"].unique()                             # Collects distinct star IDs

print("\nLabeled signals:", len(labeled))
print("Different stars:", len(star_ids))

######################################################################### FIRST SPLIT ##################################################################################


remaining_stars, test_stars = train_test_split( star_ids, test_size=0.10, random_state=42 )    

''' Randomly divides list of distinct star IDs; 10% of stars are put in 'test_stars' ; the rest are put in 'remaining_stars' ; 
random_state=42 fixes the random selection so running the script again gives the same split  '''


print("\nRemaining stars:", len(remaining_stars))
print("Test stars:", len(test_stars))

######################################################################## SECOND SPLIT ####################################################################################

train_stars, val_stars = train_test_split( remaining_stars, test_size=len(test_stars), random_state=43 )

''' Divides only the 8,878 remaining stars. The test stars are already set aside. 
Test_size=len(test_stars) asks for the same number of validation stars (val_stars) as test stars (= 987); rest are put into train_stars '''

print("\nTrain stars:", len(train_stars))
print("Validation stars:", len(val_stars))
print("Test stars:", len(test_stars))

################################################################### ADDING SPLITS TO TABLE #############################################################################

labeled["split"] = "train"                                  # Creates a new column 'split' and initially writes train on every row

val_rows = labeled["kepid"].isin(val_stars)                 # Checks each row’s star ID against the validation star IDs (Saves as True,False,True...)
test_rows = labeled["kepid"].isin(test_stars)

labeled.loc[val_rows, "split"] = "val"                      # In the rows where val_rows is True, set the split column to "val"
labeled.loc[test_rows, "split"] = "test"

print("\nThe new columns are:")

print(labeled[["kepid", "split"]].head())                   # First 5 entries of the columns 'kepid' & 'split'

print("\nThe number of signals in each group type are:")
print(labeled["split"].value_counts())                      # Counts signals in each group

print("\n\nTrain/validation overlap:", len(set(train_stars) & set(val_stars)))               # Sanity check to see whether any signals appear again in any other group
print("Train/test overlap:", len(set(train_stars) & set(test_stars)))
print("Validation/test overlap:", len(set(val_stars) & set(test_stars)))

################################################################## SAVING THE NEWLY CREATED TABLE #######################################################################

labeled.to_csv("labeled_catalog.csv", index=False)

'''' labeled.to_csv(...) writes the newly prepared table to a CSV file in the current folder. 
It will contain the original signal columns plus the planet and split columns created. index=False tells pandas not to add the
leftmost row numbers (0, 3, 5, ...) as another CSV column '''

print("\n\nSaved labeled_catalog.csv")
