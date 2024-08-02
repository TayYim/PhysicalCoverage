
import os
import re
import sys
import glob
import math
import random 
import argparse
import multiprocessing

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/processing_pipeline")])
sys.path.append(base_directory)

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils.environment_configurations import PolarConfig
from utils.environment_configurations import WaymoKinematics
from utils.environment_configurations import BeamNGKinematics
from utils.environment_configurations import HighwayKinematics

from utils.RRS_distributions import center_close_distribution
from utils.RRS_distributions import center_full_distribution

from utils.failure_oracle import FailureOracle

from process_functions import processPolarFingerprint
from process_functions import countVectorsInFile


parser = argparse.ArgumentParser()
parser.add_argument('--data_path',      type=str, default="/mnt/extradrive3/PhysicalCoverageData",          help="The location and name of the datafolder")
parser.add_argument('--total_samples',  type=int, default=-1,                                               help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--distribution',   type=str, default="",                                               help="center_close/center_full")
parser.add_argument('--scenario',       type=str, default="",                                               help="beamng/highway/waymo")
parser.add_argument('--cores',          type=int, default=4,                                                help="number of available cores")
args = parser.parse_args()

# Create the configuration classes
HK = HighwayKinematics()
BK = BeamNGKinematics()
WK = WaymoKinematics()
FO = FailureOracle(scenario=args.scenario)

POLAR = PolarConfig(lx=12, ly=60, n_rad=4, n_ring=3)

lx = POLAR.lx
ly = POLAR.ly
n_rad = POLAR.n_rad
n_ring = POLAR.n_ring

# Save the kinematics and RRS parameters
if args.scenario == "highway_random":
    new_steering_angle  = HK.steering_angle
    new_max_distance    = HK.max_velocity
elif args.scenario == "highway_generated":
    new_steering_angle  = HK.steering_angle
    new_max_distance    = HK.max_velocity
elif args.scenario == "beamng_random":
    new_steering_angle  = BK.steering_angle
    new_max_distance    = BK.max_velocity
elif args.scenario == "beamng_generated":
    new_steering_angle  = BK.steering_angle
    new_max_distance    = BK.max_velocity
elif args.scenario == "waymo_random":
    new_steering_angle  = WK.steering_angle
    new_max_distance    = WK.max_velocity
else:
    print("ERROR: Unknown scenario ({})".format(args.scenario))
    exit()

if args.distribution == "center_close":
    distribution  = center_close_distribution(args.scenario)
elif args.distribution == "center_full":
    distribution  = center_full_distribution(args.scenario)
else:
    print("ERROR: Unknown distribution ({})".format(args.distribution))
    exit()

print("")
print("----------------------------------")
print("")

# Get the total number of possible crashes per test
max_crashes_per_test = FO.max_possible_crashes
max_stalls_per_test  = FO.max_possible_stalls
failure_base         = FO.base

print("----------------------------------")
print("-----Reach Set Configuration------")
print("----------------------------------")

print("Max steering angle:\t" + str(new_steering_angle))
print("Max velocity:\t\t" + str(new_max_distance))

print("----------------------------------")
print("----------Locating Files----------")
print("----------------------------------")

all_files = None
if args.scenario == "highway_random":
    all_files = glob.glob("{}/highway/random_tests/physical_coverage/raw/*/*.txt".format(args.data_path))
elif args.scenario == "highway_generated":
    all_files = glob.glob("{}/highway/generated_tests/{}/physical_coverage/raw/{}_external_vehicles/*.txt".format(args.data_path, args.distribution, args.beam_count))
else:
    print("Error: Scenario not known")
    exit()

total_files = len(all_files)
print("Total files found: " + str(total_files))

# Select all of the files
file_names = all_files

# If no files are found exit
if len(file_names) <= 0:
    print("No files found")
    exit()

# If you don't want all files, select a random portion of the files
if (args.scenario == "highway_random") or (args.scenario == "beamng_random"): 
    if args.scenario == "highway_random":
        folders = glob.glob("{}/highway/random_tests/physical_coverage/raw/*".format(args.data_path))
    elif args.scenario == "beamng_random":
        folders = glob.glob("{}/beamng/random_tests/physical_coverage/raw/*".format(args.data_path))

    files_per_folder = int(math.ceil(args.total_samples / len(folders)))

    # Need to set the seed or else you will be picking different tests for each different beam number  
    random.seed(10)
    print("There are {} categories, thus we need to select {} from each".format(len(folders), files_per_folder))
    print("")
    file_names = []
    for f in folders:
        print("Selecting {} random files from - {}".format(files_per_folder, f))
        all_files = glob.glob(f + "/*.txt")
        names = random.sample(all_files, files_per_folder)
        file_names.append(names)
        
if args.scenario == "highway_generated" or args.scenario == "beamng_generated": 
    # You want to select all files here so do nothing
    pass

# Flatten the list
if len(np.shape(file_names)) > 1:
    file_names_flat = []
    for subl in file_names:
        for item in subl:
            file_names_flat.append(item)
    file_names = file_names_flat

# Get the file size
total_files = len(file_names)
print("Total files selected for processing: " + str(total_files))

print("----------------------------------")
print("--------Memory Requirements-------")
print("----------------------------------")
print(total_files)
print("Computing size of memory required")
# Open the first 1000 files to get an estimate of how many vectors in each file
vectors_per_file = np.zeros(min(total_files, 1000), dtype=int)
for i in tqdm(range(min(total_files, 1000))):
    # Get the filename
    file_name = file_names[i]

    # Process the file
    f = open(file_name, "r")    
    vector_count, crash = countVectorsInFile(f)
    f.close()

    # See how many vectors there are
    vectors_per_file[i] = vector_count

# Compute the average number of vectors per file
vec_per_file = np.max(vectors_per_file)


print("----------------------------------")
print("---------Processing files---------")
print("----------------------------------")

# Create the numpy array 
polar_fingerprints = np.zeros((total_files, vec_per_file, 3), dtype=int)

total_processors = int(args.cores)
pool =  multiprocessing.Pool(processes=total_processors)

# Call our function total_test_suites times
jobs = []
for i in range(total_files):
    # Get the filename
    file_name = file_names[i]
    # Start the job
    jobs.append(pool.apply_async(processPolarFingerprint, args=([file_name, vec_per_file, POLAR])))

# Get the results
for i, job in enumerate(tqdm(jobs)):
    result = job.get()
    this_fingerprints, file_name = result

    polar_fingerprints[i] = this_fingerprints

# Close your pools
pool.close()

save_name = args.scenario
save_name += "_lx" + str(POLAR.lx) 
save_name += "_ly" + str(POLAR.ly) 
save_name += "_nrad" + str(POLAR.n_rad) 
save_name += "_nring" + str(POLAR.n_ring) 
save_name += "_t" + str(total_files)
save_name += ".npy"
   
save_path = ""
if args.scenario == "highway_random":
    save_path = "../output/highway/random_tests/physical_coverage/processed/{}/{}".format(args.distribution, args.total_samples)
elif args.scenario == "highway_generated":
    save_path = "../output/highway/generated_tests/{}/physical_coverage/processed/{}/".format(args.distribution, args.total_samples)
elif args.scenario == "waymo_random":
    save_path = "../output/waymo/random_tests/physical_coverage/processed/{}/{}".format(args.distribution, args.total_samples)
else:
    print("Error 4")
    exit()

# Create the output directory if it doesn't exists
if not os.path.exists(save_path):
    os.makedirs(save_path)

print()

print("Saving data")
np.save(save_path + '/polars_{}'.format(save_name), polar_fingerprints)