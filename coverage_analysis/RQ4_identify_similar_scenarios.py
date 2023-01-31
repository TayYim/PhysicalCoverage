import sys
import glob
import random
import hashlib
import argparse
import itertools

import multiprocessing

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/coverage_analysis")])
sys.path.append(base_directory)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tqdm import tqdm
from collections import Counter
from prettytable import PrettyTable

from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from general.file_functions import get_beam_number_from_file
from general.file_functions import order_files_by_beam_number
from general.environment_configurations import RRSConfig
from general.environment_configurations import WaymoKinematics

from matplotlib_venn import venn3, venn3_circles

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


def compare_3_way(i1, i2, i3):
    # Compute the number of duplicates over the middle
    test_a = random_traces_hashes[i1]
    test_b = random_traces_hashes[i2]
    test_c = random_traces_hashes[i3]
    abc = number_of_duplicates(test_a, test_b, test_c)

    ab = number_of_duplicates(test_a, test_b) - abc
    ac = number_of_duplicates(test_a, test_c) - abc
    bc = number_of_duplicates(test_b, test_c) - abc

    total_duplicates = (3 * (abc)) + (2 * (ab + ac + bc))

    return total_duplicates, i1, i2, i3


def number_of_duplicates(list_a, list_b, list_c=None):
    # Get all the keys and a count of them
    count_a = Counter(list_a)
    count_b = Counter(list_b)

    if list_c is not None:
        count_c = Counter(list_c)
    
    
    # Get all common keys between both lists
    common_keys = set(count_a.keys()).intersection(count_b.keys())
    if list_c is not None:
        common_keys = common_keys.intersection(count_c.keys())

    # The count is the min number in both.
    # i.e. if list a has "200 A's, 50 B's" and list b has "100 A's 150 B's" then together they share "100 A's and 50 B's for a total of 150"
    if list_c is not None:
        result = sum(min(count_a[key], count_b[key], count_c[key]) for key in common_keys)
    else:
        result = sum(min(count_a[key], count_b[key]) for key in common_keys)

    return result


def number_of_duplicates_with_common_list(list_a, list_b):
    # Get all the keys and a count of them
    count_a = Counter(list_a)
    count_b = Counter(list_b)
    
    # Get all common keys between both lists
    common_keys = set(count_a.keys()).intersection(count_b.keys())

    # The count is the min number in both.
    # i.e. if list a has "200 A's, 50 B's" and list b has "100 A's 150 B's" then together they share "100 A's and 50 B's for a total of 150"
    result = sum(min(count_a[key], count_b[key]) for key in common_keys)

    # Create a common list which contains all the keys that are in common and count the min number in both
    # i.e. if list a has "200 A's, 50 B's" and list b has "100 A's 150 B's" then together they share "100 A's and 50 B's for a total of 150"
    common_list = []
    num_of_duplicates = 0
    number_of_this_key = 0
    for key in common_keys:
        # Get the number of this key
        number_of_this_key = min(count_a[key], count_b[key])
        # Add this to the duplicate counter list
        num_of_duplicates += number_of_this_key
        # Add this number of keys to the common_list
        for i in range(number_of_this_key):
            common_list.append(key)

    # Make sure they are the same size
    assert(num_of_duplicates == len(common_list))
        
    return num_of_duplicates, common_list


# Get the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',          type=str, default="/mnt/extradrive3/PhysicalCoverageData",          help="The location and name of the datafolder")
parser.add_argument('--number_of_tests',    type=int, default=-1,                                               help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--distribution',       type=str, default="",                                               help="linear/center_close/center_mid")
parser.add_argument('--scenario',           type=str, default="",                                               help="waymo")
args = parser.parse_args()

# Create the configuration classes
WK = WaymoKinematics()
RRS = RRSConfig()

# Save the kinematics and RRS parameters
if args.scenario == "waymo":
    new_steering_angle  = WK.steering_angle
    new_max_distance    = WK.max_velocity
else:
    print("ERROR: Unknown scenario")
    exit()

print("----------------------------------")
print("-----Reach Set Configuration------")
print("----------------------------------")

print("Max steering angle:\t" + str(new_steering_angle))
print("Max velocity:\t\t" + str(new_max_distance))

print("----------------------------------")
print("-----------Loading Data-----------")
print("----------------------------------")

load_name = ""
load_name += "_s" + str(new_steering_angle) 
load_name += "_b" + str('*') 
load_name += "_d" + str(new_max_distance) 
load_name += "_t" + str(args.number_of_tests)
load_name += ".npy"

# Checking the distribution
if not (args.distribution == "linear" or args.distribution == "center_close" or args.distribution == "center_mid"):
    print("ERROR: Unknown distribution ({})".format(args.distribution))
    exit()

# Get the file names
base_path = '/mnt/extradrive3/PhysicalCoverageData/{}/random_tests/physical_coverage/processed/{}/{}/'.format(args.scenario, args.distribution, args.number_of_tests)
random_trace_file_names = glob.glob(base_path + "traces_*")

if len(random_trace_file_names) <= 0:
    print("files not found")
    exit()

# Get the RRS numbers
random_trace_RRS_numbers    = get_beam_number_from_file(random_trace_file_names)

# Find the set of beam numbers which all sets of files have
RRS_numbers = list(set(random_trace_RRS_numbers))
RRS_numbers = sorted(RRS_numbers)

# Sort the data based on the beam number
random_trace_file_names         = order_files_by_beam_number(random_trace_file_names, RRS_numbers)


# For each of the different beams
for RRS_index in range(len(RRS_numbers)):

    print("\n\n")
    print("------------------------------------------")
    print("------------Processing RRS: {}------------".format(RRS_numbers[RRS_index]))
    print("------------------------------------------")

    if not( RRS_numbers[RRS_index] == 5 or RRS_numbers[RRS_index] == 10):
        print("Skipping")
        continue

    # Get the beam number and files we are currently considering
    RRS_number              = RRS_numbers[RRS_index]
    random_trace_file       = random_trace_file_names[RRS_index]

    # Skip if any of the files are blank
    if random_trace_file == "":
        print(random_trace_file)
        print("\nWarning: Could not find one of the files for RRS number: {}".format(RRS_number))
        continue

    # Load the random_traces
    global random_traces
    random_traces = np.load(random_trace_file)

    # Holds the trace as a set of hashes  
    global random_traces_hashes
    random_traces_hashes = []

    print("\n")
    print("-------Converting Traces to Hashes--------")

    # Convert to hash value for easier comparison
    for test_number in tqdm(range(np.shape(random_traces)[0])):
        hashes = []
        current_trace = random_traces[test_number, :, :]
        for t in current_trace:
            # Ignore nans
            if not np.isnan(t).any():
                # Convert to a hash
                trace_string = str(t)
                hash = hashlib.md5(trace_string.encode()).hexdigest()
                hashes.append(hash)

        # Save the hashes
        random_traces_hashes.append(hashes)


    print("\n")
    print("---------Creating Comparison Map----------")

    comparison_map = np.zeros((np.shape(random_traces)[0], np.shape(random_traces)[0]))

    # For each of the tests
    for a in tqdm(range(np.shape(random_traces)[0])):
        for b in range(np.shape(random_traces)[0]):
            
            # Get test A ad B
            test_a = random_traces_hashes[a]
            test_b = random_traces_hashes[b]

            # Count duplicates
            dup_count = number_of_duplicates(test_a, test_b)

            # Save to a comparison map
            comparison_map[a, b] = dup_count

    # Plot the map
    plt.figure("RRS {}".format(RRS_number))
    plt.imshow(comparison_map)

    print("\n")
    print("---------------Venn Diagram---------------")

    print("Comparing highway scenarios")
    # highway
    highway_scenarios = [0, 1, 3, 35, 69, 83, 94, 96, 110, 120, 128, 132, 157, 159, 163, 195, 197, 203, 266, 271]
    highway_scenarios = [132, 157, 197]

    test_a = random_traces_hashes[highway_scenarios[0]]
    test_b = random_traces_hashes[highway_scenarios[1]]
    test_c = random_traces_hashes[highway_scenarios[2]]
    abc = number_of_duplicates(test_a, test_b, test_c)

    ab = number_of_duplicates(test_a, test_b) - abc
    ac = number_of_duplicates(test_a, test_c) - abc
    bc = number_of_duplicates(test_b, test_c) - abc

    a = len(test_a) - abc - ab - ac
    b = len(test_b) - abc - ab - bc
    c = len(test_c) - abc - ac - bc

    plt.figure("RRS {} Highway Ven".format(RRS_number))
    v = venn3(subsets=(a,b,ab,c,ac,bc,abc))
    plt.title("Highway Scenarios RRS Similarities")

    print("Comparing random scenarios")
    # Set the seed
    random.seed(10)
    random_scenarios = random.sample(range(len(random_traces_hashes)), 3)

    test_a = random_traces_hashes[random_scenarios[0]]
    test_b = random_traces_hashes[random_scenarios[1]]
    test_c = random_traces_hashes[random_scenarios[2]]
    abc = number_of_duplicates(test_a, test_b, test_c)

    ab = number_of_duplicates(test_a, test_b) - abc
    ac = number_of_duplicates(test_a, test_c) - abc
    bc = number_of_duplicates(test_b, test_c) - abc

    a = len(test_a) - abc - ab - ac
    b = len(test_b) - abc - ab - bc
    c = len(test_c) - abc - ac - bc

    plt.figure("RRS {} Random Ven".format(RRS_number))
    v = venn3(subsets=(a,b,ab,c,ac,bc,abc))
    plt.title("Random Scenarios RRS Similarities")

    print("\n")
    print("---------Find best/worst 3 way-----------")

    # Create all permutations of the test indices
    indices = np.arange(len(random_traces_hashes))
    perms = itertools.combinations(indices, r=3)

    total_processors = int(120)
    pool =  multiprocessing.Pool(processes=total_processors)

    # Call our function total_test_suites times
    jobs = []
    for p in tqdm(perms):
        jobs.append(pool.apply_async(compare_3_way, args=([p[0], p[1], p[2]])))

    # Get the results
    results = []
    for job in tqdm(jobs):
        results.append(job.get())

    # Close the pool
    pool.close()

    # Look at the results
    best_indices = np.full((5,3), -1)
    worst_indices = np.full((5,3), -1)
    max_duplicates = np.full(5, -np.inf)
    min_duplicates = np.full(5, np.inf)

    for r in results:
        num_dups = r[0]
        if num_dups > max_duplicates[0]:
            # Adding
            max_duplicates      = np.roll(max_duplicates, -1)
            best_indices        = np.roll(best_indices, -1, axis=0)
            max_duplicates[-1]  = num_dups
            best_indices[-1]    = np.array([r[1], r[2], r[3]])
            # Sorting
            sort_indices        = np.argsort(max_duplicates)
            max_duplicates      = max_duplicates[sort_indices]
            best_indices        = best_indices[sort_indices]
        if num_dups < min_duplicates[-1]:
            # Adding
            min_duplicates      = np.roll(min_duplicates, 1)
            worst_indices       = np.roll(worst_indices, 1, axis=0)
            min_duplicates[0]  = num_dups
            worst_indices[0]   = np.array([r[1], r[2], r[3]])
            # Sorting
            sort_indices        = np.argsort(min_duplicates)
            min_duplicates      = min_duplicates[sort_indices]
            worst_indices       = worst_indices[sort_indices]

    
    print("Best number of duplicates: {}".format(max_duplicates))
    print("Best Indices:\n{}\n\n".format(best_indices))

    print("Worst number of duplicates: {}".format(min_duplicates))
    print("Worst Indices:\n{}\n\n".format(worst_indices))

    for i in range(len(best_indices)):

        bi = best_indices[i]

        test_a = random_traces_hashes[bi[0]]
        test_b = random_traces_hashes[bi[1]]
        test_c = random_traces_hashes[bi[2]]
        abc = number_of_duplicates(test_a, test_b, test_c)

        ab = number_of_duplicates(test_a, test_b) - abc
        ac = number_of_duplicates(test_a, test_c) - abc
        bc = number_of_duplicates(test_b, test_c) - abc

        a = len(test_a) - abc - ab - ac
        b = len(test_b) - abc - ab - bc
        c = len(test_c) - abc - ac - bc

        plt.figure("{}th Best - RRS {}".format(len(best_indices) - i, RRS_number))
        v = venn3(subsets=(a,b,ab,c,ac,bc,abc))
        plt.title("{}th Best RRS Similarities".format(len(best_indices) - i))

    for i in range(len(best_indices)):

        wi = worst_indices[i]

        test_a = random_traces_hashes[wi[0]]
        test_b = random_traces_hashes[wi[1]]
        test_c = random_traces_hashes[wi[2]]
        abc = number_of_duplicates(test_a, test_b, test_c)

        ab = number_of_duplicates(test_a, test_b) - abc
        ac = number_of_duplicates(test_a, test_c) - abc
        bc = number_of_duplicates(test_b, test_c) - abc

        a = len(test_a) - abc - ab - ac
        b = len(test_b) - abc - ab - bc
        c = len(test_c) - abc - ac - bc

        plt.figure("{}th Worst - RRS {}".format(i + 1, RRS_number))
        v = venn3(subsets=(a,b,ab,c,ac,bc,abc))
        plt.title("{}th Worst RRS Similarities".format(i + 1))

plt.show()


# Without the funny metric

# Best number of duplicates: [197. 197. 197. 197. 198.]
# Best Indices:
# [[ 11, 87, 158]
#  [ 11, 87, 187]
#  [ 11, 87, 338]
#  [ 11, 158, 187]
#  [ 11, 87, 281]]


# Worst number of duplicates: [0. 0. 0. 0. 0.]
# Worst Indices:
# [[ 0, 1, 61]
#  [ 0, 1, 56]
#  [ 0, 1, 41]
#  [ 0, 1, 30]
#  [ 0, 1, 12]]

# Adding funny metric

# Best number of duplicates: [594. 594. 594. 595. 596.]
# Best Indices:
# [[ 11  87 281]
#  [ 11  87 488]
#  [ 87 281 488]
#  [ 11 206 281]
#  [ 11 281 488]]


# Worst number of duplicates: [0. 0. 0. 0. 0.]
# Worst Indices:
# [[  0  12 144]
#  [  0  12 133]
#  [  0  12  96]
#  [  0  12  84]
#  [  0  12  61]]
