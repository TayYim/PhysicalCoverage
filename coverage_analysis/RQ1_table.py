import ast
import sys
import glob
import copy
import argparse
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

from random import sample

from tqdm import tqdm
from collections import Counter
from prettytable import PrettyTable

from general_functions import order_by_beam
from general_functions import get_beam_numbers
from general_functions import get_ignored_code_coverage_lines

from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/coverage_analysis")])
sys.path.append(base_directory)
from general.branch_converter import BranchConverter

def compute_RSR_details():
    global traces
    global crashes

    # Get the total number of tests
    total_number_of_tests = traces.shape[0]

    # Use multiprocessing to compute the trace signature and if a crash was detected
    total_processors    = int(args.cores)
    pool                =  multiprocessing.Pool(processes=total_processors)

    # Call our function on each test in the trace
    jobs = []
    for random_test_index in range(total_number_of_tests):
        jobs.append(pool.apply_async(compute_trace_signature_and_crash, args=([random_test_index])))

    # Get the results
    results = []
    for job in tqdm(jobs):
        results.append(job.get())

    # Close the pool
    pool.close()

    # Turn all the signatures into a list
    all_signatures = np.zeros(total_number_of_tests)
    all_crash_detections = np.zeros(total_number_of_tests)

    # Go through each results
    for i in range(total_number_of_tests):

        # Get the result (r)
        r = results[i]

        # Get the signature and the results
        signature, crash_detected = r

        # Collect all the signatures
        all_signatures[i] = signature
        all_crash_detections[i] = crash_detected

    # Print out the number of unique signatures
    count_of_signatures = Counter(all_signatures)

    # Get the signatures and the count
    final_signatures, count_of_signatures = zip(*count_of_signatures.items())
    count_of_signatures = np.array(count_of_signatures)

    # Determine how many classes have more than 1 test
    total_multiclasses = np.sum(count_of_signatures >= 2)
    consistent_class = np.zeros(total_multiclasses, dtype=bool)

    # Loop through each of the final signatures
    count_index = 0
    for i in range(len(final_signatures)):
        # Get the signature and count
        current_sig = final_signatures[i]
        current_count = count_of_signatures[i]

        if current_count <= 1:
            continue

        # Loop through the signatures and get the indices where this signature is in the array
        interested_indices = np.argwhere(all_signatures == current_sig).reshape(-1)
        assert(len(interested_indices) == current_count)

        # Get all the crash data for a specific signature
        single_class_crash_data = all_crash_detections[interested_indices]

        # Check if all the data is consisten
        consistent = np.all(single_class_crash_data == single_class_crash_data[0])
        consistent_class[count_index] = bool(consistent)
        count_index += 1

    # Final signatures holds the list of all signatures
    # Count of signatures holds the list intergers representing how many times each signature was seen

    # Get the total signatures
    total_signatures_count = len(final_signatures)

    # Get the total number of single test and multitest signatures
    single_test_signatures_count = len(count_of_signatures[np.argwhere(count_of_signatures == 1).reshape(-1)])
    multi_test_signatures_count = len(count_of_signatures[np.argwhere(count_of_signatures > 1).reshape(-1)])

    # Get the total number of consistent vs inconsistent classes
    consistent_class_count      = np.count_nonzero(consistent_class)
    inconsistent_class_count    = np.size(consistent_class) - np.count_nonzero(consistent_class)

    # Compute the percentage of consistency
    if np.size(consistent_class) <= 0:
        percentage_of_inconsistency = 0
    else:
        percentage_of_inconsistency = int(np.round((inconsistent_class_count / np.size(consistent_class)) * 100, 0))

    # Make sure that there is no count where the count is < 1: Make sure that single + multi == total
    assert(len(count_of_signatures[np.argwhere(count_of_signatures < 1).reshape(-1)]) == 0)
    assert(single_test_signatures_count + multi_test_signatures_count == total_signatures_count)

    return [total_signatures_count, single_test_signatures_count, multi_test_signatures_count, consistent_class_count, inconsistent_class_count, percentage_of_inconsistency]

def compute_line_coverage_details():
    global code_coverage_file_names

    # Get the total number of tests
    total_number_of_tests = len(code_coverage_file_names)

    # Use multiprocessing to compute the trace signature and if a crash was detected
    total_processors    = int(args.cores)
    pool                = multiprocessing.Pool(processes=total_processors)

    # Call our function on each file
    jobs = []
    for i in range(total_number_of_tests):
        jobs.append(pool.apply_async(compute_line_coverage_hash, args=([i])))

    # Get the results
    results = []
    for job in tqdm(jobs):
        results.append(job.get())

    # Close the pool
    pool.close()

    # Turn all the signatures into a list
    all_signatures = np.zeros(total_number_of_tests)
    all_crash_detections = np.zeros(total_number_of_tests)

    # Go through each results
    for i in range(total_number_of_tests):

        # Get the result (r)
        r = results[i]

        # Get the signature and the results
        signature, crash_detected = r

        # Collect all the signatures
        all_signatures[i] = signature
        all_crash_detections[i] = crash_detected

    # Print out the number of unique signatures
    count_of_signatures = Counter(all_signatures)

    # Get the signatures and the count
    final_signatures, count_of_signatures = zip(*count_of_signatures.items())
    count_of_signatures = np.array(count_of_signatures)

    # Determine how many classes have more than 1 test
    total_multiclasses = np.sum(count_of_signatures >= 2)
    consistent_class = np.zeros(total_multiclasses, dtype=bool)

    # Loop through each of the final signatures
    count_index = 0
    for i in range(len(final_signatures)):
        # Get the signature and count
        current_sig = final_signatures[i]
        current_count = count_of_signatures[i]

        if current_count <= 1:
            continue

        # Loop through the signatures and get the indices where this signature is in the array
        interested_indices = np.argwhere(all_signatures == current_sig).reshape(-1)
        assert(len(interested_indices) == current_count)

        # Get all the crash data for a specific signature
        single_class_crash_data = all_crash_detections[interested_indices]

        # Check if all the data is consisten
        consistent = np.all(single_class_crash_data == single_class_crash_data[0])
        consistent_class[count_index] = bool(consistent)
        count_index += 1

    # Final signatures holds the list of all signatures
    # Count of signatures holds the list intergers representing how many times each signature was seen

    # Get the total signatures
    total_signatures_count = len(final_signatures)

    # Get the total number of single test and multitest signatures
    single_test_signatures_count = len(count_of_signatures[np.argwhere(count_of_signatures == 1).reshape(-1)])
    multi_test_signatures_count = len(count_of_signatures[np.argwhere(count_of_signatures > 1).reshape(-1)])

    # Get the total number of consistent vs inconsistent classes
    consistent_class_count      = np.count_nonzero(consistent_class)
    inconsistent_class_count    = np.size(consistent_class) - np.count_nonzero(consistent_class)

    # Compute the percentage of consistency
    percentage_of_inconsistency = int(np.round((inconsistent_class_count / np.size(consistent_class)) * 100, 0))

    # Make sure that there is no count where the count is < 1: Make sure that single + multi == total
    assert(len(count_of_signatures[np.argwhere(count_of_signatures < 1).reshape(-1)]) == 0)
    assert(single_test_signatures_count + multi_test_signatures_count == total_signatures_count)

    return [total_signatures_count, single_test_signatures_count, multi_test_signatures_count, consistent_class_count, inconsistent_class_count, percentage_of_inconsistency]  

def compute_branch_coverage_details(scenario):

    if scenario != "highway":
        return [0,0,0,0,0,0]

    global code_coverage_file_names

    # Get the total number of tests
    total_number_of_tests = len(code_coverage_file_names)

    # Use multiprocessing to compute the trace signature and if a crash was detected
    total_processors    = int(args.cores)
    pool                = multiprocessing.Pool(processes=total_processors)

    # Call our function on each file
    jobs = []
    for i in range(total_number_of_tests):
        jobs.append(pool.apply_async(compute_branch_coverage_hash, args=([i, scenario])))

    # Get the results
    results = []
    for job in tqdm(jobs):
        results.append(job.get())

    # Close the pool
    pool.close()

    # Turn all the signatures into a list
    all_signatures = np.zeros(total_number_of_tests)
    all_crash_detections = np.zeros(total_number_of_tests)

    # Go through each results
    for i in range(total_number_of_tests):

        # Get the result (r)
        r = results[i]

        # Get the signature and the results
        signature, crash_detected = r

        # Collect all the signatures
        all_signatures[i] = signature
        all_crash_detections[i] = crash_detected

    # Print out the number of unique signatures
    count_of_signatures = Counter(all_signatures)

    # Get the signatures and the count
    final_signatures, count_of_signatures = zip(*count_of_signatures.items())
    count_of_signatures = np.array(count_of_signatures)

    # Determine how many classes have more than 1 test
    total_multiclasses = np.sum(count_of_signatures >= 2)
    consistent_class = np.zeros(total_multiclasses, dtype=bool)

    # Loop through each of the final signatures
    count_index = 0
    for i in range(len(final_signatures)):
        # Get the signature and count
        current_sig = final_signatures[i]
        current_count = count_of_signatures[i]

        if current_count <= 1:
            continue

        # Loop through the signatures and get the indices where this signature is in the array
        interested_indices = np.argwhere(all_signatures == current_sig).reshape(-1)
        assert(len(interested_indices) == current_count)

        # Get all the crash data for a specific signature
        single_class_crash_data = all_crash_detections[interested_indices]

        # Check if all the data is consisten
        consistent = np.all(single_class_crash_data == single_class_crash_data[0])
        consistent_class[count_index] = bool(consistent)
        count_index += 1

    # Final signatures holds the list of all signatures
    # Count of signatures holds the list intergers representing how many times each signature was seen

    # Get the total signatures
    total_signatures_count = len(final_signatures)

    # Get the total number of single test and multitest signatures
    single_test_signatures_count = len(count_of_signatures[np.argwhere(count_of_signatures == 1).reshape(-1)])
    multi_test_signatures_count = len(count_of_signatures[np.argwhere(count_of_signatures > 1).reshape(-1)])

    # Get the total number of consistent vs inconsistent classes
    consistent_class_count      = np.count_nonzero(consistent_class)
    inconsistent_class_count    = np.size(consistent_class) - np.count_nonzero(consistent_class)

    # Compute the percentage of consistency
    percentage_of_inconsistency = int(np.round((inconsistent_class_count / np.size(consistent_class)) * 100, 0))

    # Make sure that there is no count where the count is < 1: Make sure that single + multi == total
    assert(len(count_of_signatures[np.argwhere(count_of_signatures < 1).reshape(-1)]) == 0)
    assert(single_test_signatures_count + multi_test_signatures_count == total_signatures_count)

    return [total_signatures_count, single_test_signatures_count, multi_test_signatures_count, consistent_class_count, inconsistent_class_count, percentage_of_inconsistency]

def compute_trace_signature_and_crash(index):
    global traces
    global crashes

    # Get the trace and crash data
    trace = traces[index]
    crash = crashes[index]

    # Init the trace signature and crash detected variable 
    trace_signature  = set()
    crash_detected   = False

    # The signature for the trace is the set of all RSR signatures
    for sig in trace:
        trace_signature.add(tuple(sig))

    # Create the hash of the signature for each comparison
    trace_hash = hash(tuple(sorted(trace_signature)))

    # Check if this trace had a crash
    crash_detected = not np.isinf(crash).all()

    return [trace_hash, crash_detected]

def compute_line_coverage_hash(index):
    global code_coverage_file_names
    global ignored_lines

    coverage_hash = 0
    number_of_crashes = 0

    # Get the code coverage file
    code_coverage_file = code_coverage_file_names[index]
    f = open(code_coverage_file, "r")

    all_lines_coverage = set()

    # Read the file
    for line in f: 
        if "Lines covered:" in line:
            covered_l = ast.literal_eval(line[15:])

        if "Total lines covered:" in line:
            total_covered_l = int(line[21:])
            assert(len(covered_l) == total_covered_l)

        if "Total physical crashes: " in line:
            number_of_crashes = int(line[24:])

    # Close the file
    f.close()

    all_lines_coverage = set(covered_l) - ignored_lines

    # Get the coverage hash
    coverage_hash = hash(tuple(sorted(list(all_lines_coverage))))

    return [coverage_hash, number_of_crashes]

def compute_branch_coverage_hash(index, scenario):
    global code_coverage_file_names
    global ignored_lines

    coverage_hash = 0
    number_of_crashes = 0

    # Get the code coverage file
    code_coverage_file = code_coverage_file_names[index]
    f = open(code_coverage_file, "r")

    all_lines_coverage = set()

    # Read the file
    for line in f: 
        if "Lines covered:" in line:
            covered_l = ast.literal_eval(line[15:])

        if "Total lines covered:" in line:
            total_covered_l = int(line[21:])
            assert(len(covered_l) == total_covered_l)

        if "Total physical crashes: " in line:
            number_of_crashes = int(line[24:])

    # Close the file
    f.close()

    # Compute the branch coverage
    bc = BranchConverter(scenario)
    branch_coverage = bc.compute_branch_coverage(covered_l)

    # Get the coverage hash
    coverage_hash = hash(tuple(branch_coverage))

    return [coverage_hash, number_of_crashes]

parser = argparse.ArgumentParser()
parser.add_argument('--total_samples',  type=int, default=-1,   help="-1 all samples, otherwise randomly selected x samples")
parser.add_argument('--scenario',       type=str, default="",   help="beamng/highway")
parser.add_argument('--cores',          type=int, default=4,    help="number of available cores")
args = parser.parse_args()

print("----------------------------------")
print("-----------Loading Data-----------")
print("----------------------------------")

load_name = "*.npy"

# Get the file names
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/random_tests/physical_coverage/processed/' + str(args.total_samples) + "/"
trace_file_names = glob.glob(base_path + "traces_*.npy")
crash_file_names = glob.glob(base_path + "crash_*.npy")

# Get the code coverage
base_path = '../../PhysicalCoverageData/' + str(args.scenario) +'/random_tests/code_coverage/raw/'
global code_coverage_file_names
code_coverage_file_names = glob.glob(base_path + "*/*.txt")

# Make sure we have enough samples
assert(len(trace_file_names) >= 1)
assert(len(crash_file_names) >= 1)
assert(len(code_coverage_file_names) >= 1)

# Select args.total_samples total code coverage files
assert(len(code_coverage_file_names) == args.total_samples)

global ignored_lines
ignored_lines = get_ignored_code_coverage_lines(args.scenario)

# Get the beam numbers
trace_beam_numbers = get_beam_numbers(trace_file_names)
crash_beam_numbers = get_beam_numbers(crash_file_names)

# Find the set of beam numbers which all sets of files have
beam_numbers = list(set(trace_beam_numbers) | set(crash_beam_numbers))
beam_numbers = sorted(beam_numbers)

# Sort the data based on the beam number
trace_file_names = order_by_beam(trace_file_names, beam_numbers)
crash_file_names = order_by_beam(crash_file_names, beam_numbers)

# Create the output table
t = PrettyTable()
t.field_names = ["Coverage Type", "Total Signatures" , "Single Test Signatures", "Multitest Signatures", "Consistent Multitest Signatures", "Inconsistent Multitest Signatures", "Percentage Inconsistent"]

# Compute the line coverage details
print("Processing Line Coverage")
results                         = compute_line_coverage_details()
total_signatures_count          = results[0]
single_test_signatures_count    = results[1]
multi_test_signatures_count     = results[2]
consistent_class_count          = results[3]
inconsistent_class_count        = results[4]
percentage_of_inconsistency     = results[5]
print("Total signatures: {}".format(total_signatures_count))
print("Total single test signatures: {}".format(single_test_signatures_count))
print("Total multi test signatures: {}".format(multi_test_signatures_count))
print("Total consistent classes: {}".format(consistent_class_count))
print("Total inconsistent classes: {}".format(inconsistent_class_count))
print("Percentage of inconsistent classes: {}%".format(percentage_of_inconsistency))
t.add_row(["Line Coverage", total_signatures_count, single_test_signatures_count, multi_test_signatures_count, consistent_class_count, inconsistent_class_count, "{}%".format(percentage_of_inconsistency)])

# Compute the branch coverage details
print("Processing Branch Coverage")
results                         = compute_branch_coverage_details(args.scenario)
total_signatures_count          = results[0]
single_test_signatures_count    = results[1]
multi_test_signatures_count     = results[2]
consistent_class_count          = results[3]
inconsistent_class_count        = results[4]
percentage_of_inconsistency     = results[5]
print("Total signatures: {}".format(total_signatures_count))
print("Total single test signatures: {}".format(single_test_signatures_count))
print("Total multi test signatures: {}".format(multi_test_signatures_count))
print("Total consistent classes: {}".format(consistent_class_count))
print("Total inconsistent classes: {}".format(inconsistent_class_count))
print("Percentage of inconsistent classes: {}%".format(percentage_of_inconsistency))
t.add_row(["Branch Coverage", total_signatures_count, single_test_signatures_count, multi_test_signatures_count, consistent_class_count, inconsistent_class_count, "{}%".format(percentage_of_inconsistency)])


# Loop through each of the files and compute both an RSR signature as well as determine if there was a crash
for beam_number in beam_numbers:
    print("Processing RSR{}".format(beam_number))
    key = "RSR{}".format(beam_number)

    # Get the trace and crash files
    global traces
    traces  = np.load(trace_file_names[beam_number-1])
    global crashes
    crashes = np.load(crash_file_names[beam_number-1])

    # Compute the different metrics
    results                         = compute_RSR_details()
    total_signatures_count          = results[0]
    single_test_signatures_count    = results[1]
    multi_test_signatures_count     = results[2]
    consistent_class_count          = results[3]
    inconsistent_class_count        = results[4]
    percentage_of_inconsistency     = results[5]
    print("Total signatures: {}".format(total_signatures_count))
    print("Total single test signatures: {}".format(single_test_signatures_count))
    print("Total multi test signatures: {}".format(multi_test_signatures_count))
    print("Total consistent classes: {}".format(consistent_class_count))
    print("Total inconsistent classes: {}".format(inconsistent_class_count))
    print("Percentage of inconsistent classes: {}%".format(percentage_of_inconsistency))
    t.add_row([key, total_signatures_count, single_test_signatures_count, multi_test_signatures_count, consistent_class_count, inconsistent_class_count, "{}%".format(percentage_of_inconsistency)])

# Display the table
print(t)


