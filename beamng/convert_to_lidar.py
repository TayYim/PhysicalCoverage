import glob
import math
from datetime import datetime
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.geometry import Polygon, LineString, Point
import multiprocessing

def create_frame_plot(data, origin, orientation, title, fig_num):
    fig = plt.figure(fig_num)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.scatter(data[:,0], data[:,1], s=1)
    ax.quiver(origin[0], origin[1], orientation[0], orientation[1])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.title(title)
    plt.xlim([-175, 175])
    plt.ylim([-175, 175])
    return plt

def create_lidar_plot(data, title, x_range, y_range, fig_num):
    plt.figure(fig_num)
    plt.clf()
    plt.title(title)
    # Display the environment
    for i in range(len(data["polygons"])):
        # Get the polygon
        p = data["polygons"][i]
        x,y = p.exterior.xy
        # Get the color
        c = "g" if i == 0 else "r"
        # Plot
        plt.plot(x, y, color=c)
    # Display the reachset
    for i in range(len(data["r_set"])):
        # Get the polygon
        p = data["r_set"][i]
        x,y = p.xy
        # Get the color
        c = "r"
        # Plot
        plt.plot(x, y, color=c, alpha=0.5)
    # Display the reachset
    for i in range(len(data["final_r_set"])):
        # Get the polygon
        p = data["final_r_set"][i]
        x,y = p.xy
        # Get the color
        c = "g"
        # Plot
        plt.plot(x, y, color=c)
    # Set the size of the graph
    plt.xlim(x_range)
    plt.ylim(y_range)
    # Invert the y axis as negative is up and show ticks
    ax = plt.gca()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    return plt

def getStep(a, MinClip):
    return round(float(a) / MinClip) * MinClip

def vectorize_reachset(lines, accuracy=0.25):
    vector = []
    # For each line:
    for l in lines:
        l_len = l.length
        l_len = getStep(l_len, accuracy)
        l_len = round(l_len, 6)
        vector.append(l_len)
    return vector

def rotate(p, origin=(0, 0), angle=0):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def process_file(file_name, save_name, external_vehicle_count):

    steering_angle  = 90
    total_lines     = 45
    max_distance    = 60

    # Open the file and count vectors
    input_file = open(file_name, "r")
    print("Processing: " + file_name)
    print("Saving output to: " + str(save_name))
    print("-----------------------")
    
    # Open a text file to save output
    output_file = open(save_name, "w")
    output_file.write("Name: %s\n" % file_name)
    output_file.write("Date: %s/%s/%s\n" % (e.day, e.month, e.year))
    output_file.write("Time: %s:%s:%s\n" % (e.hour, e.minute, e.second))
    output_file.write("External Vehicles: %s\n" % external_vehicle_count)
    output_file.write("Reach set total lines: %d\n" % total_lines)
    output_file.write("Reach set steering angle: %d\n" % steering_angle)
    output_file.write("Reach set max distance: %d\n" % max_distance)
    output_file.write("------------------------------\n")

    frame_skip_counter =  0

    # Print the file
    for line in input_file:

        # Skip the first second of data (this needs to be atleast 1 which is the heading)
        frame_skip_counter += 1
        if frame_skip_counter < 5:
            continue
        
        # Remove unnecessary characters and split the data correctly
        data = line.split("],")
        time_step_position = data[0].split(",[")
        crash_vehicle_count = data[-1].split(",")
        data = data[1:]
        data.insert(0, time_step_position[0])
        data.insert(1, time_step_position[1])
        data = data[:-1]
        data.append(crash_vehicle_count[0])
        data.append(crash_vehicle_count[1])

        # Get the data
        current_data = {}
        current_data["position"]        = data[1]
        current_data["orientation"]     = data[2]
        current_data["velocity"]        = data[3]
        current_data["lidar"]           = data[4]
        current_data["crash"]           = data[5]
        current_data["veh_count"]       = data[6]
        current_data['origin']          = "[0, 0, 0]"
        current_data["ego_orientation"] = "[1, 0, 0]"

        # Clean the data
        for key in current_data:
            current_data[key] = current_data[key].replace('[', '') 
            current_data[key] = current_data[key].replace(']', '') 

        # Convert to numpy
        for key in current_data:
            current_data[key] = current_data[key].split(", ")
            current_data[key] = np.array(current_data[key], dtype=float)

        if current_data["veh_count"] != int(external_vehicle_count):
            print("Vehicle count does not match: " + str(current_data["veh_count"]) + " - " + external_vehicle_count)
            exit()

        # Get the lidar data into the right shape
        unique_entries = int(current_data["lidar"].shape[0] / 3)
        current_data["lidar"] = current_data["lidar"].reshape(unique_entries, -1)

        # Subtract the cars position from the data
        current_data["lidar"] = current_data["lidar"] - current_data["position"]

        # Select every 25th element
        # current_data["lidar"] = current_data["lidar"][0::25]
        # print("Analyzing " + str(current_data["lidar"].shape[0]) + " points")

        # Plot the data
        # plt = create_frame_plot(current_data["lidar"], current_data["origin"], current_data["orientation"], "World frame", 1)

        # Compute how much the world is rotated by
        deltaX = current_data["orientation"][0] - current_data['origin'][0]
        deltaY = current_data["orientation"][1] - current_data['origin'][1]
        rotation = -1 * math.atan2(deltaY, deltaX)

        # Rotate all points 
        points_xy   = current_data["lidar"][:,0:2]
        origin      = current_data["origin"][0:2]
        current_data["rotated_lidar"] = rotate(points_xy, origin, rotation)

        # Plot rotated points
        # plt = create_frame_plot(current_data["rotated_lidar"], current_data["origin"], current_data["ego_orientation"] , "Vehicle frame", 2)

        # Create the car as an object
        ego_position = [0, 0]
        s = 1
        ego_vehicle = Polygon([(ego_position[0]-(2*s), ego_position[1]-s),
                               (ego_position[0]+(2*s), ego_position[1]-s),
                               (ego_position[0]+(2*s), ego_position[1]+s),
                               (ego_position[0]-(2*s), ego_position[1]+s)])

        # Estimate the reach set
        ego_heading     = 0
        r_set = []
        # Convert steering angle to radians
        steering_angle_rad = math.radians(steering_angle)
        # Compute the intervals 
        intervals = (steering_angle_rad * 2) / float(total_lines - 1)
        # Create each line
        for i in range(total_lines):
            # Compute the angle of the beam
            theta = (-1 * steering_angle_rad) + (i * intervals) +  ego_heading
            # Compute the new point
            p2 = (ego_position[0] + (max_distance * math.cos(theta)), ego_position[1] + (max_distance * math.sin(theta)))
            # Save the linestring
            l = LineString([ego_position, p2])
            r_set.append(l)

        # Create a list of all polygons for plotting
        polygons = [ego_vehicle]

        # Turn all readings into small polygons
        for p in current_data["rotated_lidar"]:
            s = 0.2
            new_point = Polygon([(p[0]-s, p[1]-s),
                                 (p[0]+s, p[1]-s),
                                 (p[0]+s, p[1]+s),
                                 (p[0]-s, p[1]+s)])
            polygons.append(new_point)

        # Check if any of the reach set intersects with the points
        final_r_set = []
        # For each line
        for l in r_set:
            # Get the origin
            origin = l.coords[0]
            end_position = l.coords[1]
            min_distance = Point(origin).distance(Point(end_position))
            min_point = end_position
            # For each polygon (except the ego vehicle)
            for p in polygons[1:]:
                # Check if they intersect and if so where
                intersect = l.intersection(p)
                if not intersect.is_empty:
                    for i in intersect.coords:
                        # Check which distance is the closest
                        dis = Point(origin).distance(Point(i))
                        if dis < min_distance:
                            min_distance = dis
                            min_point = i
                                
            # Update the line
            true_l = LineString([origin, min_point])
            final_r_set.append(true_l)

        environment_data = {}
        environment_data["polygons"]    = polygons
        environment_data["r_set"]       = r_set
        environment_data["final_r_set"] = final_r_set

        # if plot:
        #     plt.figure(1)
        #     plt.clf()
        #     plt.title('Environment')

        #     # Invert the y axis for easier viewing
        #     plt.gca().invert_yaxis()

        #     # Display the environment
        #     for i in range(len(environment_data["polygons"])):
        #         # Get the polygon
        #         p = environment_data["polygons"][i]
        #         x,y = p.exterior.xy
        #         # Get the color
        #         c = "g" if i == 0 else "r"
        #         # Plot
        #         plt.plot(x, y, color=c)

        #     # Display the reachset
        #     for i in range(len(environment_data["r_set"])):
        #         # Get the polygon
        #         p = environment_data["r_set"][i]
        #         x,y = p.xy
        #         # Get the color
        #         c = "r"
        #         # Plot
        #         plt.plot(x, y, color=c, alpha=0.5)

        #     # Display the reachset
        #     for i in range(len(environment_data["final_r_set"])):
        #         # Get the polygon
        #         p = environment_data["final_r_set"][i]
        #         x,y = p.xy
        #         # Get the color
        #         c = "g"
        #         # Plot
        #         plt.plot(x, y, color=c)

        #     # Set the size of the graph
        #     plt.xlim([-30, 100])
        #     plt.ylim([-40, 40])

        #     # Invert the y axis as negative is up and show ticks
        #     ax = plt.gca()
        #     ax.set_ylim(ax.get_ylim()[::-1])
        #     ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        #     # plot the graph
        #     plt.pause(0.1)
        #     plt.savefig('./output/file' + str(file_number) + 'frame' + str(frame_skip_counter) + '.png')

        # Plot the environment figures
        # plt = create_lidar_plot(environment_data, "Environment Zoomed", [-15, 45], [-30, 30], 3)
        # plt = create_lidar_plot(environment_data, "Environment", [-100, 100], [-100, 100], 4)

        # Compute the vectorized reach set
        r_vector = vectorize_reachset(environment_data["final_r_set"], accuracy=0.001)
        output_file.write("Vector: " + str(r_vector) + "\n")
        output_file.write("Crash: " + str(bool(current_data["crash"])) + "\n")
        output_file.write("\n")

        # If we crashed end the trace
        if bool(current_data["crash"]):
            break

    # Close both files
    output_file.close()
    input_file.close()
    
# Create a function called "chunks" with two arguments, l and n:
def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]


raw_file_location       = "../../PhysicalCoverageData/beamng/raw/"
output_file_location    = "../../PhysicalCoverageData/beamng/processed/"
file_names = glob.glob(raw_file_location + "/*/*.csv")

total_cores = 30

# Create file names with lists of total_core length
data_to_process = list(chunks(file_names, total_cores))

for chunk_number in tqdm(range(len(data_to_process))):

    file_list = data_to_process[chunk_number]

    manager = multiprocessing.Manager()
    jobs = []

    for file_name in file_list:

        # Compute the file name in the format vehiclecount-time-run#.txt
        name_only = file_name[file_name.rfind('/')+1:]
        folder = file_name[0:file_name.rfind('/')]
        folder = folder[folder.rfind('/')+1:]
        external_vehicle_count = folder[0: folder.find('_')]
        name_only = name_only[name_only.rfind('_')+1:]
        e = datetime.now()
        save_name = ""
        save_name += str(output_file_location)
        save_name += external_vehicle_count + "-"
        save_name += str(int(e.timestamp())) +"-"
        save_name += name_only[0:-4] + ".txt"

        # Run each of the files in a seperate process
        p = multiprocessing.Process(target=process_file, args=(file_name, save_name, external_vehicle_count))
        jobs.append(p)
        p.start()     

    # For each of the currently running jobs
    for j in jobs:
        # Wait for them to finish
        j.join()

print("Complete")
