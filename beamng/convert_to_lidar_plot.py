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
import scipy

steering_angle  = 33
total_lines     = 1
max_distance    = 45

def create_frame_plot(data, origin, orientation, title, fig_num):
    fig = plt.figure(fig_num)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.scatter(data[:,0], data[:,1], s=5)
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

def combine_environment_and_reachset(r_set, polygons):
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
    return final_r_set

def estimate_reachset(ego_position, steering_angle, total_lines):
    # Estimate the reach set
    ego_heading = 0
    r_set = []
    # Convert steering angle to radians
    steering_angle_rad = math.radians(steering_angle)
    # Compute the intervals 
    intervals = 0
    if total_lines > 1:
        intervals = (steering_angle_rad * 2) / float(total_lines - 1)
    # Create each line
    for i in range(total_lines):
        # Compute the angle of the beam
        if total_lines > 1:
            theta = (-1 * steering_angle_rad) + (i * intervals) +  ego_heading
        else:
            theta = ego_heading
        # Compute the new point
        p2 = (ego_position[0] + (max_distance * math.cos(theta)), ego_position[1] + (max_distance * math.sin(theta)))
        # Save the linestring
        l = LineString([ego_position, p2])
        r_set.append(l)
    return r_set

def estimate_obstacles(ego_vehicle, current_lidar_data):
    # Create a list of all polygons for plotting
    polygons = [ego_vehicle]

    # Turn all readings into small polygons
    for pi in range(len(current_lidar_data)):
        # Get the point
        p = current_lidar_data[pi]
        # Check if the point is inside any of the polygons
        already_plotted = False
        for pj in range(len(polygons)):
            # Make sure you are not comparing the same point
            if pj != pi:
                # If the point is inside this polygon
                if polygons[pj].contains(Point(p)):
                    already_plotted = True
                    break
        # Only create the point if its not been created before
        if not already_plotted:
            s = 0.2
            new_point = Polygon([(p[0]-s, p[1]-s),
                                (p[0]+s, p[1]-s),
                                (p[0]+s, p[1]+s),
                                (p[0]-s, p[1]+s)])
            polygons.append(new_point)
    return polygons

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

def process_file(file_name, save_name, external_vehicle_count, file_number):

    # Open the file and count vectors
    input_file = open(file_name, "r")
    print("(" + str(file_number) + ") Processing: " + file_name)
    print("Saving output to: " + str(save_name))
    print("-----------------------")
    
    # Get the time
    start_time = datetime.now()

    # Open a text file to save output
    output_file = open(save_name, "w")
    output_file.write("Name: %s\n" % file_name)
    output_file.write("Date: %s/%s/%s\n" % (start_time.day, start_time.month, start_time.year))
    output_file.write("Time: %s:%s:%s\n" % (start_time.hour, start_time.minute, start_time.second))
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
        # current_data["lidar"] = current_data["lidar"][0::500]
        print("Analyzing " + str(current_data["lidar"].shape[0]) + " points")

        # Plot the data
        plt = create_frame_plot(current_data["lidar"], current_data["origin"], current_data["orientation"], "World frame", 1)

        # Compute how much the world is rotated by
        deltaX = current_data["orientation"][0] - current_data['origin'][0]
        deltaY = current_data["orientation"][1] - current_data['origin'][1]
        rotation = -1 * math.atan2(deltaY, deltaX)

        # Rotate all points 
        points_xy   = current_data["lidar"][:,0:2]
        origin      = current_data["origin"][0:2]
        current_data["rotated_lidar"] = rotate(points_xy, origin, rotation)

        # Plot rotated points
        plt = create_frame_plot(current_data["rotated_lidar"], current_data["origin"], current_data["ego_orientation"] , "Vehicle frame", 2)

        # Time our technique
        start_time = datetime.now()

        # Create the car as an object
        ego_position = [0, 0]
        s = 1
        ego_vehicle = Polygon([(ego_position[0]-(2*s), ego_position[1]-s),
                               (ego_position[0]+(2*s), ego_position[1]-s),
                               (ego_position[0]+(2*s), ego_position[1]+s),
                               (ego_position[0]-(2*s), ego_position[1]+s)])

        # Estimate the reachset
        r_set = estimate_reachset(ego_position, steering_angle, total_lines)
        # Create a list of all polygons for plotting
        polygons = estimate_obstacles(ego_vehicle, current_data["rotated_lidar"])
        # Get the final reach set
        final_r_set = combine_environment_and_reachset(r_set, polygons)
        # Vectorize the reach set and round it
        r_vector = vectorize_reachset(final_r_set, accuracy=0.001)
        
        # End out time
        end_time = datetime.now()

        # Save the data
        environment_data = {}
        environment_data["polygons"]    = polygons
        environment_data["r_set"]       = r_set
        environment_data["final_r_set"] = final_r_set
        
        plt.figure(1)
        plt.clf()
        plt.title('Environment - ' + str(len(polygons)) + " lidar points")

        # Invert the y axis for easier viewing
        plt.gca().invert_yaxis()

        # Display the environment
        for i in range(len(environment_data["polygons"])):
            # Get the polygon
            p = environment_data["polygons"][i]
            x,y = p.exterior.xy
            # Get the color
            c = "g" if i == 0 else "r"
            # Plot
            plt.plot(x, y, color=c)

        # Display the reachset
        for i in range(len(environment_data["r_set"])):
            # Get the polygon
            p = environment_data["r_set"][i]
            x,y = p.xy
            # Get the color
            c = "r"
            # Plot
            plt.plot(x, y, color=c, alpha=0.5)

        # Display the reachset
        for i in range(len(environment_data["final_r_set"])):
            # Get the polygon
            p = environment_data["final_r_set"][i]
            x,y = p.xy
            # Get the color
            c = "g"
            # Plot
            plt.plot(x, y, color=c)

        # Set the size of the graph
        plt.xlim([-30, 100])
        plt.ylim([-40, 40])

        # Invert the y axis as negative is up and show ticks
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        # plot the graph
        plt.pause(0.1)
        # plt.savefig('./output/file' + str(file_number) + 'frame' + str(frame_skip_counter) + '.png')

        # Plot the environment figures
        plt = create_lidar_plot(environment_data, "Environment Zoomed", [-15, 45], [-30, 30], 3)
        plt = create_lidar_plot(environment_data, "Environment", [-100, 100], [-100, 100], 4)

        # Compute the vectorized reach set
        output_file.write("Vector: " + str(r_vector) + "\n")
        output_file.write("Crash: " + str(bool(current_data["crash"])) + "\n")
        elapsed_time = (end_time - start_time).total_seconds()
        output_file.write("Time: " + str(elapsed_time) +"\n")
        output_file.write("\n")

        # If we crashed end the trace
        if bool(current_data["crash"]):
            break

    # Close both files
    output_file.close()
    input_file.close()
    return True

raw_file_location       = "../../PhysicalCoverageData/beamng/raw/"
output_file_location    = "../../PhysicalCoverageData/beamng/processed_test/"
file_names = glob.glob(raw_file_location + "/*/*.csv")

# Create a pool with x processes
file_number = 0

# for file_name in file_names:

file_name = file_names[-1]

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
# result_object.append(pool.apply_async(process_file, args=(file_name, save_name, external_vehicle_count, file_number)))
process_file(file_name, save_name, external_vehicle_count, file_number)

print("All files completed")