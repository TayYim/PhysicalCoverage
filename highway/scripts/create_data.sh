#!/bin/bash

# 12 different vehicle counts
vehicle_count=(2 4 6 8 10 12 14 16 18 20)

# Run it 
for value in {1..1000}
do
    for tot_vehicle in "${vehicle_count[@]}"
    do
        # Get the current time
        current_date=`date +%s`

        # Generate a random string to append to the front
        chars=abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ
        rand_string=""
        for i in {1..4} ; do
            rand_string="$rand_string${chars:RANDOM%${#chars}:1}"
        done
        
        # Create the save name
        save_name="$tot_vehicle-$current_date-$rand_string.txt"

        # Run the script
        python3 main.py --no_plot --environment_vehicles $tot_vehicle --save_name $save_name 
    done
done