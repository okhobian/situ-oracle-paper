#!/usr/bin/env python3

import psutil
import time
import csv
import signal
import sys

# Define the network interface to measure
interface_name = 'eth0'

# Function to handle the interrupt signal
def signal_handler(sig, frame):
    print('You pressed Ctrl+C! Saving the file.')
    file.close()  # Close the file to ensure it's saved
    sys.exit(0)   # Exit the program gracefully

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)


# Open the output file for writing
with open('system_metrics.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header row to the CSV file
    writer.writerow(['Timestamp', 'CPU Usage (%)', 'Memory Usage (%)', 'Disk Usage (%)', 'Network Sent (MB)', 'Network Received (MB)'])

    # try:

    #     while True:
    #          # Get the current time
    #         timestamp = int(time.time())

    #         # Get the system CPU usage as a percentage
    #         cpu_usage = psutil.cpu_percent()

    #         # Get the system memory usage as a percentage
    #         memory_usage = psutil.virtual_memory().percent

    #         # Get the system disk usage as a percentage
    #         disk_usage = psutil.disk_usage('/').percent

    #         # Get the system network usage
    #         net_io_counters = psutil.net_io_counters(pernic=True)[interface_name]
    #         net_sent_mb = round(net_io_counters.bytes_sent / (1024 * 1024), 2)
    #         net_recv_mb = round(net_io_counters.bytes_recv / (1024 * 1024), 2)

    #         # Write the system metrics to the CSV file
    #         writer.writerow([timestamp, cpu_usage, memory_usage, disk_usage, net_sent_mb, net_recv_mb])

    #         # Wait for one second before sampling again
    #         time.sleep(1)

    # except KeyboardInterrupt:
    #     # Handle any cleanup here if necessary
    #     pass

    # Sample the system metrics every second
    i = 0
    while i<7500:
        # Get the current time
        timestamp = int(time.time())

        # Get the system CPU usage as a percentage
        cpu_usage = psutil.cpu_percent()

        # Get the system memory usage as a percentage
        memory_usage = psutil.virtual_memory().percent

        # Get the system disk usage as a percentage
        disk_usage = psutil.disk_usage('/').percent

        # Get the system network usage
        net_io_counters = psutil.net_io_counters(pernic=True)[interface_name]
        net_sent_mb = round(net_io_counters.bytes_sent / (1024 * 1024), 2)
        net_recv_mb = round(net_io_counters.bytes_recv / (1024 * 1024), 2)

        # Write the system metrics to the CSV file
        writer.writerow([timestamp, cpu_usage, memory_usage, disk_usage, net_sent_mb, net_recv_mb])

        # Wait for one second before sampling again
        time.sleep(1)
        i+=1
