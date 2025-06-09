import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import argparse

def plot_events_3d(events_file, output_name='3d_plot', output_folder=None):
    # Load events from file
    events = []
    with open(events_file, 'r') as f:
        # Skip the header lines
        for _ in range(6):
            next(f)
        for line in f:
            event = tuple(map(float, line.strip().split()))
            # if event[0]<= 3.183333500 and event[0] > 3.033333492: # data000_set001_25_rotation
            # if event[0]<= 1.350000071 and event[0] > 1.200000063: # data000_set004_151_stable
            # if event[0]<= 1.183333396 and event[0] > 1.033333388: # data000_set001_074_stable
            # if event[0]<= 5.850000306 and event[0] > 5.700000298: # data000_set001_092_stable
            # if event[0]<= 0.516666694 and event[0] > 0.366666686.: # data000_set005_02_rotation
            # if event[0]<= 1.016666720 and event[0] > 0.866666713: # data000_set005_02_stable
            # if event[0]<= 6.183333656 and event[0] > 6.033333649: # data000_set005_15_stable
            if event[0]<= 5.350000280 and event[0] > 5.200000272: # data000_set005_10_stable
                events.append(event)

    first_event_ts = events[0][0]
    # Extract x, y, and z coordinates from the data
    x = [event[1] for event in events]
    y = [event[2]for event in events]
    z = [(event[0] - first_event_ts)  for event in events]
    polarities = [int(event[3]) for event in events]

    # Create a 3D figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the events with color based on polarity
    ax.scatter(x, z, y, c=['blue' if p == 1 else 'red' for p in polarities], marker='o',s=0.1)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_zlabel('Y')
    ax.set_ylabel('Time')
    ax.set_title('Events over Time and Space')

    # ax.invert_xaxis()
    ax.invert_zaxis()

    # Customize the view
    # ax.view_init(elev=30, azim=-45)
    # ax.view_init(elev=30, azim=-60)
    ax.view_init(elev=0, azim=-90)

    # Determine output path
    if output_folder is None:
        output_folder = os.path.dirname(events_file)
    # output_path = os.path.join(output_folder, f"{output_name}.pdf")

    # Save the plot as a PDF file
    plt.savefig(f"{output_folder}/plot_left.pdf", format='pdf')
    print(f"Plot saved to: {output_folder}/plot_left.pdf")
    
    ax.view_init(elev=30, azim=-45)
    plt.savefig(f"{output_folder}/plot_mid.pdf", format='pdf')
    
    ax.view_init(elev=0, azim=0)
    plt.savefig(f"{output_folder}/plot_right.pdf", format='pdf')
    
    # # Save the plot file
    # fig.savefig(output_path)
    # print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot events over time and space")
    parser.add_argument("events_file", help="Path to the file with event data")
    parser.add_argument("--output-name", default="3d_plot", help="Name of the output file (without .pdf extension)")
    parser.add_argument("--output-folder", help="Output folder for the plot (default is the same as the input file)")
    args = parser.parse_args()

    plot_events_3d(args.events_file, args.output_name, args.output_folder)