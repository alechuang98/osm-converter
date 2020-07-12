import argparse
import json
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def crearte_cmd_parser():
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument("--input")
    cmd_parser.add_argument("--output")
    return cmd_parser

def main():
    cmd_parser = crearte_cmd_parser()
    args = cmd_parser.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)
    segments = data["segments"]
    arcs = data["arcs"]
    
    fig, ax = plt.subplots()

    for segment in segments:
        seg = np.array(segment)
        ax.plot(seg[:, 0], seg[:, 1], 'bo-', lw=1)
    for arc in arcs:
        r = math.sqrt((arc[0][0] - arc[1][0]) ** 2 + (arc[0][1] - arc[1][1]) ** 2)
        start_deg = math.atan2(arc[1][1] - arc[0][1], arc[1][0] - arc[0][0]) * 360 / (2 * math.pi)
        end_deg = math.atan2(arc[2][1] - arc[0][1], arc[2][0] - arc[0][0]) * 360 / (2 * math.pi)
        print(start_deg, end_deg)
        res = mpatches.Arc(arc[0], r * 2, r * 2, theta1=start_deg, theta2=end_deg, color='b', lw=1)
        ax.add_patch(res)

    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()