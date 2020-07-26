import argparse
import json
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class Plot():
    def __init__(self, data):
        plt.close()
        self.nodes = data["nodes"]
        self.segments = data["segments"]
        self.arcs = data["arcs"]
        self.fig, self.ax = plt.subplots()

    def plot(self): 
        for segment in self.segments:
            segment = [self.nodes[segment[0]], self.nodes[segment[1]]]
            seg = np.array(segment)
            self.ax.plot(seg[:, 0], seg[:, 1], 'bo-', lw=1)
        for arc in self.arcs:
            arc[1] = self.nodes[arc[1]]
            arc[2] = self.nodes[arc[2]]
        for arc in self.arcs:
            r = math.sqrt((arc[0][0] - arc[1][0]) ** 2 + (arc[0][1] - arc[1][1]) ** 2)
            start_deg = math.atan2(arc[1][1] - arc[0][1], arc[1][0] - arc[0][0]) * 360 / (2 * math.pi)
            end_deg = math.atan2(arc[2][1] - arc[0][1], arc[2][0] - arc[0][0]) * 360 / (2 * math.pi)
            res = mpatches.Arc(arc[0], r * 2, r * 2, theta1=start_deg, theta2=end_deg, color='b', lw=1)
            self.ax.add_patch(res)
        self.ax.axis("equal")
    
    def show(self):
        plt.show()
    
    def savefig(self, file_name):
        plt.savefig(file_name)

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

    plot = Plot(data)
    plot.plot()
    plot.savefig(args.output)

if __name__ == "__main__":
    main()