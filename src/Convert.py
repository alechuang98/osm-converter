import numpy as np
import argparse
import math
import json
import Constant
import Utils
from Plot import Plot
import matplotlib.pyplot as plt

def create_cmd_parser():
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument("--input")
    cmd_parser.add_argument("--log")
    cmd_parser.add_argument("--output")
    return cmd_parser

def log2json(file, edges, arcs):
    res = {}
    res["segments"] = edges
    res["arcs"] = arcs
    with open(file, "w") as f:
        json.dump(res, f)

def output(file, edges, arcs):
    res = {}
    res["segments"] = edges
    res["arcs"] = arcs
    plot = Plot(res)
    plot.plot()
    plot.savefig(file)

def get_shift_segment(l, scale=Constant.ROAD_WIDTH):
    assert np.array_equal(l[0], l[1]) == False
    delta = l[1] - l[0]
    delta = np.array([delta[1], -delta[0]])
    delta = delta / np.linalg.norm(delta) * scale
    return np.array([l + delta, l - delta])

def get_road_endpoint(l, p1, p2):
    res1 = Utils.get_circle_and_segment_intersection(l, p1, Constant.INTERSECTION_RADIUS)
    res2 = Utils.get_circle_and_segment_intersection(l, p2, Constant.INTERSECTION_RADIUS)
    return np.array([res1, res2])

def main():
    cmd_parser = create_cmd_parser()
    args = cmd_parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)
    pos = np.array(data["nodes"])
    edges = np.array(data["edges"])
    node_amount = len(pos)
    edge_amount = len(edges)

    res_edges = [[] for _ in range(edge_amount * 2)]
    intersections = [{"in": [], "out": []} for _ in range(node_amount)]
    for i, edge in enumerate(edges):
        a = pos[edge[0]]
        b = pos[edge[1]]
        roads = get_shift_segment(np.array([a, b]))
        endpoints = np.array([get_road_endpoint(roads[0], a, b), get_road_endpoint(roads[1], a, b)])
        res_edges[i] += endpoints[0].tolist()
        res_edges[i + len(edges)] += endpoints[1].tolist()
        intersections[edge[0]]["out"].append([endpoints[0][0], b - a])
        intersections[edge[0]]["in"].append([endpoints[1][0], b - a])
        intersections[edge[1]]["in"].append([endpoints[0][1], a - b])
        intersections[edge[1]]["out"].append([endpoints[1][1], a - b])
    
    res_arcs = []
    for node_i in range(node_amount):
        for in_road in intersections[node_i]["in"]:
            for out_road in intersections[node_i]["out"]:
                in_point, in_vector = in_road[0], in_road[1]
                out_point, out_vector = out_road[0], out_road[1]
                unit_in_vector = in_vector / np.linalg.norm(in_vector)
                unit_out_vector = out_vector / np.linalg.norm(out_vector)
                inner = np.dot(unit_in_vector, unit_out_vector)
                if inner > 1:
                    inner = 1
                if inner < -1:
                    inner = -1
                degree = np.arccos(inner) * 180 / math.pi
                if degree < Constant.TURNING_THRESHOLD_ANGLE:
                    continue
                if degree > Constant.USING_LINE_THRESHOLD_ANGLE:
                    res_edges.append([in_point.tolist(), out_point.tolist()])
                    continue
                in_line_orthogonal = np.array([in_point, in_point + np.array([in_vector[1], -in_vector[0]])])
                out_line_orthogonal = np.array([out_point, out_point + np.array([out_vector[1], -out_vector[0]])])
                center = Utils.get_line_intersection(in_line_orthogonal, out_line_orthogonal)
                if Utils.outer(in_point - center, out_point - center) > 0:
                    res_arcs.append(np.array([center, in_point, out_point]).tolist())
                else:
                    res_arcs.append(np.array([center, out_point, in_point]).tolist())

    log2json(args.log, res_edges, res_arcs)
    output(args.output, res_edges, res_arcs)

if __name__ == "__main__":
    main()