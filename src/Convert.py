import numpy as np
import argparse
import math
import json
import copy
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

def log2json(file, nodes, edges):
    res = {}
    res["nodes"] = nodes
    res["edges"] = edges
    with open(file, "w") as f:
        json.dump(res, f)

def plot_log(file, nodes, edges, arcs={}):
    res = {}
    res["nodes"] = nodes.copy()
    res["segments"] = edges.copy()
    res["arcs"] = copy.deepcopy(arcs)
    plot = Plot(res)
    plot.plot()
    plot.show()
    # plot.savefig(file)

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

def min_distance(pos,edges):
    min = Utils.distance(pos[edges[0][0]], pos[edges[0][1]])
    for edge in edges:
        a = pos[edge[0]]
        b = pos[edge[1]]
        edge_distance = Utils.distance(a, b)
        if edge_distance < min:
            min = edge_distance
    return min

def add_point(nodes, edges, arcs):
    res_nodes = nodes.copy()
    res_edges = []

    for edge in edges:
        a, b = np.array(nodes[edge[0]]), np.array(nodes[edge[1]])
        dis_ab = Utils.distance(a, b)
        seg_num = math.ceil(dis_ab / Constant.MAX_DISTANCE)
        delta = (b - a) / seg_num
        prev_point = a
        prev_id = edge[0]
        for i in range(seg_num - 1):
            new_point = prev_point + delta
            new_id = len(res_nodes)
            res_nodes.append(new_point.tolist())
            res_edges.append([prev_id, new_id])
            prev_point = new_point
            prev_id = new_id
        res_edges.append([prev_id, edge[1]])

    for arc in arcs:
        a_id, b_id = arc[1], arc[2]
        arc[1] = nodes[arc[1]]
        arc[2] = nodes[arc[2]]
#        if arc[3] == 0:
        a, b = np.array(arc[1]), np.array(arc[2])
#        else:
#            b, a = np.array(arc[1]), np.array(arc[2])
        center = np.array(arc[0])
        r = math.sqrt((arc[0][0] - arc[1][0]) ** 2 + (arc[0][1] - arc[1][1]) ** 2)
        start = math.atan2(arc[1][1] - arc[0][1], arc[1][0] - arc[0][0])
        end = math.atan2(arc[2][1] - arc[0][1], arc[2][0] - arc[0][0])
        radian = (end - start)
        if radian < 0: radian += math.pi * 2
        if radian > math.pi * 2: radian -= math.pi * 2
        seg_num = math.ceil(r * radian / Constant.MAX_DISTANCE)
        delta = radian / seg_num
        prev_point = a
        prev_id = a_id
        for i in range(seg_num - 1):
            new_point = Utils.rotate(prev_point, delta, center)
            new_id = len(res_nodes)
            res_nodes.append(new_point.tolist())
            if arc[3] == 0:
                res_edges.append([prev_id, new_id])
            else:
                res_edges.append([new_id, prev_id])
            prev_point = new_point
            prev_id = new_id
        if arc[3] == 0:
            res_edges.append([prev_id, b_id])
        else:
            res_edges.append([b_id, prev_id])

    return res_nodes, res_edges

def main():
    cmd_parser = create_cmd_parser()
    args = cmd_parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)
    pos = np.array(data["nodes"])
    edges = np.array(data["edges"])
    node_amount = len(pos)
    edge_amount = len(edges)

#    plot_log(args.output, pos, edges, [])
    min_d = min_distance(pos, edges)
    if min_d <= 2 * Constant.INTERSECTION_RADIUS:
        pos *= 2 * Constant.INTERSECTION_RADIUS / min_d

    res_edges = [[] for _ in range(edge_amount * 2)]
    res_nodes = []
    cnt_nodes = 0
    intersections = [{"in": [], "out": []} for _ in range(node_amount)]
    for i, edge in enumerate(edges):
        a = pos[edge[0]]
        b = pos[edge[1]]
        roads = get_shift_segment(np.array([a, b]))
        endpoints = np.array([get_road_endpoint(roads[0], a, b), get_road_endpoint(roads[1], a, b)])
        res_nodes += endpoints[0].tolist() + endpoints[1].tolist()
        res_edges[i] += [cnt_nodes, cnt_nodes + 1]
        res_edges[i + len(edges)] += [cnt_nodes + 3, cnt_nodes + 2]
        intersections[edge[0]]["out"].append([cnt_nodes, b - a])
        intersections[edge[0]]["in"].append([cnt_nodes + 2, b - a])
        intersections[edge[1]]["in"].append([cnt_nodes + 1, a - b])
        intersections[edge[1]]["out"].append([cnt_nodes + 3, a - b])
        cnt_nodes += 4
    
    res_arcs = []
    for node_i in range(node_amount):
        for in_road in intersections[node_i]["in"]:
            for out_road in intersections[node_i]["out"]:
                in_point, in_vector = np.array(res_nodes[in_road[0]]), in_road[1]
                out_point, out_vector = np.array(res_nodes[out_road[0]]), out_road[1]
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
                    res_edges.append([in_road[0], out_road[0]])
                    continue
                in_line_orthogonal = np.array([in_point, in_point + np.array([in_vector[1], -in_vector[0]])])
                out_line_orthogonal = np.array([out_point, out_point + np.array([out_vector[1], -out_vector[0]])])
                center = Utils.get_line_intersection(in_line_orthogonal, out_line_orthogonal)
                if Utils.outer(in_point - center, out_point - center) > 0:
                    res_arcs.append([center.tolist(), in_road[0], out_road[0], 0])
                else:
                    res_arcs.append([center.tolist(), out_road[0], in_road[0], 1])

    # plot_log(args.log, res_nodes, res_edges, res_arcs)
    res_nodes, res_edges = add_point(res_nodes, res_edges, copy.deepcopy(res_arcs))
    log2json(args.output, res_nodes, res_edges)
    # plot_log(args.log, res_nodes, res_edges, {})

if __name__ == "__main__":
    main()
