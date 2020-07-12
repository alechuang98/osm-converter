import numpy as np
import argparse
import json
import constant
import matplotlib.pyplot as plt

def create_cmd_parser():
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument("--input")
    cmd_parser.add_argument("--output")
    return cmd_parser

def output(file, edges, arcs):
    res = {}
    res["segments"] = edges
    res["arcs"] = arcs
    with open(file, "w") as f:
        json.dump(res, f)

def outer(a, b):
    return a[0] * b[1] - a[1] * b[0]

def get_line_intersection(l1, l2):
    p1 = np.array(l1[0])
    p2 = np.array(l1[1])
    q1 = np.array(l2[0])
    q2 = np.array(l2[1])
    if abs(outer(p2 - p1, q2 - q1)) < constant.EPS and abs(outer(p2 - p1, q1 - p1)) < constant.EPS:
        return l2[0]
    f1 = outer(p2 - p1, q1 - p1)
    f2 = outer(p2 - p1, p1 - q2)
    f = f1 + f2
    assert abs(f) > constant.EPS
    return q1 * (f2 / f) + q2 * (f1 / f)

def get_shift_segment(l, scale=constant.ROAD_WIDTH):
    assert np.array_equal(l[0], l[1]) == False
    delta = l[1] - l[0]
    delta = np.array([delta[1], -delta[0]])
    delta = delta / np.linalg.norm(delta) * scale
    return np.array([l + delta, l - delta])

def main():
    cmd_parser = create_cmd_parser()
    args = cmd_parser.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)
    pos = np.array(data["nodes"])
    edges = np.array(data["edges"])

    res_edge = [[] for _ in range(len(edges) * 2)]
    neighbor = [[] for _ in range(len(pos))]
    for i, edge in enumerate(edges):
        if edge[0] > edge[1]:
            edge[0], edge[1] = edge[1], edge[0]
        neighbor[edge[0]].append((edge[1], i))
        neighbor[edge[1]].append((edge[0], i))
        roads = get_shift_segment(np.array([pos[edge[0]], pos[edge[1]]]))
        res_edge[i] += roads[0].tolist()
        res_edge[i + len(edges)] += roads[1].tolist()

    for node in range(len(pos)):
        for i in neighbor[node]:
            for j in neighbor[node]:
                if i[0] >= j[0] or np.dot(pos[i[0]] - pos[node], pos[j[0]] - pos[node]) > constant.EPS:
                    continue
                roads1 = get_shift_segment(np.array([pos[i[0]], pos[node]]))
                roads2 = get_shift_segment(np.array([pos[node], pos[j[0]]]))

                inter1 = get_line_intersection(roads1[0], roads2[0])
                inter2 = get_line_intersection(roads1[1], roads2[1])
                right, left = i[1], i[1] + len(edges)
                if node < i[0]:
                    right, left = left, right
                res_edge[right].append(inter1.tolist())
                res_edge[left].append(inter2.tolist())
                right, left = j[1], j[1] + len(edges)
                if node > j[0]:
                    right, left = left, right
                res_edge[right].append(inter1.tolist())
                res_edge[left].append(inter2.tolist())
    
    output(args.output, res_edge, {})

if __name__ == "__main__":
    main()