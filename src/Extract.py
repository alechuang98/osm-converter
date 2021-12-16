import os
import time
import json
from geopy.distance import geodesic
import lxml
from lxml import etree
import xmltodict, sys
bounds = []
nodes_id = []
nodes = []
edges = []
highway = ["motorway", "trunk", "primary", "secondary", "residential"]
def process_element(elem):
    elem_data = etree.tostring(elem)
    elem_dict = xmltodict.parse(elem_data,attr_prefix="",cdata_key="")

    if (elem.tag == "node"):
        if (bounds[2]>float(elem_dict["node"]["lat"])>bounds[0] and bounds[3]>float(elem_dict["node"]["lon"])>bounds[1]):
            nodes_id.append(int(elem_dict["node"]["id"]))
            nodes.append((float(elem_dict["node"]["lat"]), float(elem_dict["node"]["lon"])))
    elif (elem.tag == "way"):
        flag = 0
        if 'tag' in elem_dict["way"]:
            if isinstance(elem_dict["way"]["tag"], list) == False:
                elem_dict["way"]["tag"] = [elem_dict["way"]["tag"]]
            for datas in elem_dict["way"]["tag"]:
                if datas["k"] == "highway":
                    if datas["v"] in highway:
                        flag = 1
                        break;
            if flag:
                temp = []
                for datas in elem_dict["way"]["nd"]:
                    if int(datas["ref"]) in nodes_id:
                        temp.append(int(datas["ref"]))
                edges.append([temp])
    elif (elem.tag == "bounds"):
        bounds.append(float(elem_dict["bounds"]["minlat"]))
        bounds.append(float(elem_dict["bounds"]["minlon"]))
        bounds.append(float(elem_dict["bounds"]["maxlat"]))
        bounds.append(float(elem_dict["bounds"]["maxlon"]))


def fast_iter(context, func_element, maxline):
    placement = 0
    try:
        for event, elem in context:
            placement += 1
            if (maxline > 0):
                if (placement >= maxline): break

            func_element(elem)
            elem.clear()
            while elem.getprevious() is not None:
               del elem.getparent()[0]
    except Exception as ex:
        print("Error:",ex)

    del context

def transform(osmfile,maxline = 0):
    context = etree.iterparse(osmfile,tag=["node","way"])
    fast_iter(context, process_element, maxline)


osmfile = sys.argv[1]
bounds_context = etree.iterparse(osmfile, tag = ["bounds"])
fast_iter(bounds_context, process_element, 0)
transform(osmfile,0)

node_list = []

data = {}
data['nodes'] = []
data['edges'] = []

for edge in edges:
    for ref in edge:
        for index, node in enumerate(ref):
            if node not in node_list:
                node_list.append(node)
            if index != 0:
                data["edges"].append((node_list.index(ref[index - 1]), node_list.index(node)))

for i, d in enumerate(node_list):
	if (d in nodes_id):
	    node_idx = nodes_id.index(d)
	    lat = geodesic((nodes[node_idx][0],bounds[1]), (bounds[0], bounds[1])).meters
	    lon = geodesic((bounds[0],nodes[node_idx][1]), (bounds[0], bounds[1])).meters
	    data["nodes"].append((lon,lat))

output_path = sys.argv[2]
with open(output_path, 'w') as f:
	json.dump(data,f, indent = 4)
