# OSM CONVERTER

### Dependencies
python 3.6.8  
numpy 1.17.5  
matplotlib 3.1.3  
geopy 2.0.0  
lxml 4.5.2

### Usage
Extract edges and nodes from the osm file to json file.  
We only extract the roads that are labeled motorway, trunk, primary, secondary
```
python src/extract.py <input osm file> <output json file>
```
Convert the json file into desired format(Two lanes, Directions, Arcs)
```
python src/Convert.py --input <input json file> --log <log png file> --output <output json file>
```

##### input file format
osm file. You can get the osm file from the openstreetmap website.  
json format. `test/input.json` for reference.  
Simple demo of how to use the json file created in matlab.
```
fileName = 'output.json';
str = fileread(fileName);
data = jsondecode(str);
G = digraph;
X = [];
Y = [];
G = addnode(G,length(data.nodes));
for i = 1:length(data.nodes)
    X = [X data.nodes(i,1)];
    Y = [Y data.nodes(i,2)];
end
for i = 1:length(data.edges)
    G = addedge(G, data.edges(i,1) + 1, data.edges(i,2) + 1);
end
figure;plot(G, 'XData',X,'YData',Y)
```

### Backlog

1. Add parameter for log control.
