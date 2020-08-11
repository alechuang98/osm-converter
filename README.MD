# OSM CONVERTER

### Dependencies
python 3.6.8
numpy 1.17.5
matplotlib 3.1.3

### Usage
Extract edges and nodes from the osm file to json file
```
python src/extract.py <input osm file> <output json file>
```
Convert the json file into desired format(Two lanes, Directions, Arcs)
```
python src/Convert.py --input <input json file> --log <log png file> --output <output json file>
```

##### input file format

json format. `test/input.json` for reference.

### Backlog

1. Add parameter for log control.
