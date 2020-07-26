import matplotlib.pyplot as plt
import numpy as np
import sys
import json
with open(sys.argv[1], "r") as f:
    data = json.load(f)
pos = np.array(data["nodes"])
edge = np.array(data["edges"])
x = []
y = []
for a,b in pos:
	x.append(a)
	y.append(b)
qq = []
pp = []
for i,j in edge:
	qq.append((x[i],x[j]))
	pp.append((y[i],y[j]))
for i in range(len(qq)):
	plt.plot(qq[i],pp[i],color='r')
plt.scatter(x, y, color='b')
plt.show()
