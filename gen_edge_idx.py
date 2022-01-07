import numpy as np

edge = set()
for i in range(3876):
    if (i-1)%51!=50:
        edge.add((i-1,i))
    if (i+1)%51!=0:
        edge.add((i,i+1))
    if (i+51) <= 3875:
        edge.add((i,i+51))
    if (i-51) >= 0:
        edge.add((i-51,i))
    if (i+52) % 51 != 0 and (i+52)<=3875:
        edge.add((i,i+52))
    if (i-52) % 51 != 50 and (i-52)>=0:
        edge.add((i-52,i))
# edge = np.array(edge)
arr = []
for i in edge:
    arr.append([i[0], i[1]])
arr.sort(key=lambda x: x[0])
arr=np.array(arr)
np.savetxt('rectangular_mesh_edge_idx.txt', arr)
