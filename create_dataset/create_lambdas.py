import numpy as np
import csv
import pandas as pd

#Choose topology
od_list = list(pd.read_csv("../data/ods.csv").to_numpy())
node_list = list(pd.read_csv("../data/nodes.csv").to_numpy().flatten())
host_list = list(pd.read_csv("../data/hosts.csv").to_numpy().flatten())
switch_list = list(pd.read_csv("../data/switches.csv").to_numpy().flatten())
edge_list = list(pd.read_csv("../data/edges.csv").to_numpy())
routing_matrix = pd.read_csv("../data/routing_matrix.csv").to_numpy()
M = routing_matrix.shape[0]
L = routing_matrix.shape[1]

indices_per_host = []
for host in host_list:
    temp = []
    for i in range(len(od_list)):
        od = od_list[i]
        if od[0] == host:
            temp.append(i)
    indices_per_host.append(temp)
indices_per_host = np.array(indices_per_host)
#####################


#Choose heavy hitters
critical_devices = [host_list[0], host_list[10]]
#####################


#Choose ranges
normal_range = [20,100]
heavy_range = [250,300]
#####################

#lambdas initialization
lambdas = np.random.randint(normal_range[0],normal_range[1],size = routing_matrix.shape[1])
#adding active flows
for cd in critical_devices:
    lambdas[indices_per_host[cd]] = np.random.randint(heavy_range[0], heavy_range[1], len(indices_per_host[cd]))


with open('../data/samples/lambdas.csv','w') as csvfile:
    wr = csv.writer(csvfile,delimiter=',')
    wr.writerow(["lambdas"])
    for lam in lambdas:
        wr.writerow([lam])