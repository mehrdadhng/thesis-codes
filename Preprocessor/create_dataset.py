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
heavy_range = [130,200]
#####################


#lambdas initialization
lambdas = np.random.randint(normal_range[0],normal_range[1],size = routing_matrix.shape[1])
#adding active flows
for cd in critical_devices:
    lambdas[indices_per_host[cd]] = np.random.randint(heavy_range[0], heavy_range[1], len(indices_per_host[cd]))


#Choose distribution
distro = "Poisson"
# distro = "Normal"
# distro = "mixedPoisson"
#####################


Y = []
X = []
if distro == "Poisson":
    #Choose sample size
    sample_size = 5000
    ###################
    N = sample_size
    X = np.zeros((L,N))
    for j in range(L):
        X[j,:] = np.random.poisson(lam=lambdas[j], size = N)
    Y = routing_matrix @ X
elif distro == "Normal":
    #Choose sample size
    sample_size = 50
    ###################
    N = sample_size
    phi = np.random.rand()
    c = 1
    for j in range(len(lambdas)):
        od_std = np.sqrt(phi*np.power(lambdas[j] , c))
        X.append(np.floor(np.abs(np.random.normal(lambdas[j], od_std, size = N))))
    X = np.array(X)
    Y = routing_matrix @ X
elif distro == "mixedPoisson":
    #Choose sample size
    sample_size = 10000
    ###################
    N = sample_size
    X = np.zeros((L,N))
    rs = np.random.randint(low=normal_range[0], high=normal_range[1], size=L)
    betas = rs/(rs + lambdas)
    for j in range(L):
        X[j, :] = np.random.negative_binomial(rs[j], betas[j], size=N)
    Y = routing_matrix @ X  
    for i1 in range(len(Y)):
        for i2 in range(len(Y[i1])):
            if Y[i1][i2] < 0:
                Y[i1][i2] = 0
target = []
sum_list = []
for col in range(X.shape[1]):
    temp_sum_list = []
    for host in host_list:
        temp_sum_list.append(np.sum(X[:,col].flatten()[indices_per_host[host]]))
    sum_list.append(temp_sum_list)
for i in range(len(sum_list)):
    target.append(np.argmax(sum_list[i]))
Y = Y.T
with open('../data/samples/'+distro+"/samples.csv",'w') as csvfile:
    wr = csv.writer(csvfile,delimiter=',')
    labels = edge_list.copy()
    labels.append("target")
    wr.writerow(labels)
    samples = []
    for i in range(Y.shape[0]):
        temp = []
        for j in range(Y.shape[1]):
            temp.append(Y[i][j])
        temp.append(target[i])
        samples.append(temp)
    wr.writerows(samples)



