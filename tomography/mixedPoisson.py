import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.linalg import khatri_rao
from sklearn.metrics import mean_squared_error

use_preprocessor = False

if use_preprocessor:
    od_list = list(pd.read_csv("../data/preprocessor_data/ods.csv").to_numpy())
    routing_matrix = pd.read_csv("../data/preprocessor_data/routing_matrix.csv").to_numpy()
    edge_list = list(pd.read_csv("../data/preprocessor_data/edges.csv").to_numpy())
    edges_to_kepp = pd.read_csv("../data/preprocessor_data/edges_to_keep.csv").to_numpy().flatten()
else:
    od_list = list(pd.read_csv("../data/ods.csv").to_numpy())
    routing_matrix = pd.read_csv("../data/routing_matrix.csv").to_numpy()
    edge_list = list(pd.read_csv("../data/edges.csv").to_numpy())

node_list = list(pd.read_csv("../data/nodes.csv").to_numpy().flatten())
host_list = list(pd.read_csv("../data/hosts.csv").to_numpy().flatten())
switch_list = list(pd.read_csv("../data/switches.csv").to_numpy().flatten())

# indices_per_host = []
# for host in host_list:
#     temp = []
#     for i in range(len(od_list)):
#         od = od_list[i]
#         if od[0] == host:
#             temp.append(i)
#     indices_per_host.append(temp)
# indices_per_host = np.array(indices_per_host)


lambdas = pd.read_csv("../data/samples/lambdas.csv").to_numpy().flatten()
est_list = []
sample_sizes = [100,500,1000,2500,5000]
est_list = []
process_times = []
M = routing_matrix.shape[0]
L = routing_matrix.shape[1]
num_of_runs_per_sample_size = 3
print("runs per sample size: " + str(num_of_runs_per_sample_size))
print()

for ss in sample_sizes:
    print("sample size: " + str(ss))
    for count in range(num_of_runs_per_sample_size):
        start = time.process_time_ns()
        print("   .....run: " + str(count + 1))
        N = ss
        Y = pd.read_csv("../data/samples/mixedPoisson/samples"+str(N)+".csv").to_numpy()[:,:-1].T 
        if use_preprocessor:
            Y = Y[edges_to_kepp,:] 
        epsilon = 0.01
        gamma = 0.0005
        A_r = np.zeros((M + M**2,2*L))
        A_r[0:M,0:L] = routing_matrix
        A_r[M:,L:] = khatri_rao(routing_matrix,routing_matrix)
        first_lhs = np.mean(Y,axis=1).reshape((M,1))
        second_lhs = np.mean(khatri_rao(Y - first_lhs, Y - first_lhs), axis=1).reshape((M**2,1))
        lhs = np.concatenate((first_lhs,second_lhs),axis=0)

        A_r, A_r_idx = np.unique(A_r, return_index=True, axis=0)
        lhs = lhs[A_r_idx]

        A_r_idx = np.sum(A_r, axis=1) > 0
        A_r = A_r[A_r_idx]
        lhs = lhs[A_r_idx]

        lhs_idx = np.sum(lhs, axis=1) >= 0
        lhs = lhs[lhs_idx]
        A_r = A_r[lhs_idx]

        A_r = np.matrix(A_r)
        zeta = np.array(np.linalg.inv(A_r.H @ A_r + gamma*np.identity(2*L)) @ (A_r.H @ lhs))
        for i in range(len(zeta)):
            if zeta[i][0] < 0 :
                zeta[i][0] = 0.001

        estimation = np.zeros((L,1))
        for index in range(L):
            estimation[index] = (zeta[index]*zeta[index+L])/(zeta[index]+zeta[index+L])

        estimation = estimation.flatten()
        est_list.append(estimation)
        end = time.process_time_ns()
        process_times.append(end - start)
    print("-----------------------------")
print()

plt.title("Average process time vs Sample size")
plt.ylabel("Aveerage process time(ns)")
plt.xlabel("Samples size")
pt = []
for i in range(len(sample_sizes)):
    pt.append(np.mean(np.array(process_times)[i*num_of_runs_per_sample_size:(i+1)*num_of_runs_per_sample_size]))
plt.plot(sample_sizes,pt,'o-',color='red')
plt.show()

error_list = []
for i in range(len(sample_sizes)):
    etemp = []
    for j in range(num_of_runs_per_sample_size):
        etemp.append(mean_squared_error(lambdas,est_list[i*num_of_runs_per_sample_size+j]))
    error_list.append(np.mean(etemp))

for i in range(len(sample_sizes)):
    print("sample size: " + str(sample_sizes[i]) + "     MSE: " + str(error_list[i]))

scores_unlimited = []
for row in range(len(sample_sizes)):
    su_temp = []
    for col in range(num_of_runs_per_sample_size):
        estimation = est_list[row*num_of_runs_per_sample_size+col]
        sums = np.zeros(len(host_list))
        for i in range(len(estimation)):
            od = od_list[i]
            sums[host_list.index(od[0])] += estimation[i]
        su_temp.append(sums)
    scores_unlimited.append(np.mean(su_temp,axis=0))

print()
print('scoring hosts using all flows - 2nd score system')
for i in range(len(sample_sizes)):
    print("sample size: " + str(sample_sizes[i]))
    print(np.argsort(scores_unlimited[i])[::-1])

print()

scores_limited = []
num_of_hot_flows_list = [len(host_list) - 1, 100, 250]
for row in range(len(sample_sizes)):
    sl_temp = []
    for col in range(num_of_runs_per_sample_size):
        estimation = est_list[row*num_of_runs_per_sample_size+col]
        est_sorted_indices = np.argsort(estimation)
        temp_scores = []
        for nohf in num_of_hot_flows_list:
            score_presence = np.zeros(len(host_list))
            score_participation = np.zeros(len(host_list))
            for j in range(nohf):
                od = od_list[est_sorted_indices[j - nohf]]
                score_presence[host_list.index(od[0])] += 1
                score_participation[host_list.index(od[0])] += estimation[est_sorted_indices[j - nohf]]
            temp_scores.append([score_presence,score_participation])
        sl_temp.append(temp_scores)
    scores_limited.append(np.mean(sl_temp,axis= 0))

print()
print("scoring hosts using the first scoring system")
for i in range(len(scores_limited)):
    print("  ..sample size: " + str(sample_sizes[i]))
    for j in range(len(scores_limited[i])):
        print("    ....number of top flows used to score hosts: " + str(num_of_hot_flows_list[j]))
        s_indices = np.argsort(scores_limited[i][j][0])[::-1]
        temp = ""
        for k in range(len(s_indices) - 1):
            temp += (str(s_indices[k])+" : "+str(int(float(scores_limited[i][j][0][s_indices[k]])))+"   ,   ")
        temp += (str(s_indices[-1])+" : "+str(int(float(scores_limited[i][j][0][s_indices[-1]]))))
        print(temp)

print()
print()
print("scoring hosts using the second scoreing system")
for i in range(len(scores_limited)):
    print("  ..sample size: " + str(sample_sizes[i]))
    for j in range(len(scores_limited[i])):
        print("    ....number of top flows used to score hosts: " + str(num_of_hot_flows_list[j]))
        print(np.argsort(scores_limited[i][j][1])[::-1])




