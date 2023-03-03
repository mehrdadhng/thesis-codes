import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.linalg import khatri_rao
from sklearn.metrics import mean_squared_error
from scipy.optimize import fsolve
sqrt = np.emath.sqrt

od_list = list(pd.read_csv("../data/ods.csv").to_numpy())
node_list = list(pd.read_csv("../data/nodes.csv").to_numpy().flatten())
host_list = list(pd.read_csv("../data/hosts.csv").to_numpy().flatten())
switch_list = list(pd.read_csv("../data/switches.csv").to_numpy().flatten())
edge_list = list(pd.read_csv("../data/edges.csv").to_numpy())
routing_matrix = pd.read_csv("../data/routing_matrix.csv").to_numpy()

indices_per_host = []
for host in host_list:
    temp = []
    for i in range(len(od_list)):
        od = od_list[i]
        if od[0] == host:
            temp.append(i)
    indices_per_host.append(temp)
indices_per_host = np.array(indices_per_host)

#Choose heavy hitters
critical_devices = [host_list[0], host_list[10]]
#####################

normal_range = [20,100]
heavy_range = [130,200]

#lambdas and phi initialization
lambdas = np.random.randint(normal_range[0],normal_range[1],size = routing_matrix.shape[1])
phi = np.random.rand()
#adding active flows
for cd in critical_devices:
    lambdas[indices_per_host[cd]] = np.random.randint(heavy_range[0], heavy_range[1], len(indices_per_host[cd]))

def generate_samples(means, phi, number_of_samples_per_od, routing_matrix, c = 1):
    unobserved_samples = []
    for j in range(len(means)):
        od_std = np.sqrt(phi*np.power(means[j] , c))
        unobserved_samples.append(np.floor(np.abs(np.random.normal(means[j], od_std, size = number_of_samples_per_od))))
    unobserved_samples = np.array(unobserved_samples)
    link_samples = routing_matrix @ unobserved_samples
    return link_samples.T

def init_params(number_of_samples_per_od, observed_samples, routing_matrix, od_count):
    T = number_of_samples_per_od
    initial_means = np.ones(od_count).reshape((od_count, 1))
    sum = 0
    ones1 = np.ones(T).reshape((T, 1))
    ones2 = np.ones(len(routing_matrix)).reshape((len(routing_matrix), 1))
    ones3 = np.ones(len(routing_matrix[0])).reshape((len(routing_matrix[0]) , 1))
    temp1 = np.matmul(ones2.T, np.matmul(routing_matrix, ones3))
    if temp1 == 0 :
        temp1 = 1
    for t in range(T):
        temp2 = np.matmul(ones2.T, observed_samples[t])
        sum += temp2 / (T*temp1)

    initial_means = sum*initial_means
    i = np.random.randint(low = 0, high = len(observed_samples[0]))
    y_t_i = observed_samples[: , i]
    initial_phi = np.var(y_t_i) / np.mean(y_t_i)
    return initial_means, initial_phi

def make_sub_problems(routing_matrix):
    involved_X_indices = []
    sub_chosen_rows = []
    sub_routing_matricies = []
    for row1 in range(len(routing_matrix)):
        for row2 in range(row1+1 , len(routing_matrix)):
            temp = []
            for col in range(len(routing_matrix[0])):
                if routing_matrix[row1][col] == 1:
                    if not (col in temp):
                        temp.append(col)
                if routing_matrix[row2][col] == 1:
                    if not (col in temp):
                        temp.append(col)
            involved_X_indices.append(np.sort(temp))
            sub_chosen_rows.append([row1, row2])
            sub_routing_matricies.append(routing_matrix[[row1 , row2], :][: , involved_X_indices[-1]])
    return sub_chosen_rows,involved_X_indices, sub_routing_matricies  

def get_sub_means(current_means, involved_X_indices):
    current_sub_means = [];
    for i in range(len(involved_X_indices)):
        current_sub_means.append(current_means[involved_X_indices[i]])
    return current_sub_means

def get_sub_covariance_matricies(current_sub_means, current_phi, c = 1):
    current_sub_sigmas = []
    for i in range(len(current_sub_means)):
        temp = []
        for j in range(len(current_sub_means[i])):
            temp.append(np.power(current_sub_means[i][j][0], c))
        current_sub_sigmas.append(current_phi*np.diag(temp))
    return current_sub_sigmas

def cal_m(sub_routing_matricies, sub_chosen_rows, number_of_samples_per_od, observed_samples, current_sub_means, current_sub_sigmas):
    m_list = []
    for t in range(number_of_samples_per_od):
        m_temp = []
        for s in range(len(sub_chosen_rows)):
            y_t_s = observed_samples[t][sub_chosen_rows[s]].T.reshape((len(sub_chosen_rows[s]), 1))
            temp2 = np.add(y_t_s, -1*np.matmul(sub_routing_matricies[s], current_sub_means[s]))
            temp1 = -1
            try:
                temp1 = np.linalg.inv(np.matmul(sub_routing_matricies[s], np.matmul(current_sub_sigmas[s], sub_routing_matricies[s].T)))
            except:
                temp1 = np.linalg.pinv(np.matmul(sub_routing_matricies[s], np.matmul(current_sub_sigmas[s], sub_routing_matricies[s].T)))
            res = np.add(current_sub_means[s], np.matmul(current_sub_sigmas[s], np.matmul(sub_routing_matricies[s].T, np.matmul(temp1,temp2))))
            m_temp.append(res)
        m_list.append(m_temp)
    return m_list

def cal_R(sub_routing_matricies, current_sub_sigmas):
    R_list = []
    for s in range(len(current_sub_sigmas)):
        temp = -1
        try:
            temp = np.linalg.inv(np.matmul(sub_routing_matricies[s], np.matmul(current_sub_sigmas[s], sub_routing_matricies[s].T)))
        except:
            temp = np.linalg.pinv(np.matmul(sub_routing_matricies[s], np.matmul(current_sub_sigmas[s], sub_routing_matricies[s].T)))
        res = np.add(current_sub_sigmas[s], -1*np.matmul(current_sub_sigmas[s], np.matmul(sub_routing_matricies[s].T, np.matmul(temp, np.matmul(sub_routing_matricies[s], current_sub_sigmas[s])))))
        R_list.append(res)
    return R_list

def cal_a_and_b(od_index , involved_X_indices, number_of_samples_per_od, m_list, R_list):
    #here j and od_index are the same
    S_j = []
    for s in range(len(involved_X_indices)):
        if od_index in involved_X_indices[s]:
            S_j.append(s)
    relative_j_index_in_subProblems = []
    for s in range(len(S_j)):
        for i in range(len(involved_X_indices[S_j[s]])):
            if od_index == involved_X_indices[S_j[s]][i]:
                relative_j_index_in_subProblems.append(i)
    d_j = len(S_j)
    sum = 0
    for s in range(len(S_j)):
        sum_temp = 0
        for t in range(number_of_samples_per_od):
            sum_temp += np.power(m_list[t][S_j[s]][relative_j_index_in_subProblems[s]][0], 2)
        sum += R_list[S_j[s]][relative_j_index_in_subProblems[s]][relative_j_index_in_subProblems[s]] + sum_temp/number_of_samples_per_od
    a_j_k = sum/d_j
    sum2 = 0
    for s in range(len(S_j)):
        for t in range(number_of_samples_per_od):
            sum2 += m_list[t][S_j[s]][relative_j_index_in_subProblems[s]][0]
    b_j_k = sum2/(number_of_samples_per_od*d_j)
    return a_j_k, b_j_k

def get_eq(a, b):
    def f(phi):
        eq_9b = 0
        for i in range(len(a)):
            eq_9b = eq_9b \
                    + (((-1 * phi + sqrt((phi ** 2) + (4 * a[i]))) / 2) - b[i])
        return eq_9b
    return f

def run_EM(true_means, od_count, sub_routing_matricies, sub_chosen_rows, involved_X_indices, number_of_samples_per_od, observed_samples, initial_means, initial_phi, max_iteration, threshold = 0.1):
    current_means = initial_means.copy()
    current_phi = initial_phi
    #print("initial error: " + str(100*np.sum(np.abs((true_means - current_means)/true_means))/len(true_means)))
    for k in range(1 , max_iteration + 1):
        last_estimate = current_means.copy()
        current_sub_means = get_sub_means(current_means, involved_X_indices)
        current_sub_sigmas = get_sub_covariance_matricies(current_sub_means, current_phi)
        m_list = cal_m(sub_routing_matricies, sub_chosen_rows, number_of_samples_per_od, observed_samples, current_sub_means, current_sub_sigmas)
        R_list = cal_R(sub_routing_matricies, current_sub_sigmas)
        a_list = []
        b_list = []
        for j in range(od_count):
            a_j_k, b_j_k = cal_a_and_b(j, involved_X_indices, number_of_samples_per_od, m_list, R_list)
            a_list.append(a_j_k)
            b_list.append(b_j_k)
        ff = get_eq(a_list, b_list)
        current_phi = fsolve(ff, current_phi)[0]
        for j in range(od_count):
            current_means[j] = (-1 * current_phi + np.sqrt((current_phi ** 2) + (4 * a_list[j]))) / 2
    return current_phi, current_means.reshape((len(current_means),))


est_list = []
itrs = [1 , 2 , 5 , 10 , 20]
process_times = []
M = routing_matrix.shape[0]
L = routing_matrix.shape[1]
num_of_runs_per_sample_size = 5
print("runs per sample size: " + str(num_of_runs_per_sample_size))
print()

for itr in itrs:
    print("max iteration: " + str(itr))
    for count in range(num_of_runs_per_sample_size):
        start = time.process_time_ns()
        print("   .....run: " + str(count + 1))
        N = 50
        max_iteration = itr
        Y = generate_samples(lambdas, phi, N, routing_matrix)
        initial_means, initial_phi = init_params(Y.shape[0], observed_samples=Y, routing_matrix=routing_matrix, od_count=len(od_list))
        A = routing_matrix
        sub_chosen_rows, involved_X_indices, sub_routing_matricies = make_sub_problems(A)
        od_count = len(A[0])
        estimated_phi, estimated_means = run_EM(lambdas, od_count, sub_routing_matricies, sub_chosen_rows, involved_X_indices, Y.shape[0], Y, initial_means, initial_phi, max_iteration)
        est_list.append(estimated_means)
        end = time.process_time_ns()
        process_times.append(end - start)
    print("-----------------------------")
print()

plt.title("Average process time vs Max iteration, sample size = 50")
plt.ylabel("Aveerage process time(ns)")
plt.xlabel("Max iteration")
pt = []
for i in range(len(itrs)):
    pt.append(np.mean(np.array(process_times)[i*num_of_runs_per_sample_size:(i+1)*num_of_runs_per_sample_size]))
plt.plot(itrs,pt,'o-',color='red')
plt.show()

error_list = []
for i in range(len(itrs)):
    etemp = []
    for j in range(num_of_runs_per_sample_size):
        etemp.append(mean_squared_error(lambdas,est_list[i*num_of_runs_per_sample_size+j]))
    error_list.append(np.mean(etemp))

for i in range(len(itrs)):
    print("sample size: 50 and Max iteration:  " + str(itrs[i]) + "     MSE: " + str(error_list[i]))


scores_unlimited = []
for row in range(len(itrs)):
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
for i in range(len(itrs)):
    print("Max iteration: " + str(itrs[i]))
    print(np.argsort(scores_unlimited[i])[::-1])

print()

scores_limited = []
num_of_hot_flows_list = [len(host_list) - 1, 100, 250]
for row in range(len(itrs)):
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
    print("  ..Max iterations: " + str(itrs[i]))
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
    print("  ..Max iterations: " + str(itrs[i]))
    for j in range(len(scores_limited[i])):
        print("    ....number of top flows used to score hosts: " + str(num_of_hot_flows_list[j]))
        print(np.argsort(scores_limited[i][j][1])[::-1])




