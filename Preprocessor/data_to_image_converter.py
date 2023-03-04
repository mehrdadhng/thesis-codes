import numpy as np
import pandas as pd 
from PIL import Image
from sklearn.model_selection import train_test_split

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

#Choose distribution
# distro = "Poisson"
distro = "Normal"
# distro = "mixedPoisson"
#####################

raw_dataset_path = "../data/samples/"+distro+"/samples1000.csv"
if distro == "Normal":
    raw_dataset_path = "../data/samples/"+distro+"/samples50.csv"
folder = "../data/samples/"+distro+"/samples_images/"

df = pd.read_csv(raw_dataset_path)
class2idx = {
}
for i in range(len(host_list)):
    class2idx[host_list[i]] = i
idx2class = {v: k for k, v in class2idx.items()}

X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

num_of_blocks = 9
pixels_per_block = 10
sample_counter_per_host = np.zeros(len(host_list))
for i in range(len(X_train)):
    sample = X_train[i]
    pixels = []
    for row in range(num_of_blocks*pixels_per_block):
        temp = []
        for col in range(num_of_blocks*pixels_per_block):
            temp.append(0)
        pixels.append(temp)
    pixels = np.array(pixels)
    sorted_indices = np.argsort(sample)
    counter = 1
    for idx in sorted_indices:
        coef = counter / len(edge_list)
        i1 = int(np.floor(idx/num_of_blocks)*pixels_per_block)
        i2 = i1 + pixels_per_block
        j1 = int((idx % num_of_blocks)*pixels_per_block)
        j2 = j1 + pixels_per_block
        pixels[i1:i2,j1:j2] = 255*coef
        counter += 1
    sample_counter_per_host[int(y_train[i])] += 1
    array = np.array(pixels, dtype=np.uint8)
    new_image = Image.fromarray(array,'L')
    new_image.save(folder+str(idx2class[y_train[i]])+"N"+str(int(sample_counter_per_host[int(y_train[i])]))+".png")