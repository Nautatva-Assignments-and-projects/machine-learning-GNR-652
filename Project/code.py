from sklearn import datasets, metrics
import numpy as np
import math

from sklearn.ensemble import RandomForestClassifier

ratio = 0.5
p1 = 0.5
p2 = 0.5
min_num_sample = 4
trees = 100
ratio2 = 0.8


x = datasets.load_iris()['data']

y = datasets.load_iris()["target"].reshape(-1, 1)

x = np.c_[x, y]
np.random.shuffle(x)

xt = x[0:int(ratio2*len(x))]  # data with known labels
xu = x[int(ratio2*len(x)):len(x)]
# xu=np.delete(xu, -1, 1)    #data with unknown labels


def partition(x):
    np.random.shuffle(x)
    xs = x[0:int(ratio*len(x))]
    xe = x[int(ratio*len(x)):len(x)]
    return [xs, xe]


def attribute(x):
    s = np.random.binomial(1, p1, 1)
    if s == 0:
     #       print("attri 0")
        p = int(math.sqrt(x[0].shape[1]-1))
        l = np.arange(x[0].shape[1]-1)
        np.random.shuffle(l)
        q = []
        for i in range(0, p):
            q.append(l[i])
        return q

    elif s == 1:
      #      print("attri 1")
      #      print(x[0].shape)
        l = np.arange(x[0].shape[1]-1)
        np.random.shuffle(l)
        q = [l[0]]
        return q


def unique(x):
    if len(x) != 0:
        x = x[:, -1]
        x = x.reshape(-1, 1)
        unique, counts = np.unique(x, return_counts=True)
        d = dict(zip(unique, counts))
        return d
    else:
        d = {}
        d[0] = 0
        return d


def gini(x, s, m):  # (DATA,sample number,attribute)

    a = unique(x[0])

    q = 0
    for i in a:
        q = q+(a[i])**2
    g1 = 1-q

    xl = []
    xr = []
    for k in range(0, len(x[0])):
        if x[0][k][m] <= x[0][s][m]:
            xl.append(x[0][k])
        elif x[0][k][m] > x[0][s][m]:
            xr.append(x[0][k])
    xl = np.array(xl)
    xr = np.array(xr)

    xl1 = []
    xr1 = []
    for k in range(0, len(x[1])):
        if x[1][k][m] <= x[0][s][m]:
            xl1.append(x[1][k])
        elif x[1][k][m] > x[0][s][m]:
            xr1.append(x[1][k])
    xl1 = np.array(xl1)
    xr1 = np.array(xr1)
#    print("x[0].shape",x[0].shape)
#    print("xl.shape",xl.shape)
#    print("xr.shape",xr.shape)
    b = unique(xl)

    q = 0
    for i in b:
        q = q+(b[i])**2
    g2 = 1-q
    p = unique(xr)

    q = 0
    for i in p:
        q = q+(p[i])**2
    g3 = 1-q

    c = {}
    c["score"] = g1-(len(xl)/len(x[0]))*g2-(len(xr)/len(x[0]))*g3
    c["groups"] = [[xl, xr], [xl1, xr1]]

    return c


def split(x):
    s = np.random.binomial(1, p2, 1)

    if s == 1:  # random sampling
      #      print("split 1")

        m = attribute(x)
        m = np.array(m)
        np.random.shuffle(m)

        q = np.arange(x[0].shape[0])
        np.random.shuffle(q)

        i = q[0]
        j = m[0]
        xl = []
        xr = []
        for k in range(0, len(x[0])):
            if x[0][k][j] <= x[0][i][j]:
                xl.append(x[0][k])
            elif x[0][k][j] > x[0][i][j]:
                xr.append(x[0][k])
        xl = np.array(xl)
        xr = np.array(xr)

        xl1 = []
        xr1 = []
        for k in range(0, len(x[1])):
            if x[1][k][j] <= x[0][i][j]:
                xl1.append(x[1][k])
            elif x[1][k][j] > x[0][i][j]:
                xr1.append(x[1][k])
        xl1 = np.array(xl1)
        xr1 = np.array(xr1)

        d = {}

        d["split point"] = x[0][i][j]
        d["feature_index"] = j
        d["group"] = [[xl, xr], [xl1, xr1]]

        return d

    elif s == 0:  # optimizing impurity
       #     print("split 0")
        m = attribute(x)
        m = np.array(m)
        G = []
        K = []
        for j in range(0, len(m)):
            for i in range(0, len(x[0])):

                g = gini(x, i, m[j])

                G.append(g["score"])
                K.append([g, i, m[j]])

        max_gini = min(G)
        p = G.index(max_gini)

        l = K[p]

        d = {}

        d["split point"] = x[0][l[1], l[2]]
        d["feature_index"] = l[2]
        d["group"] = l[0]["groups"]

    #    print(len(d["group"][0][0]),len(d["group"][0][1]),len(d["group"][1][0]),len(d["group"][1][1]))
        return d


def term(x):  # terminal node

    if len(x) != 0:
        d = unique(x)
        max = -1
        for i in d:
            if d[i] >= max:
                max = d[i]
                y = i

        return y
#    else:
#        print("none encountered   lll") ####HAS TO BE CHANGED
#        return -1


def split_branch(node, min_num_sample, depth):
 #   print("split_branch")
    #    print("depth",depth)
    left_node = node['group'][0][0]
    right_node = node['group'][0][1]
    left_node1 = node['group'][1][0]
    right_node1 = node['group'][1][1]
    del(node['group'])

    if len(left_node) == 0 or len(right_node) == 0:
     #       print("case 0")
        if len(left_node) == 0:
            if len(right_node1) == 0:
                node['left'] = term(right_node)
                node['right'] = term(right_node)
            elif len(right_node1) != 0:
                node['left'] = term(right_node1)
                node['right'] = term(right_node1)

        elif len(right_node) == 0:
            if len(left_node1) == 0:
                node['left'] = term(left_node)
                node['right'] = term(left_node)
            elif len(left_node1) != 0:
                node['left'] = term(left_node1)
                node['right'] = term(left_node1)

        return

    if len(left_node) <= min_num_sample:
      #      print("2")
        if len(left_node1) == 0:
            node['left'] = term(left_node)
        else:
            node['left'] = term(left_node1)


#        print(node['left'])
    else:
       #     print("3")
        node['left'] = split([left_node, left_node1])

#        print("left=",node['left'])
        split_branch(node['left'], min_num_sample, depth+1)
    if len(right_node) <= min_num_sample:
        #    print("4")
        if len(right_node1) == 0:
            node['right'] = term(right_node)
        else:
            node['right'] = term(right_node1)


#        print(node['right'])
    else:
     #   print("5")
        #        print("right",right_node)
        node['right'] = split([right_node, right_node1])
#        print(node['right'])
        split_branch(node['right'], min_num_sample, depth+1)


def build_tree(x, min_num_sample):
    root = split(x)
    split_branch(root,  min_num_sample, 1)
    return root


def predict_sample(node, sample):

    if sample[node["feature_index"]] <= node["split point"]:
        if isinstance(node['left'], dict):
            return predict_sample(node['left'], sample)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict_sample(node['right'], sample)
        else:
            return node['right']


def predict(X, tree):
    y_pred = []
    for i in range(0, len(X)):
        y_pred.append(predict_sample(tree, X[i]))
    y_pred = np.array(y_pred)
    return y_pred


def accuracy(yp, y):

    y = y.reshape(-1, 1)

    e = yp-y

    o = np.count_nonzero(e)

    return ((len(y)-o)/len(y))*100


t = {}
ypred = np.zeros((xu.shape[0], 1))

w = partition(xt)
tree = build_tree(w, min_num_sample)
t[0] = tree
ypred[:, 0] = predict(xu, t[0])

ypred1 = np.zeros((xt.shape[0], 1))
ypred1[:, 0] = predict(xt, t[0])


for i in range(1, trees):
    w = partition(xt)
    tree = build_tree(w, min_num_sample)
    t[i] = tree
    ypred1 = np.c_[ypred1, predict(xt, tree)]
    ypred = np.c_[ypred, predict(xu, tree)]


yp = np.zeros((len(ypred), 1))

for i in range(0, len(ypred)):
    yp[i] = term(ypred[i].reshape(-1, 1))
# print("ypred",ypred)

yp1 = np.zeros((len(ypred1), 1))
# print("ypred1",ypred1)


for i in range(0, len(ypred1)):
    yp1[i] = term(ypred1[i].reshape(-1, 1))
# print(yp1)
# print(xt[:,-1].reshape(-1,1))

print("Accuracy of BRF =",
      100*metrics.accuracy_score(xu[:, -1], yp))


# Normal Random tree
# Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100)

# Fit and predict data
clf.fit(xt[:, 0:-1], xt[:, -1])
y_pred = clf.predict(xu[:, 0:-1])

print("Accuracy of original Random Forest =",
      100*metrics.accuracy_score(xu[:, -1], y_pred))
