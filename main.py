import numpy as np
import sklearn
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics
from sklearn import linear_model
from sklearn import neighbors
import matplotlib.pyplot as plt


def calc_sgd(last, alpha, y, t, X, i, limit):
    a = alpha*(y[i] - t[i])
    a = a*X[i]
    curr = last - a
    a = np.sum(np.absolute(a))
    if limit < a:
        z = np.matmul(X, curr)
        curr_y = 1/(np.exp(-1*z) + 1)
        return calc_sgd(curr, alpha, curr_y, t, X, i + 1, limit)
    else:
        return curr

# function to calculate w
def get_w(X_train, t_train):
    X_t_trans = np.transpose(X_train) # transpose of X_train
    t_t_trans = np.transpose(t_train) # transpose of t_train
    w_den = np.matmul(X_t_trans, X_train)
    det_w_den = np.linalg.det(w_den)

    if det_w_den != 0:
        w_den = np.linalg.inv(w_den)

    else:
        return 0

    w = np.matmul(w_den, np.matmul(X_t_trans, t_t_trans))
    return w

def train_func(X, t, a, limit):
    Y = 1/(np.exp(-1*np.matmul(X, get_w(X, t))) + 1)
    return calc_sgd(get_w(X, t), a, Y, t, X, 0, limit)

def get_met(Y, t):
    true_pos, false_pos, false_neg, r = 0,0,0,0
    t_len = np.size(t)

    for i,elem in enumerate(t):
        if Y[i] >= 0.5:
            pdtn = 1
        else: 
            pdtn = 0
        if pdtn != elem: 
            r += 1
        if elem == 1 and pdtn == 1:
            true_pos += 1
        elif elem == 1 and pdtn == 0:
            false_neg += 1
        elif elem == 0 and pdtn == 1:
            false_pos += 1

    rcl = true_pos/(true_pos + false_neg)
    prcn = true_pos/(true_pos + false_pos)
    return r/t_len, prcn, rcl

def nearest_func(X_train, X_valid, k):
    
    dist_list = []
    len_x = len(X_train)
    
    for i,elem in enumerate(X_train):
        abs_val = np.absolute(elem - X_valid)
        distance = np.sum(abs_val)
        dist_list.append([i, distance])

    dist_list = sorted(dist_list,key=lambda x: x[1])
    dist_k = []
    for j in range(k):
        dist_k.append(dist_list[j][0])

    return dist_k

def get_k_train_err(X, t, kf, k):

    err = 0
    x_len = len(X)
    
    for train, test in kf.split(X,t):
        X_train = X[train]
        t_train = t[train]
        X_valid = X[test]
        t_valid = t[test]
        
        for i,elem in enumerate(t_valid):
            near = nearest_func(X_train, X_valid[i], k)
            pdtn = 0
            
            for j in near:
                pdtn += t_train[j]
            cond = pdtn/k
            
            if cond >= 0.5: 
                pdtn = 1
            else: 
                pdtn = 0
            
            if pdtn != elem: 
                err += 1

    return err/x_len

def get_k_test_err(X_train, t_train, X_test, t_test, k):
    err = 0
    y_arr = []
    
    for i,elem in enumerate(t_test):
        near = nearest_func(X_train, X_test[i], k)
        pdtn = 0
        
        for j in near:
            pdtn += t_train[j]
        
        if pdtn/k >= 0.5: 
            pdtn = 1
        else: 
            pdtn = 0
        y_arr.append(pdtn)
        
        if pdtn != elem: 
            err += 1

    return err/len(X_test), np.array(y_arr)

def main():
    np.random.seed(1234)
    lim = 0.00001
    a = 0.6

    sc = sklearn.preprocessing.StandardScaler()
    X, t = datasets.load_breast_cancer(return_X_y=True)
    scikit_learn_lr = linear_model.LogisticRegression()
    X_train, X_test, t_train, t_test = model_selection.train_test_split(X, t, test_size=0.2, shuffle=True)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    w = train_func(X_train, t_train, a, lim)
    log_z = np.matmul(X_test, w)
    log_y = 1/(np.exp(-1*log_z) + 1)
    r, prcn, rcl = get_met(log_y, t_test)
    F1 = 2*prcn*rcl/(prcn+rcl)
    
    prcns, rcls, lims = metrics.precision_recall_curve(t_test, log_y)
    scikit_learn_lr.fit(X_train, t_train)
    scikit_learn_lr_Y = scikit_learn_lr.predict(X_test)
    scikit_learn_r, scikit_learn_prcn, scikit_learn_rcl = get_met(scikit_learn_lr_Y, t_test)
    scikit_learn_F1 = 2*scikit_learn_prcn*scikit_learn_rcl/(scikit_learn_prcn+scikit_learn_rcl)
    scikit_learn_prcns, scikit_learn_rcls, lims = metrics.precision_recall_curve(t_test, scikit_learn_lr_Y)

    ##### Ploting Graphs #####
    plt.plot(rcls, prcns, label='Python Script Implementation of LR', color="green")
    plt.plot(scikit_learn_rcls, scikit_learn_prcns, label='Scikit Implementation of LR', color="blue")
    plt.title("Precision/Recall (PR) Curve")
    plt.legend(loc="lower left")
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.show()

    
    ks = 5
    K = 5
    kf = model_selection.KFold(K)
    min_k = 0
    min_err = np.inf
    print("Validation Errors for Values of k")
    for k in range(1,ks+1):
        training_error = get_k_train_err(X_train, t_train, kf, k)
        print(f"k = {k} Training Error = {training_error}")
        if training_error<min_err:
            min_err = training_error
            min_k = k
    test_error, n_n_Y = get_k_test_err(X_train, t_train, X_test, t_test, min_k)
    r_k, prcn_k, rcl_k = get_met(n_n_Y, t_test)
    F1_k = 2*prcn_k*rcl_k/(prcn_k+rcl_k)
    scikit_learn_n_n = neighbors.KNeighborsClassifier(n_neighbors=min_k)
    scikit_learn_n_n.fit(X_train, t_train)
    scikit_learn_n_n_Y = scikit_learn_n_n.predict(X_test)
    scikit_learn_r_k, scikit_learn_prcn_k, scikit_learn_rcl_k = get_met(scikit_learn_n_n_Y, t_test)
    scikit_learn_F1_k = 2*scikit_learn_prcn_k*scikit_learn_rcl_k/(scikit_learn_prcn_k+scikit_learn_rcl_k)

    print("\n")
    print(f"Misclassification Rate (Python Script LR): {r} \n")
    print(f"F1 Score (Python Script LR): {F1} \n")
    print(f"Misclassification Rate (Python Script knn): {test_error} \n")
    print(f"F1 Score (Python Script knn): {F1_k} \n")

    print(f"Misclassification Rate (Scikit LR): {scikit_learn_r} \n")
    print(f"F1 Score (Scikit LR):", scikit_learn_F1, "\n")
    print(f"Misclassification Rate (Scikit Learn knn): {scikit_learn_r_k} \n")
    print(f"F1 Score (Scikit Learn knn): {scikit_learn_F1_k} \n")

if __name__ == "__main__":
    main()