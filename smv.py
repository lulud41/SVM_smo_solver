#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skd

DATA_SET_SIZE = 15
iteration = 10


#Menlo, Consolas, DejaVu Sans Mono, monospace

"""
    A faire :

    > exectution du script, avec param passés
        depuis arv et argc

        : epsilon, nb iter, C, kernel
        : nom modele a charger
        : fichiers dataset
        : nom modele save

        : si evaluate sur données

        >> argparse

    > evaluation :
        : accuracy
        : plot decision boundary

    > tester :
        : pred avec juste les vects supports
            enregistrés : 0< alpha < C

        > sauver et charger un modele
"""


def generate_test_dataset(n_samples):
    X,Y = skd.make_circles(n_samples=n_samples,noise=0.1)
    Y[Y==0] = -1
    return X,Y

def plot_datset(X,Y):
    p = X[np.where(Y==-1)]*5
    n = X[np.where(Y==1)]*10

    plt.scatter(p[:,0],p[:,1],c="red",marker="x")
    plt.scatter(n[:,0], n[:,1], c="blue", marker="o")

    plt.show()

"""
    > checker si on veut calculer le kernel
        : de 2 vect
        : tout le dataset (ligne par ligne)
        : une ligne
        : all
            : nouveau vect, kernel avec les tous les connus

"""
def RBF_kernel(mode, sigma, x1=None, x2=None, X=None, id=None):
    # boucle sur l, construit ligne par ligne
    # x_i - X : puis norme sur axe 1
    if mode == "single_val":
        return np.exp(- np.dot(x1-x2, (x1-x2).T) / (2*sigma**2) )

    if mode == "diagonal":
        return np.zeros((DATA_SET_SIZE))

    if mode == "line":
        x_i = X[id,:]

        a = (np.linalg.norm(x_i - X,axis=0) )
        a =  - np.square(a)

        return np.exp( a / (2*sigma**2))

    if mode == "new":

        a = (np.linalg.norm(x_1 - X,axis=0) )
        a =  - np.square(a)

        return np.exp( a / (2*sigma**2))

def linear_kernel(mode, x1=None, x2=None, X=None, id=None ):
    # boucle sur l, ligne par ligne
    if mode == "single_val":
        return np.dot(x1, x2)

    if mode == "diagonal":
        return np.sum(X*X, axis=1)

    if mode == "line":
        x_i = X[id]
        return np.dot(x_i, X.T)

    if mode == "new":
        return np.dot(x1, X.T)

def poly_kernel(mode, c, d, x1=None, x2=None, X=None, id=None):
    # boucle sur l, construit ligne par ligne
    # x_i - X : puis norme sur axe 1

    if mode == "single_val":
        return np.pow(np.dot(x1, x2) + c, d)

    if mode == "diagonal":
        k = np.sum(X*X, axis=1)
        k = k + c
        return np.pow(k, d)

    if mode == "line":
        x_i = X[i]
        k = np.dot(x_i, X.T)
        k = np.pow(k + c, d)

        return k

    if mode == "new":
        k = np.dot(x1, X.T)
        k = np.pow(k + c, d)

        return k

"""
    grad = Q*alpha + p
"""
def init_grad(X,Y, kernel_func, alpha):

    K = np.empty((0, DATA_SET_SIZE))

    for i in range(0, DATA_SET_SIZE):
        line = kernel_func(X, i, "line")
        K = np.vstack([K, line])

    Q = Y.T*K*Y
    grad_f = np.dot(Q, alpha) + np.ones((DATA_SET_SIZE))

    return grad_f

def stopping_criteria_test(alpha, Y, C, grad_f, epsilon):
    cond_1 = np.all([alpha < C, Y == 1], axis=0)
    cond_2 = np.all([alpha > 0, Y == -1],axis=0)

    I_up = np.any([cond_1, cond_2], axis=0)

    cond_1 = np.all([alpha < C, Y == -1], axis=0)
    cond_2 = np.all([alpha > 0, Y == 1],axis=0)

    I_low = np.any([cond_1, cond_2], axis=0)

    x = -Y*grad_f
    m_alpha = np.argmax(x[I_up])

    M_alpha = np.argmin(x[I_low])

    print(f"progress "{epsilon})

    if m - M <= epsilon:
        return True, None, None
    else:
        return False, I_low, I_up

def working_set_selection(grad_f, X, Y, I_up, I_low, kernel_func):

    x = -Y*grad_f

    i = np.argmax(x[I_up])

    k_ii = kernel_func(X[i,:], X[i,:], "single_val")

    k_tt = kernel_func(X,"diagonal")

    k_it = kernel_func(X, i, "line")

    a_it  = k_ii + k_tt - 2*k_it

    b_it = -Y[i]*grad_f[i] + Y*grad_f

    x = (-(b_it)**2)/a_it

    cond_1 = np.min(x[I_low])
    indexes = np.where(x == cond_1)
    j = np.intersect1d(indexes, I_low)[0]

    return i,j

def find_sub_prob_solutions(X, Y, grad_f, i,j, alpha, kernel_func, C):

    if Y[i] != Y[j]:

        k_ii = kernel_func(X[i,:], X[i,:], "single_val")
        k_jj = kernel_func(X[j,:], X[j,:], "single_val")
        k_ij = kernel_func(X[i,:], X[j,:], "single_val")

        a_ij  = k_ii + k_jj + 2*k_ij

        d = (-grad_f[i] - grad_f[j]) / a_ij

        alpha_new_i = alpha[i] + d
        alpha_new_j = alpha[j] - d

        #check bounds*
        dff = alpha[i] - alpha[j]

        if diff > 0:

            if alpha_new_j < 0:

                alpha_new_j = 0
                alpha_new_i = diff

            if alpha_new_i > C:
                alpha_new_i = C
                alpha_new_j = C - diff

        if diff <= 0:
            if alpha_new_i < 0:
                alpha_new_i = 0
                alpha_new_j = -diff

            if alpha_new_j > C:
                alpha_new_j = C
                alpha_new_i = C + diff

    if Y[i] == Y[j]:

        k_ii = kernel_func(X[i,:], X[i,:], "single_val")
        k_jj = kernel_func(X[j,:], X[j,:], "single_val")
        k_ij = kernel_func(X[i,:], X[j,:], "single_val")

        a_ij  = k_ii + k_jj - 2*k_ij

        d = (grad_f[i] - grad_f[j]) / a_ij

        alpha_new_i = alpha[i] - d
        alpha_new_j = alpha[j] - d

        #check bounds

        sum = alpha[i] + alpha[j]

        if sum > C:
            if alpha_new_i > C:
                alpha_new_i = C
                alpha_new_j = sum - C

            if alpha_new_j > C:
                alpha_new_j = C
                alpha_new_i = sum - C

        if sum <= C:
            if alpha_new_j < 0:
                alpha_new_j = 0
                alpha_new_i = sum

            if alpha_new_i < 0:
                alpha_new_i = 0
                alpha_new_j = sum

    new_alpha = np.copy(alpha)
    new_alpha[i] = alpha_new_i
    new_alpha[j] = alpha_new_j

    return alpha, new_alpha


def update_gradient(grad_f, alpha_new, alpha_old, i, j, X, kernel_func):
    Q_column_i = Y*Y[i]*kernel_func(X,i).T
    Q_column_j = Y*Y[j]*kernel_func(X,j).T

    grad_f = grad_f + Q_column_i*(alpha_new[i] - alpha_old[i]) \
        + Q_column_j*(alpha_new[j] - alpha_old[j])

    return grad_f

def compute_b(alpha, X, Y, kernel_func, methode=1):

    # methode 1 :
    if methode == 1:

        is_support_vect = np.all([ 0 < alpha, alpha < C])

        n = alpha[is_support_vect].shape[0]

        indexes = np.where(is_support_vect == True)

        x = 0

        for idx in indexes:
            x = x + Y[idx] - np.sum(alpha*Y*kernel_func(X, idx, "line"))

        b = x/n
        return b

    # methode 2 :
    if methode == 2:
        is_support_vect = np.all([ 0 < alpha, alpha < C])

        n = alpha[is_support_vect].shape[0]

        b = - np.sum(grad_f[is_support_vect]*Y[is_support_vect]) / n

        return b

def svm_pred(alpha, b, X, Y, kernel_func, new_x):

    pred = np.sign( np.sum(alpha*Y* kernel_func(X, new_x, "new"))) + b
    return pred

X,Y = generate_test_dataset(DATA_SET_SIZE)

#plot_datset(X,Y)

alpha = np.zeros((DATA_SET_SIZE))

grad_f = init_grad()


while iteration:

    [stopping, I_low, I_up]= stopping_criteria_test(alpha, Y, C, grad_f, epsilon):

    if stopping == False:

        [i,j] = working_set_selection()
        old_alpha, alpha = find_sub_prob_solutions(i,j)

        grad_f = update_gradient()

    else:
        print("fini")
        b = compute_b(alpha, X, )
        break


    iteration = iteration +1
