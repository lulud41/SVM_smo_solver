
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skd


"""
    A faire :

    : faire une classe propre


    >>>>>>>>>>>>>>>>>>>< DATASETS QUI MARCHENT

        > moons :

            : poly, C = 1 , d = 3, gamma = 1
                : Tr : 0.992 et test 0.990


        > circles
            : RBF, c=10, gamma 5, 100% test et train


        > xor
            RBF, c20, gamma 10, test 983, train 998

        > blobs
            linear : C == 100, test 976, train 968
"""


def generate_dataset(n_samples, type):
    np.random.seed(0)

    if type == "circles":
        X, Y = skd.make_circles(n_samples=n_samples, noise=0.02)
        Y[Y == 0] = -1

    if type == "moons":
        X, Y = skd.make_moons(n_samples=n_samples, noise=0.1)
        Y[Y == 0] = -1

    if type == "blobs":
        X, Y = skd.make_blobs(n_samples=n_samples, centers=2, cluster_std=0.8)
        Y[Y == 0] = -1

    if type == "xor":
        xx, yy = np.meshgrid(np.linspace(-3, 3, 50),
                             np.linspace(-3, 3, 50))
        rng = np.random.RandomState(0)
        X = rng.randn(n_samples, 2)
        l = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
        Y = np.zeros(n_samples)
        Y[l == True] = 1
        Y[l == False] = -1

    return X, Y


def plot_datset(X, Y):
    p = X[np.where(Y == -1)]
    n = X[np.where(Y == 1)]
    plt.scatter(p[:, 0], p[:, 1], c="red", marker="x")
    plt.scatter(n[:, 0], n[:, 1], c="blue", marker="o")
    plt.show()


def generate_train_test(X, Y):
    X_train = X[:DATA_TRAIN_SIZE]
    Y_train = Y[:DATA_TRAIN_SIZE]

    X_test = X[-DATA_TEST_SIZE:]
    Y_test = Y[-DATA_TEST_SIZE:]

    return X_train, Y_train, X_test, Y_test


class SVM():
    def __init__(self, kernel_type, C, epsilon=0.001, gamma=None, deg=None):

        self.kernel_type = kernel_type
        self.kernel_function = self.init_kernel_function(kernel_type)

        self.C = C
        self.gamma = gamma
        self.deg = deg
        self.epsilon = epsilon

        self.grad_f = np.ones(DATA_TRAIN_SIZE)*-1
        self.alpha = np.zeros(DATA_TRAIN_SIZE)

    def init_kernel_function(self, kernel_type):
        if kernel_type == "RBF":
            kernel_function = self.RBF_kernel
        if kernel_type == "linear":
            kernel_function = self.linear_kernel
        if kernel_type == "poly":
            kernel_function = self.poly_kernel

        return kernel_function

    """
        > checker si on veut calculer le kernel
            : deux vect x et y
            : un x et un array (nb_vect, dim_vect)
                : un vect avec un ensemble
                : x1 (n_dim) et x2 (n_vect, n_dim)
                    ker entre x1 et x2[i]

    """

    def RBF_kernel(self, mode, x1, x2):
        # boucle sur l, construit ligne par ligne
        # x_i - X : puis norme sur axe 1
        if mode == "single_val":
            return np.exp(np.dot(x1-x2, (x1-x2).T) * -self.gamma)

        if mode == "line":
            x = np.linalg.norm(x1 - x2, axis=1)
            x = np.exp(-np.power(x, 2) * self.gamma)
            return x

    """
        > checker si on veut calculer le kernel
            : deux vect x et y
            : un x et un array (nb_vect, dim_vect)
                : un vect avec un ensemble
                : x1 (n_dim) et x2 (n_vect, n_dim)
    """

    def linear_kernel(self, mode, x1, x2):
        # boucle sur l, ligne par ligne
        if mode == "single_val":
            return np.sum(x1*x2)

        if mode == "line":

            return np.sum(x1*x2, axis=1)

    """
        > checker si on veut calculer le kernel
            : deux vect x et y
            : un x et un array (nb_vect, dim_vect)
                : un vect avec un ensemble
                : x1 (n_dim) et x2 (n_vect, n_dim)
    """

    def poly_kernel(self, mode, x1, x2):
        # boucle sur l, construit ligne par ligne
        # x_i - X : puis norme sur axe 1
        if mode == "single_val":
            return np.power(self.gamma*np.dot(x1, x2.T) + 1, self.deg)

        if mode == "line":
            k = np.sum(x1*x2, axis=1)*self.gamma
            k = np.power(k+1, self.d)
            return k

    def stopping_criteria_test(self, Y):
        cond_1 = np.all([self.alpha < self.C, Y == 1], axis=0)
        cond_2 = np.all([self.alpha > 0, Y == -1], axis=0)

        I_up = np.any([cond_1, cond_2], axis=0)

        cond_3 = np.all([self.alpha < self.C, Y == -1], axis=0)
        cond_4 = np.all([self.alpha > 0, Y == 1], axis=0)

        I_low = np.any([cond_3, cond_4], axis=0)

        y_grad = - Y * self.grad_f

        m_alpha = np.max(y_grad[I_up])
        M_alpha = np.min(y_grad[I_low])

        d = m_alpha - M_alpha

        if d <= self.epsilon:
            return True, None, None, None
        else:
            return False, I_low, I_up, d

    def working_set_selection(self, X, Y, I_up, I_low):
        y_grad = -Y*self.grad_f
        # argmax dans I_up, mais indice dans y_grad et pas y_grad[I_up]
        max = np.max(y_grad[I_up])
        index_list = np.where(y_grad == max)
        i = np.intersect1d(index_list, np.where(I_up == True))[0]

        t = I_low
        t[t == True][np.where(y_grad[t] >= y_grad[i])] = False

        a_it = self.kernel_function("single_val", X[i], X[i]) \
            + self.kernel_function("line", X[t], X[t]) \
            - 2*self.kernel_function("line", X[i], X[t])

        a_it[a_it == 0] = 1e-12

        b_it = y_grad[i] + -1*y_grad[t]

        jj = np.argmin(-np.power(b_it, 2) / a_it)
        j = np.where(t == True)[0][jj]

        return i, j

    def find_sub_prob_solutions(self, X, Y, i, j):
        if Y[i] != Y[j]:

            diff = self.alpha[i] - self.alpha[j]

            k_ii = self.kernel_function("single_val", X[i], X[i])
            k_jj = self.kernel_function("single_val", X[j], X[j])
            k_ij = self.kernel_function("single_val", X[i], X[j])

            a_ij = k_ii + k_jj + 2*k_ij

            d = (-self.grad_f[i] - self.grad_f[j]) / a_ij

            alpha_new_i = self.alpha[i] + d
            alpha_new_j = self.alpha[j] + d

            # check bounds*
            if diff > 0:
                if alpha_new_j < 0:
                    alpha_new_j = 0
                    alpha_new_i = diff
            else:
                if alpha_new_i < 0:
                    alpha_new_i = 0
                    alpha_new_j = -diff
            if diff > 0:
                if alpha_new_i > C:
                    alpha_new_i = C
                    alpha_new_j = C - diff
            else:
                if alpha_new_j > C:
                    alpha_new_j = C
                    alpha_new_i = C + diff

        if Y[i] == Y[j]:

            sum = self.alpha[i] + self.alpha[j]

            k_ii = self.kernel_function("single_val", X[i], X[i])
            k_jj = self.kernel_function("single_val", X[j], X[j])
            k_ij = self.kernel_function("single_val", X[i], X[j])

            a_ij = k_ii + k_jj - 2*k_ij

            d = (self.grad_f[i] - self.grad_f[j]) / a_ij

            alpha_new_i = self.alpha[i] - d
            alpha_new_j = self.alpha[j] + d
            # check bounds
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

        old_alpha = np.copy(self.alpha)
        self.alpha[i] = alpha_new_i
        self.alpha[j] = alpha_new_j

        return old_alpha

    def update_gradient(self, old_alpha, i, j, X, Y):
        Q_column_i = Y*Y[i]*self.kernel_function("line", X[i], X)
        Q_column_j = Y*Y[j]*self.kernel_function("line", X[j], X)

        self.grad_f = self.grad_f + Q_column_i*(self.alpha[i] - old_alpha[i]) \
            + Q_column_j*(self.alpha[j] - old_alpha[j])

    def compute_b(self, Y):
        is_support_vect = np.all([0 < self.alpha, self.alpha < self.C], axis=0)

        n = self.alpha[is_support_vect].shape[0]

        self.b = - np.dot(self.grad_f[is_support_vect], Y[is_support_vect].T) / n

    """
        new x :
            : shape (n)
            ou : shape (m, n)
    """

    def svm_pred(self, X_train, Y_train, new_x):
        if new_x.ndim == 1:
            pred = np.sign(np.dot(self.alpha*Y_train,
                                  self.kernel_function("line", new_x, X_train)) + self.b)

        if new_x.ndim == 2:
            pred = np.empty(0)
            for i in range(new_x.shape[0]):
                p = np.sign(np.dot(self.alpha*Y_train,
                                   self.kernel_function("line", new_x[i], X_train)) + self.b)
                pred = np.concatenate((pred, np.array([p])), axis=0)

        return pred

    def compute_accuracy(self, X_train, Y_train, Y_test, X_test, dataset_type):
        correct = 0

        for i in range(X_test.shape[0]):
            label = Y_test[i]
            pred = self.svm_pred(X_train, Y_train, X_test[i])

            if label == pred:
                correct = correct+1

        print(f"accuracy : {correct/X_test.shape[0]} on {dataset_type} set")

    def plot_decision_boundary(self, X_train, Y_train):
        h = 0.05
        x_min = X[:, 0].min() - 0.5
        x_max = X[:, 0].max() + 0.5

        y_min = X[:, 1].min() - 0.5
        y_max = X[:, 1].max() + 0.5

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        data = np.c_[xx.ravel(), yy.ravel()]
        Z = self.svm_pred(X_train, Y_train, data)
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)
        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=Y+1, cmap=plt.cm.coolwarm)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())

        title = f"boundary : kernel : {self.kernel_type} : C : {self.C} "

        if self.gamma != None:
            title = title + f" gamma : {self.gamma}"
        if self.deg != None:
            title = title + f"d : {self.deg}"

        plt.title(title)
        plt.show()

    def train(self, X_train, Y_train, iteration):
        print("*** training svm ***")
        for step in range(0, iteration):
            stop, I_low, I_up, diff = self.stopping_criteria_test(Y_train)

            if stop == True:
                print("finished")
                break

            if step % 20 == 0:
                print(f"progress {diff}")

            i, j = self.working_set_selection(X_train, Y_train, I_up, I_low)
            old_alpha = self.find_sub_prob_solutions(X_train, Y_train, i, j)
            self.update_gradient(old_alpha, i, j, X_train, Y_train)

        self.compute_b(Y_train)


if __name__ == "__main__":

    iteration = 1000
    DATASET_SIZE = 1000

    DATA_TRAIN_SIZE = int(DATASET_SIZE*0.7)
    DATA_TEST_SIZE = int(DATASET_SIZE*0.3)

    X, Y = generate_dataset(DATASET_SIZE, "xor")
    X_train, Y_train, X_test, Y_test = generate_train_test(X, Y)

    C = 20
    epsilon = 0.00001

    kernel_type = "RBF"  # "poly" , "linear"

    svm_model = SVM(kernel_type, C, epsilon=0.001, gamma=10, deg=None)

    svm_model.train(X_train, Y_train, iteration)
    svm_model.compute_accuracy(X_train, Y_train, Y_train, X_train, "train")
    svm_model.compute_accuracy(X_train, Y_train, Y_test, X_test, "test")

    svm_model.plot_decision_boundary(X_train, Y_train)
