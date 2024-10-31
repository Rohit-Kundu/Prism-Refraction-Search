import numpy as np
import math
from solution import solution
import time
import math

EPS = 1e-6
decay = 0.09

np.random.seed(42)


def compute_incidence(E, A, mu, r1):
    # print(E,A,mu,r1)
    E *= 0.0174533  # convert to radian
    A *= 0.0174533  # convert to radian
    theta = r1 * math.sin(A) * math.sqrt(mu ** 2 - math.sin(E) ** 2) - (math.sin(E) * math.cos(A))
    theta = max(min(theta, 1), -1)  # ensuring theta lies in range [-1,1]
    incidence = math.asin(theta)
    return incidence * 57.2958  # convert to degree


def compute_mu(A_mat, delta_mat):
    mu = np.zeros(shape=(delta_mat.shape))
    for i in range(delta_mat.shape[0]):
        # converting angles to radians before applying trignometric functions
        mu[i] = 1 + abs(math.sin(0.0174533 * (A_mat[i] + delta_mat[i]) / 2) / (EPS + math.sin(0.0174533 * A_mat[i] / 2)))
    return mu


def PROA(objf, lb, ub, dim, PopSize, iters):
    s = solution()

    PopSize = 1 # single-solution algorithm

    """ Initializations """
    fit = np.zeros(PopSize)
    Best = np.zeros(dim)
    Worst = np.zeros(dim)
    BestScore = float("inf")
    WorstScore = -1.0 * float("inf")

    ctr = 0

    if not isinstance(lb, list):
        lb = [lb] * dim
        ub = [ub] * dim

    incidence = np.zeros((PopSize, dim))

    for i in range(PopSize):
        for j in range(dim):
            incidence[i][j] = np.random.uniform(0, 90) * (ub[j] - lb[j]) + lb[j]

    A = np.random.uniform(15, 90, (PopSize,)) * (min(ub) - max(lb)) + max(lb)
    emergence = np.zeros(shape=(PopSize, dim))

    convergence_curve = np.zeros(iters)

    print("PROA is optimizing  \"" + objf.__name__ + "\"")

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(iters):

        for i in range(PopSize):
            l1 = [None] * dim
            l1 = np.clip(incidence[i, :], lb, ub)
            incidence[i, :] = l1.copy()

            # Calculate objective function for each particle
            fitness = []
            fitness = objf(l1)
            fit[i] = fitness

            if (fitness < BestScore):
                BestScore = fitness
                Best = l1.copy()

        mu = compute_mu(A, fit)

        for i in range(PopSize):

            for j in range(dim):

                # update emergent angle
                emergence[i][j] = fit[i] - incidence[i][j] + A[i]

                # random number in [-1, 1]
                r1 = np.random.uniform(-1,1,1)

                # update incident angle
                incidence[i][j] = compute_incidence(emergence[i][j], A[i], mu[i], r1)

        A = A * math.exp(-decay * l/iters)

        # add to convergence curve
        convergence_curve[l] = BestScore
        if (l % 1 == 0):
            print(['At iteration ' + str(l + 1) + ' the best fitness is ' + str(BestScore)])

        print('-'*100)

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.Algorithm = "PROA"
    s.objfname = objf.__name__
    s.params = {"decay" : decay, "EPS" : EPS, "max_iter" : iters, "BestFitness" : BestScore}

    return s
