import numpy as np


def stompDtheta(trajProb, em):
    nJoints = len(em)
    nDiscretize = trajProb.shape[1]

    dtheta = np.zeros((nJoints, nDiscretize))

    for i in range(0, nJoints):
        em_i = em[i]
        T = trajProb * em_i
        T_sum = np.sum(T, axis=0)  # tested
        dtheta[i, :] = T_sum

    return dtheta
