import math
import numpy as np


def stompUpdateProb(Stheta):
    #     magic number
    h = 5

    #  max and min local cost at each time step
    # maxS = max(Stheta, [], 1)
    # minS = min(Stheta, [], 1)
    maxS = np.amax(Stheta, axis=0)
    minS = np.amin(Stheta, axis=0)
    # print('---------------------')
    # print(Stheta - minS)
    # print(maxS - minS)
    # print('---------------------')
    #  exp calculates the element-wise exponential
    expCost = np.exp(-h * (Stheta - minS) / (maxS - minS))
    # print(expCost)
    #  To handle the case where maxS = minS = 0. This is possible when local
    #  trajectory cost only includes obstacle cost, which at the end of the time
    #  duration, the speed is 0, or when there is no collision.
    expCost = np.nan_to_num(expCost)
    # normalize the exponentialized cost to have the probabilities

    trajProb = expCost / np.sum(expCost, axis=0)  # tested
    trajProb = np.nan_to_num(trajProb)

    return trajProb
