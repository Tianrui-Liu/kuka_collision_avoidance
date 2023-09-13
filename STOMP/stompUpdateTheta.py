import numpy as np

def stompUpdateTheta(theta, dtheta, M):
    # Smooth
    dtheta_smoothed = M @ dtheta[:, 1:-1].T.conjugate()

    theta[:, 1:-1] = theta[:, 1:-1] + dtheta_smoothed.T.conjugate()

    return theta, dtheta_smoothed
