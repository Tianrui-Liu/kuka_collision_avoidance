# % Input: 
# %   sigma： sample covariance matrix
# %   theta: mean trajectory from last iteration
# % Output:
# %   theta_paths: sampled trajectories
# %   em: sampled Gaussian trajectory for each joint
import numpy as np


def stompSamples(nSamplePaths, sigma, theta):
    nJoints, nDiscretize = theta.shape[0], theta.shape[1]
    em = []
    ek = []
    #
    # em = np.array(1,nJoints)
    # ek = np.array(1,nSamplePaths)
    theta_paths = []

    # theta_paths = np.array(1, nSamplePaths)
    mu = np.zeros(len(sigma), dtype=float)

    for m in range(0, nJoints):
        # 混合高斯分布
        gau_dis = np.random.multivariate_normal(mu, sigma, nSamplePaths)

        # zero_dis = np.zeros((gau_dis.shape[1]))  # 未验证
        # dis = np.cat(zero_dis, gau_dis, zero_dis)  # 按行连接

        # 等价于前后两列补零
        dis = np.pad(gau_dis, ((0, 0), (1, 1)), 'constant')
        em.append(dis)

    # regroup it by samples,(1,nJoints)-> (nJoints,nDiscretize*nJoints)
    # emk = [em{:}] #？？
    # em:7*(20*20)

    emk = em[0]
    for i in range(1, nJoints):
        emk = np.concatenate((emk, em[i]), axis=1)

    for k in range(0, nSamplePaths):
        # tmp=emk[k,:]
        # print(emk[0,:])
        # print(emk[1,:])
        # ek[k]=np.reshape(emk[])
        # ek[k]=emk[k,:].reshape(nDiscretize, nJoints)
        ek.append(np.reshape(emk[k, :], (nDiscretize, nJoints)).T.conjugate())
        theta_paths.append(theta + ek[k])

    return theta_paths, em
