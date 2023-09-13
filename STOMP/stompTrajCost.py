import numpy as np
import math
import time
from timeit import timeit


def RtoM(R, pos):
    M = np.zeros((4, 4))
    M[0][0] = R[0]
    M[0][1] = R[1]
    M[0][2] = R[2]
    M[1][0] = R[3]
    M[1][1] = R[4]
    M[1][2] = R[5]
    M[2][0] = R[6]
    M[2][1] = R[7]
    M[2][2] = R[8]
    M[0][3] = pos[0]
    M[1][3] = pos[1]
    M[2][3] = pos[2]
    M[3][3] = 1
    return M


def VecToso3(omg):
    so3mat = [[0, -omg[2], omg[1]],
              [omg[2], 0, -omg[0]],
              [-omg[1], omg[0], 0]]

    return so3mat


def Combine3MatwithVec(mat, vec, LastOne):
    fourby3 = np.concatenate([mat, vec], axis=1)
    mat4 = np.concatenate([fourby3, np.zeros(4).reshape((1, 4))])
    mat4[3][3] = LastOne
    return mat4


def VecTose3(V):  # input:6
    three = np.array(VecToso3(V[0:3]))
    Trans = np.array(V[3:6]).reshape((3, 1))
    se3mat = Combine3MatwithVec(three, Trans, 0)
    return se3mat


def AxisAng3(expc3):  # input size 3:
    theta = np.linalg.norm(expc3)
    omghat = expc3 / theta
    return omghat, theta


def so3ToVec(so3mat):
    omg = [so3mat[2][1], so3mat[0][2], so3mat[1][0]]
    return omg


def NearZero(near):
    if np.linalg.norm(near) < 1e-6:
        return True
    else:
        return False


def MatrixExp3(so3mat):
    omgtheta = so3ToVec(so3mat)
    if NearZero(np.linalg.norm(omgtheta)):
        R = np.eye(3)
    else:
        omghat, theta = AxisAng3(omgtheta)
        omgmat = so3mat / theta
        R = np.eye(3) + math.sin(theta) * omgmat + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)

    return R


def MatrixExp6(so3mat):  # input :4*4
    partial_so3mat = np.array(so3mat)[0:3][:, 0:3]
    omgtheta = so3ToVec(partial_so3mat)
    if NearZero(np.linalg.norm(omgtheta)):
        T = Combine3MatwithVec(np.eye(3), np.array(so3mat)[0:3][:, 3].reshape((3, 1)), 1)
    else:
        omghat, theta = AxisAng3(omgtheta)
        omgmat = partial_so3mat / theta
        mat3 = MatrixExp3(partial_so3mat)
        partial_vec = (np.eye(3) * theta + (1 - math.cos(theta)) * omgmat + (theta - math.sin(theta)) * np.dot(omgmat,
                                                                                                               omgmat))
        vec = np.dot(partial_vec, np.array(so3mat)[0:3][:, 3] / theta).reshape((3, 1))
        T = Combine3MatwithVec(mat3, vec, 1)

    return T


def FkinSpaceIntermediates(Slist, thetalist, nJoints):  # 0.002992 second

    Is = []
    I = np.eye(4);
    for i in range(0, nJoints):
        I = np.dot(I, MatrixExp6(VecTose3(Slist[i]) * thetalist[i]))
        Is.append(I)

    return Is

def setArm2TargetwithSimulation(stepNum,EndTargetPos,Startpos,kuka_id,p,EndEffector):#input a destination,endeffecter will erach there,returns every degree of joints
    step_array = (np.array(EndTargetPos) - np.array(Startpos)) / stepNum
    for j in range(stepNum):
        robotStepPos = step_array + Startpos
        targetPositionsJoints = p.calculateInverseKinematics(kuka_id, 6, robotStepPos)
        p.setJointMotorControlArray(kuka_id, range(7), p.POSITION_CONTROL, targetPositions=targetPositionsJoints)
        p.stepSimulation()
        time.sleep(1. / 1000.)
        Startpos = np.array(robotStepPos)

    TaskFinal = p.calculateInverseKinematics(kuka_id, EndEffector, EndTargetPos)
    return TaskFinal



def setArmDegreewithSimulation(stepNum,TargetDegree,kuka_id,p):
    for j in range(stepNum):
        p.setJointMotorControlArray(kuka_id, range(7), p.POSITION_CONTROL, targetPositions=TargetDegree)
        p.stepSimulation()
        time.sleep(1. / 1000.)




def updateJointsWorldPosition(RobotID, theta, p):  # 0.002993 second
    #theta:7*1
    nJoints = p.getNumJoints(RobotID)
    if nJoints != len(theta):
        print("The dimension of the theta is not compatiable")
    #p.setJointMotorControlArray(RobotID, range(nJoints), p.POSITION_CONTROL, targetPositions=theta)
    setArmDegreewithSimulation(10,theta,RobotID,p)
    #  更新X

    T = np.zeros((nJoints, 4, 4))
    X = np.zeros((nJoints, 4))
    M = np.identity(4)
    a = [0, 0, -1]#其实不太知道为什么是magicnumber
    b = [0, 1, 0]
    ws = [a, b, a, b, a, b, a, b]#check-----
    Ms = np.zeros((nJoints, 4, 4))
    # modified
    Slist = np.zeros((nJoints, 6))
    Thetalist = np.zeros(nJoints)

    for i in range(0, nJoints):
        ort = p.getLinkState(RobotID, i)[5]#respect to previous link
        M_ = p.getMatrixFromQuaternion(ort)
        pos = p.getLinkState(RobotID, i)[4]

        M_ = RtoM(M_, pos)

        M = np.dot(M, M_)
        Ms[i] = M
        homo_q = M[:, 3]
        q = homo_q[0:3]
        w = ws[i]
        v = -np.cross(w, q)
        s = np.concatenate([w, v], axis=0)
        Slist[i] = s
        Thetalist[i] = p.getJointState(RobotID, i)[0]
    I = FkinSpaceIntermediates(Slist, Thetalist, nJoints)

    for i in range(0, nJoints):
        T_= I[i] @ Ms[i]
        # T_ = I[i] * Ms[i]

        T[i] = T_
        homo_q = T_[:, 3]
        X[i] = homo_q

    return X, T


def stompRobotSphere(X):  # 输入关节坐标,返回球体中心以及半径
    X = X[:, 0:3]
    nSpheresList=[]
    K = X.shape[0]
    center_cell = []
    for k in range(0, K):
        if k == 0:
            parent_joint_position = np.array([0, 0, 0])
        else:
            parent_joint_position = np.array(X[k - 1])

        child_joint_position = X[k]
        rad = 0.05

        nSpheres = int(np.linalg.norm(child_joint_position - parent_joint_position, ord=2) / rad) + 1
        nSpheresList.append(nSpheres)
        center_cell_k = np.linspace(parent_joint_position, child_joint_position, nSpheres)  # 返回两关节之间所有的圆球中心\
        for i in range(0, nSpheres):
            center_cell.append(center_cell_k[i])



    return center_cell,nSpheresList


def stompObstacleCost(sphere_centers, radi, objects_pool, vel, p):  # 第1,2,4参数长度相同n,radi和p为n*1，sphere_center为n*3

    safety_margin = 0.0867
    theta = 0
    for i in range(0, len(sphere_centers)):

        dx = math.inf
        if len(objects_pool) != 0:
            for j in range(0, len(objects_pool)):
                # 障碍物半径0.2,位置objects_pool[j]，小球sphere_centers[i]
                PosCollision=np.linalg.norm(objects_pool[j]-sphere_centers[i],ord=2,keepdims=False)
                distance=PosCollision-0.18-0.05
                if distance < dx:
                    dx = distance
        maxdistance = max(safety_margin- dx, 0)
        partial_theta = maxdistance * vel[i]

        theta = theta + partial_theta

    return theta




def stompConstrainCost(constrains, orts):  # input : 4*4    4*4 总长度为关节数  已完成

    Costs = []

    for i in range(0, orts.shape[0]):
        ort = orts[i][0:3, 0:3]
        constrain = constrains[i][0:3, 0:3]
        dot = np.transpose(constrain).dot(ort)
        I = np.eye(3)
        ret = np.linalg.norm(dot - I, ord=None)
        Costs.append(ret * ret)

    return Costs

def FitSphereSize(nspherelistpre,nspherelist,sphere_centers):
    sphere_centers_new=[]
    idx=0
    for i in range(0,len(nspherelistpre)):
        if nspherelistpre[i]==nspherelist[i]:
            temp=sphere_centers[idx:idx+nspherelist[i]]
            for j in range(0,len(temp)):
                sphere_centers_new.append(temp[j])
        else:

            start=sphere_centers[idx]
            end=sphere_centers[idx+nspherelist[i]-1]

            temp=np.linspace(start, end, nspherelistpre[i])
            for j in range(0, len(temp)):
                sphere_centers_new.append(temp[j])

        idx=idx+nspherelist[i]
    return sphere_centers_new


def stompTrajCost(RobotID, theta, objects_pool, p, R):
    nDiscretize = theta.shape[1]  # 7*20
    Qocost = np.zeros(nDiscretize)
    Qccost = np.zeros(nDiscretize)
    # 获得关节坐标位置

    X, T = updateJointsWorldPosition(RobotID, theta[:, 0], p)

    rad=0.05
    sphere_centers,nspherelist = stompRobotSphere(X)

    sphere_num = len(sphere_centers)

    vel = np.zeros(sphere_num)

    Qocost[0] = stompObstacleCost(sphere_centers, rad, objects_pool, vel, p)

    for i in range(1, nDiscretize):
        sphere_centers_prev = sphere_centers
        nspherelistpre=nspherelist
        # 目前的问题是X一直没有变化 done
        # print(theta[:,i])
        X, T = updateJointsWorldPosition(RobotID, theta[:, i], p)
        # print(X)
        sphere_centers,nspherelist = stompRobotSphere(X)

        if sum(nspherelist)!=sum(nspherelistpre):
            sphere_centers=FitSphereSize(nspherelistpre,nspherelist,sphere_centers)
            nspherelist=nspherelistpre

        radi = np.ones(len(sphere_centers)) * rad
        # 目前的问题是dx一直为0 done
        dx = np.array(sphere_centers_prev) - np.array(sphere_centers)
        # 目前的问题是vel一直为0 done
        vel = np.linalg.norm(dx, ord=2, axis=1, keepdims=False)

        Qocost[i] = stompObstacleCost(sphere_centers, radi, objects_pool, vel, p)

    Stheta = 1e7*Qocost*3 + Qccost
    theta = theta[:, 1:-1]

    #print(Stheta.sum())
    Qtheta = Stheta.sum() + (0.5 * theta @ R @ theta.T.conjugate()).sum()
    print(Qtheta)
    return 1e7*Qocost*3, Qtheta
