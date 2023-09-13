import os
import time
import math
#   /tmp/pycharm_project_562/STOMPv1/
import numpy as np
import Data
import pybullet as p
import pybullet_data

import stompTrajCost as sC
import stompSamples as sS
import stompDtheta as sD
import stompUpdateTheta as uT
import stompUpdateProb as uP

nDiscretize = 20
# K
nPaths = 20
# ----------------------------------------------------------------------------------------
A_k = np.eye(nDiscretize - 1, nDiscretize - 1)
A = -2 * np.eye(nDiscretize, nDiscretize)
tmp1 = np.pad(A_k, ((0, 1), (1, 0)), 'constant')
tmp2 = np.pad(A_k, ((1, 0), (0, 1)), 'constant')
# (nDiscretize-1,nDiscretize-1) converted to (nDiscretize,nDiscretize) in order to add up
tmp = tmp1 + tmp2
# print(tmp)
# A[1:, :-1] = A[1:, :-1] + tmp
# A[:, 1:] = A[:, 1:] + A_k
A = A + tmp
A = A[:, 1:-1]
# print(A.shape)
R = A.T.conjugate() @ A
# print(R.shape)
Rinv = np.linalg.inv(R)
# print(Rinv.shape)
M = (1 / nDiscretize * Rinv) / np.amax(Rinv, axis=1)
Rinv = 1.5 * Rinv / sum(sum(Rinv))
# print(Rinv)
# ----------------------------------------------------------------------------------------

p.connect(p.DIRECT)  # or p.GUI for graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setPhysicsEngineParameter(frictionERP=0, restitutionVelocityThreshold=0, numSubSteps=10)
p.setGravity(0, 0, -9.8)

EndTargetPos = Data.EndTargetPos
KukaPos = Data.KukaPos
CollisionPos = Data.CollisionPos

ball_coll = p.createVisualShape(p.GEOM_SPHERE, rgbaColor=[100, 1, 1, 1], radius=Data.colli_radi)
collision1 = p.createMultiBody(baseMass=0, baseVisualShapeIndex=ball_coll, basePosition=CollisionPos)

target_coll = p.createVisualShape(p.GEOM_SPHERE, radius=Data.target_radi)
target_object = p.createMultiBody(baseMass=0, baseVisualShapeIndex=target_coll, basePosition=EndTargetPos)

plane_id = p.loadURDF("plane.urdf")
kuka_id = p.loadURDF("kuka_iiwa/model_vr_limits.urdf", basePosition=KukaPos, baseOrientation=[0, 0, 1, 3.14])

objects_pool = []
objects_pool.append(CollisionPos)

p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-80,
                             cameraPitch=-20, cameraTargetPosition=EndTargetPos)
# reset collision
for jointIndex in range(0, 7):
    p.setCollisionFilterPair(kuka_id, collision1, jointIndex, -1, 0)
    p.setCollisionFilterPair(kuka_id, target_object, jointIndex, -1, 0)

EndEffector = 6

sC.setArmDegreewithSimulation(10, np.zeros(7), kuka_id, p)

Startpos = p.getLinkState(kuka_id, EndEffector)[4]

TaskFinal = sC.setArm2TargetwithSimulation(100, EndTargetPos, Startpos, kuka_id, p, EndEffector)

sC.setArmDegreewithSimulation(100, np.zeros(7), kuka_id, p)  # 通过赋角度值回来，准确度比下面高

# CurrentPos=p.getLinkState(kuka_id,EndEffector)[4]#通过位置赋值
# sC.setArm2TargetwithSimulation(100,Startpos,CurrentPos,kuka_id,p,EndEffector)#通过位置赋值

TaskInit = p.calculateInverseKinematics(kuka_id, EndEffector, Startpos)  # All should equals to 0

print(TaskInit)

q0 = TaskInit

qT = TaskFinal

num_joints = p.getNumJoints(kuka_id)

theta = np.zeros((num_joints, nDiscretize))
for k in range(0, num_joints):
    theta[k] = np.linspace(q0[k], qT[k], nDiscretize)

RobotID = kuka_id

# p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)不明作用，可以仔细check一下，不确定


_, Qtheta = sC.stompTrajCost(RobotID, theta, objects_pool, p, R)
print(Qtheta)
Qtheta_old = 0
iter = 0

while abs(Qtheta - Qtheta_old) > 1:
    Qtheta_old = Qtheta
    # thetha_paths, em = stompSamples(nPaths, )
    theta_paths, em = sS.stompSamples(nPaths, Rinv, theta)

    # % Calculate Local trajectory cost
    Stheta = np.zeros((nPaths, nDiscretize))
    for i in range(0, nPaths):
        theta_path = theta_paths[i]
        local_trajectory_cost, _ = sC.stompTrajCost(RobotID, theta, objects_pool, p, R)
        # print("local",local_trajectory_cost.sum())
        Stheta[i, :] = local_trajectory_cost

    # % Given the local traj cost, update local trajectory probability
    trajProb = uP.stompUpdateProb(Stheta)

    # % Compute delta theta (aka gradient estimator)
    dtheta = sD.stompDtheta(trajProb, em)

    theta, dtheta_smoothed = uT.stompUpdateTheta(theta, dtheta, M)

    # % Compute the cost of the new trajectory
    _, Qtheta = sC.stompTrajCost(RobotID, theta, objects_pool, p, R)

    # % control cost
    # print(np.sum(theta[:, 1: -1] @ R @ theta[:, 1: - 1].T.conjugate()))
    # 去掉了一个sum  need test
    RAR = 1 / 2 * np.sum(theta[:, 1: -1] @ R @ theta[:, 1: - 1].T.conjugate())
    # overall cost

    # control cost
    # print(RAR)
    iter = iter + 1
    # % Stop iteration criteria:
    if iter > 40 or np.sum(dtheta_smoothed) == 0:
        print('Maximum iteration (40) has reached.')
        break

    if np.sum(dtheta_smoothed) == 0:
        print('Estimated gradient is 0.')
        break

for i in range(120):
    p.stepSimulation()

np.save("theta.npy", theta)

ar_load = np.load("theta.npy")
print(ar_load)

p.disconnect()

print('STOMP Finished.')

# # %% Plot path
# # % axis tight manual % this ensures that getframe() returns a consistent size
# for t=1:size(theta,2)
#     show(robot, theta(:,t),'PreservePlot', true, 'Frames', 'on');
#     drawnow;
#     pause(1/50);
# end
#
#
# %% save data
# filename = ['Theta_nDisc', num2str(nDiscretize),'_nPaths_', num2str(nPaths), '.mat'];
# save(filename,'theta')
