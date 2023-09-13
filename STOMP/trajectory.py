import os
import time
import math
#   /tmp/pycharm_project_562/STOMPv1/
import numpy as np

import pybullet as p
import pybullet_data

import stompTrajCost as sC
import Data


path="/tmp/pycharm_project_562/STOMPv1"
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

p.connect(p.GUI)  # or p.GUI for graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

EndTargetPos=Data.EndTargetPos
KukaPos=Data.KukaPos
CollisionPos= Data.CollisionPos

ball_coll = p.createVisualShape(p.GEOM_SPHERE,rgbaColor=[100, 1, 1, 1],radius=Data.colli_radi)
collision1 = p.createMultiBody(baseMass=0,baseVisualShapeIndex=ball_coll,basePosition=CollisionPos)

target_coll=p.createVisualShape(p.GEOM_SPHERE,radius=Data.target_radi)
target_object = p.createMultiBody(baseMass=0,baseVisualShapeIndex=target_coll,basePosition=EndTargetPos)

plane_id = p.loadURDF("plane.urdf")
kuka_id = p.loadURDF("kuka_iiwa/model_vr_limits.urdf", basePosition=KukaPos,baseOrientation=[0,0,1,3.14])

objects_pool = []
objects_pool.append(CollisionPos)

p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-80,
                             cameraPitch=-20,cameraTargetPosition=EndTargetPos)
# reset collision
for jointIndex in range(0, 7):
    p.setCollisionFilterPair(kuka_id, collision1, jointIndex, -1, 0)
    p.setCollisionFilterPair(kuka_id, target_object, jointIndex, -1, 0)

EndEffector=6

sC.setArmDegreewithSimulation(10,np.zeros(7),kuka_id,p)

Startpos = p.getLinkState(kuka_id,EndEffector)[4]



num_joints = p.getNumJoints(kuka_id)


RobotID = kuka_id


p.stepSimulation()

ar_load = np.load("theta.npy")
# ar_load=np.genfromtxt("data/theta.csv",delimiter=',')

for i in range(ar_load.shape[1]):
    sC.setArmDegreewithSimulation(300, ar_load[:,i], kuka_id, p)


print('STOMP Finished.')
while 1:
    p.stepSimulation()

p.disconnect()
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