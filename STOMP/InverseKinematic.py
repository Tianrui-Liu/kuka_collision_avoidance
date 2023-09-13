import time
import random

import numpy as np
import pybullet as p
import pybullet_data
import stompTrajCost as sC
import Data

p.connect(p.GUI)  # or p.GUI for graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
import numpy as np

data = np.load('test.npy')

# 设置新的物理引擎参数
p.setPhysicsEngineParameter(frictionERP=0, restitutionVelocityThreshold=0, numSubSteps=100)
EndTargetPos = Data.EndTargetPos
KukaPos = Data.KukaPos
CollisionPos = Data.CollisionPos
ball_collob = p.createCollisionShape(p.GEOM_SPHERE, radius=Data.colli_radi)
ball_coll = p.createVisualShape(p.GEOM_SPHERE, rgbaColor=[100, 1, 1, 1], radius=Data.colli_radi)
collision1 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=ball_collob, baseVisualShapeIndex=ball_coll,
                               basePosition=CollisionPos)

target_coll = p.createVisualShape(p.GEOM_SPHERE, radius=Data.target_radi)
target_collob = p.createCollisionShape(p.GEOM_SPHERE, radius=Data.target_radi)
target_object = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=target_collob, baseVisualShapeIndex=target_coll,
                                  basePosition=EndTargetPos)

plane_id = p.loadURDF("plane.urdf")
kuka_id = p.loadURDF("kuka_iiwa/model_vr_limits.urdf", basePosition=KukaPos)

objects_pool = []
objects_pool.append(CollisionPos)
# p.setTimeStep(0.005)
p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-80,
                             cameraPitch=-20, cameraTargetPosition=EndTargetPos)
# reset collision
for jointIndex in range(0, 7):
    p.setCollisionFilterPair(kuka_id, collision1, jointIndex, -1, 0)
    p.setCollisionFilterPair(kuka_id, target_object, jointIndex, -1, 0)

for i in range(7):
    for j in range(7):
        if i < j:
            p.setCollisionFilterPair(kuka_id, kuka_id, i, j, 0)

EndEffector = 6
act_low = np.array(
    [-0.96705972839, -2.09439510239, -2.96705972839, 0.19439510239, -2.96705972839, -2.09439510239, -3.05432619099])
act_high = np.array(
    [0.96705972839, 2.09439510239, 2.96705972839, 2.29439510239, 2.96705972839, 2.09439510239,
     3.05432619099])
stepNum = 5000
Startpos = p.getLinkState(kuka_id, EndEffector)[4]
step_array = (np.array(EndTargetPos) - np.array(Startpos)) / stepNum
# for j in range(stepNum):
#     robotStepPos = step_array + Startpos
#     targetPositionsJoints = p.calculateInverseKinematics(kuka_id, 6, robotStepPos)
#     targetPositionsJoints=  [   0.96705973, -2.0943951 ,  1.81159448 , 1.98544833 ,-2.8559654 , -2.0943951,
#  -2.33335729]
#
#
#     p.setJointMotorControlArray(kuka_id, range(7), p.POSITION_CONTROL, targetPositions=targetPositionsJoints)
#     p.stepSimulation()
#     Startpos = np.array(robotStepPos)
#     time.sleep(1/100)
#     if(p.getClosestPoints(kuka_id,target_object,linkIndexA=6, distance=100)[0][8]<0.1):
#         break

# ------------------------------------------------------------------
targetPositionsJoints = p.calculateInverseKinematics(kuka_id, 6, EndTargetPos, lowerLimits=act_low,
                                                     upperLimits=act_high)

# targetPositionsJoints = [-0.80157685 , 0.42880428 , 2.9153202   ,2.211226 ,  -2.8790529 , -2.0414536,
#  -2.6620152 ]
targetPositionsJoints =  [-0.8445282,  0.5451798,  2.9253187,  2.196291  , 2.8228748 ,-1.8574523,
  2.5986934]

p.setJointMotorControlArray(kuka_id, range(7), p.POSITION_CONTROL, targetPositions=targetPositionsJoints)
print("target", targetPositionsJoints)
# -0.7373701223858019, 0.6107943188538397, -0.2441093421677202,
#     -1.7755089418629284, -0.00751451845503019, 0.3730953750224103, 0.0


# p.setJointMotorControlArray(kuka_id, range(7), p.POSITION_CONTROL, targetPositions=targetPositionsJoints)
for i in range(50):
    p.stepSimulation()

print("error",
      np.linalg.norm(np.array(targetPositionsJoints) - np.array(p.getJointStates(kuka_id, range(7)))[:, 0], ord=2))
d = np.linalg.norm(np.array(p.getLinkState(kuka_id, EndEffector)[0]) - np.array(EndTargetPos), ord=2)
print(d)
ret = p.getClosestPoints(kuka_id, target_object, linkIndexA=6, distance=100)
print("ret")
print(ret)
print(ret[0][8])

print("closest point to target dis:",np.linalg.norm(np.array(ret[0][5]) - np.array(EndTargetPos), ord=2))
# for kk in range(990):
#     targetPositionsJoints= data[0][kk][24:31]
#     stepNum = 5000
#     Startpos = p.getLinkState(kuka_id, EndEffector)[4]
#     step_array = (np.array(EndTargetPos) - np.array(Startpos)) / stepNum
#     for j in range(stepNum):
#         robotStepPos = step_array + Startpos
#         # targetPositionsJoints = p.calculateInverseKinematics(kuka_id, 6, robotStepPos)
#         # targetPositionsJoints = [-0.83891782, 2.0943951, 2.96705973, 2.2943951, 2.96705973, -2.0943951,
#         #                          -2.77658305]
#
#         p.setJointMotorControlArray(kuka_id, range(7), p.POSITION_CONTROL, targetPositions=targetPositionsJoints)
#         p.stepSimulation()
#         Startpos = np.array(robotStepPos)
#         d = np.linalg.norm(np.array(p.getLinkState(kuka_id, EndEffector)[4]) - np.array(EndTargetPos), ord=2)
#         print(d)
#     time.sleep(1/100)


TaskFinal = p.calculateInverseKinematics(kuka_id, EndEffector, EndTargetPos)

while 1:
    p.stepSimulation()
