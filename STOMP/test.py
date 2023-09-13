#[ 0.02816976 -0.52856475  2.6231933   0.5        -0.71        1.86      ] [-2.4743392  -2.6530988  -0.13908377  2.6990442  -0.01375813  2.8692024
  #1.8287054 ] [ 0.2297914  -0.45784682  2.6736505   0.5        -0.71        1.86      ]


import time
import random

import numpy as np
import pybullet as p
import pybullet_data
import stompTrajCost as sC
import Data
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

stepNum = 5000
Startpos = p.getLinkState(kuka_id,EndEffector)[4]
step_array = (np.array(EndTargetPos) - np.array(Startpos)) / stepNum

qKey = ord('q')
keys = p.getKeyboardEvents()
if qKey in keys and keys[qKey]&p.KEY_WAS_TRIGGERED:
    for j in range(stepNum):
        robotStepPos = step_array + Startpos
        targetPositionsJoints = p.calculateInverseKinematics(kuka_id, 6, robotStepPos)
        p.setJointMotorControlArray(kuka_id, range(7), p.POSITION_CONTROL, targetPositions=targetPositionsJoints)
        p.stepSimulation()
        time.sleep(1. / 1000.)
        Startpos = np.array(robotStepPos)

    TaskFinal = p.calculateInverseKinematics(kuka_id, EndEffector, EndTargetPos)



while 1:
    p.stepSimulation()
