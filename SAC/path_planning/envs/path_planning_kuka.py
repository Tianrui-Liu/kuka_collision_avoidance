import numpy as np
import gym
from gym import spaces
import pybullet as p
import pybullet_data
import time
import math
import random
PosSet=[
    [0.27385896, 0.09399141, 2.2551664 ],
[ 0.55532365, -0.70032536, 2.32059288],
[ 0.458438 ,  -0.68871992 , 2.05519553],

[ 0.41450493 ,-0.1324283  , 2.3451498 ],

[ 0.285375823457418 , 0.07184049002484633 , 2.0916159343274723 ],

[ 0.395482254415116 , -0.7018634515347186 , 2.2399857220617583 ],

[ 0.29608132764333406 , -0.7084738242577795 , 1.928848746642047 ],

[ 0.555323648211636 , -0.7003253566988865 , 2.3205928834176333 ],
#     altered for collision
[ 0.8104174581587543 , -0.4350967454591223 , 1.7411771782832968 ],

# [ 0.9104174581587543 , -0.5350967454591223 , 1.7411771782832968 ],
[ 0.40539490462940986 , 0.10841376791114016 , 1.9376573393274754 ],
[ 0.19515206659930173 , -0.20845429901967344 , 1.542245510625456 ],
[ 0.3652299378375483 , 0.07893650972484884 , 1.5566135164743469 ],
# [ 0.5808606940487037 , -0.730899004323712 , 2.3429436605696257 ],
[ 0.3138696381808088 , -0.7135164565979224 , 2.1931013508195765 ],
[ 0.42446920960732604 , -0.4917427569163437 , 1.9989416565034632 ],
[ 0.2719722078041556 , -0.538962960573058 , 1.3648211385247944 ],
[ 0.4468474978878 , -0.20088144363300878 , 2.2315142750026884 ],
#     collision
[ 0.4464246788187468 , -0.7406699610944505 , 1.3483359365213765 ],

# [ 0.8464246788187468 , -0.7406699610944505 , 1.3483359365213765 ],
# [ 0.17716568256217025 , -0.1479438268477496 , 1.272969748349555 ],


[ 0.2621755055300726 , -0.6407679555940371 , 1.4284885381328953 ],
[ 0.9104174581587543 , -0.5350967454591223 , 1.7411771782832968 ],
[ 0.4380997561007971 , -0.4075851526759998 , 2.3790539849471615 ],
[ 0.4326523630966881 , -0.07429723056464918 , 1.6133187175555366 ],
[ 0.1653258318453824 , 0.23583290255166772 , 1.8430914008411086 ],
[ 0.4087798869522312 , -0.37767975392812664 , 2.1447890609362497 ],                                                                                                           [ 0.35402481108150485 , 0.07838743095289552 , 2.4453649814926517 ],
]


class Kukaenv(gym.Env):
    """
  Custom Environment that follows gym interface.
  This is a continuous env where each joint of the manipulator must learn how to rotate to reach the endpoint.
  """

    # implement GUI ('human' render mode),need change on server(console mode)
    # metadata = {'render.modes': ['human'],'render_fps': 4}
    metadata = {'render.modes': ['console']}

    def __init__(self):
        # visualization need change on server
        # p.connect(p.GUI)
        # p.resetDebugVisualizerCamera(cameraDistance=1.5,cameraYaw=0,\
        #                                  cameraPitch=-40,cameraTargetPosition=[0.55,-0.35,0.2])
        act_low = np.array(
            [-0.96705972839, -2.09439510239, -2.96705972839, 0.19439510239, -2.96705972839, -2.09439510239,
             -3.05432619099])
        act_high = np.array(
            [0.96705972839, 2.09439510239, 2.96705972839, 2.29439510239, 2.96705972839, 2.09439510239, 3.05432619099])

        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)
        self.KUKADEPTH=0.05
        self.SAFETY_MARGIN=0.03

        # self.action_space = spaces.Box(np.array([-3.15] * 7), np.array([3.15] * 7), dtype=np.float32)

        # target coordinate abc + joints coordinate xyz --> （7+1）*3
        self.observation_space = spaces.Box(np.array([-10] * 24), np.array([10] * 24), dtype=np.float32)
        # self.timeStep = 1 / 360
        self.done = False
        self.prev_dist_to_goal = 0

        self.present = False
        # EndTargetPos will be random in a range afterwards
        # self.EndTargetPos = random.choice(PosSet)
        # self.KukaPos = (0.2, -0.3, 1.5)
        # # CollisionPos into the state
        # self.CollisionPos = (0.51, -0.47, 1.76)
        
        if self.present:
            self.client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0,
                                         cameraPitch=-40, cameraTargetPosition=[ 0.41450493 ,-0.1324283  , 2.3451498 ])
        else:
            self.client = p.connect(p.DIRECT)

        p.setPhysicsEngineParameter(numSubSteps=2)
        self.reset()

    def reset(self):
        """
        reload urdf should be in reset,
        while the client of pybullet should remain the same in init()
        return:coordinate of the joints and the target, (7+1)*3 = 24*1 array
        """
        p.resetSimulation(self.client)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)

        # Reduce length of episodes for RL algorithms,seem no use
        # p.setTimeStep(self.timeStep, self.client)

        # EndTargetPos will be random in a range afterwards
        self.EndTargetPos = random.choice(PosSet)
        self.KukaPos = [0.2, -0.3, 1.5]
        # CollisionPos into the state
        self.CollisionPos = [0.51, -0.47, 1.76]
        # self.CollisionPos2 = [0.51, -0.07, 1.96]
        self.radi_colli=0.1
        self.radi_target=0.03

        ball_coll = p.createCollisionShape(p.GEOM_SPHERE, radius=self.radi_colli, physicsClientId=self.client)
        ball_visual = p.createVisualShape(p.GEOM_SPHERE, rgbaColor=[100, 1, 1, 1], radius=0.2,
                                          physicsClientId=self.client)
        self.collision1 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=ball_coll,
                                       baseVisualShapeIndex=ball_visual, basePosition=self.CollisionPos,
                                       physicsClientId=self.client)
        # self.collision2 =p.createMultiBody(baseMass=0, baseCollisionShapeIndex=ball_coll,
        #                                baseVisualShapeIndex=ball_visual, basePosition=self.CollisionPos2,
        #                                physicsClientId=self.client)

        target_coll = p.createCollisionShape(p.GEOM_SPHERE, radius=self.radi_target, physicsClientId=self.client)
        self.target_object = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=target_coll,
                                          basePosition=self.EndTargetPos, physicsClientId=self.client)
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.kuka_id = p.loadURDF("kuka_iiwa/model_vr_limits.urdf", basePosition=self.KukaPos,
                                physicsClientId=self.client)
        self.info = [self.target_object, self.collision1]

        for i in range(7):
            p.setCollisionFilterPair(self.kuka_id, self.target_object, i, -1, 0, physicsClientId=self.client)
            p.setCollisionFilterPair(self.kuka_id, self.collision1, i, -1, 0, physicsClientId=self.client)
            # p.setCollisionFilterPair(self.kuka_id, self.collision2, i, -1, 0, physicsClientId=self.client)
            for j in range(7):
                if i < j:
                    p.setCollisionFilterPair(self.kuka_id, self.kuka_id, i, j, 0, physicsClientId=self.client)

        # turn off visualizer
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        # print(p.getBasePositionAndOrientation(self.kuka_id, physicsClientId=self.client))
        #
        self.done = False

        p.setJointMotorControlArray(
            self.kuka_id,
            range(7),
            p.POSITION_CONTROL,
            targetPositions=np.zeros(7),
            physicsClientId=self.client)
        for i in range(50):
            p.stepSimulation(physicsClientId=self.client)
        # targetPositions=np.zeros(7))
        # turn on visualizer
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        pos = np.array(p.getLinkStates(self.kuka_id, range(7), physicsClientId=self.client), dtype=object)[:, 0]
        threed_pos = [i for i in pos]
        # end_effector_pos = np.array(p.getLinkState(self.kuka_id, 6, physicsClientId=self.client)[4])
        # print(end_effector_pos)
        # state = np.append(end_effector_pos, self.EndTargetPos)
        state = np.append(threed_pos, np.array(self.EndTargetPos))
        # print(state.size)
        # tmp = 0.0
        # for i in range(3):
        #     tmp += (state[i] - state[i + 3]) ** 2
        # self.prev_dist_to_goal = math.sqrt(tmp)
        # self.prev_dist_to_goal = np.linalg.norm(state[-6:-3] - state[-3:], ord=2)

        self.prev_dist_to_goal = np.linalg.norm(state[-6:-3] - state[-3:], ord=2)-self.KUKADEPTH-self.radi_target
        return state.astype(np.float32)

    def getjointstat(self):
        return p.getJointStates(self.kuka_id, range(7), physicsClientId=self.client)

    def step(self, action):
        """
        input: np.array 7*1, target angle of the joints
        output: new state of the joints and the target as debugging info, reward, whether is done
    """
        # ensure the input action is np.array 7*1
        # print("action:")
        # print(action)
        assert action.shape[0] == 7
        # for real action 
        # real_action = action * np.array(
        #     [0.96705972839, 2.09439510239, 2.96705972839, 2.29439510239, 2.96705972839, 2.09439510239, 3.05432619099])
        # print(real_action)

        # ensure the joint move to desired cooridnate
        
        p.setJointMotorControlArray(
            bodyUniqueId=self.kuka_id,
            jointIndices=range(7),
            controlMode=p.POSITION_CONTROL,
            targetPositions=action,
            physicsClientId=self.client
        )
        for i in range(120):
            p.stepSimulation(physicsClientId=self.client)
        # time.sleep(self.timeStep)

        # time.sleep(1/24)
        # states_ = Kukaenv.get_state(self.kuka_id)
        pos = np.array(p.getLinkStates(self.kuka_id, range(7), physicsClientId=self.client), dtype=object)[:, 0]
        threed_pos = [i for i in pos]
        # end_effector_pos = np.array(p.getLinkState(self.kuka_id, 6, physicsClientId=self.client)[4])
        # print(end_effector_pos)
        # states_ = np.append(end_effector_pos, self.EndTargetPos)
        states_ = np.append(threed_pos, self.EndTargetPos)
        # print(states_)
        done = False
        r = 0
        collison_contact = False

        dist_to_goal = np.linalg.norm(states_[-6:-3] - states_[-3:], ord=2)-self.KUKADEPTH-self.radi_target

        # numerical balance
        reward = (self.prev_dist_to_goal-dist_to_goal)*1000 *math.exp(-40*dist_to_goal+2)
        # distance = dist_to_goal / 5

        # # ret[0][8]-kukadepth
        # dist_to_goal=dist_to_goal
        # a, b, c, d = 0, 0, 0, -100
        # # related to the SAFETYMARGIN 0.02
        # if 0.4 >= dist_to_goal > 0:
        #     a, b, c, d = 4, 6, 0, -4500
        # elif dist_to_goal > 0.4:
        #     a, b, c, d = 1, 6, -1, -13

        # # 1.6 max dis
        # reward = a * math.exp(b * (1.6 - dist_to_goal) + c) + d


        self.prev_dist_to_goal = dist_to_goal

        # base(-1) from link 0 to link 6 check collsion
        for i in range(len(self.info) - 1):
            ret=p.getClosestPoints(self.kuka_id,self.collision1,linkIndexA=i, distance=100, physicsClientId=self.client)
            # ret2=p.getClosestPoints(self.kuka_id,self.collision2,linkIndexA=i, distance=100, physicsClientId=self.client)
            if(len(ret)!=0):
                # distance between the center of mass of certain link and the surface of the obstacle
                d=ret[0][8]
                # d2=ret2[0][8]
                # inner=False
                if d<self.SAFETY_MARGIN*1.5:
                    # if(np.linalg.norm(np.array(ret[0][5]) - np.array(self.CollisionPos), ord=2)<=self.radi_colli):
                        # inner=True
                    # print("collision")
                    collison_contact = True
                    reward = -10000
                    break
            # if bool(p.getContactPoints(self.kuka_id, self.info[i + 1], physicsClientId=self.client)):
            #     collison_contact = True
            #     reward = -10000
            #     break
        if not collison_contact:
            ret=p.getClosestPoints(self.kuka_id,self.target_object,linkIndexA=6, distance=100, physicsClientId=self.client)
            # ensure the dis do not exceed max distance 100
            if len(ret)!=0 :
                # margin between the center of mass of linkindex 6 and the surface of the target
                margin=ret[0][8]
                if(margin<=(self.SAFETY_MARGIN/3)):
                    # into the target, permit but weakly discourage
                    done = True
                    if margin<0:
                    # if np.linalg.norm(np.array(ret[0][5]) - np.array(self.EndTargetPos), ord=2)<self.radi_target:
                        reward=10
                        print("into the target")
                    # within SAFETY_MARGIN from target
                    else:
                        print("pre_act",action)
                        print("error", np.linalg.norm(np.array(action) - np.array(p.getJointStates(self.kuka_id, range(7),physicsClientId=self.client),dtype=object)[:, 0], ord=2))

                        reward = 50000

            # done = bool(p.getContactPoints(self.kuka_id, self.info[0], linkIndexA=6, physicsClientId=self.client))
            # done = done or dist_to_goal < 0.1
            # if done:
                # reward = 45000
                # print("done——————————————————————————————————————————————————")
                # print("———————————————————————————————————————————————————————")

        # Optionally we can pass additional info, we are not using that for now
        info = {}
        return states_.astype(np.float32), reward, done, info

    def render(self, mode='console'):
        # not done yet
        if mode == 'human':
            # Implement visualization
            pass
        if mode == 'console':
            # Implement visualization
            pass

    def close(self):
        p.disconnect(self.client)


if __name__ == '__main__':
    from stable_baselines.common.env_checker import check_env

    env = Kukaenv()
    check_env(env, warn=True)
