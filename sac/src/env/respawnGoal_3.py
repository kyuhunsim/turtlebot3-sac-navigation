#!/usr/bin/env python3
import rospy
import random
import time
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
import rospkg

class Respawn():
    def __init__(self):
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('turtlebot3_gazebo')
        model_path = os.path.join(package_path, 'models', 'turtlebot3_square', 'goal_box', 'model.sdf')

        if not os.path.exists(model_path):
            rospy.logerr(f"Model file not found: {model_path}")
            return

        self.modelPath = model_path
        
        try:
            with open(self.modelPath, 'r') as f:
                self.model = f.read()
        except IOError as e:
            rospy.logerr(f"Error reading model file: {e}")
            return

        self.stage = 4
        self.goal_position = Pose()
        self.goal_order = [
            (-1, 3),
            (1.1, 7.5),
            (-1.1, 7.75),
            (-2, 4),
            (-2, 3),
            (1, 3),
            (1, 6),
            (2, 4.2),
        ]
        self.init_goal_x, self.init_goal_y = self.goal_order[0]
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.modelName = 'goal'
        self.obstacle_1 = 0.6, 0.6
        self.obstacle_2 = 0.6, -0.6
        self.obstacle_3 = -0.6, 0.6
        self.obstacle_4 = -0.6, -0.6
        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y
        self.last_index = 0
        self.index = 0  # Index for sequential goal selection.
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.check_model = False
        self.index = 0

    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True
    
    def respawnModel(self):
        while True:
            if not self.check_model:
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
                rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x,
                              self.goal_position.position.y)
                break
            else:
                pass

    def deleteModel(self):
        while True:
            if self.check_model:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                del_model_prox(self.modelName)
                break
            else:
                pass

    def getPosition(self, position_check=False, delete=False):
        if delete:
            self.deleteModel()

        if self.stage != 4:
            while position_check:
                goal_x = random.randrange(-12, 13) / 10.0
                goal_y = random.randrange(-12, 13) / 10.0
                if abs(goal_x - self.obstacle_1[0]) <= 0.4 and abs(goal_y - self.obstacle_1[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_2[0]) <= 0.4 and abs(goal_y - self.obstacle_2[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_3[0]) <= 0.4 and abs(goal_y - self.obstacle_3[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_4[0]) <= 0.4 and abs(goal_y - self.obstacle_4[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - 0.0) <= 0.4 and abs(goal_y - 0.0) <= 0.4:
                    position_check = True
                else:
                    position_check = False

                if abs(goal_x - self.last_goal_x) < 1 and abs(goal_y - self.last_goal_y) < 1:
                    position_check = True

                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y

        else:
            while position_check:
                goal_x, goal_y = self.goal_order[self.index]

                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y

                self.index = (self.index + 1) % len(self.goal_order)
                position_check = False

        time.sleep(0.5)
        self.respawnModel()

        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return self.goal_position.position.x, self.goal_position.position.y

if __name__ == '__main__':
    rospy.init_node('respawn_goal')
    respawn = Respawn()
    respawn.getPosition(position_check=True)
