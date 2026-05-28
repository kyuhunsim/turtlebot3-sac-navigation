#!/usr/bin/env python3

import rospy
import time
import random
from gazebo_msgs.msg import ModelState, ModelStates

class RandomMover():
    def __init__(self):
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        self.current_target = self.generate_random_target()
        self.moving()

    def generate_random_target(self):
        x = random.uniform(-1, 1)
        y = random.uniform(3, 6) if x > 0 else random.uniform(2, 6)
        return (x, y)

    def moving(self):
        while not rospy.is_shutdown():
            model = rospy.wait_for_message('gazebo/model_states', ModelStates)
            for i in range(len(model.name)):
                if model.name[i] == 'obstacle_1':
                    obstacle_1 = ModelState()
                    obstacle_1.model_name = model.name[i]
                    obstacle_1.pose = model.pose[i]

                    target_x, target_y = self.current_target

                    if abs(obstacle_1.pose.position.x - target_x) < 0.05 and abs(obstacle_1.pose.position.y - target_y) < 0.05:
                        self.current_target = self.generate_random_target()
                        target_x, target_y = self.current_target

                    direction_x = target_x - obstacle_1.pose.position.x
                    direction_y = target_y - obstacle_1.pose.position.y

                    distance = (direction_x ** 2 + direction_y ** 2) ** 0.5
                    step_size = 0.01  # 한번에 이동할 크기
                    move_x = step_size * direction_x / distance
                    move_y = step_size * direction_y / distance

                    obstacle_1.pose.position.x += move_x
                    obstacle_1.pose.position.y += move_y

                    self.pub_model.publish(obstacle_1)
                    time.sleep(0.1)

def main():
    rospy.init_node('random_mover')
    try:
        mover = RandomMover()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
