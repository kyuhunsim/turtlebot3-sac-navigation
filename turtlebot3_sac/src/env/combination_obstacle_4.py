#!/usr/bin/env python3

import rospy
import time
from gazebo_msgs.msg import ModelState, ModelStates

class Combination():
    def __init__(self):
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        self.waypoints = [
            (-1.5, 7.8), (-1, 7.8), (0, 7.8), (0, 6), (1, 6)
        ]
        self.current_waypoint_index = 0
        self.forward = True  # 경로의 진행 방향 (True: 정방향, False: 역방향)
        self.moving()

    def moving(self):
        while not rospy.is_shutdown():
            model = rospy.wait_for_message('gazebo/model_states', ModelStates)
            for i in range(len(model.name)):
                if model.name[i] == 'obstacle_2':
                    obstacle_2 = ModelState()
                    obstacle_2.model_name = model.name[i]
                    obstacle_2.pose = model.pose[i]

                    target_x, target_y = self.waypoints[self.current_waypoint_index]

                    if abs(obstacle_2.pose.position.x - target_x) < 0.05 and abs(obstacle_2.pose.position.y - target_y) < 0.05:
                        if self.forward:
                            self.current_waypoint_index += 1
                            if self.current_waypoint_index >= len(self.waypoints):
                                self.current_waypoint_index -= 2
                                self.forward = False
                        else:
                            self.current_waypoint_index -= 1
                            if self.current_waypoint_index < 0:
                                self.current_waypoint_index += 2
                                self.forward = True

                        target_x, target_y = self.waypoints[self.current_waypoint_index]

                    direction_x = target_x - obstacle_2.pose.position.x
                    direction_y = target_y - obstacle_2.pose.position.y

                    distance = (direction_x ** 2 + direction_y ** 2) ** 0.5
                    step_size = 0.01  # 한번에 이동할 크기
                    move_x = step_size * direction_x / distance
                    move_y = step_size * direction_y / distance

                    obstacle_2.pose.position.x += move_x
                    obstacle_2.pose.position.y += move_y

                    self.pub_model.publish(obstacle_2)
                    time.sleep(0.1)

def main():
    rospy.init_node('combination_obstacle_2')
    try:
        combination = Combination()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
