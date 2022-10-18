import random
import math
import numpy as np
import pandas as pd
import pygame

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
# Return true if positions intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
# calculate distance between two coordinates
def calc_dist(point1, point2):
    return math.sqrt((point1[0] - point2[0]) * (point1[0] - point2[0]) + (point1[1] - point2[1]) * (point1[1] - point2[1]))
# calculate travel time between path nodes by both forward motion and rotational motion
def calc_travel_time(degg, point1, point2, agent, return_deg=True, seperate=False):
    d = calc_dist(point1, point2)
    degrees = math.degrees(math.atan2(-(point2[0] - point1[0]), point2[1] - point1[1])) - degg
    angle = math.degrees(math.atan2(-(point2[0] - point1[0]), point2[1] - point1[1]))
    if abs(d) <= ((50 / 10) * 2):
        time = (abs(d) / agent.max_speed) + 1000
    else:
        time = abs(d) / agent.max_speed
    plus = 0
    if (0 > angle > -45 and 180 > degg > 135):
        diff = angle - 45 + (degg-135)
        if abs(diff) > 35:
            plus = 100
        rotation_time = (abs(diff) / agent.max_rot_speed) + plus
    elif (0 > degg > -45 and 180 > angle > 135):
        diff = degg - 45 + (angle-135)
        if abs(diff) > 35:
            plus = 100
        rotation_time = (abs(diff) / agent.max_rot_speed) + plus
    elif (0 < angle < 45 and -180 < degg < -135):
        diff = angle + 45 + (degg+135)
        if abs(diff) > 35:
            plus = 100
        rotation_time = (abs(diff) / agent.max_rot_speed) + plus
    elif (0 < degg < 45 and -180 < angle < -135):
        diff = degg + 45 + (angle+135)
        if abs(diff) > 35:
            plus = 100
        rotation_time = (abs(diff) / agent.max_rot_speed) + plus
    else:
        rotation_time = abs(degrees)/ agent.max_rot_speed
    if return_deg:
        return time + rotation_time, degrees + degg
    elif seperate:
        return time, rotation_time
    else:
        return time + rotation_time

class RRT_method:
    def __init__(self, start_pos, end_pos, screen, cones, D, d, T):
        self.initial_pos = start_pos
        self.goal = end_pos
        self.screen = screen
        self.cones = [[i for i in cones if i.type == 0], [i for i in cones if i.type == 1]]  # red,blue
        self.nodes_list = [{'name': 'n0', 'coord': np.array(self.initial_pos), 'parent': 'None', 'degrees': -90, 'cost': 0, 'time':0}]
        self.path_nodes = []
        self.max_dist = D
        self.min_dist = d
        self.max_time = T

    def tree_gen(self, agent):
        while True:
            next_node, parent_name, parent_degrees, parent_cost = self.node_gen(agent)
            # distance to the end
            dis_to_den = [(self.goal[0] - next_node[0]), (self.goal[1] - next_node[1])]
            if self.goal[0] > next_node[0] and 500 > next_node[1]:
                print(next_node)
            if -self.max_dist < dis_to_den[0] < self.max_dist and -self.max_dist < dis_to_den[1] < self.max_dist:
                if not self.cone_check(next_node, self.goal):
                    name = "n{}".format(len(self.nodes_list))
                    cost = parent_cost + calc_dist(next_node, self.goal)
                    travel_t, degrees = calc_travel_time(parent_degrees, next_node, self.goal, agent)
                    self.nodes_list.append({'name': name, 'coord': self.goal, 'parent': parent_name, 'degrees': degrees,
                         'cost': cost, 'time': travel_t})
                    self.path_nodes = self.find_path()
                    break
        print('number of path nodes: ', len(self.path_nodes))
        print('Finished')

    '''generates new tree nodes: if new node is not within required distance of all previous nodes
     or exceeds travel time a new node is created'''
    def node_gen(self, agent):
        while True:
            rand_coords = (int(random.random() * self.screen.get_width()), int(random.random() * self.screen.get_height()))

            closest_node = [calc_travel_time(i['degrees'], i['coord'], rand_coords, agent, False) for i in self.nodes_list]
            parent = self.nodes_list[min(range(len(closest_node)), key=closest_node.__getitem__)]
            distance = calc_dist(parent['coord'], rand_coords)
            # if not within distance variables calculate a new new node position within set distance
            if distance > self.max_dist or distance < self.min_dist:
                theta = math.atan2(rand_coords[1] - parent['coord'][1], rand_coords[0] - parent['coord'][0])
                add_tox = random.randrange(-self.max_dist, self.max_dist,(self.max_dist/10))
                add_toy = random.randrange(-self.max_dist, self.max_dist,(self.max_dist/10))
                choice = random.choice([(0, add_toy), (add_tox, 0), (add_tox, add_toy)])
                rand_coords = int(parent['coord'][0] + choice[0] * math.cos(theta)), int(
                    parent['coord'][1] + choice[1] * math.sin(theta))

            TRAVELT, deg = calc_travel_time(parent['degrees'], parent['coord'], rand_coords, agent)

            next_node = rand_coords
            # check travel time, within track layout
            if TRAVELT > self.max_time or self.cone_check(parent['coord'], next_node) or \
                    self.vision_check(agent, next_node) or self.counter_productive_check(parent, next_node):
                pass
            else:
                break

        name = "n{}".format(len(self.nodes_list))
        parent_name = parent['name']
        cost = parent['cost'] + calc_dist(parent['coord'], next_node)
        travel_t, degrees = calc_travel_time(parent['degrees'], parent['coord'], rand_coords, agent)
        self.nodes_list.append({'name': name, 'coord': np.array(next_node), 'parent': parent_name, 'degrees': degrees, 'cost': cost, 'time':travel_t})
        self.draw_point(next_node)
        return next_node, name, degrees, cost
    # helps ensure constance progression (more relative node positions)
    def counter_productive_check(self, parent_node,current_node):
        paren_parent_node = None
        if parent_node['name'] != 'n0':
            paren_parent_node = [i for i in self.nodes_list if i['name'] == parent_node['parent']]
        if paren_parent_node:
            if paren_parent_node[0]['name'] != 'n0':
                par_paren_parent_node = [i for i in self.nodes_list if i['name'] == paren_parent_node[0]['parent']]

                if(par_paren_parent_node[0]['coord'][0] < current_node[0] < parent_node['coord'][0] or
                        par_paren_parent_node[0]['coord'][1] < current_node[1] < parent_node['coord'][1] or
                        paren_parent_node[0]['coord'][0] < current_node[0] < parent_node['coord'][0] or
                        paren_parent_node[0]['coord'][1] < current_node[1] < parent_node['coord'][1]):
                    return True

        return False
    # ensures path between previous node and next node does not intersect with nearby cones
    def cone_check(self, node, next_node):
        cones_relevant = [], []
        for red in self.cones[0]:
            if ((node[0] - 50) < red.position[0][0] < (node[0] + 50) and (node[1] - 50) < red.position[1][1] < (
                    node[1] + 50)) or (
                    (next_node[0] - 50) < red.position[0][0] < (next_node[0] + 50) and (next_node[1] - 50) <
                    red.position[1][1] < (
                            next_node[1] + 50)):
                cones_relevant[0].append([red.position[0][0], red.position[1][1]])
        for blue in self.cones[1]:
            if (node[0] - 50) < blue.position[0][0] < (node[0] + 50) and (node[1] - 50) < blue.position[1][1] < (
                    node[1] + 50) or (
                    (next_node[0] - 50) < blue.position[0][0] < (next_node[0] + 50) and (next_node[1] - 50) <
                    blue.position[1][1] < (
                            next_node[1] + 50)):
                cones_relevant[1].append([blue.position[0][0], blue.position[1][1]])

        for i in range(len(cones_relevant[0])):
            if i + 1 != len(cones_relevant[0]):
                if intersect(cones_relevant[0][i], cones_relevant[0][i + 1], node, next_node):
                    return True
        for i in range(len(cones_relevant[1])):
            if i + 1 != len(cones_relevant[1]):
                if intersect(cones_relevant[1][i], cones_relevant[1][i + 1], node, next_node):
                    return True
        return False

    '''field of view check: makes sure that if within view distance the node is within FOV
    plan is to adapt this in order to allow path planning during motion without map knowledge'''
    def vision_check(self, agent, next_node):
        cos_rot = math.cos(math.radians(agent.pos[2][2]))

        m = math.atan2(next_node[1] - agent.pos[1][1], next_node[0] - agent.pos[0][0])
        m2 = math.atan2(agent.pos[1][1] - next_node[1], agent.pos[0][0] - next_node[0])

        dis_to_den = [agent.pos[0][0] - next_node[0], agent.pos[1][1] - next_node[1]]

        if agent.visual_dist >= abs(dis_to_den[0]) and agent.visual_dist >= abs(dis_to_den[1]):
            if (agent.left_angle[0] >= m >= agent.right_angle[0] and cos_rot >= 0) or (
                    agent.left_angle[1] > m2 > agent.right_angle[1] and cos_rot <= 0):
                return False
            else:
                if -agent.visual_dist <= dis_to_den[0] and next_node[0] < (agent.pos[0][0] - self.max_dist) and cos_rot >= 0.8:
                    return False
                return True
        return False
    # find travel route from goal to start
    def find_path(self):
        get_nodes = []
        current = self.nodes_list[-1]  # set end as current node
        get_nodes.append(current)  # append end coordinates to list of path nodes
        for _, j in reversed(list(enumerate(self.nodes_list))):
            if current['parent'] == j['name']:
                get_nodes.append(j)
                current = j
        return get_nodes

    def draw_point(self,pos):
        pygame.draw.circle(self.screen, [0, 255, 0], pos, 3, 3)
        pygame.display.flip()
    # load previously found route
    def load_path(self, file_name):
        csv_data = pd.read_csv(file_name)
        csv_data = csv_data.iloc[:, :-1]

        path_list = csv_data.values.tolist()

        for i in path_list:
            coord = [i[1], i[2]]
            print(coord)
            time = i[6].replace('}', '')
            name = i[0].replace('{', '')
            self.path_nodes.append(
                {'name': name, 'coord': coord, 'parent': i[3], 'degrees': i[4], 'cost': i[5], 'time': float(time)})
