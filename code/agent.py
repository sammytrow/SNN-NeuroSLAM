import pygame
import math
import numpy as np
from path_planning import calc_travel_time, calc_dist

# generated noise, could be improved by applying forward and angular dependence
def add_gaussian_noise(input,std=10, dist = 0, isx = False):
    mu = np.mean(input)
    noise = np.random.normal(mu, std, size = np.array(input).shape)
    #sp = np.mean(x ** 2)  # Signal Power
    #need to set snr by x or y distance
    snr_values = [0.95,]
    #for snr in snr_values:
    #    std_n = (sp / snr) ** 0.5  # Noise std. deviation
    return noise

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
def calc_dist(point1, point2):
    return math.sqrt((point1[0] - point2[0]) * (point1[0] - point2[0]) + (point1[1] - point2[1]) * (point1[1] - point2[1]))

def calc_vision_cone(endv, pos, angle):
    endv = list(endv)
    endv[0] -= pos[0]
    endv[1] -= pos[1]

    calcx = endv[0] * math.cos(angle) - endv[1] * math.sin(angle)
    calcy = endv[0] * math.sin(angle) + endv[1] * math.cos(angle)

    endv[0] = calcx + pos[0]
    endv[1] = calcy + pos[1]
    return endv

class Agent:
    def __init__(self, position, dir, deg, vis_dist, vis_deg, max_s, max_rot_s, ppcn):
        self.agent = pygame.Surface((8,4), pygame.SRCALPHA)
        self.agent.fill((0,0,0))
        self.exact_pos = np.array([[position[0], 0, 0], [0, position[1], 0], [0, 0, 0]])
        self.pos = np.array([[position[0], 0, 0], [0, position[1], 0], [0, 0, 0]])
        self.rotation = 0
        self.rotated_agent = self.agent
        self.velocity = [0, 0]
        self.no_noise_velocity = [0,0]
        self.max_speed = max_s
        self.max_rot_speed = max_rot_s
        self.start_direction = dir
        self.start_dir_deg = deg
        self.orientation = 0
        self.vision_block = [0, 0]
        self.left_angle = [0, 0]
        self.right_angle = [0, 0]
        self.visual_dist = vis_dist
        self.visual_deg = vis_deg
        self.route = []
        self.lastnode_ind = None
        self.ppc_number_nodes = ppcn
        self.travel_time = 0
        self.rot_time = 0
        self.landmarks = []
        self.seen_landmarks = [],[],[]
        self.prev_pos = np.array([[position[0], 0, 0], [0, position[1], 0], [0, 0, 0]])
        self.avoidance_time = 0
        self.laps_completed = 0
    # calculate the new position of the agent
    def move(self):
        self.prev_pos = self.pos
        new_x = self.pos[0][0] + (self.velocity[0] * math.cos(math.radians(self.pos[2][2])))
        new_y = self.pos[1][1] + (self.velocity[0] * math.sin(math.radians(self.pos[2][2])))
        self.pos[0][0] = new_x
        self.pos[1][1] = new_y
    # calculate new rotation of agent
    def rotate(self, direction):
        if direction[0]:
            self.pos[2][2] += self.velocity[1]
        elif direction[1]:
            self.pos[2][2] -= self.velocity[1]
        self.rotated_agent = pygame.transform.rotate(self.agent, float(self.pos[2][2]))
    # rotate simulated FOV
    def rotate_camera(self):
        pos = np.array([self.pos[0][0], self.pos[1][1]])
        new_dir = np.array(pos) + np.array(self.start_direction)
        new_dir = calc_vision_cone(new_dir, pos, math.radians(self.pos[2][2]))
        new_dir = np.array(new_dir)
        new_dir -= pos
        self.orientation = new_dir
        end = np.array(pos) + np.array(new_dir) * self.visual_dist
        left = calc_vision_cone(end, pos, self.visual_deg / 2)
        right = calc_vision_cone(end, pos, -self.visual_deg / 2)
        return left, right, end
    # detect objects within agent FOV
    def detection(self, left, right, item):
        detected = False

        self.left_angle[0] = math.atan2(left[1] - self.pos[1][1], left[0] - self.pos[0][0])
        self.right_angle[0] = math.atan2(right[1] - self.pos[1][1], right[0] - self.pos[0][0])
        self.left_angle[1] = math.atan2(self.pos[1][1] - left[1], self.pos[0][0] - left[0])
        self.right_angle[1] = math.atan2(self.pos[1][1] - right[1], self.pos[0][0] - right[0])

        cos_rot = math.cos(math.radians(self.pos[2][2]))

        item_pos = [add_gaussian_noise(item.position[0][0], 3, (self.pos[0][0]-item.position[0][0]), True), add_gaussian_noise(item.position[1][1], 3, (self.pos[1][1]-item.position[1][1])), 0]
        #simple bug fix, may no longer need this
        try:
            mb = math.atan2(item_pos[1] - self.pos[1][1], item_pos[0] - self.pos[0][0])
            m2b = math.atan2(self.pos[1][1] - item_pos[1], self.pos[0][0] - item_pos[0])
            m = math.atan2(item_pos[1] - self.pos[1][1], item_pos[0] - self.pos[0][0])
            m2 = math.atan2(self.pos[1][1] - item_pos[1], self.pos[0][0] - item_pos[0])
            skip = False
        except:
            skip = True
        cone_det = []
        if not skip:
            self.vision_block[0] = np.array([self.pos[0][0]-self.visual_dist, self.pos[1][1]-self.visual_dist, 0])
            self.vision_block[1] = np.array([self.pos[0][0]+self.visual_dist, self.pos[1][1]+self.visual_dist, 0])

            '''
                if object is within FOV cone change object colour to green to indicate detection and 
                return detected item details 
            '''
            if self.vision_block[0][0] < item_pos[0] < self.vision_block[1][0] and self.vision_block[0][1] < item_pos[1] < self.vision_block[1][1] \
                and self.left_angle[0] >= m >= self.right_angle[0] and self.left_angle[0] >= mb >= self.right_angle[0] and cos_rot >= 0 or\
                self.vision_block[0][0] < item_pos[0] < self.vision_block[1][0] and self.vision_block[0][1] < item_pos[1] < self.vision_block[1][1] \
                and self.left_angle[1] > m2 > self.right_angle[1] and self.left_angle[1] > m2b > self.right_angle[1] and cos_rot <= 0:
                item.colour = [0, 255, 0]
                detected = True

                est_FOV_pos = self.FOV_pos(item_pos)
                if 0 > m >= self.right_angle[0] and 0 > mb >= self.right_angle[0] and cos_rot >= 0 or\
                0 > m2 > self.right_angle[1] and 0 > m2b > self.right_angle[1] and cos_rot <= 0:
                    est_FOV_pos[1] = -est_FOV_pos[1]

                dist = calc_dist([self.pos[0][0], self.pos[1][1]], item_pos)
                angle = math.degrees(math.atan2(-(item_pos[0] - self.pos[0][1]), item_pos[1] - self.pos[1][1])) - self.pos[2][2]
                cone_det = [item.id, item.type, est_FOV_pos[0], est_FOV_pos[1]]

                self.seen_landmarks[item.type].append({'id': item.id, 'pos': item_pos, 'type':item.type, 'dist' : dist, 'angle' : angle})
            elif item.type == 0 and item.colour == [0, 255, 0]:
                item.colour = [255, 0, 0]
            elif item.type == 1 and item.colour == [0, 255, 0]:
                item.colour = [0, 0, 255]
            elif item.type == 2 and item.colour == [0, 255, 0]:
                item.colour = [211, 211, 211]

        return item, detected, cone_det
    # calculate estimated x and y grid distance from agent to object, estimated coordinates
    def FOV_pos(self, item_pos):
        pos = np.array([self.pos[0][0], self.pos[1][1]])
        new_dir = np.array(pos) + np.array(self.start_direction)
        new_dir = calc_vision_cone(new_dir, pos, math.radians(self.pos[2][2]))
        new_dir = np.array(new_dir)
        new_dir -= pos

        self.orientation = new_dir

        baseline_dist = self.visual_dist+50
        end = np.array(pos) + np.array(new_dir) * baseline_dist
        agent_to_item_dist = abs(calc_dist(pos, item_pos))
        end_to_item_dist = abs(calc_dist(end, item_pos))

        h = 0.5 * math.sqrt((agent_to_item_dist + baseline_dist + end_to_item_dist) * (-agent_to_item_dist + baseline_dist + end_to_item_dist)\
            * (agent_to_item_dist - baseline_dist + end_to_item_dist) * (agent_to_item_dist + baseline_dist - end_to_item_dist)) / baseline_dist
        angle = math.asin(h/agent_to_item_dist)
        B = math.cos(angle) * agent_to_item_dist
        return [B, h]

    # checks for objects that the agent may collide with
    def collision_check(self, detected_items, tick_tock):
        next_pos = [0,0]
        if self.travel_time > 0:
            next_pos[0] = self.pos[0][0] + (self.velocity[0] * math.cos(math.radians(self.pos[2][2])) * tick_tock)
            next_pos[1] = self.pos[1][1] + (self.velocity[0] * math.sin(math.radians(self.pos[2][2])) * tick_tock)

            cur_pos = [self.pos[0][0], self.pos[1][1]]
            cones_relevant = [], []
            for red in self.seen_landmarks[0]:
                if ((cur_pos[0] - 50) < red['pos'][0] < (cur_pos[0] + 50) and (cur_pos[1] - 50) < red['pos'][1] < (
                        cur_pos[1] + 50)):
                    cones_relevant[0].append([red['pos'][0], red['pos'][1]])
            for blue in self.seen_landmarks[1]:
                if (cur_pos[0] - 50) < blue['pos'][0] < (cur_pos[0] + 50) and (cur_pos[1] - 50) < blue['pos'][1] < (
                        cur_pos[1] + 50):
                    cones_relevant[1].append([blue['pos'][0], blue['pos'][1]])

            for i in range(len(cones_relevant[0])):
                if i + 1 != len(cones_relevant[0]):
                    if intersect(cones_relevant[0][i], cones_relevant[0][i + 1], cur_pos, next_pos):
                        return True, 0
            for i in range(len(cones_relevant[1])):
                if i + 1 != len(cones_relevant[1]):
                    if intersect(cones_relevant[1][i], cones_relevant[1][i + 1], cur_pos, next_pos):
                        return True, 1

        return False, 0
    # attempt at collision avoidance but doesnt quite work, needs further work
    def collision_avoidance(self, type):
        if type == 0:
            if self.pos[2][2] < 0:
                rot = abs(self.pos[2][2]) + 10
                rot = -rot
            else:
                rot = self.pos[2][2] + 10

        elif type == 1:
            if self.pos[2][2] < 0:
                rot = abs(self.pos[2][2]) - 10
                rot = -rot
            else:
                rot = self.pos[2][2] - 10

        alpha = math.atan2(- self.pos[1][1], - self.pos[0][0]) - math.radians(rot)

        delta = math.atan2(2.0 * 1 * math.sin(alpha) / 100, 1.0)

        return delta
    # calculates next position using velocity and travel time for autonomous movement
    def autonomous_movement(self, tick_tock, delta, rotate_first = False, type = 0):
        self.prev_pos = self.pos

        if self.travel_time > 0 and not rotate_first:
            self.pos[0][0] += self.velocity[0] * math.cos(math.radians(self.pos[2][2])) * tick_tock
            self.pos[1][1] += self.velocity[0] * math.sin(math.radians(self.pos[2][2])) * tick_tock
        else:
            self.velocity[0] = 0
        # another attempt to avoid going off track
        if rotate_first:
            if self.avoidance_time <=0:
                delta = self.collision_avoidance(type)
                self.avoidance_time = 3
            else:
                self.avoidance_time -= 1
                self.pos[2][2] += math.degrees(self.velocity[1] / 1 * math.tan(delta) * tick_tock)
        elif self.rot_time > 0:
            self.pos[2][2] += math.degrees(self.velocity[1] / 1 * math.tan(delta) * tick_tock)
        else:
            self.velocity[1] = 0

    ''' calculates forward and rotational travel time between path nodes,
    also when a lap is complete it calculates the forward and rotational travel time to return to starting position'''
    def path_tracking_ppc(self, crash=False, reset=False):
        if reset:
            alpha = math.atan2(self.route[-1]['coord'][1] - self.pos[1][1],
                               self.route[-1]['coord'][0] - self.pos[0][0]) - math.radians(self.pos[2][2])

            delta = math.atan2(2.0 * 1 * math.sin(alpha) / 100, 1.0)
            time = 0
            rot_time = 0

            t, rt = calc_travel_time((self.pos[2][2] + self.start_dir_deg),
                                    [self.pos[0][0], self.pos[1][1]], self.route[-1]['coord'],
                                     self,
                                     False, True)
            time += t
            rot_time += rt
            if (self.route[-1]['coord'][0] - 30) < self.pos[0][0] < (self.route[-1]['coord'][0] + 30) and\
                (self.route[-1]['coord'][1] - 30) < self.pos[1][1] < (self.route[-1]['coord'][1] + 30):
                self.lastnode_ind = len(self.route)-1
                self.travel_time = 0
                self.rot_time = 0
            else:
                self.travel_time = time
                self.rot_time = rot_time
            return delta
        else:
            if crash:
                closest_node = [1000, 1000, 0]
                for i in range(0, len(self.route)):
                    disx = self.pos[0][0] - self.route[i]['coord'][0]
                    disy = self.pos[1][1] - self.route[i]['coord'][1]

                    if closest_node[0] > abs(disx) and closest_node[1] > abs(disy) and i > self.lastnode_ind:
                        closest_node = [disx, disy, i]

                reset_node = closest_node[2]
                alpha = math.atan2(self.route[reset_node]['coord'][1] - self.pos[1][1],
                                   self.route[reset_node]['coord'][0] - self.pos[0][0]) - math.radians(self.pos[2][2])

                delta = math.atan2(2.0 * 1 * math.sin(alpha) / 100, 1.0)
                time = 0
                rot_time = 0

                t, rt = calc_travel_time((self.pos[2][2] + self.start_dir_deg),
                                         [self.pos[0][0], self.pos[1][1]], self.route[reset_node]['coord'],
                                         self,
                                         False, True)
                time += t
                rot_time += rt
                if (self.route[reset_node]['coord'][0] - 30) < self.pos[0][0] < (self.route[reset_node]['coord'][0] + 30) and \
                        (self.route[reset_node]['coord'][1] - 30) < self.pos[1][1] < (self.route[reset_node]['coord'][1] + 30):
                    self.lastnode_ind = len(self.route) - 1
                    self.travel_time = 0
                    self.rot_time = 0
                else:
                    self.travel_time = time
                    self.rot_time = rot_time
                return delta
            else:
                finish_dist = abs(calc_dist([self.pos[0][0], self.pos[1][1]], self.route[0]['coord']))
                if self.lastnode_ind == None or self.lastnode_ind == 0:
                    self.lastnode_ind = len(self.route)-1
                    next_ind = self.lastnode_ind - self.ppc_number_nodes
                elif self.lastnode_ind - self.ppc_number_nodes < 0:
                    self.laps_completed += 1
                    next_ind = 0
                else:
                    next_ind = self.lastnode_ind - self.ppc_number_nodes

            alpha = math.atan2(self.route[next_ind]['coord'][1] - self.pos[1][1],
                               self.route[next_ind]['coord'][0] - self.pos[0][0]) - math.radians(self.pos[2][2])

            delta = math.atan2(2.0 * 1 * math.sin(alpha) / 100, 1.0)
            time = 0
            rot_time = 0

            while self.lastnode_ind != next_ind:
                prev_last_node = self.lastnode_ind
                self.lastnode_ind -= 1
                t, rt = calc_travel_time(self.route[prev_last_node]['degrees'],
                                         self.route[prev_last_node]['coord'], self.route[self.lastnode_ind]['coord'], self,
                                         False, True)
                time += t
                rot_time += rt

            self.travel_time = time
            self.rot_time = rot_time
            return delta