import math

import numpy as np

class ekf_slam:
    def __init__(self, num_objects, objects, starting_pos):
        self.matrix_size = (num_objects+3)
        self.prev_mew = np.zeros((self.matrix_size, 1))
        #vector of same len as landmarks and covariance
        self.mew = 0
        self.est_mew = np.zeros((self.matrix_size, 1))
        '''self.est_mew[0] = starting_pos[0]
        self.prev_mew[0] = starting_pos[0]
        self.est_mew[1] = starting_pos[1]
        self.prev_mew[1] = starting_pos[1]
        self.est_mew[2] = starting_pos[2]
        self.prev_mew[2] = starting_pos[2]'''
        self.covariance =np.eye(self.matrix_size)
        #vector of same shape as landmarks
        self.velocity = []
        self.measurement = [0,0]
        #self.observations = 0
        #self.landmarks = [j for i in objects for j in i]
        #self.landmarks = [i for i in objects]
        self.tick_tock = 1
        # 3 types of landmarks but many of them
        self.initialize_f()
        self.seen_landmarks = []
        #self.system_variance = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.001]])
        self.system_variance = np.diag([0.5, 0.5, np.deg2rad(30.0)]) ** 2
        self.talor_exp = []
        # line 6 in probabilistic robotics
        # assuming accurate camera
        self.sensorReading = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def update(self): # mu, covariance, movement, measurement
        #self.est_mew = self.prev_mew
        ''' The main bug seems to be somewhere within the functions of next two lines '''
        talor_exp, fx = self.jacobian_motion()
        self.est_mew[0:3] = self.motion_model(self.prev_mew[0:3])
        print("position: ",self.est_mew[0:3])
        # line 5 in probabilistic robotics
        self.covariance[0:3, 0:3] = np.transpose(talor_exp) @ self.covariance[0:3, 0:3] @ talor_exp + np.transpose(fx) @ self.system_variance @ fx

        self.Fk = np.zeros((self.matrix_size, self.matrix_size))  # 3 times the number of landmarks
        self.Fk[0][0] = 1
        self.Fk[1][1] = 1
        self.Fk[2][2] = 1

        # line 7 in probabilistic robotics
        matrixk = []
        matrixh = []
        matrix_z_est = []
        for item in self.seen_landmarks:
            # line 8 in probabilistic robotics
            j = item[0]
            kp = j
            # line 9 in probabilistic robotics
            #print("kp_value: ", kp)
            if self.est_mew[kp][0] == 0:
                # line 10 in probabilistic robotics
                self.measurement[0] = item[-2]
                self.measurement[1] = item[-1]
                pos = self.calc_ladmark_pos()
                #print('yay new landmark: ', pos)
                self.est_mew[kp] = pos[0]
                self.est_mew[kp + 1] = pos[1]
                self.est_mew[kp + 2] = 0
            # line 12 in probabilistic robotics
            delta = [self.est_mew[kp] - self.est_mew[0], self.est_mew[kp + 1] - self.est_mew[1]]

            H, K, z_est = self.calculations(delta, kp)
            zx = self.est_mew[kp] - z_est[0][0]
            zy = self.est_mew[kp+1] - z_est[0][1]
            zt = self.est_mew[kp+2]
            z = [[float(zx), float(zy), float(zt)]]
            matrixk.append(K)
            matrix_z_est.append(z)
            matrixh.append(H)
            # line 19 in probabilistic robotics
            self.est_mew = self.est_mew + K.dot(np.transpose(z))
            # line 20 in probabilistic robotics
            self.covariance = (np.identity(len(self.est_mew)) - (K@ H)) @ self.covariance

    # line 2 in probabilistic robotics
    def initialize_f(self):
        self.f = np.zeros((3, self.matrix_size))
        self.f[0][0] = 1
        self.f[1][1] = 1
        self.f[2][2] = 1

    # line 4 in probabilistic robotics
    def jacobian_motion(self):
        #self.ft = np.transpose(self.f[0:2])
        fx = [[1,0,0],[0,1,0],[0,0,1]]

        #print(self.velocity)
        '''jmF = np.array(
            [[0, 0,  self.tick_tock * self.velocity[0][0] * math.sin(self.prev_mew[2][0])],
             [0, 0, self.tick_tock * self.velocity[0][0] * math.cos(self.prev_mew[2][0])],
             [0, 0, 0]], dtype=float)'''

        if self.velocity[0][0] == 0:
            vel = self.velocity[1][0]
        elif self.velocity[1][0] == 0:
            vel = self.velocity[0][0]
        else:
            vel = (self.velocity[0][0] / self.velocity[1][0])

        jmF = np.array(
            [[0, 0, ((vel * math.cos(self.prev_mew[2][0])) - (vel * math.cos(
                self.prev_mew[2][0] + self.velocity[1][0] * self.tick_tock)))],
             [0, 0, ((vel * math.sin(self.prev_mew[2][0])) - (vel * math.sin(
                 self.prev_mew[2][0] + self.velocity[1][0] * self.tick_tock)))],
             [0, 0, 0]], dtype=float)
        '''
        jmF = np.array(
            [[0, 0, -self.tick_tock * self.velocity[0][0] * math.sin(self.prev_mew[2][0])],
             [0, 0, self.tick_tock * self.velocity[0][0] * math.cos(self.prev_mew[2][0])],
             [0, 0, 0]], dtype=float)'''
        jmF = np.where(np.isnan(jmF), 0, jmF)
        #self.G = self.ft @ jmF.T
        talor_exp = np.eye(3) + np.transpose(fx) @ jmF @ fx
        return talor_exp, fx

    # line 3 in probabilistic robotics
    def motion_model(self, x):
        tempF = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])
        if self.velocity[0][0] == 0:
            vel = self.velocity[1][0]
        elif self.velocity[1][0] == 0:
            vel = self.velocity[0][0]
        else:
            vel = self.velocity[0][0] / self.velocity[1][0]
        temp = np.array([[((-vel * math.sin(self.prev_mew[2][0])) + (vel * math.sin(
                self.prev_mew[2][0] + self.velocity[1][0] * self.tick_tock))),
                         ((vel * math.cos(self.prev_mew[2][0])) - (vel * math.cos(
                self.prev_mew[2][0] + self.velocity[1][0] * self.tick_tock))),
                         self.velocity[1][0] * self.tick_tock]])
        '''temp = np.array([[self.tick_tock * math.cos(x[2][0]),0],
                         [self.tick_tock * math.sin(x[2][0]),0],
                         [0.0, self.tick_tock]])'''
        x = (np.transpose(tempF) @ temp.T)
        #x = x + (tempF @ temp)
        #x = temp + (tempF @ x)
        return x

    def calc_ladmark_pos(self):
        pos = [0, 0]
        pos[0] = self.est_mew[0][0] + self.measurement[0] * math.cos(self.est_mew[2][0]) + self.measurement[1]
        pos[1] = self.est_mew[1][0] + self.measurement[0] * math.sin(self.est_mew[2][0]) + self.measurement[1]

        return pos

    def calculations(self, delta, kp):
        # line 13 in probabilistic robotics
        q = (np.transpose(delta) @ delta)[0, 0]

        # line 14 in probabilistic robotics
        z_est = [[math.sqrt(q), (math.atan2(delta[1][0], delta[0][0]) - self.est_mew[2][0])]]

        self.Fk[kp][kp] = 1
        self.Fk[kp + 1][kp + 1] = 1
        self.Fk[kp + 2][kp + 2] = 1

        h = self.H_calc(delta, q, kp)
        # line 17 in probabilistic robotics K = (PEst @ H.T) @ np.linalg.inv(S)
        #S = H @ PEst @ H.T + Cx[0:2, 0:2]

        k = (self.covariance @ np.transpose(h)) @ np.linalg.inv((h @ self.covariance @ np.transpose(h) + self.system_variance))
        return h, k, z_est

    def H_calc(self, delta, q, kp):
        # line 15 in probabilistic robotics
        f_temp = np.concatenate((self.Fk[0:3], self.Fk[kp:kp + 3]), axis=0)

        qsqrt = math.sqrt(q)

        grape = np.array([[-qsqrt * delta[0][0], -qsqrt * delta[1][0], 0, qsqrt * delta[0][0], qsqrt * delta[1][0],0],
                          [delta[1][0], -delta[0][0], -q, -delta[1][0], delta[0][0],0], [0, 0, 0, 0, 0, 1]])

        # line 16 in probabilistic robotics
        try:
            return 1/q * (grape @ f_temp)
        except:
            return 0 * (grape @ f_temp)
