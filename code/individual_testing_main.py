import sys, pygame
import track_builder as tb
import numpy as np
import math
import agent
from path_planning import RRT_method
from Simulation.EKF_SLAM.ekf import ekf_slam
from data_management import *
from Simulation.SNN.snn import localisation_SNN

'''Primary function: this functions brings all processes of the system together'''
#@measure_energy
def main():
    # time relative variables
    prev_time = 0
    prev_time_move = 0
    clock = pygame.time.Clock()
    tick_tock = 1  # seconds
    time = 0
    FPS = 60

    # initialise pygame and display simulation window
    pygame.init()
    window_size = 1200, 600
    bg_colour=[255,255,255]
    screen = pygame.display.set_mode(window_size)

    # initialise track and provide track id, build that track
    track = tb.TrackBuilder(1)
    track.build_track()

    # agents visual specifications
    visual_dist = 100
    visual_deg = math.radians(90)
    '''initialise agent, provide starting position & direction, visual specifications, forward speed, rotational speed 
    and steps ahead to consider in path following'''
    agent_car = agent.Agent(track.start_pos, track.start_dir, track.start_deg, visual_dist, visual_deg, 4, 3, 3)
    # calculate draw positions for agent display of field of view
    left, right, end = agent_car.rotate_camera()
    agent_car.left_angle[0] = math.atan2(left[1] - agent_car.pos[1][1], left[0] - agent_car.pos[0][0])
    agent_car.right_angle[0] = math.atan2(right[1] - agent_car.pos[1][1], right[0] - agent_car.pos[0][0])
    agent_car.left_angle[1] = math.atan2(agent_car.pos[1][1] - left[1], agent_car.pos[0][0] - left[0])
    agent_car.right_angle[1] = math.atan2(agent_car.pos[1][1] - right[1], agent_car.pos[0][0] - right[0])

    # manual control variables
    speed_input = [False, False]
    direction = [False, False]
    plus_speed = 1
    manual = False

    # autonomouse activation and path planning control
    run_path = True
    load_the_path = True
    display_planner = True
    store_training_data = False

    # Run EKF or SNN SLAM
    run_ekf_slam = False
    run_snn_slam = True

    # simulation control variables
    num_laps = 1
    start_movement = True
    end_sim = False
    crash = False

    # initialise path planner
    path_planner = RRT_method(track.start_pos, track.end_pos, screen, track.track, 40, 0, 20)

    # other miscellaneous variables
    log_location_training_data = []
    testing_landmark_calcs = []
    re_cal_counter = 0
    re_calibrate = False
    lap_counter = 0

    plot_environment(track.track, track.name)
    # either load a previous route or find a new route
    if load_the_path:
        if track.name == "square":
            path_planner.load_path('map_paths/square_active.csv')
        if track.name == "square_curved_corners":
            path_planner.load_path('map_paths/square_curved_corners_active.csv')
        if track.name == "circle":
            path_planner.load_path('map_paths/circle_active.csv')
        if track.name == "test_straight":
            path_planner.load_path('map_paths/test_straight.csv')
        agent_car.route = path_planner.path_nodes
    elif run_path:
        path_planner.tree_gen(agent_car)
        agent_car.route = path_planner.path_nodes
        log_tree(path_planner, "Time-attempt-1", track.name)

    # initialise EKF or SNN SLAM
    if run_ekf_slam:
        ekfstart_pos = [track.start_pos[0], track.start_pos[1], 0]
        slam_object = ekf_slam(track.num_objects, track.track, ekfstart_pos)
        ekf_time_tracker = []
        x_ekf_pos = 0
        y_ekf_pos = 0
        theta_ekf_pos = 0
    elif run_snn_slam:
        snn_time_tracker = []
        snn_local_object = localisation_SNN()
        snn_local_object.initialize_snn()
        snn_results = []

    #get simulation start time
    start_date_time = datetime.now()
    print("timestamp: ", start_date_time)
    while True:
        clock.tick(FPS)

        #allows manual control if enable, else active motion or end simulation early
        if manual:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
                if event.type==pygame.KEYDOWN:
                    if event.key==pygame.K_w:
                        speed_input[0] = True
                    if event.key == pygame.K_s:
                        speed_input[1] = True
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_w:
                        speed_input[0] = False
                    if event.key == pygame.K_s:
                        speed_input[1] = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        direction[1] = True
                    if event.key == pygame.K_d:
                        direction[0] = True
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_a:
                        direction[1] = False
                    if event.key == pygame.K_d:
                        direction[0] = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        end_sim = True
            if speed_input[0] and agent_car.velocity[0] < agent_car.max_speed: agent_car.velocity[0] += plus_speed
            if speed_input[1] and agent_car.velocity[0] < -agent_car.max_speed: agent_car.velocity[0] -= plus_speed
            if direction[0] and agent_car.velocity[1] < agent_car.max_rot_speed: agent_car.velocity[0] += plus_speed
            if direction[1] and agent_car.velocity[1] < -agent_car.max_rot_speed: agent_car.velocity[0] -= plus_speed


            if not speed_input[0] and not speed_input[1] and agent_car.velocity[0] > 0:
                agent_car.velocity[0] -= plus_speed
            if not speed_input[0] and not speed_input[1] and agent_car.velocity[0] < 0:
                agent_car.velocity[0] += plus_speed
        else:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        start_movement = True
                    if event.key == pygame.K_d:
                        end_sim = True

        # draw screen
        screen.fill(bg_colour)
        screen.blit(agent_car.rotated_agent, agent_car.rotated_agent.get_rect(center=[agent_car.pos[0][0], agent_car.pos[1][1]]))

        # get FOV coordinates and find possibly detected objects
        left, right, end = agent_car.rotate_camera()
        detected_cones = []
        new_landmarks = []
        new_landmarks_temp = []
        for item in track.track:
            item, detected, L = agent_car.detection(left, right, item)
            item.draw(screen)
            if L:
                new_landmarks.append(L)
                new_landmarks_temp.append([item.position[0][0], item.position[1][1]])
            else:
                detected_cones.append(item)

        '''if autonomous calculate and enact motion, if no landmarks are spotted provide warning and stop movement 
        else enact manuel motion input'''
        if run_path:
            if detected_cones or re_calibrate:
                re_cal_counter = 0
                if start_movement:
                    #crash, cone = agent_car.collision_check(detected_cones, 1) #(time-prev_time_move)
                    if agent_car.velocity[0] < agent_car.max_speed: agent_car.velocity[0] += plus_speed
                    if agent_car.velocity[1] < agent_car.max_rot_speed: agent_car.velocity[1] += plus_speed

                    if (agent_car.travel_time <= 0 and agent_car.rot_time <= 0):
                        delta = agent_car.path_tracking_ppc()
                    elif agent_car.lastnode_ind == 0:
                        delta = agent_car.path_tracking_ppc(False, True)
                        if (agent_car.travel_time > 0 and agent_car.rot_time > 0):
                            agent_car.autonomous_movement(1, delta)
                        else:
                            lap_counter += 1
                            print("LAP: ", lap_counter)
                        agent_car.travel_time -= 1
                        agent_car.rot_time -= 1
                        prev_time_move = time
                    elif crash:
                        delta = agent_car.path_tracking_ppc(True)
                        if (agent_car.travel_time > 0 and agent_car.rot_time > 0):
                            agent_car.autonomous_movement(1, delta)
                        else:
                            crash = False
                        agent_car.travel_time -= 1
                        agent_car.rot_time -= 1
                        prev_time_move = time
                    else:
                        agent_car.autonomous_movement(1, delta)
                        agent_car.travel_time -= 1
                        agent_car.rot_time -= 1
                        prev_time_move = time

                else:
                    prev_time_move = time
            else:
                print("no cones so stop!")
                re_cal_counter += 1

            if display_planner:
                #for i in range(len(path_planner.nodes_list)):
                #pygame.draw.circle(screen, [0, 255, 0], path_planner.nodes_list[i]['coord'], 3, 3)
                for i in range(len(path_planner.path_nodes)):
                    pygame.draw.circle(screen, [128, 0, 128], path_planner.path_nodes[i]['coord'], 3, 3)
        else:
            agent_car.rotate(direction)
            agent_car.move()

        ''' if ekf SLAM  run EKF update function and print results !!currently doesnt work correctly and needs repair'''
        if run_ekf_slam:
            ekf_tic = datetime.now()
            slam_object.velocity = np.array([[agent_car.velocity[0]], [agent_car.velocity[1]]])
            for l in new_landmarks:
                skip = False
                for sl in slam_object.seen_landmarks:
                    if sl[0] == l[0]:
                        skip = True
                if not skip:
                    slam_object.seen_landmarks.append(l)
            slam_object.update()
            slam_object.prev_mew = slam_object.est_mew
            x_ekf_pos += slam_object.prev_mew[0][0]
            y_ekf_pos += slam_object.prev_mew[1][0]
            theta_ekf_pos += slam_object.prev_mew[2][0]

            print("x: " + str(x_ekf_pos) + " , x pos: " + str(agent_car.pos[0][0]) + ", x ekf pos: " + str(
                slam_object.prev_mew[0][0]) + "")
            print("y: " + str(y_ekf_pos) + " , y pos: " + str(agent_car.pos[1][1]) + ", y ekf pos: " + str(
                slam_object.prev_mew[1][0]) + "")
            print("theta: " + str(theta_ekf_pos) + " , theta pos: " + str(
                agent_car.pos[2][2]) + ", theta ekf pos: " + str(slam_object.prev_mew[2][0]) + "")
            ekf_toc = datetime.now()
            ekf_time = ekf_tic - ekf_toc
            ekf_time_sec = ekf_time.total_seconds()
            ekf_time_tracker.append(ekf_time_sec)

        #gether data for SNN SLAM input or storing
        temp_pos = [agent_car.pos[0][0].copy(), agent_car.pos[1][1].copy(), agent_car.pos[2][2].copy()]
        data_line = [time, new_landmarks.copy(), agent_car.velocity.copy(), temp_pos]
        # if SNN SLAM prepare input spikes, if store data apply to data array
        if run_snn_slam:
            snn_local_object.spike_gen(data_line)
            snn_local_object.spike_gen2(data_line)
        if store_training_data:
            log_location_training_data.append(data_line.copy())
            data_line2 = [time, new_landmarks.copy(), new_landmarks_temp.copy(), temp_pos]
            testing_landmark_calcs.append(data_line2)

        ''' if SNN SLAM run to find next position and store output data, also provides tracking of SNN runtime 
        during each time step if required
        improvement: provide previous SNN results'''
        if run_snn_slam:
            #snn_tic = datetime.now()
            snn_return = snn_local_object.run_snn_loc(data_line)
            snn_results.append([time, snn_return])
            #snn_toc = datetime.now()
            #snn_time = snn_tic - snn_toc
            #snn_time_sec = snn_time.total_seconds()
            #print("snn run-time: ", snn_time_sec)
            #snn_time_tracker.append(snn_time_sec)

        '''if end sim or laps completed store run time with test specifications then plot any results or store required data
        currently due to EKF failure EKF results are not plotted'''
        if end_sim or lap_counter == num_laps:
            runtime = datetime.now() - start_date_time
            runtime_sec = runtime.total_seconds()
            date = save_date()
            f = open("log_data/test_results/" + date + "_" + track.name + "_test_log_SNN_" + str(
                run_snn_slam) + "_EKF_" + str(run_ekf_slam) + ".txt", "w+")
            f.write("# test details" + "\n")
            f.write("start: " + str(start_date_time) + "\n")
            f.write("end: " + str(runtime) + "\n")
            f.write('runtime sec: ' + str(runtime_sec) + "\n")
            f.write('track: ' + str(track.name) + "\n")
            f.write('laps: ' + str(lap_counter) + "\n")
            f.write('EKF: ' + str(run_ekf_slam) + "\n")
            f.write('SNN: ' + str(run_snn_slam) + "\n")
            f.close()
            if store_training_data:
                save_location_training_data(log_location_training_data, "SNN_training_data")
                log_land_det(testing_landmark_calcs, "calc_data")
            if run_snn_slam:
                pass
                #snn_local_object.end_run(track.track, ""+track.name+"-"+str(lap_counter)+"LAPS-loc-",snn_results, str(track.name))
                #snn_map_object.end_run(track.track, ""+track.name+"-"+str(lap_counter)+"LAPS-map-",snn_results, str(track.name))
                #save_snn_results(snn_results, track.name)

            break

        # plot route
        if left:
            pygame.draw.line(screen, (200,200,200), [agent_car.pos[0][0], agent_car.pos[1][1]], left)
            pygame.draw.line(screen, (200,200,200), [agent_car.pos[0][0], agent_car.pos[1][1]], right)
            pygame.draw.line(screen, (200,200,200), left, end)
            pygame.draw.line(screen, (200,200,200), right, end)

        #draw screen and update timestep
        prev_time = time
        time += tick_tock
        pygame.draw.rect(screen,[0,0,0],(agent_car.pos[0][0], agent_car.pos[1][1],8,8))
        pygame.display.flip()

main()