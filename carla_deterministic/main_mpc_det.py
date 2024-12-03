import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
from kernel_computation import kernel_matrix
import sys
sys.path.insert(1, '/home/ims-robotics/Basant/ICRA_RAL_2025/carla_deterministic/optimizer')
from optimizer import cem
import argparse
import jax
sys.path.insert(1, '/home/ims-robotics/CARLA_0.9.13/PythonAPI/carla')
import carla_simulation
import carla_simulation_static
import carla
import pygame
import random as rnd
import math

def compute_lane_bar(prob,y):

    cost_centerline_penalty_lb = -y+ prob.y_lb
    cost_centerline_penalty_ub = y- prob.y_ub
    
    cost_lb = np.maximum(0., cost_centerline_penalty_lb)
    cost_ub = np.maximum(0., cost_centerline_penalty_ub)

    return cost_lb,cost_ub 

def compute_lane_violations(prob,y):
    
    cost_lane_lb,cost_lane_ub = compute_lane_bar(prob,y)

    count_lane_lb = np.count_nonzero(cost_lane_lb)
    count_lane_ub = np.count_nonzero(cost_lane_ub)

    count_lane = count_lane_lb + count_lane_ub
    count_lane = np.count_nonzero(count_lane)

    return count_lane

def apply_control(w,v,prob, csm,target_acc,prev_vel,prev_acc,throttle1,steer):

    physics_control = csm.vehicle.get_physics_control()

    max_steer_angle_list = []
    for wheel in physics_control.wheels:
        max_steer_angle_list.append(wheel.max_steer_angle)
    max_steer_angle = max(max_steer_angle_list)*np.pi/180

    vel = csm.vehicle.get_velocity()
    vel = (vel.x**2 + vel.y**2 + vel.z**2)**0.5

    throttle_lower_border = -(0.01*9.81*physics_control.mass + 0.5*0.3*2.37*1.184*vel**2 + \
        9.81*physics_control.mass*np.sin(csm.vehicle.get_transform().rotation.pitch*2*np.pi/360))/physics_control.mass

    brake_upper_border = throttle_lower_border + -500/physics_control.mass
    csm.pid.setpoint = target_acc

    acc = (vel - prev_vel)/prob.t

    if acc>10:
        control = csm.pid(0)
    else:
        prev_acc = (prev_acc*4 + acc)/5
        control = csm.pid(prev_acc)

    # steer = np.arctan(w*prob.wheel_base/v)
    steer = steer/max_steer_angle
    throttle = 0
    brake = 0

    throttle1 = np.clip(throttle1 + control,-4.0, 4.0)

    if throttle1>throttle_lower_border:
        throttle = (throttle1-throttle_lower_border)/4
        brake = 0
    elif throttle1> brake_upper_border:
        brake = 0
        throttle = 0
    else:
        brake = (brake_upper_border-throttle1)/4
        throttle = 0

    brake = np.clip(brake, 0.0, 1.0)
    throttle = np.clip(throttle, 0.0, 1.0)
    csm.vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake)))
   
    prev_vel = vel
    return prev_vel,prev_acc,throttle1,acc

def compute_obs_data(csm,prob,x_global_init,y_global_init,v_global_init,psi_global_init):
    total_obs = len(csm.obs_list)
    y_obs = np.zeros(total_obs)
    x_obs = np.zeros(total_obs)
    v_obs = 10.0*np.asarray(np.random.uniform(0, 2, total_obs))
    psi_obs = 0.0*np.asarray(np.random.uniform(0, 2, total_obs))
    vx_obs = 10.0*np.asarray(np.random.uniform(0, 2, total_obs))
    vy_obs = 10.0*np.asarray(np.random.uniform(0, 2, total_obs))
    dim_x_obs = np.zeros(total_obs)
    dim_y_obs = np.zeros(total_obs)
    
    k=0
    for j,vehicle in enumerate(csm.obs_list):
        vec1 = (vehicle.get_location().x - x_global_init,vehicle.get_location().y-y_global_init)
        vec2 = v_global_init*np.asarray([np.cos(psi_global_init),np.sin(psi_global_init)])
        theta = np.arccos(np.clip(np.dot(vec1,vec2), -1.0, 1.0))
        if (theta<=5*np.pi/6):
            x_obs[k] = vehicle.get_location().x 
            y_obs[k] = vehicle.get_location().y
            v_obs[k] = np.sqrt(vehicle.get_velocity().x**2 + vehicle.get_velocity().y**2)
            vx_obs[k] = vehicle.get_velocity().x
            vy_obs[k] = vehicle.get_velocity().y
            psi_obs[k] = math.radians(vehicle.get_transform().rotation.yaw)
            psi_obs[k] = np.arctan2(np.sin(psi_obs[k]),np.cos(psi_obs[k]))
            dim_x_obs[k] = 2*vehicle.bounding_box.extent.x 
            dim_y_obs[k] = 2*vehicle.bounding_box.extent.y 
            k=k+1
    
    x_obs = x_obs[0:k]
    y_obs = y_obs[0:k]
    v_obs = v_obs[0:k]
    vx_obs = vx_obs[0:k]
    vy_obs = vy_obs[0:k]
    psi_obs = psi_obs[0:k]
    dim_x_obs = dim_x_obs[0:k]
    dim_y_obs = dim_y_obs[0:k]

    total_obs = np.shape(x_obs)[0]

    if total_obs<prob.num_obs:
        if total_obs==0 or total_obs is None:
            numobs = prob.num_obs
            y_obs = 300*np.ones(numobs)
            x_obs = 300*np.ones(numobs)
            v_obs = 0.0*np.asarray(np.random.uniform(0, 2, numobs))
            vx_obs = np.zeros(numobs)
            vy_obs = np.zeros(numobs)
            psi_obs = 0.0*np.asarray(np.random.uniform(0, 2, numobs))
            dim_x_obs = np.ones(numobs)
            dim_y_obs = np.ones(numobs)
        else:
            for j in range(0,prob.num_obs-total_obs):
                total_obs = np.shape(x_obs)[0]
                x_obs = np.append(x_obs,x_obs[-1])
                y_obs = np.append(y_obs,y_obs[-1])
                vx_obs = np.append(vx_obs,vx_obs[-1])
                vy_obs = np.append(vy_obs,vy_obs[-1])
                v_obs = np.append(v_obs,v_obs[-1])
                psi_obs = np.append(psi_obs,psi_obs[-1])
                dim_x_obs = np.append(dim_x_obs,dim_x_obs[-1])
                dim_y_obs = np.append(dim_y_obs,dim_y_obs[-1])

    total_obs = np.shape(x_obs)[0]
    dist_obs = (x_global_init-x_obs)**2+(y_global_init-y_obs)**2
    idx_sort = np.argsort(dist_obs)

    x_obs_sort = x_obs[idx_sort[0:prob.num_obs]]
    y_obs_sort = y_obs[idx_sort[0:prob.num_obs]]
    psi_obs_sort = psi_obs[idx_sort[0:prob.num_obs]]
    v_obs_sort  = v_obs[idx_sort[0:prob.num_obs]]
    vx_obs_sort  = vx_obs[idx_sort[0:prob.num_obs]]
    vy_obs_sort  = vy_obs[idx_sort[0:prob.num_obs]]
    dim_y_obs_sort = dim_y_obs[idx_sort[0:prob.num_obs]]
    dim_x_obs_sort = dim_x_obs[idx_sort[0:prob.num_obs]]

    return x_obs_sort,y_obs_sort,vx_obs_sort,vy_obs_sort,psi_obs_sort,dim_x_obs_sort,dim_y_obs_sort,\
        x_obs,y_obs,vx_obs,vy_obs,psi_obs

bool_check = False

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_levels",type=float, required=True)
    parser.add_argument("--num_exps",type=int, required=True)
    parser.add_argument('-r','--num_reduced_set',type=int, nargs='+', required=True)
    # parser.add_argument("--num_mother_set",  type=int, required=True)
    parser.add_argument("--num_obs",  type=int, required=True)
    parser.add_argument("--obs_type",  type=str, required=True)
    parser.add_argument('--costs', type=str, required=True)
    parser.add_argument("--town",  type=str, required=True)
    parser.add_argument("--total_obs",  type=int, required=True)
    parser.add_argument("--num_prime",type=int, required=True)
    parser.add_argument("--noises",type=str, required=True)
    parser.add_argument("--acc_const_noise",type=float, required=True)
    parser.add_argument("--steer_const_noise",type=float, required=True)

    args = parser.parse_args()
    
    acc_const_noise = args.acc_const_noise
    steer_const_noise = args.steer_const_noise

    noise = args.noises
    num_prime = args.num_prime
    noise_level = args.noise_levels
    town = args.town
    num_exps = args.num_exps
    list_num_reduced = args.num_reduced_set
    # num_mother = args.num_mother_set
    total_obs = args.total_obs # total number of obs in simulation
    num_obs = args.num_obs # number of nearest obs to take into account in optimizer
    cost = args.costs
    obs_type = args.obs_type

    #### data collection variables
    obstacle_init_frenet_all, ego_init_frenet_all, curvature_ref_all, path_ref_all = [],[], [], []
    neural_output_all,traj_input_all = [],[]
    beta_cem_data_all = []
    
    for num_reduced in list_num_reduced:
        global bool_check
        bool_check = False

        prob = cem.CEM(num_reduced,1,num_obs,noise_level,num_prime,noise,town)
        
        func_cem = prob.compute_cem_det_1

        num_collisions = 0
        list_lane_violations, list_speed = [],[]

        for ii in range(num_exps):
            print("Reduced set ", num_reduced, "Experiment ", ii)

            bool_check = False

            x_ego = np.zeros((0,1))
            y_ego = np.zeros((0,1))
            psi_ego = np.zeros((0,1))
            v_ego = np.zeros((0,1))
            psidot_ego = np.zeros((0,1))
            steer_ego = np.zeros((0,1))
           
            x_obs_init_all,y_obs_init_all\
                  = np.zeros((0,total_obs)),np.zeros((0,total_obs))
            
            vx_obs_init_all,vy_obs_init_all \
                = np.zeros((0,total_obs)),np.zeros((0,total_obs))

            def on_collision(event):
                global bool_check
                collision_actor = event.other_actor
                print(f"Collision with {collision_actor.type_id} (ID: {collision_actor.id})")
                bool_check = True
            
            throttle1 = 0.0
            prev_vel = 0.0 
            prev_acc = 0.0
            
            pygame.init()
            display = pygame.display.set_mode(
                    (800, 600),
                    pygame.HWSURFACE | pygame.DOUBLEBUF)
            font = carla_simulation_static.get_font()
            clock = pygame.time.Clock()
            
            if obs_type=="static":
                csm = carla_simulation_static.CarlaSimulation(display, font,clock,total_obs,town)
            else:
                csm = carla_simulation.CarlaSimulation(display, font,clock,total_obs,town)

            collision_sensor = csm.blueprint_library.find('sensor.other.collision')
            collision_sensor = csm.world.spawn_actor(
                collision_sensor,
                carla.Transform(carla.Location(x=1.0, y=0.0, z=1.0)),
                attach_to=csm.vehicle)
            
            collision_sensor.listen(lambda event: on_collision(event))
            csm.actor_list.append(collision_sensor)

                    ### Load reference path
            start_position = csm.vehicle.get_location()

            spawn_points = csm.m.get_spawn_points()

            # goal_position = carla.Location(x= -143.8 ,y= -91.6)

            # while(True):
            #     spawn_point_random = rnd.choice(spawn_points)
            #     start_position_random = start_position #rnd.choice(spawn_points)

            #     route_trace = csm.grp.trace_route(carla.Location(x=start_position_random.x,y = start_position_random.y),
            #                                       carla.Location(x=spawn_point_random.location.x,y=spawn_point_random.location.y))
            #     # print(start_position_random.location.x,start_position_random.location.y)
            #     print(spawn_point_random.location.x,spawn_point_random.location.y)
            #     print("-----------------------")
            #     x_path_data = np.zeros(len(route_trace))
            #     y_path_data = np.zeros(len(route_trace))

            #     len_path = x_path_data.shape[0]

            #     for i in range(0,len(route_trace)):
            #         x_path_data[i] = route_trace[i][0].transform.location.x
            #         y_path_data[i] = route_trace[i][0].transform.location.y

            #     plt.figure(1)
            #     plt.plot(x_path_data,y_path_data)
            #     plt.axis("equal")
            #     plt.show()

            # spawn_point_random = rnd.choice(spawn_points)
            # start_position_random = rnd.choice(spawn_points)

            if town == "Town10HD":
                goal_position = carla.Location(x=-48.839951, y=-32.034920)
            else: # Town05
                # goal_position = carla.Location(x= 97.27 ,y= 63.11) # town03
                goal_position = carla.Location(x= -175.9 ,y= 6.47) # town05

            route_trace = csm.grp.trace_route(start_position,goal_position)
            x_path_data = np.zeros(len(route_trace))
            y_path_data = np.zeros(len(route_trace))

            ######################
            spectator = csm.world.get_spectator()
            csm.vehicle.get_transform().rotation.pitch = -90
            csm.vehicle.get_location().z = 30
            

            ######################
            len_path = x_path_data.shape[0]

            for i in range(0,len(route_trace)):
                x_path_data[i] = route_trace[i][0].transform.location.x
                y_path_data[i] = route_trace[i][0].transform.location.y

            # plt.figure(1)
            # plt.plot(x_path_data,y_path_data)
            # plt.axis("equal")
            # plt.show()
            # kk

            #### spectator

             ######################

            # spectator = csm.world.get_spectator()
            # spect_transform = csm.vehicle.get_transform()


            # spect_transform.rotation.pitch = -90
            # spect_transform.location.z = 60
            # spect_transform.location.x = 70
            # spect_transform.location.y = 180

            # spectator.set_transform(spect_transform)
            # csm.actor_list.append(spectator)

            # Artificially extending the path so that optimizer does not run into issues when near the end of the actual path
            if town=="Town10HD":
                num_p = 20000
                m = (y_path_data[-1]-y_path_data[-2])/(x_path_data[-1]-x_path_data[-2])

                if(y_path_data[-1]>y_path_data[-2]):
                    y_linspace = np.linspace(y_path_data[-1],y_path_data[-1]+5000,num_p)
                else:
                    y_linspace = np.linspace(y_path_data[-1],y_path_data[-1]-5000,num_p)

                intercept = y_path_data[-1] - m*x_path_data[-1]
                x_linspace = (y_linspace - intercept)/m
            
            else:
                num_p = 20000
                m = (y_path_data[-1]-y_path_data[-2])/(x_path_data[-1]-x_path_data[-2])

                if(x_path_data[-1]>x_path_data[-2]):
                    x_linspace = np.linspace(x_path_data[-1],x_path_data[-1]+5000,num_p)
                else:
                    x_linspace = np.linspace(x_path_data[-1],x_path_data[-1]-5000,num_p)

                intercept = y_path_data[-1] - m*x_path_data[-1]
                y_linspace = m*x_linspace + intercept
            
        #########################################################3333

            cs_x_path, cs_y_path,cs_phi_path, arc_length, arc_vec = prob.cem_helper.path_spline(x_path_data, y_path_data)

            num_p = int(arc_length/0.25) # to create uniform spacing between points
            arc_vec = np.linspace(0, arc_length, num_p)

            x_path_data = cs_x_path(arc_vec)
            y_path_data = cs_y_path(arc_vec)

            y_path_data = np.hstack((y_path_data,y_linspace))
            x_path_data = np.hstack((x_path_data,x_linspace))

            cs_x_path, cs_y_path,cs_phi_path, arc_length, arc_vec = prob.cem_helper.path_spline(x_path_data, y_path_data)
            # print(arc_length)
            # plt.figure(1)
            # plt.plot(cs_x_path(arc_vec),cs_y_path(arc_vec))
            # plt.axis("equal")
            # plt.show()

            # kk


            idx = 0

            x_global_init = x_path_data[idx]
            y_global_init = y_path_data[idx]
            psi_global_init = np.arctan2( y_path_data[idx+1]-y_path_data[idx], x_path_data[idx+1]-x_path_data[idx]  )#+30*jnp.pi/180
            psi_global_init = np.arctan2(np.sin(psi_global_init),np.cos(psi_global_init))

            v_global_init = 0.1
            vdot_global_init = 0.0
            psidot_global_init = 0*jnp.pi/180

            x_global_shifted = 0.0
            y_global_shifted = 0.0

            v_des = 10
        
            ################################################## generating random samples for v_des, y_des

            mean_vx_1 = v_des
            mean_vx_2 = v_des
            mean_vx_3 = v_des
            mean_vx_4 = v_des

            y_1 = 0.0
            mean_y_des_1 = y_1
            mean_y_des_2 = y_1
            mean_y_des_3 = y_1
            mean_y_des_4 = y_1
            cov_vel = 20.
            cov_y = 100.
            mean_param = jnp.hstack(( mean_vx_1, mean_vx_2, mean_vx_3, mean_vx_4, mean_y_des_1, mean_y_des_2, mean_y_des_3, mean_y_des_4))
            diag_param = np.hstack(( cov_vel, cov_vel, cov_vel, cov_vel, cov_y, cov_y, cov_y, cov_y))
            cov_param = jnp.asarray(np.diag(diag_param)) 

            num_mean_update = 4
            t_target = (num_mean_update-1)*prob.t

            i = 0

            dist_goal = np.sqrt( (x_global_init-x_path_data[0:len_path][-1])**2 + (y_global_init-y_path_data[0:len_path][-1])**2 )  

            num_lane_violations = 0

            #### data collection variables
            obstacle_init_frenet, ego_init_frenet, curvature_ref, path_ref = [],[], [], []
            neural_output,traj_input = [],[]
            beta_cem_data = []
            
            # plt.figure(1)
            # plt.plot(x_path_data,y_path_data)
            # plt.axis("equal")
            # plt.show()
            # kk
            while( dist_goal >= 7. and i<=5000 ):
                ### draw ref path

                # for i in range(0,len(route_trace)):
                #     csm.world.debug.draw_string(route_trace[i][0].transform.location, 'O', draw_shadow=False,
                #                         color=carla.Color(r=255, g=0, b=0), life_time=0)
                    
                ############################
                 
                if bool_check:
                    break

                if csm.should_quit():
                    break

                x_waypoints, y_waypoints, phi_Waypoints = prob.cem_helper.waypoint_generator(x_global_init, y_global_init,
                                                            x_path_data, y_path_data, arc_vec, cs_x_path, cs_y_path, cs_phi_path, arc_length)

                ########### shifting waypoints to the center of the ego vehicle
                x_waypoints_shifted = x_waypoints-x_global_init
                y_waypoints_shifted = y_waypoints-y_global_init

                initial_state_global = np.hstack(( x_global_shifted, y_global_shifted, v_global_init, vdot_global_init, 
                                                    psi_global_init, psidot_global_init))
                
                x_obs_sort,y_obs_sort,vx_obs_sort,vy_obs_sort,psi_obs_sort,_,_,\
                                x_obs_all,y_obs_all,vx_obs_all,vy_obs_all,psi_obs_all,\
                        = compute_obs_data(csm,prob,x_global_init,y_global_init,
                                        v_global_init,psi_global_init)
                
                x_obs_shifted = x_obs_sort - x_global_init
                y_obs_shifted = y_obs_sort - y_global_init

                threshold = 0.1
                x_path, y_path = prob.cem_helper.custom_path_smoothing(x_waypoints_shifted, y_waypoints_shifted, threshold)
                
                Fx_dot, Fy_dot, Fx_ddot, Fy_ddot, _arc_vec, \
                    kappa, _arc_length = prob.cem_helper.compute_path_parameters(x_path, y_path)
            
                x_init_obs, y_init_obs, vx_init_obs, vy_init_obs,\
                      psi_init_obs = \
                        prob.cem_helper.global_to_frenet_obs_vmap(x_obs_shifted,y_obs_shifted
                                ,vx_obs_sort,vy_obs_sort,psi_obs_sort,x_path, y_path,_arc_vec,
                            Fx_dot, Fy_dot, kappa)

                x_obs_traj,y_obs_traj,psi_obs_traj = prob.cem_helper.compute_obs_trajectories(x_init_obs, y_init_obs,
                                                             vx_init_obs, vy_init_obs,psi_init_obs)

                start = time.time()
            
                cx_best,cy_best,mmd_lane,mmd_obs,v_best,steering_best,\
                x_roll_global,y_roll_global,mean_param,_ego_init_frenet,\
                behavioural_inputs, _beta_cem_data\
                = func_cem(i,initial_state_global,
                    mean_param,cov_param,
                    x_obs_traj,y_obs_traj,v_des,
                    x_path,y_path,_arc_vec,Fx_dot,Fy_dot,kappa)
                
                v_control = jnp.mean(v_best[0:num_mean_update])
                steer_control = jnp.mean(steering_best[0:num_mean_update])
                steer_control = jnp.clip(steer_control, -prob.steer_max, prob.steer_max)
                
                steer_control_update = steer_control
                a_control = (v_control-v_global_init)/t_target

                np.random.seed(3*ii+5*i+7)

                if noise=="gaussian":
                    acc_pert = np.random.normal(0,1,(1,)).reshape(-1)
                    steer_pert = np.random.normal(0,1,(1,)).reshape(-1)
                    
                    acc_pert = ( prob.sigma_acc*jnp.abs(a_control) )*acc_pert
                    steer_pert = (prob.sigma_steer*jnp.abs(steer_control_update) )*steer_pert
            
                else:
                    acc_pert = np.random.beta(prob.beta_a,prob.beta_b,(1,)).reshape(-1)
                    steer_pert = np.random.beta(prob.beta_a,prob.beta_b,(1,)).reshape(-1)
                    
                    acc_pert = ( prob.sigma_acc*a_control)*acc_pert
                    steer_pert = (prob.sigma_steer*steer_control_update)*steer_pert
               
                a_control = a_control + acc_pert + acc_const_noise*np.random.normal(0,1,(1,)).reshape(-1)
                steer_control_update = steer_control_update + steer_pert + steer_const_noise*np.random.normal(0,1,(1,)).reshape(-1)

                prev_vel,prev_acc,throttle1,acc = apply_control(0.0,0.0,prob,csm,a_control,prev_vel,
                                                                prev_acc,throttle1,steer_control_update)
                
                csm.tick(timeout=2.0)
                # csm.visualize()

                i += 1
                vel = csm.vehicle.get_velocity()
                v_global_init = (vel.x**2 + vel.y**2)**0.5

                yaw = csm.vehicle.get_transform().rotation.yaw

                psi_global_init = math.radians(yaw)
                psi_global_init = np.arctan2(np.sin(psi_global_init),np.cos(psi_global_init))
                psidot_global_init = 1.0*math.radians(csm.vehicle.get_angular_velocity().z)

                x_global_init = csm.vehicle.get_transform().location.x
                y_global_init = csm.vehicle.get_transform().location.y

                # print(i," vel feedback",v_global_init,"vel optimizer",v_control)
            
                x_ego = np.append(x_ego,x_global_init*np.ones(1).reshape(1,-1),axis=0)
                y_ego = np.append(y_ego,y_global_init*np.ones(1).reshape(1,-1),axis=0)
                psi_ego = np.append(psi_ego,psi_global_init*np.ones(1).reshape(1,-1),axis=0)
                v_ego = np.append(v_ego,v_global_init*np.ones(1).reshape(1,-1),axis=0)
                psidot_ego = np.append(psidot_ego,psidot_global_init*np.ones(1).reshape(1,-1),axis=0)
                steer_ego = np.append(steer_ego,steer_control_update*np.ones(1).reshape(1,-1),axis=0)

                # vx_obs_init_all = np.append(vx_obs_init_all,vx_obs_all.reshape(1,-1),axis=0)
                # vy_obs_init_all = np.append(vy_obs_init_all,vy_obs_all.reshape(1,-1),axis=0)
                # x_obs_init_all = np.append(x_obs_init_all,x_obs_all.reshape(1,-1),axis=0)
                # y_obs_init_all = np.append(y_obs_init_all,y_obs_all.reshape(1,-1),axis=0)

                dist_goal = np.sqrt( (x_global_init-x_path_data[0:len_path][-1])**2 + (y_global_init-y_path_data[0:len_path][-1])**2 )

                _initial_state = np.array([x_global_init,y_global_init,v_global_init,vdot_global_init,psi_global_init,psidot_global_init])
                x_ego_frenet, y_ego_frenet, vx_ego_frenet, vy_ego_frenet,\
                      ax_ego_frenet, ay_ego_frenet, _,_,_ = prob.cem_helper.global_to_frenet(x_waypoints,
                                                                                              y_waypoints,
                                                                                                _initial_state, _arc_vec,
                                                                                                Fx_dot, Fy_dot, kappa )
                
                num_lane_violations += compute_lane_violations(prob,y_ego_frenet)
                # print(num_lane_violations, len(x_ego))

            list_lane_violations.append(num_lane_violations/len(x_ego))
            list_speed.append(np.mean(v_ego))

            if bool_check :
                num_collisions += 1
                print("Aborting due to collision x x x x ")
            else:
                print('Experiment successfully completed.')

            print("Average lane violation after {} experiments = {}%".format(len(list_lane_violations),
                                                            100*sum(list_lane_violations)/len(list_lane_violations)))
           
            print("% Collision after {} experiments =  {}%".format(len(list_lane_violations),100*num_collisions/len(list_lane_violations)))
            print("Average speed after {} experiments =  {}".format(len(list_lane_violations),sum(list_speed)/len(list_speed)))
            print("-------------------------")

            np.savez("../carla_stochastic_dynamics/data/{}/{}_noise/noise_{}/{}_{}_{}".format(town,noise,int(noise_level*100),
                                                                            obs_type,
                                                                            cost,ii),
                                                        x_ego=x_ego,y_ego=y_ego,psi_ego=psi_ego,
                                                        v_ego=v_ego,psidot_ego=psidot_ego,steer_ego=steer_ego,
                                                        x_path=x_path_data[0:len_path], y_path=y_path_data[0:len_path],
                                                        lane_violations_per_exp = num_lane_violations/len(x_ego),
                                                        num_collision_per_exp = int(bool_check),
                                                        avg_vel_per_exp = np.mean(v_ego))
                
            csm.destroy()
            pygame.quit()
        
if __name__ == '__main__':
    main()

