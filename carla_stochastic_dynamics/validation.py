import numpy as np
import os
import sys
sys.path.insert(1, '/home/ims-robotics/Basant/ICRA_RAL_2025/carla_stochastic_dynamics/optimizer')
from optimizer import cem
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import matplotlib.pyplot as plt

@partial(jit, static_argnums=(1,))
def compute_lane_bar(y,len_ego):

    cost_centerline_penalty_lb = -y+y_lb
    cost_centerline_penalty_ub = y-y_ub
    
    cost_lb = jnp.maximum(jnp.zeros(len_ego), cost_centerline_penalty_lb)
    cost_ub = jnp.maximum(jnp.zeros(len_ego), cost_centerline_penalty_ub)

    return cost_lb,cost_ub 

@partial(jit, static_argnums=(1,))
def compute_stats(y,len_ego):
    
    cost_lane_lb,cost_lane_ub = compute_lane_bar(y,len_ego)

    # count_lane_lb = np.count_nonzero(cost_lane_lb.reshape(-1,1),axis=1)
    # count_lane_ub = np.count_nonzero(cost_lane_ub.reshape(-1,1),axis=1)

    # count_lane = count_lane_lb + count_lane_ub
    # count_lane = np.count_nonzero(count_lane)

    return cost_lane_lb+cost_lane_ub

town = "Town10HD"

prob = cem.CEM(1,1,1,1,1,"gaussian",town,1,1)
y_lb = prob.y_lb
y_ub = prob.y_ub

root = "./data/{}/".format(town)

## MMD Gaussian
# list_noise = ["gaussian"]
# list_noise_level = ["30"]
# list_num_prime = ["40"]
# list_cost = ["mmd_opt"]
# list_num_reduced = ["2"]
# list_obs_type = ["static"]

## MMD Beta
list_noise = ["beta"]
list_noise_level = ["5"] 
list_num_prime = ["40"]
list_cost = ["mmd_opt"]
list_num_reduced = ["2"]
list_obs_type = ["static"]

## CvaR Gaussian
# list_noise = ["gaussian"]
# list_noise_level = ["30"]
# list_num_prime = ["40"]
# list_cost = ["cvar"]
# list_num_reduced = ["2"]
# list_obs_type = ["static"]

## CVaR Beta
# list_noise = ["beta"]
# list_noise_level = ["5"] 
# list_num_prime = ["40"]
# list_cost = ["cvar"]
# list_num_reduced = ["2"]
# list_obs_type = ["static"]

num_prime = list_num_prime[0]
cost = list_cost[0]
obs_type=list_obs_type[0]
noise = list_noise[0]

for noise_level in list_noise_level:
    for num_reduced in list_num_reduced:
        print(town, noise, noise_level, num_reduced)
        coll_list,vel_list,lane_list,vel_max_list = [],[],[],[]

        i = 0
        j_lane,j_coll = 0,0
        lane,vel,coll,vel_max = 0,0,0,0
        _lane_list,arc_length_list =[], []
        # x_ego,y_ego,v_ego,psi_ego,psidot_ego,steer_ego = [],[],[],[],[],[]
        
        while(i<=60):
            filename = root + "{}_noise/noise_{}/ts_{}/{}_{}_{}_samples_{}.npz".format(noise,noise_level,
                            num_prime,obs_type,cost,num_reduced,i)
            
            if(os.path.isfile(filename)):

                data = np.load(filename)
                x_path = data["x_path"]
                y_path = data["y_path"]

                _x_ego = np.asarray(data["x_ego"]).reshape(-1)
                _y_ego = np.asarray(data["y_ego"]).reshape(-1)
                _v_ego = np.asarray(data["v_ego"]).reshape(-1)
                _psi_ego = np.asarray(data["psi_ego"]).reshape(-1)
                _psidot_ego = np.asarray(data["psidot_ego"]).reshape(-1)
                _steer_ego = np.asarray(data["steer_ego"]).reshape(-1)
                
                coll += data["num_collision_per_exp"]
                j_coll+=1
                
                vel += data["avg_vel_per_exp"]

                len_ego = _x_ego.shape[0]
                
                initial_state = np.array([_x_ego, _y_ego, _v_ego, np.zeros(_v_ego.shape[0]), _psi_ego, _psidot_ego]).T
                Fx_dot, Fy_dot, Fx_ddot, Fy_ddot, arc_vec, kappa, arc_length \
                    = prob.cem_helper.compute_path_parameters(x_path, y_path)

                x_ego_frenet, y_ego_frenet, vx_ego_frenet, vy_ego_frenet, \
                    ax_ego_frenet, ay_ego_frenet, _,_,_ = prob.cem_helper.global_to_frenet_vmap_1(x_path, y_path,
                                                    initial_state, arc_vec,
                                                    Fx_dot, Fy_dot, kappa )

                count_lane = compute_stats(y_ego_frenet,len_ego)
                _lane_list.append(np.sum(count_lane))
                vel_max+=np.max(_v_ego)

                # _,_,_,_,_,_, arc_length \
                #     = prob.cem_helper.compute_path_parameters(_x_ego,_y_ego)

                arc_length_list.append(arc_length)

                j_lane+=1

                # print("{} {}, {}".format(np.sum(count_lane), arc_length, i))
                # print(len_ego)
                # print("--------")

                # if True:
                #     plt.figure(1)
                #     plt.plot(x_path,y_path,"-k", linewidth=4)
                #     plt.plot(_x_ego,_y_ego,"-b", linewidth=4)
                #     plt.axis("equal")
                #     plt.show()

            else:
                pass
            
            i+=1
        
        if j_coll==0:
            print("error")
        coll_list.append(100*coll/j_coll)
        lane_list.append(100*np.sum(_lane_list)/np.sum(arc_length_list))
        vel_list.append(vel/j_lane)
        vel_max_list.append(vel_max/j_lane)

        # print("final",lane_list)

        np.savez("./stats/{}/{}_noise/noise_{}/ts_{}/{}_{}_{}_samples".format(town,noise,noise_level,
                    num_prime,cost,obs_type,
                    num_reduced), coll=coll_list,lane=lane_list,vel=vel_list,vel_max = vel_max_list)
