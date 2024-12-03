import numpy as np
import jax.numpy as jnp
import time
import sys
sys.path.insert(1, 'path/to/optimizer')
from optimizer import cem
import argparse
import jax

def compute_obs_data(num_obs,seed):
    
    np.random.seed(seed)
    x_obs_init = np.random.choice(np.array([35,40,45,50,55,60,65,70,75]), (num_obs, ),replace=False)
    y_obs_init = np.random.choice(np.array([-1.75,1.75]), (num_obs, ))

    vx_obs_init = np.zeros(num_obs)
   
    vy_obs_init = np.zeros(num_obs)
    psi_obs_init = np.zeros(num_obs)

    return x_obs_init,y_obs_init,vx_obs_init,vy_obs_init,psi_obs_init

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_levels',type=float, nargs='+', required=True)
    parser.add_argument('--num_reduced_sets',type=int, nargs='+', required=True)
    parser.add_argument('--num_obs',type=int, nargs='+', required=True)
    parser.add_argument('--costs',type=str, nargs='+', required=True)
    parser.add_argument('--num_prime',type=int, nargs='+', required=True)
    parser.add_argument('--noises',type=str, nargs='+', required=True)
    parser.add_argument("--acc_const_noise",  type=float, required=True)
    parser.add_argument("--steer_const_noise",  type=float, required=True)

    args = parser.parse_args()

    list_noises = args.noises
    list_num_prime = args.num_prime
    list_noise_levels = args.noise_levels
    list_num_reduced = args.num_reduced_sets
    list_num_obs = args.num_obs 
    list_costs = args.costs
    acc_const_noise = args.acc_const_noise
    steer_const_noise = args.steer_const_noise
    
    x_init = 0.0
    vx_init = 5
    ax_init = 0.0

    y_init = 1.75
    vy_init = 0.
    ay_init = 0.0
    
    init_state = jnp.hstack(( x_init, y_init, vx_init, vy_init, ax_init, ay_init   ))

    v_des = 15.
    
    mean_vx_1 = v_des
    mean_vx_2 = v_des
    mean_vx_3 = v_des
    mean_vx_4 = v_des
    
    mean_y = 0.
    mean_y_des_1 = mean_y
    mean_y_des_2 = mean_y
    mean_y_des_3 = mean_y
    mean_y_des_4 = mean_y
   
    cov_vel = 20.
    cov_y = 100.
    mean_param = jnp.hstack(( mean_vx_1, mean_vx_2, mean_vx_3, mean_vx_4, mean_y_des_1, mean_y_des_2, mean_y_des_3, mean_y_des_4))

    diag_param = np.hstack(( cov_vel,cov_vel,cov_vel,cov_vel, cov_y,cov_y,cov_y,cov_y ))
    cov_param = np.asarray(np.diag(diag_param)) 
    
    num_configs = 200
    for noise in list_noises:
        for noise_level in list_noise_levels:
            for num_prime in list_num_prime:
                for num_obs in list_num_obs:
                    for num_reduced in list_num_reduced:
                        prob = cem.CEM(num_reduced,num_obs,noise_level,num_prime,noise
                                    ,acc_const_noise,steer_const_noise)
                    #####################################################
                        for cost in list_costs:
                            if cost=="mmd_opt" :
                                func_cem = prob.compute_cem_mmd_opt
                                threshold_lane = -2*prob.ker_wt + 1.
                                threshold_obs = -prob.ker_wt + 1.
                            elif cost=="mmd_random":
                                func_cem = prob.compute_cem_mmd_random
                                threshold_lane = -2*prob.ker_wt + 1.
                                threshold_obs = -prob.ker_wt + 1.
                            else:
                                func_cem = prob.compute_cem_cvar
                                threshold_lane = 10**(-5)
                                threshold_obs = 10**(-5)
                        
                            cx_all,cy_all,init_state_all = np.zeros((0,prob.nvar)),\
                                                    np.zeros((0,prob.nvar)),\
                                                    np.zeros((0,6)),\
                                                
                            x_obs_all,y_obs_all = np.zeros((0,prob.num_obs)),np.zeros((0,prob.num_obs))
                            vx_obs_all,vy_obs_all = np.zeros((0,prob.num_obs)),np.zeros((0,prob.num_obs))

                            for k in range(num_configs): 

                                x_obs_init,y_obs_init,vx_obs_init,vy_obs_init,psi_obs_init = compute_obs_data(num_obs,k)
                                x_obs_traj,y_obs_traj,_ = prob.cem_helper.compute_obs_trajectories(x_obs_init,y_obs_init,vx_obs_init,vy_obs_init,psi_obs_init)       

                                if cost=="mmd_opt":
                                    start = time.time()
                                    cx_best,cy_best,cost_lane,cost_obs,beta ,sigma,res_beta\
                                        = func_cem(np.random.randint(1,10000),init_state,mean_param,cov_param,
                                        x_obs_traj,y_obs_traj,v_des)
                                    
                                else:
                                    cx_best,cy_best,cost_lane,cost_obs = func_cem(np.random.randint(1,10000),init_state,mean_param,cov_param,
                                        x_obs_traj,y_obs_traj,v_des)
                                    
                                if cost_obs<= threshold_obs:
                                    cx_all = np.append(cx_all,cx_best.reshape(1,-1),axis=0)
                                    cy_all = np.append(cy_all,cy_best.reshape(1,-1),axis=0)
                                    init_state_all = np.append(init_state_all,init_state.reshape(1,-1),axis=0)
                                    x_obs_all = np.append(x_obs_all,x_obs_init.reshape(1,-1),axis=0)
                                    y_obs_all = np.append(y_obs_all,y_obs_init.reshape(1,-1),axis=0)
                                    vx_obs_all = np.append(vx_obs_all,vx_obs_init.reshape(1,-1),axis=0)
                                    vy_obs_all = np.append(vy_obs_all,vy_obs_init.reshape(1,-1),axis=0)
                                    
                            np.savez("./data/{}_noise/noise_{}/ts_{}/{}_{}_samples_{}_obs".format(noise,int(noise_level*100),
                                prob.num_prime,
                                cost,num_reduced,num_obs),
                                    cx = cx_all,cy = cy_all,
                                init_state = init_state_all,x_obs=x_obs_all,y_obs=y_obs_all,
                                vx_obs=vx_obs_all,vy_obs=vy_obs_all)

                            print("cost {}, reduced_set {}, num_obs {}, num_prime {}, noise_level {}, noise {}".format(cost,num_reduced,num_obs,
                                                                    num_prime,noise_level,
                                                                    noise))
                            print("--------------------------------")
                            
if __name__ == '__main__':
    main()


