from functools import partial
from kernel_computation import kernel_matrix
from compute_beta import beta_cem
import numpy as np
import jax.numpy as jnp
from jax import lax,jit,vmap
import matplotlib.pyplot as plt
import jax
import scipy
import sys
sys.path.insert(1, '/home/ims-robotics/Basant/ICRA_RAL_2025/synthetic_stochastic_dynamics/optimizer')
from optimizer import cem
import argparse

def normal_vectors(x, y, scalar):
    tck = scipy.interpolate.splrep(x, y)
    y_deriv = scipy.interpolate.splev(x, tck, der=1)
    normals_rad = np.arctan(y_deriv)+np.pi/2.
    return np.cos(normals_rad)*scalar, np.sin(normals_rad)*scalar

def compute_rollout_one_step(acc,steer,state):
    
    x,y,vx,vy,psi = state[:,0],state[:,1],state[:,2]  ,state[:,3], state[:,4]

    v = np.sqrt(vx**2 +vy**2)
    v = v + acc*prob.t
    psidot = v*np.tan(steer)/prob.wheel_base
    psi_next = psi+psidot*prob.t

    vx_next = v*np.cos(psi_next)
    vy_next = v*np.sin(psi_next)
    
    x_next = x + vx_next*prob.t
    y_next = y + vy_next*prob.t
    
    state_next = np.vstack((x_next,y_next,
                    vx_next,vy_next,
                    psi_next)).T
    
    return state_next

def compute_rollout_complete(acc,steer,initial_state,noise_level,num_prime,noise,key):
    np.random.seed(key)

    # if noise=="gaussian":
    #     noise_samples_acc = np.random.multivariate_normal(np.zeros(num_prime), np.eye(num_prime),
    #                                                 (_num_batch,))
                
    #     noise_samples_steer = np.random.multivariate_normal(np.zeros(num_prime), np.eye(num_prime),
    #                                                 (_num_batch,))
        
    #     acc_pert = noise_level*np.abs(acc)*noise_samples_acc
    #     steer_pert = noise_level*np.abs(steer)*noise_samples_steer
    # else:
    #     noise_samples_acc = np.random.beta(prob.beta_a*np.abs(acc),prob.beta_b*np.abs(acc),
    #                                         (_num_batch,num_prime))
                
    #     noise_samples_steer = np.random.beta(prob.beta_a*np.abs(steer+1e-5),prob.beta_b*np.abs(steer+1e-5),
    #                                             (_num_batch,num_prime))
        
    #     acc_pert = noise_level*(2*noise_samples_acc-1) ## transform (0,1) to (-1,1)
    #     steer_pert = noise_level*(2*noise_samples_steer-1)

    if noise=="gaussian":
        noise_samples_acc = np.random.multivariate_normal(np.zeros(num_prime), np.eye(num_prime),
                                                    (_num_batch,))
                
        noise_samples_steer = np.random.multivariate_normal(np.zeros(num_prime), np.eye(num_prime),
                                                    (_num_batch,))
        
        acc_pert = noise_level*np.abs(acc)*noise_samples_acc
        steer_pert = noise_level*np.abs(steer)*noise_samples_steer
    else:
        noise_samples_acc = np.random.beta(prob.beta_a*np.abs(acc),prob.beta_b*np.abs(acc),
                                            (_num_batch,num_prime))
                
        noise_samples_steer = np.random.beta(prob.beta_a*np.abs(steer)+1e-5,prob.beta_b*np.abs(steer)+1e-5,
                                                (_num_batch,num_prime))
        
        acc_pert = noise_level*(2*noise_samples_acc-1) ## transform (0,1) to (-1,1)
        steer_pert = prob.cem_helper.K_steer*noise_level*(2*noise_samples_steer-1)

    noise_samples = np.random.multivariate_normal(np.zeros(num_prime), np.eye(num_prime),
                                                    (_num_batch,))
    
    acc = acc + acc_pert + prob.acc_const_noise*noise_samples
    steer = steer + steer_pert+ prob.steer_const_noise*noise_samples

    x_roll = np.zeros((_num_batch,num_prime))
    y_roll = np.zeros((_num_batch,num_prime))
    
    state = initial_state
    state = np.vstack([state] * (_num_batch))

    for idx in range(num_prime):
        x_roll[:,idx] = state[:,0]
        y_roll[:,idx] = state[:,1]
        
        state = compute_rollout_one_step(acc[:,idx],steer[:,idx],state)
        
    return x_roll,y_roll

def compute_f_bar_temp(x_obs,y_obs,x,y,num_prime,num_obs): 
    
    wc_alpha = (x-x_obs[:,0:num_prime][:,np.newaxis])
    ws_alpha = (y-y_obs[:,0:num_prime][:,np.newaxis])

    cost = -(wc_alpha**2)/(prob.a_obs**2) - (ws_alpha**2)/(prob.b_obs**2) +  np.ones((num_obs,_num_batch,num_prime))
    cost_bar = np.maximum(np.zeros((num_obs,_num_batch,num_prime)), cost)
    return cost_bar

def compute_lane_bar(y,num_prime):

    cost_centerline_penalty_lb = -y+prob.y_lb
    cost_centerline_penalty_ub = y-prob.y_ub
    
    cost_lb = np.maximum(np.zeros((_num_batch,num_prime)), cost_centerline_penalty_lb)
    cost_ub = np.maximum(np.zeros((_num_batch,num_prime)), cost_centerline_penalty_ub)

    return cost_lb,cost_ub 

def compute_controls(xdot,ydot,xddot,yddot):
    v = np.sqrt(xdot**2+ydot**2)
    v = np.hstack((v,v[-1]))
    
    acc = np.diff(v)/prob.t
    acc = np.hstack((acc,acc[-1]))

    curvature_best = (yddot*xdot-ydot*xddot)/((xdot**2+ydot**2)**(1.5)) 
    steer = np.arctan(curvature_best*prob.wheel_base  )

    return acc,steer

def compute_stats(cx,cy,init_state,x_obs,y_obs,vx_obs,vy_obs,num_prime,
                  noise_level,noise,num_obs,key):
    x_obs,y_obs = x_obs.reshape(-1),y_obs.reshape(-1)
    vx_obs,vy_obs = vx_obs.reshape(-1),vy_obs.reshape(-1)

    cx,cy = cx.reshape(-1),cy.reshape(-1)
    xdot,xddot = np.dot(prob.Pdot_jax,cx), np.dot(prob.Pddot_jax,cx)
    ydot,yddot = np.dot(prob.Pdot_jax,cy), np.dot(prob.Pddot_jax,cy)

    init_state = init_state.reshape(-1)
    initial_state = np.asarray([init_state[0], init_state[1],init_state[2],init_state[3],np.arctan2(init_state[3],init_state[2])])

    acc,steer = compute_controls(xdot,ydot,xddot,yddot)

    x_roll,y_roll = compute_rollout_complete(acc[0:num_prime],steer[0:num_prime],initial_state,noise_level,num_prime,
                                             noise,key)

    x_obs_traj,y_obs_traj,_ = prob.cem_helper.compute_obs_trajectories(x_obs,y_obs,vx_obs,vy_obs,np.arctan2(vy_obs,vx_obs))       
    
    cost = compute_f_bar_temp(x_obs_traj,y_obs_traj,x_roll,y_roll,num_prime,num_obs) # num_obs x rollouts x timesteps
    cost = cost.transpose(0,2,1) # num_obs x timesteps x rollouts

    intersection = np.count_nonzero(cost,axis=2) # num_obs x timesteps
    count = np.max(intersection,axis=1) # num_obs
    count = np.max(count)

    cost_lane_lb,cost_lane_ub = compute_lane_bar(y_roll,num_prime) # rollouts x timesteps
    cost_lane_lb,cost_lane_ub = cost_lane_lb.T,cost_lane_ub.T # timesteps x rollouts

    intersection = np.count_nonzero(cost_lane_lb,axis=1) # timesteps
    count_lane_lb = np.max(intersection) 

    intersection = np.count_nonzero(cost_lane_ub,axis=1) # timesteps
    count_lane_ub = np.max(intersection) 

    count_lane = count_lane_lb + count_lane_ub

    return count,count_lane,x_roll,y_roll,x_obs_traj,y_obs_traj

_num_batch = 1000

num_p = 25000
x_path = np.linspace(0,1000,num_p)
y_path = np.zeros(num_p)

x_path_normal_lb,y_path_normal_lb = normal_vectors(x_path,y_path,-3.5)
x_path_lb = x_path + x_path_normal_lb
y_path_lb = y_path + y_path_normal_lb

x_path_normal_ub,y_path_normal_ub = normal_vectors(x_path,y_path,-2.25)
x_path_ub = x_path + x_path_normal_ub
y_path_ub = y_path + y_path_normal_ub

x_path_normal_d_lb,y_path_normal_d_lb = normal_vectors(x_path,y_path,0)
x_path_d_lb = x_path + x_path_normal_d_lb
y_path_d_lb = y_path + y_path_normal_d_lb

x_path_normal_d_ub,y_path_normal_d_ub = normal_vectors(x_path,y_path,2.25)
x_path_d_ub = x_path + x_path_normal_d_ub
y_path_d_ub = y_path + y_path_normal_d_ub

x_path_normal_d_mid,y_path_normal_d_mid = normal_vectors(x_path,y_path,3.5)
x_path_d_mid = x_path + x_path_normal_d_mid
y_path_d_mid = y_path + y_path_normal_d_mid

len_path = 6000
linewidth = 0.5

parser = argparse.ArgumentParser()
parser.add_argument('--noise_levels',type=float, nargs='+', required=True)
parser.add_argument('--num_reduced_sets',type=int, nargs='+', required=True)
parser.add_argument('--num_obs',type=int, nargs='+', required=True)
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
acc_const_noise = args.acc_const_noise
steer_const_noise = args.steer_const_noise

num_exps = 1

root = "./data"

for noise in list_noises:
    for noise_level in list_noise_levels:
        for num_prime in list_num_prime:
            for num_obs in list_num_obs:
                for num_reduced in list_num_reduced:
                    prob = cem.CEM(num_reduced,num_obs,noise_level,
                                num_prime,noise,acc_const_noise,steer_const_noise)
                    # data_mmd = np.load(root +"/{}_noise/noise_{}/ts_{}/{}_{}_{}_samples_{}_obs.npz".format(noise,int(noise_level*10),
                    #             num_prime,obs_type,
                    #             "mmd",num_reduced,num_obs))
                    
                    data_mmd_opt = np.load(root +"/{}_noise/noise_{}/ts_{}/{}_{}_samples_{}_obs.npz".format(noise,int(noise_level*100),
                                num_prime,
                                "mmd_opt",num_reduced,num_obs))
                    
                    data_cvar = np.load(root +"/{}_noise/noise_{}/ts_{}/{}_{}_samples_{}_obs.npz".format(noise,int(noise_level*100),
                                num_prime,
                                "cvar",num_reduced,num_obs))                    
                    
                    data_mmd_random = np.load(root +"/{}_noise/noise_{}/ts_{}/{}_{}_samples_{}_obs.npz".format(noise,int(noise_level*100),
                                num_prime,
                                "mmd_random",num_reduced,num_obs))
                    
                    cx_all_mmd_opt = np.asarray(data_mmd_opt["cx"])
                    cy_all_mmd_opt = np.asarray(data_mmd_opt["cy"])
                    init_state_all_mmd_opt = np.asarray(data_mmd_opt["init_state"])
                    x_obs_mmd_opt =  np.asarray(data_mmd_opt["x_obs"])
                    y_obs_mmd_opt =  np.asarray(data_mmd_opt["y_obs"])
                    vx_obs_mmd_opt =  np.asarray(data_mmd_opt["vx_obs"])
                    vy_obs_mmd_opt =  np.asarray(data_mmd_opt["vy_obs"])

                    # cx_all_mmd = np.asarray(data_mmd["cx"])
                    # cy_all_mmd = np.asarray(data_mmd["cy"])
                    # init_state_all_mmd = np.asarray(data_mmd["init_state"])
                    # x_obs_mmd =  np.asarray(data_mmd["x_obs"])
                    # y_obs_mmd =  np.asarray(data_mmd["y_obs"])
                    # vx_obs_mmd =  np.asarray(data_mmd["vx_obs"])
                    # vy_obs_mmd =  np.asarray(data_mmd["vy_obs"])

                    cx_all_mmd_random = np.asarray(data_mmd_random["cx"])
                    cy_all_mmd_random = np.asarray(data_mmd_random["cy"])
                    init_state_all_mmd_random = np.asarray(data_mmd_random["init_state"])
                    x_obs_mmd_random =  np.asarray(data_mmd_random["x_obs"])
                    y_obs_mmd_random =  np.asarray(data_mmd_random["y_obs"])
                    vx_obs_mmd_random =  np.asarray(data_mmd_random["vx_obs"])
                    vy_obs_mmd_random =  np.asarray(data_mmd_random["vy_obs"])

                    cx_all_cvar = np.asarray(data_cvar["cx"])
                    cy_all_cvar = np.asarray(data_cvar["cy"])
                    init_state_all_cvar = np.asarray(data_cvar["init_state"])
                    x_obs_cvar =  np.asarray(data_cvar["x_obs"])
                    y_obs_cvar =  np.asarray(data_cvar["y_obs"])
                    vx_obs_cvar =  np.asarray(data_cvar["vx_obs"])
                    vy_obs_cvar =  np.asarray(data_cvar["vy_obs"])
                    
                    coll_mmd_opt,coll_mmd_opt_lane = [],[]
                    coll_mmd,coll_mmd_lane = [],[]
                    coll_mmd_random,coll_mmd_random_lane  = [],[]
                    coll_cvar,coll_cvar_lane = [],[]
                    
                    mmd_opt_matrix = np.hstack((init_state_all_mmd_opt,x_obs_mmd_opt[:,0:num_obs],y_obs_mmd_opt[:,0:num_obs],
                                            vx_obs_mmd_opt[:,0:num_obs],vy_obs_mmd_opt[:,0:num_obs]))
                    
                    # mmd_matrix = np.hstack((init_state_all_mmd,x_obs_mmd[:,0:num_obs],y_obs_mmd[:,0:num_obs],
                    #                         vx_obs_mmd[:,0:num_obs],vy_obs_mmd[:,0:num_obs]))

                    mmd_random_matrix = np.hstack((init_state_all_mmd_random,x_obs_mmd_random[:,0:num_obs],y_obs_mmd_random[:,0:num_obs],
                                            vx_obs_mmd_random[:,0:num_obs],vy_obs_mmd_random[:,0:num_obs]))

                    cvar_matrix = np.hstack((init_state_all_cvar,x_obs_cvar[:,0:num_obs],y_obs_cvar[:,0:num_obs],
                                            vx_obs_cvar[:,0:num_obs],vy_obs_cvar[:,0:num_obs]))

                    # print(mmd_matrix.shape,mmd_random_matrix.shape,cvar_matrix.shape)

                    # aset = set([tuple(x) for x in mmd_matrix])
                    bset = set([tuple(x) for x in mmd_random_matrix])
                    cset = set([tuple(x) for x in cvar_matrix])
                    dset = set([tuple(x) for x in mmd_opt_matrix])
                    eset = np.array([x for x in cset & dset])

                    print(eset.shape)

                    for k in range(0,eset.shape[0]):
                        # idx_mmd = np.where(np.all(eset[k]==mmd_matrix,axis=1))[0]
                        # idx_mmd_random = np.where(np.all(eset[k]==mmd_random_matrix,axis=1))[0]
                        idx_cvar = np.where(np.all(eset[k]==cvar_matrix,axis=1))[0]
                        idx_mmd_opt = np.where(np.all(eset[k]==mmd_opt_matrix,axis=1))[0]

                        # if (len(idx_mmd) >1):
                        #     idx_mmd = idx_mmd[0]
                        # if (len(idx_mmd_random) >1):
                        #     idx_mmd_random = idx_mmd_random[0]
                        if (len(idx_cvar) >1):
                            idx_cvar = idx_cvar[0]

                        if (len(idx_mmd_opt) >1):
                            idx_mmd_opt = idx_mmd_opt[0]

                        # print(k)
                        count_mmd_opt,count_mmd_opt_lane,x_roll_mmd_opt,y_roll_mmd_opt\
                        ,_,_ = compute_stats(cx_all_mmd_opt[idx_mmd_opt],cy_all_mmd_opt[idx_mmd_opt],
                                                                init_state_all_mmd_opt[idx_mmd_opt],
                                                    x_obs_mmd_opt[idx_mmd_opt],y_obs_mmd_opt[idx_mmd_opt],
                                                    vx_obs_mmd_opt[idx_mmd_opt],vy_obs_mmd_opt[idx_mmd_opt],
                                                    num_prime,noise_level,noise,num_obs,k)
                        
                        # count_mmd,count_mmd_lane,x_roll_mmd,y_roll_mmd,x_obs_traj,y_obs_traj = compute_stats(cx_all_mmd[idx_mmd],cy_all_mmd[idx_mmd],
                        #                                         init_state_all_mmd[idx_mmd],
                        #                             x_obs_mmd[idx_mmd],y_obs_mmd[idx_mmd],
                        #                             vx_obs_mmd[idx_mmd],vy_obs_mmd[idx_mmd],
                        #                             num_prime,noise_level,noise,num_obs,k)
                        
                        # count_mmd_random,count_mmd_random_lane,x_roll_mmd_random,y_roll_mmd_random,_,_ = compute_stats(cx_all_mmd_random[idx_mmd_random],cy_all_mmd_random[idx_mmd_random],
                        #                                                                 init_state_all_mmd_random[idx_mmd_random],
                        #                             x_obs_mmd_random[idx_mmd_random]
                        #                             ,y_obs_mmd_random[idx_mmd_random],
                        #                             vx_obs_mmd_random[idx_mmd_random]
                        #                             ,vy_obs_mmd_random[idx_mmd_random],
                        #                             num_prime,noise_level,noise,num_obs,k)
                        
                        count_cvar,count_cvar_lane,x_roll_cvar,y_roll_cvar,_,_ = compute_stats(cx_all_cvar[idx_cvar],cy_all_cvar[idx_cvar],
                                                                    init_state_all_cvar[idx_cvar],
                                                    x_obs_cvar[idx_cvar]
                                                    ,y_obs_cvar[idx_cvar],
                                                    vx_obs_cvar[idx_cvar]
                                                    ,vy_obs_cvar[idx_cvar],
                                                    num_prime,noise_level,noise,num_obs,k)
                        
                        # coll_mmd = np.append(coll_mmd,count_mmd)
                        # coll_mmd_lane = np.append(coll_mmd_lane,count_mmd_lane)
                    
                        coll_mmd_opt = np.append(coll_mmd_opt,count_mmd_opt)
                        coll_mmd_opt_lane = np.append(coll_mmd_opt_lane,count_mmd_opt_lane)

                        # coll_mmd_random = np.append(coll_mmd_random,count_mmd_random)
                        # coll_mmd_random_lane = np.append(coll_mmd_random_lane,count_mmd_random_lane)

                        coll_cvar = np.append(coll_cvar,count_cvar)
                        coll_cvar_lane = np.append(coll_cvar_lane,count_cvar_lane)
                        
                        if count_mmd_opt<=-1:   
                            th = np.linspace(0, 2*np.pi, 100)

                            # x_best_mmd = np.dot(prob.P,cx_all_mmd[idx_mmd].reshape(-1))[0:num_prime]
                            # y_best_mmd = np.dot(prob.P,cy_all_mmd[idx_mmd].reshape(-1))[0:num_prime]

                            x_best_mmd_opt = np.dot(prob.P,cx_all_mmd_opt[idx_mmd_opt].reshape(-1))[0:num_prime]
                            y_best_mmd_opt = np.dot(prob.P,cy_all_mmd_opt[idx_mmd_opt].reshape(-1))[0:num_prime]
                                
                            x_best_mmd_random = np.dot(prob.P,cx_all_mmd_random[idx_mmd_random].reshape(-1))[0:num_prime]
                            y_best_mmd_random = np.dot(prob.P,cy_all_mmd_random[idx_mmd_random].reshape(-1))[0:num_prime]
                            
                            x_best_cvar = np.dot(prob.P,cx_all_cvar[idx_cvar].reshape(-1))[0:num_prime]
                            y_best_cvar = np.dot(prob.P,cy_all_cvar[idx_cvar].reshape(-1))[0:num_prime]

                            plt.figure(1)
                            plt.plot(x_roll_mmd_opt.T, y_roll_mmd_opt.T, linewidth=0.1,color="r")
                            plt.plot(x_best_mmd_opt,y_best_mmd_opt,linewidth=4,color = "b")
                            plt.plot(x_path_d_lb[0:len_path],y_path_d_lb[0:len_path], color='tab:brown',linewidth=3*linewidth,linestyle="--")
                            plt.plot(x_path_d_ub[0:len_path],y_path_d_ub[0:len_path], color='tab:brown',linewidth=3*linewidth,linestyle="--")
                            plt.plot(x_path_d_mid[0:len_path],y_path_d_mid[0:len_path], color='tab:red',linewidth=3*linewidth,linestyle="--")
                            plt.plot(x_path_lb[0:len_path],y_path_lb[0:len_path], color='tab:red',linewidth=3*linewidth,linestyle="--")
                            plt.plot(x_path_ub[0:len_path],y_path_ub[0:len_path], color='tab:brown',linewidth=3*linewidth,linestyle="--")
                            
                            _x_obs = x_obs_mmd_opt[idx_mmd_opt].reshape(-1)
                            _y_obs = y_obs_mmd_opt[idx_mmd_opt].reshape(-1)

                            for i in range(0, prob.num_obs):
                                x_circ = _x_obs[i]+prob.a_obs*np.cos(th)
                                y_circ = _y_obs[i]+prob.b_obs*np.sin(th)
                                plt.plot(x_circ, y_circ, '-k')
                            
                            plt.axis('equal')	

                            # plt.figure(2)
                            # plt.plot(x_roll_mmd.T, y_roll_mmd.T, linewidth=0.1,color="r")
                            # plt.plot(x_best_mmd,y_best_mmd,linewidth=4,color = "b")
                            # plt.plot(x_path_d_lb[0:len_path],y_path_d_lb[0:len_path], color='tab:brown',linewidth=3*linewidth,linestyle="--")
                            # plt.plot(x_path_d_ub[0:len_path],y_path_d_ub[0:len_path], color='tab:brown',linewidth=3*linewidth,linestyle="--")
                            # plt.plot(x_path_d_mid[0:len_path],y_path_d_mid[0:len_path], color='tab:red',linewidth=3*linewidth,linestyle="--")
                            # plt.plot(x_path_lb[0:len_path],y_path_lb[0:len_path], color='tab:red',linewidth=3*linewidth,linestyle="--")
                            # plt.plot(x_path_ub[0:len_path],y_path_ub[0:len_path], color='tab:brown',linewidth=3*linewidth,linestyle="--")
                            
                            # _x_obs = x_obs_mmd[idx_mmd].reshape(-1)
                            # _y_obs = y_obs_mmd[idx_mmd].reshape(-1)

                            # for i in range(0, prob.num_obs):
                            #     x_circ = _x_obs[i]+prob.a_obs*np.cos(th)
                            #     y_circ = _y_obs[i]+prob.b_obs*np.sin(th)
                            #     plt.plot(x_circ, y_circ, '-k')
                            
                            # plt.axis('equal')	

                            # plt.figure(2)
                            # plt.plot(x_roll_mmd_random.T, y_roll_mmd_random.T, linewidth=0.5,color="r")
                            # plt.plot(x_best_mmd_random,y_best_mmd_random,linewidth=4,color = "b")

                            # plt.plot(x_path_d_lb[0:len_path],y_path_d_lb[0:len_path], color='tab:brown',linewidth=3*linewidth,linestyle="--")
                            # plt.plot(x_path_d_ub[0:len_path],y_path_d_ub[0:len_path], color='tab:brown',linewidth=3*linewidth,linestyle="--")
                            # plt.plot(x_path_d_mid[0:len_path],y_path_d_mid[0:len_path], color='tab:red',linewidth=3*linewidth,linestyle="--")
                            # plt.plot(x_path_lb[0:len_path],y_path_lb[0:len_path], color='tab:red',linewidth=3*linewidth,linestyle="--")
                            # plt.plot(x_path_ub[0:len_path],y_path_ub[0:len_path], color='tab:brown',linewidth=3*linewidth,linestyle="--")
                            
                            # _x_obs = x_obs_mmd_opt[idx_mmd_opt].reshape(-1)
                            # _y_obs = y_obs_mmd_opt[idx_mmd_opt].reshape(-1)

                            # for i in range(0, prob.num_obs):
                            #     x_circ = _x_obs[i]+prob.a_obs*np.cos(th)
                            #     y_circ = _y_obs[i]+prob.b_obs*np.sin(th)
                            #     plt.plot(x_circ, y_circ, '-k')
                            
                            # plt.axis('equal')	

                            plt.figure(2)
                            plt.plot(x_roll_cvar.T, y_roll_cvar.T, linewidth=0.5,color="r")
                            plt.plot(x_best_cvar,y_best_cvar,linewidth=4,color = "b")
                            plt.plot(x_path_d_lb[0:len_path],y_path_d_lb[0:len_path], color='tab:brown',linewidth=3*linewidth,linestyle="--")
                            plt.plot(x_path_d_ub[0:len_path],y_path_d_ub[0:len_path], color='tab:brown',linewidth=3*linewidth,linestyle="--")
                            plt.plot(x_path_d_mid[0:len_path],y_path_d_mid[0:len_path], color='tab:red',linewidth=3*linewidth,linestyle="--")
                            plt.plot(x_path_lb[0:len_path],y_path_lb[0:len_path], color='tab:red',linewidth=3*linewidth,linestyle="--")
                            plt.plot(x_path_ub[0:len_path],y_path_ub[0:len_path], color='tab:brown',linewidth=3*linewidth,linestyle="--")
                            
                            _x_obs = x_obs_mmd_opt[idx_mmd_opt].reshape(-1)
                            _y_obs = y_obs_mmd_opt[idx_mmd_opt].reshape(-1)

                            for i in range(0, prob.num_obs):
                                x_circ = _x_obs[i]+prob.a_obs*np.cos(th)
                                y_circ = _y_obs[i]+prob.b_obs*np.sin(th)
                                plt.plot(x_circ, y_circ, '-k')
                            
                            plt.axis('equal')
                            plt.show()

                    # if num_prime==60 and noise=="beta" and noise_level==0.3 and num_reduced==5:
                    #     print(coll_mmd,coll_cvar,coll_mmd_random)
                    np.savez("./stats/{}_noise/noise_{}/ts_{}/{}_samples_{}_obs".format(noise,int(noise_level*100),
                                num_prime,
                                num_reduced,num_obs),
                                coll_cvar = coll_cvar,coll_cvar_lane = coll_cvar_lane,
                                coll_mmd_opt = coll_mmd_opt,coll_mmd_opt_lane = coll_mmd_opt_lane,
                                coll_mmd_random=coll_mmd_random,coll_mmd_random_lane=coll_mmd_random_lane)

