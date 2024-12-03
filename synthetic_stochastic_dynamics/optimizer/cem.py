import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random,vmap,lax
import jax
import jax.lax as lax
import bernstein_coeff_order10_arbitinterval
from kernel_computation import kernel_matrix
import time
from compute_beta import beta_cem
from projection import Projection
from cem_helper import Helper
from costs import Costs
from compute_beta import beta_cem

class CEM():
    def __init__(self,num_reduced,num_obs
                 ,noise_level,num_prime,noise,acc_const_noise,steer_const_noise):
        
        self.acc_const_noise = acc_const_noise
        self.steer_const_noise = steer_const_noise
        
        self.noise = noise
        self.beta_a ,self.beta_b = 2,5
        self.a_obs,self.b_obs = 4.25,2.75
        self.wheel_base = 2.5
        self.kappa_max = 0.230
        self.a_centr = 1.5
        self.num_circles = 1
        self.v_max = 30.0 
        self.v_min = 0.1
        self.a_max = 18.0
        self.num_obs = num_obs
        self.steer_max = 0.6
        self.steer_rate_max = 0.6

        self.t_fin = 15
        self.num = 100

        self.t = self.t_fin/self.num
        
        tot_time = np.linspace(0, self.t_fin, self.num)
        self.tot_time = tot_time
        tot_time_copy = tot_time.reshape(self.num, 1)

        self.P, self.Pdot, self.Pddot = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)

        self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)

        self.nvar = jnp.shape(self.P_jax)[1]
        
        self.num_prime = num_prime
                
        ################################################################
        self.A_eq_x = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.Pddot_jax[0]  ))
        self.A_eq_y = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.Pddot_jax[0], self.Pdot_jax[-1]  ))
             
        self.A_vel = self.Pdot_jax 
        self.A_acc = self.Pddot_jax
        self.A_projection = jnp.identity(self.nvar)
        
        ###################################3 obstacle avoidance		
        self.A_y_centerline = self.P_jax
        self.A_lane = jnp.vstack(( self.P_jax, -self.P_jax    ))
        self.A_obs = jnp.tile(self.P_jax, ((self.num_obs)*self.num_circles, 1))

        ###################################################

        self.rho_nonhol = 1.0
        self.rho_ineq = 1
        self.rho_obs = 1.0
        self.rho_projection = 1.0
        self.rho_goal = 1.0
        self.rho_lane = 1.0
        self.rho_long = 1.0

        self.rho_v = 1 
        self.rho_offset = 1

        self.weight_smoothness = 100
        self.cost_smoothness = self.weight_smoothness*jnp.dot(self.Pddot_jax.T, self.Pddot_jax)

        #################################################
        self.weight_smoothness_x = 100
        self.weight_smoothness_y = 100

        #################################################
        self.maxiter = 1
        self.maxiter_cem = 20

        self.k_p_v = 2
        self.k_d_v = 2.0*jnp.sqrt(self.k_p_v)

        self.k_p = 2
        self.k_d = 2.0*jnp.sqrt(self.k_p)

        self.P_jax_1 = self.P_jax[0:25, :]
        self.P_jax_2 = self.P_jax[25:50, :]
        self.P_jax_3 = self.P_jax[50:75, :]
        self.P_jax_4 = self.P_jax[75:100, :]

        self.Pdot_jax_1 = self.Pdot_jax[0:25, :]
        self.Pdot_jax_2 = self.Pdot_jax[25:50, :]
        self.Pdot_jax_3 = self.Pdot_jax[50:75, :]
        self.Pdot_jax_4 = self.Pdot_jax[75:100, :]
            
        self.Pddot_jax_1 = self.Pddot_jax[0:25, :]
        self.Pddot_jax_2 = self.Pddot_jax[25:50, :]
        self.Pddot_jax_3 = self.Pddot_jax[50:75, :]
        self.Pddot_jax_4 = self.Pddot_jax[75:100, :]

        self.num_partial = 25
        ###########################################3
        key = random.PRNGKey(0)

        self.key = key

        self.alpha_mean = 0.6
        self.alpha_cov = 0.6

        self.lamda = 0.9

        self.gamma = 1.
        self.gamma_obs = 1.
        # upper lane bound
        self.P_ub_1 = self.P_jax[1:self.num,:]
        self.P_ub_0 = self.P_jax[0:self.num-1,:]
        self.A_ub = self.P_ub_1 + (self.gamma-1)*self.P_ub_0

        # lower lane bound
        self.P_lb_1 = -self.P_jax[1:self.num,:]
        self.P_lb_0 = self.P_jax[0:self.num-1,:]
        self.A_lb = self.P_lb_1 + (1-self.gamma)*self.P_lb_0
        self.A_lane_bound = jnp.vstack((self.A_ub,self.A_lb))
      
        self.num_params = 8 
        self.num_batch = 100
        self.ellite_num = 5
        self.ellite_num_projection = self.num_batch
        self.ellite_num_cost = 20

        self.num_reduced = num_reduced
        self.num_mother = num_reduced**2

        self.modes = jnp.asarray([1,2,3])
        self.modes_probs = jnp.asarray([0.4,0.2,0.4])
        self.mu = jnp.asarray([[0.5,0.,0.5,0.],[0.5,-0.1,0.9,0.01],[-0.2,0.1,1.,-0.015]])
        self.sigma = jnp.asarray([[0.1,0.1,1,0.1],[0.02,0.01,0.8,0.05],[0.1,0.01,0.1,0.01]])  
        self.size_1 = int(self.modes_probs[0]*self.num_reduced) 
        self.size_2 = int(self.modes_probs[1]*self.num_reduced) 
        self.size_3 = int(self.modes_probs[2]*self.num_reduced) 
        self.size_1 = jax.lax.cond(self.size_1+self.size_2+self.size_3==self.num_reduced,
                                lambda _: self.size_1,lambda _: self.num_reduced-(self.size_2 + self.size_3), 0)
        
        self.y_lb,self.y_ub = -2.25,2.25 #-2.25,-1.25 # 
        self.y_des_1, self.y_des_2 = -1.75,1.75

        self.alpha_quant = 0.98
        self.alpha_quant_lane = 0.98

        self.weight_mmd_lane,self.weight_mmd_obs = 0*10**(3),10**(3)
        self.weight_cvar_lane,self.weight_cvar_obs = 0*10**(3),10**(3)
        self.weight_saa_lane,self.weight_saa_obs = 10**(6),10**(6)# 10**(3),10**(3) #

        self.sigma_ker = 10**(1)
        self.ker_wt = 1000.

        self.sigma_acc = noise_level
        self.sigma_steer = noise_level

        self.gamma_lane_des = 0.3

        self.prob = kernel_matrix(self.num_reduced,self.ker_wt,self.P_jax)
        
        self.prob2 = beta_cem(self.num_reduced,self.num_mother,self.ker_wt,self.P_jax)
        
        self.projection = Projection(self.num_obs,self.num_circles,self.v_max,self.v_min,self.a_max,
                 self.num,self.P_jax,self.Pdot_jax,self.Pddot_jax,self.rho_ineq,self.rho_obs,self.rho_projection,self.rho_lane,
                 self.gamma,self.gamma_obs,self.num_batch,self.maxiter,self.nvar,
                 self.A_eq_x,self.A_eq_y,self.A_obs,self.A_lane_bound,self.y_lb,self.y_ub,
                 self.A_vel,self.A_acc,self.A_projection)
        
        self.cem_helper = Helper(self.v_min,self.v_max, self.P,self.Pdot,self.Pddot,self.rho_v,self.rho_offset,
                 self.weight_smoothness_x,self.weight_smoothness_y,self.k_p_v,self.k_d_v,self.k_p,self.k_d,
                 self.P_jax_1,self.P_jax_2,self.P_jax_3,self.P_jax_4,
                 self.Pdot_jax_1,self.Pdot_jax_2,self.Pdot_jax_3,self.Pdot_jax_4,
                 self.Pddot_jax_1,self.Pddot_jax_2,self.Pddot_jax_3,self.Pddot_jax_4,
                 self.num_params,self.num_batch,self.ellite_num,self.num_partial,self.alpha_mean,self.alpha_cov,
                 self.lamda,self.modes,self.modes_probs,self.mu,self.sigma,self.size_1,self.size_2,self.size_3,
                 tot_time,self.wheel_base,self.num_reduced,
                 self.num_prime,self.num_circles,num_obs,self.A_eq_x,self.A_eq_y,self.y_des_1,self.y_des_2,self.ellite_num_cost,self.num,
                 self.steer_max,self.nvar,self.t,self.sigma_acc,
                 self.sigma_steer,self.beta_a,self.beta_b,self.noise,self.num_mother,self.prob,self.prob2,
                 self.acc_const_noise,self.steer_const_noise)
        
        self.costs = Costs(self.prob,self.num_reduced,self.num_obs,self.num_prime,
                           self.a_obs,self.b_obs,self.y_lb,self.y_ub,
                           self.alpha_quant,self.alpha_quant_lane,
                           self.y_des_1,self.y_des_2,self.gamma_lane_des)

    @partial(jit, static_argnums=(0, ))	
    def compute_cem_mmd_opt(self,idx_mpc,init_state,
                    mean_param_init,cov_param_init,
                    x_obs_traj,y_obs_traj,v_des):
        
        lamda_x_init = jnp.zeros((self.num_batch, self.nvar))
        lamda_y_init = jnp.zeros((self.num_batch, self.nvar))
        s_lane_init = jnp.zeros((self.num_batch, 2*(self.num-1)))

        res_init = jnp.zeros(self.maxiter_cem)
        res_2_init = jnp.zeros(self.maxiter_cem)

        neural_output_batch_init = self.cem_helper.sampling_param(mean_param_init, cov_param_init)
       
        x_init, y_init, vx_init, vy_init, ax_init, ay_init = init_state
        b_eq_x, b_eq_y = self.cem_helper.compute_boundary_vec(x_init, vx_init, ax_init, y_init, vy_init, ay_init)
        
        _initial_state = jnp.asarray([x_init,y_init,
                                    vx_init,vy_init,jnp.arctan2(vy_init,vx_init)])
        
        def lax_cem(carry,idx):

            res,res_2,lamda_x, lamda_y, neural_output_batch,mean_param,cov_param,s_lane = carry

            key = jax.random.PRNGKey(3*idx_mpc + 5*idx + 7)

            c_x_bar, c_y_bar = self.cem_helper.compute_x_guess(b_eq_x, b_eq_y,
                                                    neural_output_batch)

            c_x, c_y, x, y, xdot, ydot, xddot, yddot, res_norm_batch,lamda_x, lamda_y,s_lane = self.projection.compute_projection(x_obs_traj,y_obs_traj,b_eq_x, b_eq_y,
                                                                            lamda_x, lamda_y, c_x_bar, c_y_bar,self.a_obs,self.b_obs,s_lane)

            idx_ellite_projection = jnp.argsort(res_norm_batch)

            x_ellite_projection = x[idx_ellite_projection[0:self.ellite_num_projection]]
            y_ellite_projection = y[idx_ellite_projection[0:self.ellite_num_projection]]

            xdot_ellite_projection = xdot[idx_ellite_projection[0:self.ellite_num_projection]]
            ydot_ellite_projection = ydot[idx_ellite_projection[0:self.ellite_num_projection]]
            
            xddot_ellite_projection = xddot[idx_ellite_projection[0:self.ellite_num_projection]]
            yddot_ellite_projection = yddot[idx_ellite_projection[0:self.ellite_num_projection]]

            c_x_ellite_projection = c_x[idx_ellite_projection[0:self.ellite_num_projection]]
            c_y_ellite_projection = c_y[idx_ellite_projection[0:self.ellite_num_projection]]
            res_ellite_projection = res_norm_batch[idx_ellite_projection[0:self.ellite_num_projection]]

            neural_output_projection = neural_output_batch[idx_ellite_projection[0:self.ellite_num_projection]]

            acc_ellite_projection,steering_ellite_projection \
                = self.cem_helper.compute_controls(xdot_ellite_projection, ydot_ellite_projection,
                                        xddot_ellite_projection,yddot_ellite_projection)
            
            key,subkey = jax.random.split(key)

            x_roll_ellite_projection,y_roll_ellite_projection,beta,sigma, res_beta \
                = self.cem_helper.compute_rollout_opt_vmap(acc_ellite_projection[:,0:self.num_prime],
                                                            steering_ellite_projection[:,0:self.num_prime]
                                                      ,_initial_state,key)
            
            mmd_obs = self.costs.compute_mmd_obs_vmap(beta,sigma,x_roll_ellite_projection,y_roll_ellite_projection,
                                              x_obs_traj[:,0:self.num_prime],y_obs_traj[:,0:self.num_prime])

            idx_ellite_mmd = jnp.argsort(mmd_obs)

            mmd_obs_ellite = mmd_obs[idx_ellite_mmd[0:self.ellite_num_cost]]

            x_ellite_mmd = x_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            y_ellite_mmd = y_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]

            xdot_ellite_mmd = xdot_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            ydot_ellite_mmd = ydot_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            
            xddot_ellite_mmd = xddot_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            yddot_ellite_mmd = yddot_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]

            c_x_ellite_mmd = c_x_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            c_y_ellite_mmd = c_y_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            res_ellite_mmd = res_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]

            neural_ellite_mmd = neural_output_projection[idx_ellite_mmd[0:self.ellite_num_cost]]

            x_roll_ellite_mmd = x_roll_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            y_roll_ellite_mmd = y_roll_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            steer_ellite_mmd = steering_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            
            beta_ellite_mmd = beta[idx_ellite_mmd[0:self.ellite_num_cost]]
            sigma_ellite_mmd = sigma[idx_ellite_mmd[0:self.ellite_num_cost]]
            res_beta_ellite_mmd = res_beta[idx_ellite_mmd[0:self.ellite_num_cost]]

            mmd_lane  = self.costs.compute_mmd_lane_vmap(beta_ellite_mmd,sigma_ellite_mmd,y_roll_ellite_mmd)
            # mmd_des_lane  = self.costs.compute_lane_des_mmd_vmap(gamma_ellite_mmd,beta_1_ellite_mmd,beta_2_ellite_mmd,y_roll_ellite_mmd)

            cost_batch = self.cem_helper.compute_cost(self.weight_mmd_obs*mmd_obs_ellite,self.weight_mmd_lane*mmd_lane
                                                      ,x_ellite_mmd,y_ellite_mmd, res_ellite_mmd, \
                                        xdot_ellite_mmd, ydot_ellite_mmd, 
                                        xddot_ellite_mmd, yddot_ellite_mmd, \
                                            v_des, steer_ellite_mmd)

            neural_output_ellite, idx_ellite = self.cem_helper.compute_ellite_samples(cost_batch, neural_ellite_mmd)
            
            key, subkey = jax.random.split(key)

            mean_param, cov_param, neural_output_batch, cost_batch_temp\
                 = self.cem_helper.compute_shifted_samples(key, neural_output_ellite, 
                cost_batch, idx_ellite, mean_param, cov_param)
        
            idx_min = jnp.argmin(cost_batch_temp)
          
            res = res.at[idx].set(jnp.min(cost_batch_temp))
            res_2 = res_2.at[idx].set(res_ellite_mmd[idx_min])

            return (res,res_2,lamda_x, lamda_y, neural_output_batch,mean_param,cov_param,s_lane),\
                (c_x_ellite_mmd[idx_min],c_y_ellite_mmd[idx_min],mmd_lane[idx_min],mmd_obs_ellite[idx_min],
                 beta_ellite_mmd[idx_min],sigma_ellite_mmd[idx_min],res_beta_ellite_mmd[idx_min])

        carry_init = (res_init,res_2_init,lamda_x_init, lamda_y_init, \
            neural_output_batch_init,mean_param_init, cov_param_init,s_lane_init)

        carry_final,result = lax.scan(lax_cem,carry_init,jnp.arange(self.maxiter_cem))

        res,res_2,lamda_x, lamda_y, neural_output_batch,mean_param,cov_param,s_lane = carry_final 
        
        cx_best = result[0][-1]
        cy_best = result[1][-1]
       
        mmd_lane = result[2][-1]
        mmd_obs = result[3][-1]
        beta= result[4][-1]
        sigma = result[5][-1]
        res_beta = result[6][-1]

        return cx_best,cy_best,mmd_lane,mmd_obs,beta,sigma,res_beta
    
    @partial(jit, static_argnums=(0, ))	
    def compute_cem_mmd_random(self,idx_mpc,init_state,
                    mean_param_init,cov_param_init,
                    x_obs_traj,y_obs_traj,v_des):
        
        lamda_x_init = jnp.zeros((self.num_batch, self.nvar))
        lamda_y_init = jnp.zeros((self.num_batch, self.nvar))
        s_lane_init = jnp.zeros((self.num_batch, 2*(self.num-1)))

        res_init = jnp.zeros(self.maxiter_cem)
        res_2_init = jnp.zeros(self.maxiter_cem)

        neural_output_batch_init = self.cem_helper.sampling_param(mean_param_init, cov_param_init)
       
        x_init, y_init, vx_init, vy_init, ax_init, ay_init = init_state
        b_eq_x, b_eq_y = self.cem_helper.compute_boundary_vec(x_init, vx_init, ax_init, y_init, vy_init, ay_init)
        
        _initial_state = jnp.asarray([x_init,y_init,
                                    vx_init,vy_init,jnp.arctan2(vy_init,vx_init)])
        
        beta = (1/self.num_reduced)*jnp.ones((self.ellite_num_projection,self.num_reduced))
        sigma= 0.01*jnp.ones(self.ellite_num_projection)
        
        def lax_cem(carry,idx):

            res,res_2,lamda_x, lamda_y, neural_output_batch,mean_param,cov_param,s_lane = carry

            key = jax.random.PRNGKey(3*idx_mpc + 5*idx + 7)

            c_x_bar, c_y_bar = self.cem_helper.compute_x_guess(b_eq_x, b_eq_y,
                                                    neural_output_batch)

            c_x, c_y, x, y, xdot, ydot, xddot, yddot, res_norm_batch,lamda_x, lamda_y,s_lane = self.projection.compute_projection(x_obs_traj,y_obs_traj,b_eq_x, b_eq_y,
                                                                            lamda_x, lamda_y, c_x_bar, c_y_bar,self.a_obs,self.b_obs,s_lane)

            idx_ellite_projection = jnp.argsort(res_norm_batch)

            x_ellite_projection = x[idx_ellite_projection[0:self.ellite_num_projection]]
            y_ellite_projection = y[idx_ellite_projection[0:self.ellite_num_projection]]

            xdot_ellite_projection = xdot[idx_ellite_projection[0:self.ellite_num_projection]]
            ydot_ellite_projection = ydot[idx_ellite_projection[0:self.ellite_num_projection]]
            
            xddot_ellite_projection = xddot[idx_ellite_projection[0:self.ellite_num_projection]]
            yddot_ellite_projection = yddot[idx_ellite_projection[0:self.ellite_num_projection]]

            c_x_ellite_projection = c_x[idx_ellite_projection[0:self.ellite_num_projection]]
            c_y_ellite_projection = c_y[idx_ellite_projection[0:self.ellite_num_projection]]
            res_ellite_projection = res_norm_batch[idx_ellite_projection[0:self.ellite_num_projection]]

            neural_output_projection = neural_output_batch[idx_ellite_projection[0:self.ellite_num_projection]]

            acc_ellite_projection,steering_ellite_projection \
                = self.cem_helper.compute_controls(xdot_ellite_projection, ydot_ellite_projection,
                                        xddot_ellite_projection,yddot_ellite_projection)
            
            key,subkey = jax.random.split(key)

            x_roll_ellite_projection,y_roll_ellite_projection \
                = self.cem_helper.compute_rollout_baseline_vmap(acc_ellite_projection[:,0:self.num_prime],
                                                            steering_ellite_projection[:,0:self.num_prime]
                                                      ,_initial_state,key)
            
            mmd_obs = self.costs.compute_mmd_obs_vmap(beta,sigma,x_roll_ellite_projection,y_roll_ellite_projection,
                                              x_obs_traj[:,0:self.num_prime],y_obs_traj[:,0:self.num_prime])

            idx_ellite_mmd = jnp.argsort(mmd_obs)

            mmd_obs_ellite = mmd_obs[idx_ellite_mmd[0:self.ellite_num_cost]]

            x_ellite_mmd = x_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            y_ellite_mmd = y_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]

            xdot_ellite_mmd = xdot_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            ydot_ellite_mmd = ydot_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            
            xddot_ellite_mmd = xddot_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            yddot_ellite_mmd = yddot_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]

            c_x_ellite_mmd = c_x_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            c_y_ellite_mmd = c_y_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            res_ellite_mmd = res_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]

            neural_ellite_mmd = neural_output_projection[idx_ellite_mmd[0:self.ellite_num_cost]]

            x_roll_ellite_mmd = x_roll_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            y_roll_ellite_mmd = y_roll_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            steer_ellite_mmd = steering_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            
            beta_ellite_mmd = beta[idx_ellite_mmd[0:self.ellite_num_cost]]
            # sigma_ellite_mmd = sigma[idx_ellite_mmd[0:self.ellite_num_cost]]

            mmd_lane  = jnp.zeros((self.ellite_num_cost)) #self.costs.compute_mmd_lane_vmap(beta_ellite_mmd,sigma_ellite_mmd,y_roll_ellite_mmd)

            cost_batch = self.cem_helper.compute_cost(self.weight_mmd_obs*mmd_obs_ellite,self.weight_mmd_lane*mmd_lane
                                                      ,x_ellite_mmd,y_ellite_mmd, res_ellite_mmd, \
                                        xdot_ellite_mmd, ydot_ellite_mmd, 
                                        xddot_ellite_mmd, yddot_ellite_mmd, \
                                            v_des, steer_ellite_mmd)

            neural_output_ellite, idx_ellite = self.cem_helper.compute_ellite_samples(cost_batch, neural_ellite_mmd)
            
            key, subkey = jax.random.split(key)

            mean_param, cov_param, neural_output_batch, cost_batch_temp\
                 = self.cem_helper.compute_shifted_samples(key, neural_output_ellite, 
                cost_batch, idx_ellite, mean_param, cov_param)
        
            idx_min = jnp.argmin(cost_batch_temp)
          
            res = res.at[idx].set(jnp.min(cost_batch_temp))
            res_2 = res_2.at[idx].set(res_ellite_mmd[idx_min])

            return (res,res_2,lamda_x, lamda_y, neural_output_batch,mean_param,cov_param,s_lane),\
                (c_x_ellite_mmd[idx_min],c_y_ellite_mmd[idx_min],mmd_lane[idx_min],mmd_obs_ellite[idx_min])

        carry_init = (res_init,res_2_init,lamda_x_init, lamda_y_init, \
            neural_output_batch_init,mean_param_init, cov_param_init,s_lane_init)

        carry_final,result = lax.scan(lax_cem,carry_init,jnp.arange(self.maxiter_cem))
        
        cx_best = result[0][-1]
        cy_best = result[1][-1]
       
        mmd_lane = result[2][-1]
        mmd_obs = result[3][-1]

        return cx_best,cy_best,mmd_lane,mmd_obs
    
    @partial(jit, static_argnums=(0, ))	
    def compute_cem_cvar(self,idx_mpc,init_state,
                    mean_param_init,cov_param_init,
                    x_obs_traj,y_obs_traj,v_des):

        lamda_x_init = jnp.zeros((self.num_batch, self.nvar))
        lamda_y_init = jnp.zeros((self.num_batch, self.nvar))
        s_lane_init = jnp.zeros((self.num_batch, 2*(self.num-1)))

        res_init = jnp.zeros(self.maxiter_cem)
        res_2_init = jnp.zeros(self.maxiter_cem)

        neural_output_batch_init = self.cem_helper.sampling_param(mean_param_init, cov_param_init)
       
        x_init, y_init, vx_init, vy_init, ax_init, ay_init = init_state
        b_eq_x, b_eq_y = self.cem_helper.compute_boundary_vec(x_init, vx_init, ax_init, y_init, vy_init, ay_init)
        
        _initial_state = jnp.asarray([x_init,y_init,
                                    vx_init,vy_init,jnp.arctan2(vy_init,vx_init)])
        
        def lax_cem(carry,idx):

            res,res_2,lamda_x, lamda_y, neural_output_batch,mean_param,cov_param,s_lane = carry

            key = jax.random.PRNGKey(3*idx_mpc + 5*idx + 7)

            c_x_bar, c_y_bar = self.cem_helper.compute_x_guess(b_eq_x, b_eq_y,
                                                    neural_output_batch)

            c_x, c_y, x, y, xdot, ydot, xddot, yddot, res_norm_batch,lamda_x, lamda_y,s_lane = self.projection.compute_projection(x_obs_traj,y_obs_traj,b_eq_x, b_eq_y,
                                                                            lamda_x, lamda_y, c_x_bar, c_y_bar,self.a_obs,self.b_obs,s_lane)

            idx_ellite_projection = jnp.argsort(res_norm_batch)

            x_ellite_projection = x[idx_ellite_projection[0:self.ellite_num_projection]]
            y_ellite_projection = y[idx_ellite_projection[0:self.ellite_num_projection]]

            xdot_ellite_projection = xdot[idx_ellite_projection[0:self.ellite_num_projection]]
            ydot_ellite_projection = ydot[idx_ellite_projection[0:self.ellite_num_projection]]
            
            xddot_ellite_projection = xddot[idx_ellite_projection[0:self.ellite_num_projection]]
            yddot_ellite_projection = yddot[idx_ellite_projection[0:self.ellite_num_projection]]

            c_x_ellite_projection = c_x[idx_ellite_projection[0:self.ellite_num_projection]]
            c_y_ellite_projection = c_y[idx_ellite_projection[0:self.ellite_num_projection]]
            res_ellite_projection = res_norm_batch[idx_ellite_projection[0:self.ellite_num_projection]]

            neural_output_projection = neural_output_batch[idx_ellite_projection[0:self.ellite_num_projection]]

            acc_ellite_projection,steering_ellite_projection \
                = self.cem_helper.compute_controls(xdot_ellite_projection, ydot_ellite_projection,
                                        xddot_ellite_projection,yddot_ellite_projection)
            
            key,subkey = jax.random.split(key)

            x_roll_ellite_projection,y_roll_ellite_projection \
                = self.cem_helper.compute_rollout_baseline_vmap(acc_ellite_projection[:,0:self.num_prime],
                                                            steering_ellite_projection[:,0:self.num_prime]
                                                      ,_initial_state,key)
            
            mmd_obs = self.costs.compute_cvar_obs_vmap(x_roll_ellite_projection,y_roll_ellite_projection,
                                              x_obs_traj[:,0:self.num_prime],y_obs_traj[:,0:self.num_prime])

            idx_ellite_mmd = jnp.argsort(mmd_obs)

            mmd_obs_ellite = mmd_obs[idx_ellite_mmd[0:self.ellite_num_cost]]

            x_ellite_mmd = x_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            y_ellite_mmd = y_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]

            xdot_ellite_mmd = xdot_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            ydot_ellite_mmd = ydot_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            
            xddot_ellite_mmd = xddot_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            yddot_ellite_mmd = yddot_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]

            c_x_ellite_mmd = c_x_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            c_y_ellite_mmd = c_y_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            res_ellite_mmd = res_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]

            neural_ellite_mmd = neural_output_projection[idx_ellite_mmd[0:self.ellite_num_cost]]

            x_roll_ellite_mmd = x_roll_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            y_roll_ellite_mmd = y_roll_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            steer_ellite_mmd = steering_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
           
            mmd_lane  = self.costs.compute_cvar_lane_vmap(y_roll_ellite_mmd)
            # mmd_des_lane  = self.costs.compute_lane_des_mmd_vmap(gamma_ellite_mmd,beta_1_ellite_mmd,beta_2_ellite_mmd,y_roll_ellite_mmd)

            cost_batch = self.cem_helper.compute_cost(self.weight_cvar_obs*mmd_obs_ellite,self.weight_cvar_lane*mmd_lane,
                                                      x_ellite_mmd,y_ellite_mmd, res_ellite_mmd, \
                                        xdot_ellite_mmd, ydot_ellite_mmd, 
                                        xddot_ellite_mmd, yddot_ellite_mmd, \
                                            v_des, steer_ellite_mmd)

            neural_output_ellite, idx_ellite = self.cem_helper.compute_ellite_samples(cost_batch, neural_ellite_mmd)
            
            key, subkey = jax.random.split(key)

            mean_param, cov_param, neural_output_batch, cost_batch_temp\
                 = self.cem_helper.compute_shifted_samples(key, neural_output_ellite, 
                cost_batch, idx_ellite, mean_param, cov_param)
        
            idx_min = jnp.argmin(cost_batch_temp)
           
            res = res.at[idx].set(jnp.min(cost_batch_temp))
            res_2 = res_2.at[idx].set(res_ellite_mmd[idx_min])

            return (res,res_2,lamda_x, lamda_y, neural_output_batch,mean_param,cov_param,s_lane),\
                (c_x_ellite_mmd[idx_min],c_y_ellite_mmd[idx_min],mmd_lane[idx_min],mmd_obs_ellite[idx_min])

        carry_init = (res_init,res_2_init,lamda_x_init, lamda_y_init, \
            neural_output_batch_init,mean_param_init, cov_param_init,s_lane_init)

        carry_final,result = lax.scan(lax_cem,carry_init,jnp.arange(self.maxiter_cem))

        res,res_2,lamda_x, lamda_y, neural_output_batch,mean_param,cov_param,s_lane = carry_final 
        
        cx_best = result[0][-1]
        cy_best = result[1][-1]
       
        mmd_lane = result[2][-1]
        mmd_obs = result[3][-1]

        return cx_best,cy_best,mmd_lane,mmd_obs
    
    @partial(jit, static_argnums=(0, ))	
    def compute_cem_saa(self,idx_mpc,init_state,
                    mean_param_init,cov_param_init,
                    x_obs_traj,y_obs_traj,v_des):

        lamda_x_init = jnp.zeros((self.num_batch, self.nvar))
        lamda_y_init = jnp.zeros((self.num_batch, self.nvar))
        s_lane_init = jnp.zeros((self.num_batch, 2*(self.num-1)))

        res_init = jnp.zeros(self.maxiter_cem)
        res_2_init = jnp.zeros(self.maxiter_cem)

        neural_output_batch_init = self.cem_helper.sampling_param(mean_param_init, cov_param_init)
       
        x_init, y_init, vx_init, vy_init, ax_init, ay_init = init_state
        b_eq_x, b_eq_y = self.cem_helper.compute_boundary_vec(x_init, vx_init, ax_init, y_init, vy_init, ay_init)
        
        _initial_state = jnp.asarray([x_init,y_init,
                                    vx_init,vy_init,jnp.arctan2(vy_init,vx_init)])
        
        def lax_cem(carry,idx):

            res,res_2,lamda_x, lamda_y, neural_output_batch,mean_param,cov_param,s_lane = carry

            key = jax.random.PRNGKey(3*idx_mpc + 5*idx + 7)

            c_x_bar, c_y_bar = self.cem_helper.compute_x_guess(b_eq_x, b_eq_y,
                                                    neural_output_batch)

            c_x, c_y, x, y, xdot, ydot, xddot, yddot, res_norm_batch,lamda_x, lamda_y,s_lane = self.projection.compute_projection(x_obs_traj,y_obs_traj,b_eq_x, b_eq_y,
                                                                            lamda_x, lamda_y, c_x_bar, c_y_bar,self.a_obs,self.b_obs,s_lane)

            idx_ellite_projection = jnp.argsort(res_norm_batch)

            x_ellite_projection = x[idx_ellite_projection[0:self.ellite_num_projection]]
            y_ellite_projection = y[idx_ellite_projection[0:self.ellite_num_projection]]

            xdot_ellite_projection = xdot[idx_ellite_projection[0:self.ellite_num_projection]]
            ydot_ellite_projection = ydot[idx_ellite_projection[0:self.ellite_num_projection]]
            
            xddot_ellite_projection = xddot[idx_ellite_projection[0:self.ellite_num_projection]]
            yddot_ellite_projection = yddot[idx_ellite_projection[0:self.ellite_num_projection]]

            c_x_ellite_projection = c_x[idx_ellite_projection[0:self.ellite_num_projection]]
            c_y_ellite_projection = c_y[idx_ellite_projection[0:self.ellite_num_projection]]
            res_ellite_projection = res_norm_batch[idx_ellite_projection[0:self.ellite_num_projection]]

            neural_output_projection = neural_output_batch[idx_ellite_projection[0:self.ellite_num_projection]]

            acc_ellite_projection,steering_ellite_projection \
                = self.cem_helper.compute_controls(xdot_ellite_projection, ydot_ellite_projection,
                                        xddot_ellite_projection,yddot_ellite_projection)
            
            key,subkey = jax.random.split(key)

            x_roll_ellite_projection,y_roll_ellite_projection \
                = self.cem_helper.compute_rollout_baseline_vmap(acc_ellite_projection[:,0:self.num_prime],
                                                            steering_ellite_projection[:,0:self.num_prime]
                                                      ,_initial_state,key)
            
            mmd_obs = self.costs.compute_saa_obs_vmap(x_roll_ellite_projection,y_roll_ellite_projection,
                                              x_obs_traj[:,0:self.num_prime],y_obs_traj[:,0:self.num_prime])

            idx_ellite_mmd = jnp.argsort(mmd_obs)

            mmd_obs_ellite = mmd_obs[idx_ellite_mmd[0:self.ellite_num_cost]]

            x_ellite_mmd = x_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            y_ellite_mmd = y_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]

            xdot_ellite_mmd = xdot_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            ydot_ellite_mmd = ydot_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            
            xddot_ellite_mmd = xddot_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            yddot_ellite_mmd = yddot_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]

            c_x_ellite_mmd = c_x_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            c_y_ellite_mmd = c_y_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            res_ellite_mmd = res_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]

            neural_ellite_mmd = neural_output_projection[idx_ellite_mmd[0:self.ellite_num_cost]]

            x_roll_ellite_mmd = x_roll_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            y_roll_ellite_mmd = y_roll_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
            steer_ellite_mmd = steering_ellite_projection[idx_ellite_mmd[0:self.ellite_num_cost]]
           
            mmd_lane  = self.costs.compute_saa_lane_vmap(y_roll_ellite_mmd)
            # mmd_des_lane  = self.costs.compute_lane_des_mmd_vmap(gamma_ellite_mmd,beta_1_ellite_mmd,beta_2_ellite_mmd,y_roll_ellite_mmd)

            cost_batch = self.cem_helper.compute_cost(self.weight_saa_obs*mmd_obs_ellite,self.weight_saa_lane*mmd_lane,
                                                      x_ellite_mmd,y_ellite_mmd, res_ellite_mmd, \
                                        xdot_ellite_mmd, ydot_ellite_mmd, 
                                        xddot_ellite_mmd, yddot_ellite_mmd, \
                                            v_des, steer_ellite_mmd)

            neural_output_ellite, idx_ellite = self.cem_helper.compute_ellite_samples(cost_batch, neural_ellite_mmd)
            
            key, subkey = jax.random.split(key)

            mean_param, cov_param, neural_output_batch, cost_batch_temp\
                 = self.cem_helper.compute_shifted_samples(key, neural_output_ellite, 
                cost_batch, idx_ellite, mean_param, cov_param)
        
            idx_min = jnp.argmin(cost_batch_temp)
            
            res = res.at[idx].set(jnp.min(cost_batch_temp))
            res_2 = res_2.at[idx].set(res_ellite_mmd[idx_min])

            return (res,res_2,lamda_x, lamda_y, neural_output_batch,mean_param,cov_param,s_lane),\
                (c_x_ellite_mmd[idx_min],c_y_ellite_mmd[idx_min],mmd_lane[idx_min],mmd_obs_ellite[idx_min])

        carry_init = (res_init,res_2_init,lamda_x_init, lamda_y_init, \
            neural_output_batch_init,mean_param_init, cov_param_init,s_lane_init)

        carry_final,result = lax.scan(lax_cem,carry_init,jnp.arange(self.maxiter_cem))

        res,res_2,lamda_x, lamda_y, neural_output_batch,mean_param,cov_param,s_lane = carry_final 
        
        cx_best = result[0][-1]
        cy_best = result[1][-1]
       
        mmd_lane = result[2][-1]
        mmd_obs = result[3][-1]

        return cx_best,cy_best,mmd_lane,mmd_obs