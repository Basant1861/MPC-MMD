import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random,vmap,lax
import jax
import jax.lax as lax
from scipy.interpolate import CubicSpline
import bernstein_coeff_order10_arbitinterval

class Helper():
    def __init__(self,v_min,v_max, P,Pdot,Pddot,rho_v,rho_offset,
                 weight_smoothness_x,weight_smoothness_y,k_p_v,k_d_v,k_p,k_d,
                 P_jax_1,P_jax_2,P_jax_3,P_jax_4,
                 Pdot_jax_1,Pdot_jax_2,Pdot_jax_3,Pdot_jax_4,
                 Pddot_jax_1,Pddot_jax_2,Pddot_jax_3,Pddot_jax_4,
                 num_params,num_batch,ellite_num,num_partial,alpha_mean,alpha_cov,
                 lamda,modes,modes_probs,mu,sigma,size_1,size_2,size_3,
                 tot_time,wheel_base,num_reduced_sqrt,num_reduced,
                 num_prime,num_circles,num_obs,A_eq_x,A_eq_y,y_des_1,y_des_2,ellite_num_cost,num,
                 steer_max,nvar,dt,sigma_acc,sigma_steer,beta_a,beta_b,noise,a_centr,prob,prob2):
        
        self.prob = prob
        self.prob2 = prob2
        self.a_centr = a_centr
        self.noise = noise
        self.beta_a,self.beta_b = beta_a,beta_b
        self.sigma_acc = sigma_acc
        self.sigma_steer = sigma_steer

        self.t = dt
        self.nvar = nvar
        self.steer_max = steer_max
        self.num_prime = num_prime
        self.num_obs = num_obs
        self.num_circles = num_circles
        self.num = num
        self.ellite_num_cost = ellite_num_cost
        self.y_des_1, self.y_des_2 = y_des_1,y_des_2

        self.tot_time = tot_time
        self.wheel_base = wheel_base
        self.num_reduced_sqrt = num_reduced_sqrt
        self.num_reduced = num_reduced

        self.v_max = v_max
        self.v_min = v_min
    
        self.P_jax, self.Pdot_jax, self.Pddot_jax = P,Pdot,Pddot

        self.rho_v = rho_v 
        self.rho_offset = rho_offset
        
        self.weight_smoothness_x = weight_smoothness_x
        self.weight_smoothness_y = weight_smoothness_y

        self.k_p_v = k_p_v
        self.k_d_v = k_d_v

        self.k_p = k_p
        self.k_d = k_d
        self.A_eq_x = A_eq_x
        self.A_eq_y = A_eq_y

        self.P_jax_1 = P_jax_1
        self.P_jax_2 = P_jax_2
        self.P_jax_3 = P_jax_3
        self.P_jax_4 = P_jax_4

        self.Pdot_jax_1 = Pdot_jax_1
        self.Pdot_jax_2 = Pdot_jax_2
        self.Pdot_jax_3 = Pdot_jax_3
        self.Pdot_jax_4 = Pdot_jax_4
            
        self.Pddot_jax_1 = Pddot_jax_1
        self.Pddot_jax_2 = Pddot_jax_2
        self.Pddot_jax_3 = Pddot_jax_3
        self.Pddot_jax_4 = Pddot_jax_4

        self.num_partial = num_partial
        ###########################################3
        key = random.PRNGKey(0)

        self.key = key

        self.alpha_mean = alpha_mean
        self.alpha_cov = alpha_cov

        self.lamda = lamda
        self.vec_product = jit(jax.vmap(self.comp_prod, 0, out_axes=(0)))

        self.num_params = num_params 
        self.num_batch = num_batch
        self.ellite_num = ellite_num

        self.modes = modes
        self.modes_probs = modes_probs
        self.mu = mu
        self.sigma = sigma
        self.size_1 = size_1
        self.size_2 = size_2
        self.size_3 = size_3

        self.compute_controls_vmap = jit(vmap(self.compute_controls,in_axes=(0,0)))
        self.compute_rollout_vmap = jit(vmap(self.compute_rollout_complete,in_axes=(0,0,None,None)))
        self.compute_rollout_opt_vmap = jit(vmap(self.compute_rollout_complete_opt,in_axes=(0,0,None,None)))
        self.compute_rollout_det_vmap = jit(vmap(self.compute_rollout_complete_det,in_axes=(0,0,None,None)))

        ################################################ Matrices for Custom path smoothing
        self.rho_smoothing = 1.0

        self.num_path = 600

        self.A_smoothing = jnp.identity(self.num_path)
        self.A_jerk_smoothing = jnp.diff(jnp.diff(jnp.diff(self.A_smoothing, axis = 0), axis = 0), axis = 0)
        self.A_acc_smoothing = jnp.diff(jnp.diff(self.A_smoothing, axis = 0), axis = 0)
        self.A_vel_smoothing = jnp.diff(self.A_smoothing, axis = 0)
        cost_jerk_smoothing = jnp.dot(self.A_jerk_smoothing.T, self.A_jerk_smoothing)
        cost_acc_smoothing = jnp.dot(self.A_acc_smoothing.T, self.A_acc_smoothing)
        cost_vel_smoothing = jnp.dot(self.A_vel_smoothing.T, self.A_vel_smoothing)

        self.A_eq_smoothing = self.A_smoothing[0].reshape(1, self.num_path)
        cost_smoothing = (20)*(cost_jerk_smoothing+0.0*cost_acc_smoothing)+self.rho_smoothing*jnp.dot(self.A_smoothing.T, self.A_smoothing)
        
        cost_mat_x = jnp.vstack((  jnp.hstack(( cost_smoothing, self.A_eq_smoothing.T )), jnp.hstack(( self.A_eq_smoothing, jnp.zeros(( jnp.shape(self.A_eq_smoothing)[0], jnp.shape(self.A_eq_smoothing)[0] )) )) ))
        self.cost_smoothing_inv = jnp.linalg.inv(cost_mat_x)

        self.maxiter_smoothing = 10

        #####################################################################
        self.jax_interp = jit(jnp.interp) ############## jitted interp
        self.interp_vmap = jit(vmap(self.interp,in_axes=(0,None,None)))
        self.frenet_to_global_vmap = jit(vmap(self.frenet_to_global,in_axes=(0,0,0,0,0)))
      
        self.global_to_frenet_obs_vmap = jit(vmap(self.global_to_frenet_obs,in_axes=(0,0,0,0,0,None,None,None,None,None,None)))
        self.global_to_frenet_vmap = jit(vmap(self.global_to_frenet_trajs,in_axes=(0,0,None,None,None,None,None)))
        self.global_to_frenet_vmap_1 = jit(vmap(self.global_to_frenet,in_axes=(None,None,0,None,None,None,None)))

        ##########################################################################
        self.t_fin_prime = self.num_prime*self.t
        tot_time_prime = jnp.linspace(0, self.t_fin_prime, self.num_prime)
        self.tot_time_prime = tot_time_prime.reshape(self.num_prime, 1)

        self.P_prime, self.Pdot_prime, self.Pddot_prime = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, self.tot_time_prime[0], self.tot_time_prime[-1], self.tot_time_prime)
        self.P_jax_prime, self.Pdot_jax_prime, self.Pddot_jax_prime = jnp.asarray(self.P_prime), \
                                                            jnp.asarray(self.Pdot_prime), jnp.asarray(self.Pddot_prime)

        self.nvar_prime = jnp.shape(self.P_jax_prime)[1]

    @partial(jit, static_argnums=(0, ))	
    def frenet_to_global(self, y_frenet, ref_x, ref_y, dx_by_ds, dy_by_ds):

        normal_x = -1*dy_by_ds
        normal_y = dx_by_ds

        norm_vec = jnp.sqrt(normal_x**2 + normal_y**2)
        normal_unit_x = (1/norm_vec)*normal_x
        normal_unit_y = (1/norm_vec)*normal_y

        global_x = ref_x + y_frenet*normal_unit_x
        global_y = ref_y + y_frenet*normal_unit_y

        psi_global = jnp.arctan2(jnp.diff(global_y),jnp.diff(global_x))

        return global_x, global_y, psi_global	    
    
    @partial(jit, static_argnums=(0,))
    def global_to_frenet_obs(self,x_obs,y_obs,vx_obs,vy_obs,psi_obs, x_path, y_path, arc_vec, Fx_dot, Fy_dot, kappa ):
        v_obs = jnp.sqrt(vx_obs**2+vy_obs**2)

        idx_closest_point = jnp.argmin( jnp.sqrt((x_path-x_obs)**2+(y_path-y_obs)**2))
        closest_point_x, closest_point_y = x_path[idx_closest_point], y_path[idx_closest_point]

        x_init = arc_vec[idx_closest_point]

        kappa_interp = self.jax_interp(x_init, arc_vec, kappa)
        kappa_pert = self.jax_interp(x_init+0.001, arc_vec, kappa)

        kappa_prime = (kappa_pert-kappa_interp)/0.001

        Fx_dot_interp = self.jax_interp(x_init, arc_vec, Fx_dot)
        Fy_dot_interp = self.jax_interp(x_init, arc_vec, Fy_dot)

        normal_x = -Fy_dot_interp
        normal_y = Fx_dot_interp

        normal = jnp.hstack((normal_x, normal_y   ))
        vec = jnp.asarray([x_obs-closest_point_x,y_obs-closest_point_y ])
        y_init = (1/(jnp.linalg.norm(normal)))*jnp.dot(normal,vec)
        
        psi_init = psi_obs-jnp.arctan2(Fy_dot_interp, Fx_dot_interp)
        psi_init = jnp.arctan2(jnp.sin(psi_init), jnp.cos(psi_init))
        
        vx_init = v_obs*jnp.cos(psi_init)/(1-y_init*kappa_interp)
        vy_init = v_obs*jnp.sin(psi_init)

        return x_init, y_init, vx_init, vy_init, psi_init
    @partial(jit, static_argnums=(0, ))
    def interp(self,x,xp,fp):
        return self.jax_interp(x,xp,fp)   
      
    @partial(jit, static_argnums=(0,))
    def global_to_frenet_trajs(self, x_roll,y_roll,x_path, y_path, arc_vec, Fx_dot, Fy_dot ):
        
        xg_init = jnp.zeros(self.num_prime)
        yg_init = jnp.zeros(self.num_prime)

        def lax_rollout_vmap(_x_roll,_y_roll):
            def lax_rollout(carry,idx):
                xg,yg = carry
                idx_closest_point = jnp.argmin( jnp.sqrt((x_path-_x_roll[idx])**2+(y_path-_y_roll[idx])**2))
                closest_point_x, closest_point_y = x_path[idx_closest_point], y_path[idx_closest_point]

                x_init = arc_vec[idx_closest_point]

                Fx_dot_interp = self.jax_interp(x_init, arc_vec, Fx_dot)
                Fy_dot_interp = self.jax_interp(x_init, arc_vec, Fy_dot)

                normal_x = -Fy_dot_interp
                normal_y = Fx_dot_interp

                normal = jnp.hstack((normal_x, normal_y   ))
                vec = jnp.asarray([_x_roll[idx]-closest_point_x,_y_roll[idx]-closest_point_y ])
                
                y_init = (1/(jnp.linalg.norm(normal)))*jnp.dot(normal,vec)

                xg = xg.at[idx].set(x_init)
                yg = yg.at[idx].set(y_init)

                return (xg,yg),0

            carry_init = xg_init,yg_init
            carry_final,result = lax.scan(lax_rollout,carry_init,jnp.arange(self.num_prime))
            _xg,_yg = carry_final
            return _xg,_yg
        
        rollout_vmap = jit(vmap(lax_rollout_vmap,in_axes=(0,0)))
        xg,yg = rollout_vmap(x_roll,y_roll)
        return xg,yg

    def path_spline(self, x_path, y_path):

        x_diff = np.diff(x_path)
        y_diff = np.diff(y_path)

        phi = np.unwrap(np.arctan2(y_diff, x_diff))
        phi_init = phi[0]
        phi = np.hstack(( phi_init, phi  ))

        arc = np.cumsum( np.sqrt( x_diff**2+y_diff**2 )   )
        arc_length = arc[-1]

        arc_vec = np.linspace(0, arc_length, np.shape(x_path)[0])

        cs_x_path = CubicSpline(arc_vec, x_path)
        cs_y_path = CubicSpline(arc_vec, y_path)
        cs_phi_path = CubicSpline(arc_vec, phi)
        
        return cs_x_path, cs_y_path, cs_phi_path, arc_length, arc_vec
        
    def waypoint_generator(self, x_global_init, y_global_init, x_path_data, y_path_data, arc_vec, cs_x_path, cs_y_path, cs_phi_path, arc_length):

        idx = np.argmin( np.sqrt((x_global_init-x_path_data)**2+(y_global_init-y_path_data)**2))
        arc_curr = arc_vec[idx]
        arc_pred = arc_curr + 300
       
        arc_look = np.linspace(arc_curr, arc_pred, self.num_path)

        x_waypoints = cs_x_path(arc_look)
        y_waypoints =  cs_y_path(arc_look)
        phi_Waypoints = cs_phi_path(arc_look)

        return x_waypoints, y_waypoints, phi_Waypoints

    @partial(jit, static_argnums=(0,))
    def compute_x_smoothing(self, x_waypoints, y_waypoints, alpha_smoothing, d_smoothing, lamda_x_smoothing, lamda_y_smoothing):

        b_x_smoothing = x_waypoints+d_smoothing*jnp.cos(alpha_smoothing)
        b_y_smoothing = y_waypoints+d_smoothing*jnp.sin(alpha_smoothing)

        lincost_x = -lamda_x_smoothing-self.rho_smoothing*jnp.dot(self.A_smoothing.T, b_x_smoothing)
        lincost_y = -lamda_y_smoothing-self.rho_smoothing*jnp.dot(self.A_smoothing.T, b_y_smoothing)
        
        b_eq_smoothing_x = x_waypoints[0]
        b_eq_smoothing_y = y_waypoints[0]

        sol_x = jnp.dot(self.cost_smoothing_inv, jnp.hstack((-lincost_x, b_eq_smoothing_x)))
        sol_y = jnp.dot(self.cost_smoothing_inv, jnp.hstack((-lincost_y, b_eq_smoothing_y)))

        x_smooth = sol_x[0:self.num_path]
        y_smooth = sol_y[0:self.num_path]

        return x_smooth, y_smooth

    @partial(jit, static_argnums=(0,))
    def compute_alpha_smoothing(self, x_smooth, y_smooth, x_waypoints, y_waypoints, threshold, lamda_x_smoothing, lamda_y_smoothing):

        wc_alpha_smoothing = (x_smooth-x_waypoints)
        ws_alpha_smoothing = (y_smooth-y_waypoints)
        
        alpha_smoothing  = jnp.arctan2( ws_alpha_smoothing, wc_alpha_smoothing )

        c1_d_smoothing = (jnp.cos(alpha_smoothing)**2 + jnp.sin(alpha_smoothing)**2 )
        c2_d_smoothing = (wc_alpha_smoothing*jnp.cos(alpha_smoothing) + ws_alpha_smoothing*jnp.sin(alpha_smoothing)  )

        d_smoothing  = c2_d_smoothing/c1_d_smoothing
        d_smoothing  = jnp.minimum( d_smoothing, threshold*jnp.ones(self.num_path ) )

        res_x_smoothing = wc_alpha_smoothing-d_smoothing*jnp.cos(alpha_smoothing)
        res_y_smoothing = ws_alpha_smoothing-d_smoothing*jnp.sin(alpha_smoothing)

        lamda_x_smoothing = lamda_x_smoothing-self.rho_smoothing*jnp.dot(self.A_smoothing.T, res_x_smoothing).T
        lamda_y_smoothing = lamda_y_smoothing-self.rho_smoothing*jnp.dot(self.A_smoothing.T, res_y_smoothing).T

        return alpha_smoothing, d_smoothing, lamda_x_smoothing, lamda_y_smoothing

    @partial(jit, static_argnums=(0,))
    def compute_path_parameters(self, x_path, y_path):

        Fx_dot = jnp.diff(x_path)
        Fy_dot = jnp.diff(y_path)

        Fx_dot = jnp.hstack(( Fx_dot[0], Fx_dot  ))

        Fy_dot = jnp.hstack(( Fy_dot[0], Fy_dot  ))

            
        Fx_ddot = jnp.diff(Fx_dot)
        Fy_ddot = jnp.diff(Fy_dot)

        Fx_ddot = jnp.hstack(( Fx_ddot[0], Fx_ddot  ))

        Fy_ddot = jnp.hstack(( Fy_ddot[0], Fy_ddot  ))

        arc = jnp.cumsum( jnp.sqrt( Fx_dot**2+Fy_dot**2 )   )
        arc_vec = jnp.hstack((0, arc[0:-1] ))

        arc_length = arc_vec[-1]

        kappa = (Fy_ddot*Fx_dot-Fx_ddot*Fy_dot)/((Fx_dot**2+Fy_dot**2)**(1.5))

        return Fx_dot, Fy_dot, Fx_ddot, Fy_ddot, arc_vec, kappa, arc_length

    @partial(jit, static_argnums=(0,))
    def global_to_frenet(self, x_path, y_path, initial_state, arc_vec, Fx_dot, Fy_dot, kappa ):

        x_global_init, y_global_init, v_global_init, vdot_global_init, psi_global_init, psidot_global_init = initial_state
        idx_closest_point = jnp.argmin( jnp.sqrt((x_path-x_global_init)**2+(y_path-y_global_init)**2))
        closest_point_x, closest_point_y = x_path[idx_closest_point], y_path[idx_closest_point]

        x_init = arc_vec[idx_closest_point]

        kappa_interp = self.jax_interp(x_init, arc_vec, kappa)
        kappa_pert = self.jax_interp(x_init+0.001, arc_vec, kappa)

        kappa_prime = (kappa_pert-kappa_interp)/0.001

        Fx_dot_interp = self.jax_interp(x_init, arc_vec, Fx_dot)
        Fy_dot_interp = self.jax_interp(x_init, arc_vec, Fy_dot)

        normal_x = -Fy_dot_interp
        normal_y = Fx_dot_interp

        normal = jnp.hstack((normal_x, normal_y   ))
        vec = jnp.asarray([x_global_init-closest_point_x,y_global_init-closest_point_y ])
        y_init = (1/(jnp.linalg.norm(normal)))*jnp.dot(normal,vec)
        
        psi_init = psi_global_init-jnp.arctan2(Fy_dot_interp, Fx_dot_interp)
        psi_init = jnp.arctan2(jnp.sin(psi_init), jnp.cos(psi_init))
        
        vx_init = v_global_init*jnp.cos(psi_init)/(1-y_init*kappa_interp)
        vy_init = v_global_init*jnp.sin(psi_init)

        psidot_init = psidot_global_init-kappa_interp*vx_init

        ay_init = vdot_global_init*jnp.sin(psi_init)+v_global_init*jnp.cos(psi_init)*psidot_init
        
        ax_init_part_1 = vdot_global_init*jnp.cos(psi_init)-v_global_init*jnp.sin(psi_init)*psidot_init
        ax_init_part_2 = -vy_init*kappa_interp-y_init*kappa_prime*vx_init

        ax_init = (ax_init_part_1*(1-y_init*kappa_interp)-(v_global_init*jnp.cos(psi_init))*(ax_init_part_2) )/((1-y_init*kappa_interp)**2)
            
        psi_fin = 0.0

        return x_init, y_init, vx_init, vy_init, ax_init, ay_init, psi_init, psi_fin, psidot_init

    @partial(jit, static_argnums=(0,))
    def custom_path_smoothing(self,  x_waypoints, y_waypoints, threshold):

        alpha_smoothing_init = jnp.zeros(self.num_path)
        d_smoothing_init = threshold*jnp.ones(self.num_path)
        lamda_x_smoothing_init = jnp.zeros(self.num_path)
        lamda_y_smoothing_init = jnp.zeros(self.num_path)

        def lax_smoothing(carry,idx):
            alpha_smoothing, d_smoothing, lamda_x_smoothing, lamda_y_smoothing,x_smooth,y_smooth = carry
            x_smooth, y_smooth = self.compute_x_smoothing(x_waypoints, y_waypoints, alpha_smoothing, d_smoothing, lamda_x_smoothing, lamda_y_smoothing)
            alpha_smoothing, d_smoothing, lamda_x_smoothing, lamda_y_smoothing = self.compute_alpha_smoothing(x_smooth, y_smooth, x_waypoints, y_waypoints, threshold, lamda_x_smoothing, lamda_y_smoothing)
            
            return (alpha_smoothing, d_smoothing, lamda_x_smoothing, lamda_y_smoothing,x_smooth,y_smooth),x_smooth

        carry_init = (alpha_smoothing_init,d_smoothing_init,lamda_x_smoothing_init,lamda_y_smoothing_init,x_waypoints,y_waypoints)
        carry_final,result = lax.scan(lax_smoothing,carry_init,jnp.arange(self.maxiter_smoothing))

        alpha_smoothing, d_smoothing, lamda_x_smoothing, lamda_y_smoothing,x_smooth,y_smooth = carry_final

        return x_smooth, y_smooth
    
    @partial(jit, static_argnums=(0,))	
    def sampling_param(self,mean_param, cov_param):

        key, subkey = random.split(self.key)
        param_samples = jax.random.multivariate_normal(key, mean_param, cov_param, (self.num_batch,))
        
        v_des_1 = param_samples[:, 0]
        v_des_2 = param_samples[:, 1]
        v_des_3 = param_samples[:, 2]
        v_des_4 = param_samples[:, 3]
            
        y_des_1 = param_samples[:, 4]
        y_des_2 = param_samples[:, 5]
        y_des_3 = param_samples[:, 6]
        y_des_4 = param_samples[:, 7]

        v_des_1 = jnp.clip(v_des_1, self.v_min*jnp.ones(self.num_batch),
                            self.v_max*jnp.ones(self.num_batch)   )
        v_des_2 = jnp.clip(v_des_2, self.v_min*jnp.ones(self.num_batch), 
                           self.v_max*jnp.ones(self.num_batch)   )
        v_des_3 = jnp.clip(v_des_3, self.v_min*jnp.ones(self.num_batch), 
                           self.v_max*jnp.ones(self.num_batch)   )
        v_des_4 = jnp.clip(v_des_4, self.v_min*jnp.ones(self.num_batch),
                            self.v_max*jnp.ones(self.num_batch)   )

        neural_output_batch = jnp.vstack(( v_des_1, v_des_2, v_des_3, v_des_4, y_des_1, y_des_2, y_des_3, y_des_4,
                                          )).T
        
        return neural_output_batch
	
    @partial(jit, static_argnums=(0,))	
    def compute_boundary_vec(self, x_init, vx_init, ax_init, y_init, vy_init, ay_init):

        x_init_vec = x_init*jnp.ones((self.num_batch, 1))
        y_init_vec = y_init*jnp.ones((self.num_batch, 1)) 

        vx_init_vec = vx_init*jnp.ones((self.num_batch, 1))
        vy_init_vec = vy_init*jnp.ones((self.num_batch, 1))

        ax_init_vec = ax_init*jnp.ones((self.num_batch, 1))
        ay_init_vec = ay_init*jnp.ones((self.num_batch, 1))

        b_eq_x = jnp.hstack(( x_init_vec, vx_init_vec, ax_init_vec ))
        b_eq_y = jnp.hstack(( y_init_vec, vy_init_vec, ay_init_vec, jnp.zeros((self.num_batch, 1  ))   ))

        return b_eq_x, b_eq_y

    @partial(jit, static_argnums=(0,))	
    def compute_x_guess(self, b_eq_x, b_eq_y, neural_output_batch):

        v_des_1 = neural_output_batch[:, 0]
        v_des_2 = neural_output_batch[:, 1]
        v_des_3 = neural_output_batch[:, 2]
        v_des_4 = neural_output_batch[:, 3]

        y_des_1 = neural_output_batch[:, 4]
        y_des_2 = neural_output_batch[:, 5]
        y_des_3 = neural_output_batch[:, 6]
        y_des_4 = neural_output_batch[:, 7]
        
        #############################
        A_vd_1 = self.Pddot_jax_1-self.k_p_v*self.Pdot_jax_1 #- self.k_d_v*self.Pddot_jax_1
        b_vd_1 = -self.k_p_v*jnp.ones((self.num_batch, self.num_partial ))*(v_des_1)[:, jnp.newaxis]

        A_vd_2 = self.Pddot_jax_2-self.k_p_v*self.Pdot_jax_2 #- self.k_d_v*self.Pddot_jax_2
        b_vd_2 = -self.k_p_v*jnp.ones((self.num_batch, self.num_partial ))*(v_des_2)[:, jnp.newaxis]

        A_vd_3 = self.Pddot_jax_3-self.k_p_v*self.Pdot_jax_3 #- self.k_d_v*self.Pddot_jax_3
        b_vd_3 = -self.k_p_v*jnp.ones((self.num_batch, self.num_partial ))*(v_des_3)[:, jnp.newaxis]

        A_vd_4 = self.Pddot_jax_4-self.k_p_v*self.Pdot_jax_4 #- self.k_d_v*self.Pddot_jax_4
        b_vd_4 = -self.k_p_v*jnp.ones((self.num_batch, self.num_partial ))*(v_des_4)[:, jnp.newaxis]

        A_pd_1 = self.Pddot_jax_1-self.k_p*self.P_jax_1# -self.k_d*self.Pdot_jax_1
        b_pd_1 = -self.k_p*jnp.ones((self.num_batch, self.num_partial ))*(y_des_1)[:, jnp.newaxis]
        
        A_pd_2 = self.Pddot_jax_2-self.k_p*self.P_jax_2 #-self.k_d*self.Pdot_jax_2
        b_pd_2 = -self.k_p*jnp.ones((self.num_batch, self.num_partial ))*(y_des_2)[:, jnp.newaxis]
            
        A_pd_3 = self.Pddot_jax_3-self.k_p*self.P_jax_3 #-self.k_d*self.Pdot_jax_3
        b_pd_3 = -self.k_p*jnp.ones((self.num_batch, self.num_partial ))*(y_des_3)[:, jnp.newaxis]
        
        A_pd_4 = self.Pddot_jax_4-self.k_p*self.P_jax_4 #-self.k_d*self.Pdot_jax_4
        b_pd_4 = -self.k_p*jnp.ones((self.num_batch, self.num_partial ))*(y_des_4)[:, jnp.newaxis]
        
        cost_smoothness_x = self.weight_smoothness_x*jnp.dot(self.Pddot_jax.T, self.Pddot_jax)
        cost_smoothness_y = self.weight_smoothness_y*jnp.dot(self.Pddot_jax.T, self.Pddot_jax)
        
        cost_temp_x = self.rho_v*jnp.dot(A_vd_1.T, A_vd_1)+self.rho_v*jnp.dot(A_vd_2.T, A_vd_2)+self.rho_v*jnp.dot(A_vd_3.T, A_vd_3)+self.rho_v*jnp.dot(A_vd_4.T, A_vd_4)
        cost_temp_y = self.rho_offset*jnp.dot(A_pd_1.T, A_pd_1)+self.rho_offset*jnp.dot(A_pd_2.T, A_pd_2)+self.rho_offset*jnp.dot(A_pd_3.T, A_pd_3)+self.rho_offset*jnp.dot(A_pd_4.T, A_pd_4)
        
        cost_x = cost_smoothness_x + cost_temp_x
        cost_y = cost_smoothness_y + cost_temp_y

        cost_mat_x = jnp.vstack((  jnp.hstack(( cost_x, self.A_eq_x.T )), jnp.hstack(( self.A_eq_x, jnp.zeros(( jnp.shape(self.A_eq_x)[0], jnp.shape(self.A_eq_x)[0] )) )) ))
        cost_mat_y = jnp.vstack((  jnp.hstack(( cost_y, self.A_eq_y.T )), jnp.hstack(( self.A_eq_y, jnp.zeros(( jnp.shape(self.A_eq_y)[0], jnp.shape(self.A_eq_y)[0] )) )) ))
        
        lincost_x = -self.rho_v*jnp.dot(A_vd_1.T, b_vd_1.T).T-self.rho_v*jnp.dot(A_vd_2.T, b_vd_2.T).T-self.rho_v*jnp.dot(A_vd_3.T, b_vd_3.T).T-self.rho_v*jnp.dot(A_vd_4.T, b_vd_4.T).T
        lincost_y = -self.rho_offset*jnp.dot(A_pd_1.T, b_pd_1.T).T-self.rho_offset*jnp.dot(A_pd_2.T, b_pd_2.T).T-self.rho_offset*jnp.dot(A_pd_3.T, b_pd_3.T).T-self.rho_offset*jnp.dot(A_pd_4.T, b_pd_4.T).T 
    
        sol_x = jnp.linalg.solve(cost_mat_x, jnp.hstack(( -lincost_x, b_eq_x )).T).T
        sol_y = jnp.linalg.solve(cost_mat_y, jnp.hstack(( -lincost_y, b_eq_y )).T).T

        #######################3

        primal_sol_x = sol_x[:,0:self.nvar]
        primal_sol_y = sol_y[:,0:self.nvar]

        return primal_sol_x, primal_sol_y

    @partial(jit, static_argnums=(0, ))
    def compute_cost(self,mmd_lane_des,mmd_obs,mmd_lane,x_ellite_mmd,y_ellite_mmd, res_ellite_mmd, \
        xdot_ellite_mmd, ydot_ellite_mmd, 
        xddot_ellite_mmd, yddot_ellite_mmd, \
             v_des, steering_ellite_mmd,kappa_interp):
        
        cost_centerline_1 = jnp.linalg.norm(y_ellite_mmd - self.y_des_1,axis=1)
        cost_centerline_2 = jnp.linalg.norm(y_ellite_mmd - self.y_des_2,axis=1)
        cost_des_lane = cost_centerline_1*cost_centerline_2

        cost_steering = jnp.linalg.norm(steering_ellite_mmd, axis = 1)
        steering_vel = jnp.diff(steering_ellite_mmd, axis = 1)
        cost_steering_vel = jnp.linalg.norm(steering_vel, axis = 1)
        steering_acc = jnp.diff(steering_vel, axis = 1)
        
        v = jnp.sqrt(xdot_ellite_mmd**2+ydot_ellite_mmd**2)
    
        cost_steering_acc = jnp.linalg.norm(steering_acc, axis = 1)
        cost_steering_penalty = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.ellite_num_cost, self.num  )), jnp.abs(steering_ellite_mmd)-self.steer_max ), axis = 1)
        cost_steering_vel_penalty = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.ellite_num_cost, self.num-1  )), jnp.abs(steering_vel)-0.05 ), axis = 1)
        
        centr_acc = jnp.abs((xdot_ellite_mmd**2)*(kappa_interp))
        centr_acc_cost = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.ellite_num_cost, self.num  )), centr_acc-self.a_centr   ), axis = 1)

        cost_batch = res_ellite_mmd \
            + 0.1*jnp.linalg.norm(v-v_des, axis = 1)\
             + 0.1*(cost_steering + cost_steering_vel + cost_steering_acc) \
                  + 0.1*(cost_steering_penalty+cost_steering_vel_penalty)\
                     + 0.02*(jnp.linalg.norm(yddot_ellite_mmd, axis = 1)) \
                 +0.02*(jnp.linalg.norm(xddot_ellite_mmd, axis = 1)) \
                    + mmd_obs + mmd_lane + mmd_lane_des \
                    + 0.01*cost_des_lane\
                    + 0.1*centr_acc_cost
           
        return cost_batch
    
    @partial(jit, static_argnums=(0, ))
    def compute_ellite_samples(self, cost_batch, neural_output_projection):

        idx_ellite = jnp.argsort(cost_batch)

        neural_output_ellite = neural_output_projection[idx_ellite[0:self.ellite_num]]

        return neural_output_ellite, idx_ellite
    
    @partial(jit, static_argnums=(0,))
    def comp_prod(self, diffs, d ):
        term_1 = jnp.expand_dims(diffs, axis = 1)
        term_2 = jnp.expand_dims(diffs, axis = 0)
        prods = d * jnp.outer(term_1,term_2)
        return prods	
  
    @partial(jit, static_argnums=(0, ))
    def compute_shifted_samples(self, key, neural_output_ellite, cost_batch, idx_ellite, mean_param_prev, cov_param_prev):
        cost_batch_temp = cost_batch[idx_ellite[0:self.ellite_num]]
        w = cost_batch_temp
        w_min = jnp.min(cost_batch_temp)
        w = jnp.exp(-(1/self.lamda) * (w - w_min ) )
        sum_w = jnp.sum(w, axis = 0)
        mean_param = (1-self.alpha_mean)*mean_param_prev + self.alpha_mean*(jnp.sum( (neural_output_ellite * w[:,jnp.newaxis]) , axis= 0)/ sum_w)
        diffs = (neural_output_ellite - mean_param)
        prod_result = self.vec_product(diffs, w)
        cov_param = (1-self.alpha_cov)*cov_param_prev + self.alpha_cov*(jnp.sum( prod_result , axis = 0)/jnp.sum(w, axis = 0)) + 0.01*jnp.identity(self.num_params)

        param_samples = jax.random.multivariate_normal(key, mean_param, cov_param, (self.num_batch-self.ellite_num, ))

        v_des_1 = param_samples[:, 0]
        v_des_2 = param_samples[:, 1]
        v_des_3 = param_samples[:, 2]
        v_des_4 = param_samples[:, 3]
        
        y_des_1 = param_samples[:, 4]
        y_des_2 = param_samples[:, 5]
        y_des_3 = param_samples[:, 6]
        y_des_4 = param_samples[:, 7]
                
        v_des_1 = jnp.clip(v_des_1, self.v_min*jnp.ones(self.num_batch-self.ellite_num), self.v_max*jnp.ones(self.num_batch-self.ellite_num)   )
        v_des_2 = jnp.clip(v_des_2, self.v_min*jnp.ones(self.num_batch-self.ellite_num), self.v_max*jnp.ones(self.num_batch-self.ellite_num)   )
        v_des_3 = jnp.clip(v_des_3, self.v_min*jnp.ones(self.num_batch-self.ellite_num), self.v_max*jnp.ones(self.num_batch-self.ellite_num)   )
        v_des_4 = jnp.clip(v_des_4, self.v_min*jnp.ones(self.num_batch-self.ellite_num), self.v_max*jnp.ones(self.num_batch-self.ellite_num)   )

        neural_output_shift = jnp.vstack(( v_des_1, v_des_2, v_des_3, v_des_4, y_des_1, y_des_2, y_des_3, y_des_4,
                                           )).T

        neural_output_batch = jnp.vstack(( neural_output_ellite, neural_output_shift  ))

        return mean_param, cov_param, neural_output_batch, cost_batch_temp
    
    @partial(jit, static_argnums=(0, ))	
    def compute_noisy_init_state(self,idx_mpc,x_init,y_init,vx_init,vy_init):
        key = jax.random.PRNGKey(idx_mpc)
        key,subkey = jax.random.split(key)

        epsilon = jax.random.multivariate_normal(key,jnp.zeros(4),jnp.eye(4), (1,) )
        epsilon_x = epsilon[0]
        epsilon_y = epsilon[1]
        epsilon_v = epsilon[2]
        epsilon_psi = epsilon[3]

        v_init = jnp.sqrt(vx_init**2+vy_init**2)

        eps_x_1 = epsilon_x*self.sigma[0][0] + self.mu[0][0]
        eps_x_2 = epsilon_x*self.sigma[1][0] + self.mu[1][0]
        eps_x_3 = epsilon_x*self.sigma[2][0] + self.mu[2][0]

        eps_y_1 = epsilon_y*self.sigma[0][1] + self.mu[0][1]
        eps_y_2 = epsilon_y*self.sigma[1][1] + self.mu[1][1]
        eps_y_3 = epsilon_y*self.sigma[2][1] + self.mu[2][1]

        eps_v_1 = epsilon_v*self.sigma[0][2] + self.mu[0][2]
        eps_v_2 = epsilon_v*self.sigma[1][2] + self.mu[1][2]
        eps_v_3 = epsilon_v*self.sigma[2][2] + self.mu[2][2]

        eps_psi_1 = epsilon_psi*self.sigma[0][3] + self.mu[0][3]
        eps_psi_2 = epsilon_psi*self.sigma[1][3] + self.mu[1][3]
        eps_psi_3 = epsilon_psi*self.sigma[2][3] + self.mu[2][3]

        weight_samples = jax.random.choice(key,self.modes, (1,),p=self.modes_probs)

        idx_1 = jnp.where(weight_samples==1,size=self.size_1)
        idx_2 = jnp.where(weight_samples==2,size=self.size_2)
        idx_3 = jnp.where(weight_samples==3,size=self.size_3)

        eps_x = jnp.hstack((eps_x_1[idx_1],eps_x_2[idx_2],eps_x_3[idx_3]))
        eps_y = jnp.hstack((eps_y_1[idx_1],eps_y_2[idx_2],eps_y_3[idx_3]))
        eps_v = jnp.hstack((eps_v_1[idx_1],eps_v_2[idx_2],eps_v_3[idx_3]))
        eps_psi = jnp.hstack((eps_psi_1[idx_1],eps_psi_2[idx_2],eps_psi_3[idx_3]))
        
        psi_init = jnp.arctan2(vy_init,vx_init)
        x_init = x_init + eps_x
        y_init = y_init + eps_y
        v_init = v_init + 0*eps_v
        psi_init = psi_init + 0*eps_psi
        vx_init = v_init*jnp.cos(psi_init)
        vy_init = v_init*jnp.sin(psi_init)

        return x_init.reshape(-1),y_init.reshape(-1),vx_init.reshape(-1),vy_init.reshape(-1),psi_init.reshape(-1)
    
    @partial(jit, static_argnums=(0,))	
    def compute_obs_trajectories(self,x_obs,y_obs,vx_obs,vy_obs,psi_obs):

        x_obs_traj = (x_obs + vx_obs * self.tot_time[:, jnp.newaxis]).T # num_obs x num
        y_obs_traj = (y_obs + vy_obs * self.tot_time[:, jnp.newaxis]).T
        
        psi_obs_traj = jnp.tile(psi_obs, (self.num,1)).T # num_obs x num
        
        x_obs_circles = x_obs_traj
        y_obs_circles = y_obs_traj
        psi_obs_circles = psi_obs_traj

        return x_obs_circles,y_obs_circles,psi_obs_circles

    @partial(jit, static_argnums=(0, ))	
    def compute_rollout_one_step(self,acc,steer,state):
        
        x,y,vx,vy,psi = state[:,0],state[:,1],state[:,2]  ,state[:,3], state[:,4]

        v = jnp.sqrt(vx**2 +vy**2)
        v = v + acc*self.t
        psidot = v*jnp.tan(steer)/self.wheel_base
        psi_next = psi+psidot*self.t

        vx_next = v*jnp.cos(psi_next)
        vy_next = v*jnp.sin(psi_next)
        
        x_next = x + vx_next*self.t
        y_next = y + vy_next*self.t
        
        state_next = jnp.vstack((x_next,y_next,
                        vx_next,vy_next,
                        psi_next)).T
     
        return state_next

    @partial(jit, static_argnums=(0, ))	
    def compute_rollout_complete(self,acc,steer,initial_state,key):

        if self.noise=="gaussian":
            noise_samples = jax.random.multivariate_normal(key,jnp.zeros(self.num_prime), jnp.eye(self.num_prime),
                                                     (self.num_reduced_sqrt,))
    
            acc_pert = self.sigma_acc*jnp.abs(acc)*noise_samples
            steer_pert = self.sigma_steer*jnp.abs(steer)*noise_samples
        else:
            noise_samples = jax.random.beta(key,self.beta_a,self.beta_b,(self.num_reduced_sqrt,self.num_prime))
            acc_pert = self.sigma_acc*acc*noise_samples
            steer_pert = self.sigma_steer*steer*noise_samples

        acc = acc + acc_pert
        steer = steer + steer_pert

        x_roll_init = jnp.zeros((self.num_reduced_sqrt,self.num_prime))
        y_roll_init = jnp.zeros((self.num_reduced_sqrt,self.num_prime))
        
        state_init = initial_state
        state_init = jnp.vstack([state_init] * (self.num_reduced_sqrt))

        def lax_rollout(carry,idx):
            state,x_roll,y_roll = carry
            x_roll = x_roll.at[:,idx].set(state[:,0])
            y_roll = y_roll.at[:,idx].set(state[:,1])
           
            state = self.compute_rollout_one_step(acc[:,idx],steer[:,idx],state)
            
            return (state,x_roll,y_roll),0.

        carry_init = state_init,x_roll_init,y_roll_init
        carry_final,result = lax.scan(lax_rollout,carry_init,jnp.arange(self.num_prime))
        state,x_roll,y_roll = carry_final
    
        return x_roll,y_roll
    
    @partial(jit, static_argnums=(0, ))	
    def compute_rollout_complete_opt(self,acc,steer,initial_state,key):

        if self.noise=="gaussian":
            noise_samples = jax.random.multivariate_normal(key,jnp.zeros(self.num_prime), jnp.eye(self.num_prime),
                                                     (self.num_reduced_sqrt,))
    
            acc_pert = self.sigma_acc*jnp.abs(acc)*noise_samples
            steer_pert = self.sigma_steer*jnp.abs(steer)*noise_samples
        else:
            noise_samples = jax.random.beta(key,self.beta_a,self.beta_b,(self.num_reduced_sqrt,self.num_prime))
            acc_pert = self.sigma_acc*acc*noise_samples
            steer_pert = self.sigma_steer*steer*noise_samples
       
        acc = acc + acc_pert + 0.1
        steer = steer + steer_pert + 0.1

        acc = jnp.repeat(acc,self.num_reduced_sqrt,axis=0)
        steer = jnp.tile(steer,(self.num_reduced_sqrt,1))

        x_roll_init = jnp.zeros((self.num_reduced,self.num_prime))
        y_roll_init = jnp.zeros((self.num_reduced,self.num_prime))
        
        state_init = initial_state
        state_init = jnp.vstack([state_init] * (self.num_reduced))

        def lax_rollout(carry,idx):
            state,x_roll,y_roll = carry
            x_roll = x_roll.at[:,idx].set(state[:,0])
            y_roll = y_roll.at[:,idx].set(state[:,1])
           
            state = self.compute_rollout_one_step(acc[:,idx],steer[:,idx],state)
            
            return (state,x_roll,y_roll),0.

        carry_init = state_init,x_roll_init,y_roll_init
        carry_final,result = lax.scan(lax_rollout,carry_init,jnp.arange(self.num_prime))
        state,x_roll,y_roll = carry_final

        cx_roll,cy_roll = self.compute_coeff(x_roll,y_roll)

        beta_best,res,_sigma_best,ker_red_best,\
        x_roll_red,y_roll_red,cx_roll_red,cy_roll_red,beta_samples \
            = self.prob2.compute_cem(cx_roll,cy_roll,x_roll,y_roll)
     
        beta_cem_data = (beta_samples,_sigma_best,
                        cx_roll,cy_roll)
        
        return x_roll_red[-1],y_roll_red[-1],beta_best[-1],\
            _sigma_best[-1], res,beta_cem_data
    
    @partial(jit, static_argnums=(0, ))	
    def compute_rollout_complete_det(self,acc,steer,initial_state,key):

        acc = acc + jnp.zeros((self.num_reduced_sqrt,self.num_prime))
        steer = steer +  jnp.zeros((self.num_reduced_sqrt,self.num_prime))

        acc = jnp.repeat(acc,self.num_reduced_sqrt,axis=0)
        steer = jnp.tile(steer,(self.num_reduced_sqrt,1))

        x_roll_init = jnp.zeros((self.num_reduced,self.num_prime))
        y_roll_init = jnp.zeros((self.num_reduced,self.num_prime))
        
        state_init = initial_state
        state_init = jnp.vstack([state_init] * (self.num_reduced))

        def lax_rollout(carry,idx):
            state,x_roll,y_roll = carry
            x_roll = x_roll.at[:,idx].set(state[:,0])
            y_roll = y_roll.at[:,idx].set(state[:,1])
           
            state = self.compute_rollout_one_step(acc[:,idx],steer[:,idx],state)
            
            return (state,x_roll,y_roll),0.

        carry_init = state_init,x_roll_init,y_roll_init
        carry_final,result = lax.scan(lax_rollout,carry_init,jnp.arange(self.num_prime))
        state,x_roll,y_roll = carry_final

        cx_roll,cy_roll = self.compute_coeff(x_roll,y_roll)

        beta_best,res,_sigma_best,ker_red_best,\
        x_roll_red,y_roll_red,cx_roll_red,cy_roll_red,beta_samples \
            = self.prob2.compute_cem(cx_roll,cy_roll,x_roll,y_roll)
     
        beta_cem_data = (beta_samples,_sigma_best,
                        cx_roll,cy_roll)
        
        return x_roll_red[-1],y_roll_red[-1],beta_best[-1],\
            _sigma_best[-1], res,beta_cem_data
    
    @partial(jit, static_argnums=(0, ))	
    def compute_controls(self, xdot,ydot,xddot,yddot):
        v = jnp.sqrt(xdot**2+ydot**2)
        v = jnp.hstack((v,v[:,-1].reshape(-1,1)))
        
        acc = jnp.diff(v,axis=1)/self.t
        acc = jnp.hstack((acc,acc[:,-1].reshape(-1,1)))

        curvature_best = (yddot*xdot-ydot*xddot)/((xdot**2+ydot**2)**(1.5)) 
        steer = jnp.arctan(curvature_best*self.wheel_base  )

        return acc,steer
    
    @partial(jit, static_argnums=(0,))
    def compute_coeff(self,x,y):

        cost = jnp.dot(self.P_jax_prime.T, self.P_jax_prime) + 0.05*jnp.identity(self.nvar_prime)
        
        lincost_x = -jnp.dot(self.P_jax_prime.T, x.T ).T 
        lincost_y = -jnp.dot(self.P_jax_prime.T, y.T ).T 
        
        cx = jnp.linalg.solve(-cost, lincost_x.T).T
        cy = jnp.linalg.solve(-cost, lincost_y.T).T

        return cx,cy