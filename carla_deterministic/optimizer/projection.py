import jax.numpy as jnp
from functools import partial
from jax import jit,lax

class Projection():
    def __init__(self,num_obs,num_circles,v_max,v_min,a_max,
                 num,P,Pdot,Pddot,rho_ineq,rho_obs,rho_projection,rho_lane,
                 gamma,gamma_obs,num_batch,maxiter,nvar,
                 A_eq_x,A_eq_y,A_obs,A_lane_bound,y_lb,y_ub,
                 A_vel,A_acc,A_projection,a_centr,wheel_base):

        self.a_centr = a_centr
        self.wheel_base = wheel_base
        self.num_circles = num_circles
        self.v_max = v_max
        self.v_min = v_min
        self.a_max = a_max
        self.num_obs = num_obs
        
        self.num = num
        
        self.P_jax, self.Pdot_jax, self.Pddot_jax = P,Pdot,Pddot
        self.nvar = nvar

        ################################################################
        self.A_eq_x = A_eq_x
        self.A_eq_y = A_eq_y
             
        self.A_vel = A_vel 
        self.A_acc = A_acc
        self.A_projection = A_projection
        
        ################################### obstacle avoidance		
        self.A_obs = A_obs

        ###################################################

        self.rho_ineq = rho_ineq
        self.rho_obs = rho_obs
        self.rho_projection = rho_projection
        self.rho_lane = rho_lane

        self.maxiter = maxiter

        self.gamma = gamma
        self.gamma_obs = gamma_obs

        # upper lane bound
        self.A_lane_bound = A_lane_bound
        self.y_lb,self.y_ub = y_lb,y_ub

        self.num_batch = num_batch

        self.jax_interp = jit(jnp.interp) ############## jitted interp

    @partial(jit, static_argnums=(0,))
    def initial_alpha_d_obs(self, a_obs,b_obs,x_guess, y_guess, xdot_guess, ydot_guess, xddot_guess, yddot_guess, 
                            x_obs, y_obs, lamda_x, lamda_y):

        wc_alpha_temp = (x_guess-x_obs[:,jnp.newaxis])
        ws_alpha_temp = (y_guess-y_obs[:,jnp.newaxis])

        wc_alpha = wc_alpha_temp.transpose(1, 0, 2)
        ws_alpha = ws_alpha_temp.transpose(1, 0, 2)

        wc_alpha = wc_alpha.reshape(self.num_batch, self.num*((self.num_obs)*self.num_circles))
        ws_alpha = ws_alpha.reshape(self.num_batch, self.num*((self.num_obs)*self.num_circles))

        alpha_obs = jnp.arctan2( ws_alpha*a_obs, wc_alpha*b_obs)
        c1_d = 1.0*self.rho_obs*(a_obs**2*jnp.cos(alpha_obs)**2 + b_obs**2*jnp.sin(alpha_obs)**2 )
        c2_d = 1.0*self.rho_obs*(a_obs*wc_alpha*jnp.cos(alpha_obs) + b_obs*ws_alpha*jnp.sin(alpha_obs)  )

        d_temp = c2_d/c1_d
        d_obs = jnp.maximum(jnp.ones((self.num_batch,  self.num*self.num_obs*self.num_circles   )), d_temp   )
        

        ################# velocity terms

        wc_alpha_vx = xdot_guess
        ws_alpha_vy = ydot_guess
        alpha_v = jnp.unwrap(jnp.arctan2( ws_alpha_vy, wc_alpha_vx))
        
        
        c1_d_v = 1.0*self.rho_ineq*(jnp.cos(alpha_v)**2 + jnp.sin(alpha_v)**2 )
        c2_d_v = 1.0*self.rho_ineq*(wc_alpha_vx*jnp.cos(alpha_v) + ws_alpha_vy*jnp.sin(alpha_v)  )
        
        d_temp_v = c2_d_v/c1_d_v
        
        d_v = jnp.clip(d_temp_v, self.v_min, self.v_max )
        
        ################# acceleration terms

        wc_alpha_ax = xddot_guess
        ws_alpha_ay = yddot_guess
        alpha_a = jnp.unwrap(jnp.arctan2( ws_alpha_ay, wc_alpha_ax))
        

        c1_d_a = 1.0*self.rho_ineq*(jnp.cos(alpha_a)**2 + jnp.sin(alpha_a)**2 )
        c2_d_a = 1.0*self.rho_ineq*(wc_alpha_ax*jnp.cos(alpha_a) + ws_alpha_ay*jnp.sin(alpha_a)  )

        d_temp_a = c2_d_a/c1_d_a
        d_a = jnp.clip(d_temp_a, jnp.zeros((self.num_batch, self.num)), self.a_max  )

        
        #########################################33
        res_ax_vec = xddot_guess-d_a*jnp.cos(alpha_a)
        res_ay_vec = yddot_guess-d_a*jnp.sin(alpha_a)
        
        res_vx_vec = xdot_guess-d_v*jnp.cos(alpha_v)
        res_vy_vec = ydot_guess-d_v*jnp.sin(alpha_v)

        res_x_obs_vec = wc_alpha-a_obs*d_obs*jnp.cos(alpha_obs)
        res_y_obs_vec = ws_alpha-b_obs*d_obs*jnp.sin(alpha_obs)

        res_vel_vec = jnp.hstack(( res_vx_vec,  res_vy_vec  ))
        res_acc_vec = jnp.hstack(( res_ax_vec,  res_ay_vec  ))
        res_obs_vec = jnp.hstack(( res_x_obs_vec, res_y_obs_vec  ))
        
        lamda_x = lamda_x-self.rho_ineq*jnp.dot(self.A_acc.T, res_ax_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vx_vec.T).T\
        # -self.rho_obs*jnp.dot(self.A_obs.T, res_x_obs_vec.T).T

        lamda_y = lamda_y-self.rho_ineq*jnp.dot(self.A_acc.T, res_ay_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vy_vec.T).T\
        # -self.rho_obs*jnp.dot(self.A_obs.T, res_y_obs_vec.T).T

        return alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, alpha_v, d_v

    @partial(jit, static_argnums=(0,))	
    def compute_x(self, a_obs,b_obs,lamda_x, lamda_y, b_eq_x, b_eq_y, alpha_a, d_a, alpha_v, d_v, x_obs, y_obs,
                   alpha_obs, d_obs, c_x_bar, c_y_bar,s_lane):

        b_lane_lb = -self.gamma*self.y_lb*jnp.ones(( self.num_batch, self.num-1  ))
        b_lane_ub = self.gamma*self.y_ub*jnp.ones(( self.num_batch, self.num-1  ))
        b_lane_bound = jnp.hstack((b_lane_ub,b_lane_lb))
        b_lane_aug = b_lane_bound-s_lane

        b_ax_ineq = d_a*jnp.cos(alpha_a)
        b_ay_ineq = d_a*jnp.sin(alpha_a)

        b_vx_ineq = d_v*jnp.cos(alpha_v)
        b_vy_ineq = d_v*jnp.sin(alpha_v)


        temp_x_obs = d_obs*jnp.cos(alpha_obs)*a_obs
        b_obs_x = x_obs.reshape(self.num*self.num_obs*self.num_circles)+temp_x_obs
            
        temp_y_obs = d_obs*jnp.sin(alpha_obs)*b_obs
        b_obs_y = y_obs.reshape(self.num*self.num_obs*self.num_circles)+temp_y_obs

        cost_x = self.rho_projection*jnp.dot(self.A_projection.T, self.A_projection)\
            +self.rho_ineq*jnp.dot(self.A_acc.T, self.A_acc)+self.rho_ineq*jnp.dot(self.A_vel.T, self.A_vel)\
            +self.rho_obs*jnp.dot(self.A_obs.T, self.A_obs)
        
        cost_y = self.rho_projection*jnp.dot(self.A_projection.T, self.A_projection)\
            +self.rho_ineq*jnp.dot(self.A_acc.T, self.A_acc)+self.rho_ineq*jnp.dot(self.A_vel.T, self.A_vel)\
                +self.rho_lane*jnp.dot(self.A_lane_bound.T, self.A_lane_bound)\
            +self.rho_obs*jnp.dot(self.A_obs.T, self.A_obs)

        cost_mat_x = jnp.vstack((  jnp.hstack(( cost_x, self.A_eq_x.T )), jnp.hstack(( self.A_eq_x, jnp.zeros(( jnp.shape(self.A_eq_x)[0], jnp.shape(self.A_eq_x)[0] )) )) ))
        
        cost_mat_y = jnp.vstack((  jnp.hstack(( cost_y, self.A_eq_y.T )), jnp.hstack(( self.A_eq_y, jnp.zeros(( jnp.shape(self.A_eq_y)[0], jnp.shape(self.A_eq_y)[0] )) )) ))
        
        lincost_x = -lamda_x-self.rho_projection*jnp.dot(self.A_projection.T, c_x_bar.T).T\
            -self.rho_ineq*jnp.dot(self.A_acc.T, b_ax_ineq.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, b_vx_ineq.T).T\
            -self.rho_obs*jnp.dot(self.A_obs.T, b_obs_x.T).T

        lincost_y = -lamda_y-self.rho_projection*jnp.dot(self.A_projection.T, c_y_bar.T).T\
            -self.rho_ineq*jnp.dot(self.A_acc.T, b_ay_ineq.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, b_vy_ineq.T).T\
                            -self.rho_lane*jnp.dot(self.A_lane_bound.T, b_lane_aug.T).T \
            -self.rho_obs*jnp.dot(self.A_obs.T, b_obs_y.T).T

        sol_x = jnp.linalg.solve(cost_mat_x, jnp.hstack(( -lincost_x, b_eq_x )).T).T
        sol_y = jnp.linalg.solve(cost_mat_y, jnp.hstack(( -lincost_y, b_eq_y )).T).T

        primal_sol_x = sol_x[:,0:self.nvar]
        primal_sol_y = sol_y[:,0:self.nvar]

        x = jnp.dot(self.P_jax, primal_sol_x.T).T
        xdot = jnp.dot(self.Pdot_jax, primal_sol_x.T).T
        xddot = jnp.dot(self.Pddot_jax, primal_sol_x.T).T
        xdddot = jnp.dot(self.Pddot_jax, primal_sol_x.T).T

        y = jnp.dot(self.P_jax, primal_sol_y.T).T
        ydot = jnp.dot(self.Pdot_jax, primal_sol_y.T).T
        yddot = jnp.dot(self.Pddot_jax, primal_sol_y.T).T

        s_lane = jnp.maximum( jnp.zeros(( self.num_batch, 2*(self.num-1) )),-jnp.dot(self.A_lane_bound, primal_sol_y.T).T+b_lane_bound )
        res_lane_vec = jnp.dot(self.A_lane_bound, primal_sol_y.T).T-b_lane_bound+s_lane

        return primal_sol_x, primal_sol_y, x, y, xdot, ydot, xddot, yddot,res_lane_vec,s_lane

    @partial(jit, static_argnums=(0,))
    def comp_d_obs_prev(self,d_obs_prev):
        d_obs_batch = jnp.reshape(d_obs_prev,(self.num_batch,self.num_circles*self.num_obs,self.num))
        d_obs_batch_modified = jnp.dstack((jnp.ones((self.num_batch,self.num_circles*self.num_obs,1)), d_obs_batch[:,:,0:self.num-1]))
        return d_obs_batch_modified.reshape(self.num_batch,-1)

    @partial(jit, static_argnums=(0,))	
    def compute_alph_d(self,x_obs, y_obs, x, y, xdot, ydot, xddot, yddot, lamda_x, lamda_y,
                      d_obs_prev,a_obs,b_obs,res_lane_vec,kappa):
        
        wc_alpha_temp = (x-x_obs[:,jnp.newaxis])
        ws_alpha_temp = (y-y_obs[:,jnp.newaxis])

        wc_alpha = wc_alpha_temp.transpose(1, 0, 2)
        ws_alpha = ws_alpha_temp.transpose(1, 0, 2)

        wc_alpha = wc_alpha.reshape(self.num_batch, self.num*((self.num_obs)*self.num_circles))
        ws_alpha = ws_alpha.reshape(self.num_batch, self.num*((self.num_obs)*self.num_circles))

        alpha_obs = jnp.arctan2( ws_alpha*a_obs, wc_alpha*b_obs)
        c1_d = 1.0*self.rho_obs*(a_obs**2*jnp.cos(alpha_obs)**2 + b_obs**2*jnp.sin(alpha_obs)**2 )
        c2_d = 1.0*self.rho_obs*(a_obs*wc_alpha*jnp.cos(alpha_obs) + b_obs*ws_alpha*jnp.sin(alpha_obs)  )

        d_temp = c2_d/c1_d

        d_obs_prev = self.comp_d_obs_prev(d_obs_prev)
        d_obs = jnp.maximum(jnp.ones((self.num_batch,self.num*((self.num_obs)*self.num_circles)  )) + (1-self.gamma_obs)*(d_obs_prev-1),d_temp)

        ################# velocity terms

        wc_alpha_vx = xdot
        ws_alpha_vy = ydot

        alpha_v = jnp.arctan2(ydot, xdot)

        c1_d_v = 1.0*self.rho_ineq*(jnp.cos(alpha_v)**2 + jnp.sin(alpha_v)**2 )
        c2_d_v = 1.0*self.rho_ineq*(wc_alpha_vx*jnp.cos(alpha_v) + ws_alpha_vy*jnp.sin(alpha_v)  )
        
        d_temp_v = c2_d_v/c1_d_v
        
        d_max = jnp.sqrt(self.a_centr/jnp.abs(kappa))
        v_max = jnp.minimum(self.v_max,d_max)
        d_v = jnp.clip(d_temp_v, self.v_min, self.v_max )
        ################# acceleration terms

        wc_alpha_ax = xddot
        ws_alpha_ay = yddot

        alpha_a = jnp.arctan2(yddot, xddot)

        c1_d_a = 1.0*self.rho_ineq*(jnp.cos(alpha_a)**2 + jnp.sin(alpha_a)**2 )
        c2_d_a = 1.0*self.rho_ineq*(wc_alpha_ax*jnp.cos(alpha_a) + ws_alpha_ay*jnp.sin(alpha_a)  )

        d_temp_a = c2_d_a/c1_d_a
        d_a = jnp.clip(d_temp_a, jnp.zeros((self.num_batch, self.num)), self.a_max )

            
        #######################################################

        #########################################33
        res_ax_vec = xddot-d_a*jnp.cos(alpha_a)
        res_ay_vec = yddot-d_a*jnp.sin(alpha_a)
        
        res_vx_vec = xdot-d_v*jnp.cos(alpha_v)
        res_vy_vec = ydot-d_v*jnp.sin(alpha_v)

        res_x_obs_vec = wc_alpha-a_obs*d_obs*jnp.cos(alpha_obs)
        res_y_obs_vec = ws_alpha-b_obs*d_obs*jnp.sin(alpha_obs)

        res_vel_vec = jnp.hstack(( res_vx_vec,  res_vy_vec  ))
        res_acc_vec = jnp.hstack(( res_ax_vec,  res_ay_vec  ))
        res_obs_vec = jnp.hstack(( res_x_obs_vec, res_y_obs_vec  ))

        res_norm_batch = jnp.linalg.norm(res_acc_vec, axis =1)\
                +jnp.linalg.norm(res_vel_vec, axis =1)\
                            +jnp.linalg.norm(res_lane_vec, axis = 1) \
                + jnp.linalg.norm(res_obs_vec, axis =1) \

        lamda_x = lamda_x-self.rho_ineq*jnp.dot(self.A_acc.T, res_ax_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vx_vec.T).T \
            -self.rho_obs*jnp.dot(self.A_obs.T, res_x_obs_vec.T).T\

        lamda_y = lamda_y-self.rho_ineq*jnp.dot(self.A_acc.T, res_ay_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vy_vec.T).T \
                    -self.rho_lane*jnp.dot(self.A_lane_bound.T, res_lane_vec.T).T \
            -self.rho_obs*jnp.dot(self.A_obs.T, res_y_obs_vec.T).T\
    
        return alpha_obs, d_obs,alpha_a, d_a, lamda_x, lamda_y, res_norm_batch, alpha_v, d_v
    
    @partial(jit, static_argnums=(0, ))	
    def compute_projection(self,x_obs,y_obs,b_eq_x, b_eq_y,lamda_x_init, lamda_y_init, c_x_bar,
                            c_y_bar,a_obs,b_obs,s_lane,arc_vec, kappa):
        
        s_lane_init = s_lane

        x_guess = jnp.dot(self.P_jax, c_x_bar.T).T 
        y_guess = jnp.dot(self.P_jax, c_y_bar.T).T 

        xdot_guess = jnp.dot(self.Pdot_jax, c_x_bar.T).T 
        ydot_guess = jnp.dot(self.Pdot_jax, c_y_bar.T).T 

        xddot_guess = jnp.dot(self.Pddot_jax, c_x_bar.T).T 
        yddot_guess = jnp.dot(self.Pddot_jax, c_y_bar.T).T 

        alpha_obs_init, d_obs_init, alpha_a_init, d_a_init, lamda_x_init, lamda_y_init, alpha_v_init, d_v_init = self.initial_alpha_d_obs(a_obs,b_obs,x_guess, y_guess, xdot_guess, ydot_guess,
                                                                                                                                           xddot_guess, yddot_guess, x_obs, y_obs, lamda_x_init, lamda_y_init)
        
        def lax_projection(carry,idx):

            c_x, c_y, x, y, xdot, ydot, xddot, yddot, alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, \
                    res_norm_batch, alpha_v, d_v,s_lane = carry
            
            d_obs_prev = d_obs

            c_x, c_y, x, y, xdot, ydot, xddot, yddot,res_lane_vec,s_lane = self.compute_x(a_obs,b_obs,lamda_x, lamda_y, b_eq_x, b_eq_y,
                         alpha_a, d_a, alpha_v, d_v, x_obs, y_obs, alpha_obs, 
                         d_obs, c_x_bar, c_y_bar,s_lane)

            kappa_interp = self.jax_interp(jnp.clip(x, jnp.zeros((self.num_batch, self.num)), 
                                                    arc_vec[-1]*jnp.ones((self.num_batch, self.num)) ), arc_vec, kappa)

            alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, res_norm_batch, alpha_v, d_v = self.compute_alph_d(x_obs, y_obs, 
                                                                                            x, y, xdot, ydot, xddot, yddot, lamda_x, lamda_y,
                                                                                        d_obs_prev,a_obs,b_obs,res_lane_vec,kappa_interp)

            curvature_frenet = d_a*jnp.sin(alpha_a-alpha_v)/(d_v**2)
            steering = jnp.arctan((curvature_frenet+kappa_interp*jnp.cos(alpha_v)/(1-y*kappa_interp) )*self.wheel_base)

            return (c_x, c_y, x, y, xdot, ydot, xddot, yddot, alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, \
                    res_norm_batch, alpha_v, d_v,s_lane)\
                    ,(steering,kappa_interp)

        carry_init = (jnp.zeros((self.num_batch,self.nvar)), jnp.zeros((self.num_batch,self.nvar)), 
                      jnp.zeros((self.num_batch,self.num)),jnp.zeros((self.num_batch,self.num)),jnp.zeros((self.num_batch,self.num)), jnp.zeros((self.num_batch,self.num)),
            jnp.zeros((self.num_batch,self.num)), jnp.zeros((self.num_batch,self.num))
        , alpha_obs_init, d_obs_init, alpha_a_init, d_a_init, lamda_x_init, lamda_y_init, jnp.zeros((self.num_batch)), 
        alpha_v_init, d_v_init,s_lane_init)

        carry_final,result = lax.scan(lax_projection,carry_init,jnp.arange(self.maxiter))
        c_x, c_y, x, y, xdot, ydot, xddot, yddot, alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, \
                    res_norm_batch, alpha_v, d_v,s_lane= carry_final
        
        steering = result[0][-1]
        kappa_interp = result[1][-1]

        return 	c_x, c_y, x, y, xdot, ydot, xddot, yddot, \
            res_norm_batch,lamda_x, lamda_y,s_lane,\
            steering,kappa_interp
