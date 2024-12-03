import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random,vmap,lax
import jax
import jax.lax as lax

class Costs():
    def __init__(self,prob,num_reduced_sqrt,num_obs,num_prime,a_obs,b_obs,
                 y_lb,y_ub,alpha_quant,alpha_quant_lane,y_des_1,y_des_2,gamma):
        
        self.y_des_1,self.y_des_2 = y_des_1, y_des_2
        self.gamma = gamma
        self.alpha_quant = alpha_quant
        self.alpha_quant_lane = alpha_quant_lane

        self.prob = prob
        self.y_lb,self.y_ub = y_lb,y_ub

        self.num_prime = num_prime
        self.num_reduced = num_reduced_sqrt
        self.num_obs = num_obs
        self.a_obs,self.b_obs = a_obs,b_obs
        
        self.compute_f_bar_vmap = jit(vmap(self.compute_f_bar,in_axes=(0,0,None,None)))
        self.compute_lane_bar_vmap = jit(vmap(self.compute_lane_bar,in_axes=(0)))
        # self.compute_lane_des_mmd_vmap = jit(vmap(self.compute_lane_des_mmd,in_axes=(None,0,None,None)))

        self.compute_mmd_obs_vmap = jit(vmap(self.compute_mmd_obs,
                        in_axes=(0,0,0,0,None,None)))
        
        self.compute_mmd_lane_vmap = jit(vmap(self.compute_mmd_lane,in_axes=(0,0,0)))
        self.compute_lane_des_mmd_vmap = jit(vmap(self.compute_lane_des_mmd,in_axes=(0,0,0)))

        self.compute_cvar_obs_vmap = jit(vmap(self.compute_cvar_obs,
                        in_axes=(0,0,None,None)))
        
        self.compute_cvar_lane_vmap = jit(vmap(self.compute_cvar_lane,in_axes=(0)))
        self.compute_lane_des_cvar_vmap = jit(vmap(self.compute_lane_des_cvar,in_axes=(0)))

        self.compute_saa_obs_vmap = jit(vmap(self.compute_saa_obs,
                        in_axes=(0,0,None,None)))
        
        self.compute_saa_lane_vmap = jit(vmap(self.compute_saa_lane,in_axes=(0)))
        self.compute_lane_des_saa_vmap = jit(vmap(self.compute_lane_des_saa,in_axes=(0)))

    @partial(jit, static_argnums=(0,))
    def compute_f_bar(self,x,y,x_obs,y_obs): 

        wc_alpha = (x-x_obs)
        ws_alpha = (y-y_obs)

        cost = -(wc_alpha**2)/(self.a_obs**2) - (ws_alpha**2)/(self.b_obs**2) + jnp.ones((self.num_obs,self.num_prime))
        
        cost_bar = jnp.maximum(jnp.zeros((self.num_obs,self.num_prime)), cost)

        return cost_bar # num_obs x num_prime
    
    @partial(jit, static_argnums=(0,))
    def compute_lane_bar(self,y):

        cost_centerline_penalty_lb = -y+self.y_lb
        cost_centerline_penalty_ub = y-self.y_ub

        cost_lb = jnp.maximum(jnp.zeros((self.num_reduced,self.num_prime)), cost_centerline_penalty_lb)
        cost_ub = jnp.maximum(jnp.zeros((self.num_reduced,self.num_prime)), cost_centerline_penalty_ub)

        return cost_lb,cost_ub 
    
    @partial(jit, static_argnums=(0,))
    def compute_lane_des_mmd(self,beta,sigma,y):
        cost_centerline_1 = jnp.linalg.norm(y-self.y_des_1)
        cost_centerline_2 = jnp.linalg.norm(y-self.y_des_2)
        cost_centerline = cost_centerline_1*cost_centerline_2 - self.gamma
        costbar = jnp.maximum(jnp.zeros((self.num_reduced,self.num_prime)), cost_centerline)

        costbar = jnp.max(costbar,axis=1)
        mmd_lane_des = self.prob.compute_mmd(beta,costbar,sigma)

        # mmd_lane_des = self.prob.compute_mmd_vmap(beta,costbar.T)

        return jnp.sum(mmd_lane_des)
    
    @partial(jit, static_argnums=(0,))
    def compute_lane_des_cvar(self,y):

        cost_centerline_1 = jnp.linalg.norm(y-self.y_des_1)
        cost_centerline_2 = jnp.linalg.norm(y-self.y_des_2)
        cost_centerline = cost_centerline_1*cost_centerline_2 - self.gamma
        costbar = jnp.maximum(jnp.zeros((self.num_reduced,self.num_prime)), cost_centerline)

        costbar = jnp.max(costbar,axis=1)

        var_alpha = jnp.quantile(costbar,self.alpha_quant_lane)
        cvar_alpha = jnp.where(costbar>=var_alpha,costbar,jnp.full((costbar.shape[0],),jnp.nan))
        num_cvar = jnp.count_nonzero(~jnp.isnan(cvar_alpha))
        cvar_alpha = jnp.nan_to_num(cvar_alpha)
        cvar_alpha = jax.lax.cond( num_cvar>0, lambda _: cvar_alpha.sum()/num_cvar, lambda _ : 0., 0.)
      
        return cvar_alpha
    
    @partial(jit, static_argnums=(0,))
    def compute_lane_des_saa(self,y):

        cost_centerline_1 = jnp.linalg.norm(y-self.y_des_1)
        cost_centerline_2 = jnp.linalg.norm(y-self.y_des_2)
        cost_centerline = cost_centerline_1*cost_centerline_2 - self.gamma
        costbar = jnp.maximum(jnp.zeros((self.num_reduced,self.num_prime)), cost_centerline)

        costbar = jnp.max(costbar,axis=1)
        costbar = jnp.where(costbar>0., 1., 0.)
        
        return costbar.sum()/self.num_reduced

   
    @partial(jit, static_argnums=(0, ))	
    def compute_mmd_lane(self,beta,sigma,y_red):

        cost_lb,cost_ub = self.compute_lane_bar(y_red) # (num_reduced, num_prime)
        
        cost_lb = jnp.max(cost_lb,axis=1) # (num_reduced,)
        cost_ub = jnp.max(cost_ub,axis=1) # (num_reduced,)

        mmd_lb = self.prob.compute_mmd(beta,cost_lb,sigma)
        mmd_ub = self.prob.compute_mmd(beta,cost_ub,sigma)
        
        # mmd_lb = self.prob.compute_mmd_vmap(beta,cost_lb.T)
        # mmd_ub = self.prob.compute_mmd_vmap(beta,cost_ub.T)

        return jnp.sum(mmd_lb+mmd_ub)
    
    @partial(jit, static_argnums=(0, ))	
    def compute_cvar_lane(self,y_red):
       
        cost_lb,cost_ub = self.compute_lane_bar(y_red)
        
        cost_lb = jnp.max(cost_lb,axis=1) # (num_reduced,)

        var_alpha = jnp.quantile(cost_lb,self.alpha_quant)
        cvar_alpha = jnp.where(cost_lb>=var_alpha,cost_lb,jnp.full((cost_lb.shape[0],),jnp.nan))
        num_cvar = jnp.count_nonzero(~jnp.isnan(cvar_alpha))
        cvar_alpha = jnp.nan_to_num(cvar_alpha)
        cvar_alpha_lb = jax.lax.cond( num_cvar>0, lambda _: cvar_alpha.sum()/num_cvar, lambda _ : 0., 0.)

        cost_ub = jnp.max(cost_ub,axis=1) # (num_reduced,)

        var_alpha = jnp.quantile(cost_ub,self.alpha_quant)
        cvar_alpha = jnp.where(cost_ub>=var_alpha,cost_ub,jnp.full((cost_ub.shape[0],),jnp.nan))
        num_cvar = jnp.count_nonzero(~jnp.isnan(cvar_alpha))
        cvar_alpha = jnp.nan_to_num(cvar_alpha)
        cvar_alpha_ub = jax.lax.cond( num_cvar>0, lambda _: cvar_alpha.sum()/num_cvar, lambda _ : 0., 0.)
      
        return jnp.sum(cvar_alpha_lb+cvar_alpha_ub)
    
    @partial(jit, static_argnums=(0, ))	
    def compute_saa_lane(self,y_red):
       
        cost_mmd_lb,cost_mmd_ub = self.compute_lane_bar(y_red)
        
        cost_mmd_lb = jnp.max(cost_mmd_lb,axis=1) # (num_reduced,)
        costbar_lb = jnp.where(cost_mmd_lb>0., 1., 0.)
        
        cost_mmd_ub = jnp.max(cost_mmd_ub,axis=1) # (num_reduced,)
        costbar_ub = jnp.where(cost_mmd_ub>0., 1., 0.)

        return (costbar_lb.sum() + costbar_ub.sum())/self.num_reduced

    @partial(jit, static_argnums=(0, ))	
    def compute_mmd_obs(self,beta,sigma,x_roll,y_roll,
                    x_obs,y_obs):
        
        cost = self.compute_f_bar_vmap(x_roll[:,0:self.num_prime],y_roll[:,0:self.num_prime],
                                        x_obs,y_obs) # num_reduced x num_obs x num_prime
        
        cost = jnp.max(jnp.max(cost,axis=2),axis=1) # num_reduced,
        mmd_total = self.prob.compute_mmd(beta,cost,sigma)

        # cost = cost.transpose(2,1,0) # num_prime x num_obs x num_reduced
        # mmd_total = self.prob.compute_mmd_double_vmap(beta,cost)
        
        return jnp.sum(mmd_total)
    
    @partial(jit, static_argnums=(0, ))	
    def compute_cvar_obs(self,x_roll,y_roll,
                    x_obs,y_obs):
        
        costbar = self.compute_f_bar_vmap(x_roll[:,0:self.num_prime],y_roll[:,0:self.num_prime],
                                                x_obs,y_obs) # num_reduced x num_obs x num
        
        costbar = jnp.max(jnp.max(costbar,axis=2),axis=1) # num_reduced 

        var_alpha = jnp.quantile(costbar,self.alpha_quant)
        cvar_alpha = jnp.where(costbar>=var_alpha,costbar,jnp.full((costbar.shape[0],),jnp.nan))
        num_cvar = jnp.count_nonzero(~jnp.isnan(cvar_alpha))
        cvar_alpha = jnp.nan_to_num(cvar_alpha)
        cvar_alpha = jax.lax.cond( num_cvar>0, lambda _: cvar_alpha.sum()/num_cvar, lambda _ : 0., 0.)
      
        return cvar_alpha
    
    @partial(jit, static_argnums=(0, ))	
    def compute_saa_obs(self,x_roll,y_roll,
                    x_obs,y_obs):
        
        costbar = self.compute_f_bar_vmap(x_roll[:,0:self.num_prime],y_roll[:,0:self.num_prime],
                                                x_obs,y_obs) # num_reduced x num_obs x num
        
        costbar = jnp.max(jnp.max(costbar,axis=2),axis=1) # num_reduced 

        costbar = jnp.where(costbar>0., 1., 0.)
        
        return costbar.sum()/self.num_reduced
    
   