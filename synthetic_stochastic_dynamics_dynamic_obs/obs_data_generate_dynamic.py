import jax.numpy as jnp
import bernstein_coeff_order10_arbitinterval
from jax import jit
import numpy as np
import jax
from functools import partial

class obs_data():

    def __init__(self,_num_batch):
        t_fin = 15
        self.num = 100
        t = t_fin/self.num
        tot_time = np.linspace(0, t_fin, self.num)
        tot_time = tot_time
        tot_time_copy = tot_time.reshape(self.num, 1)

        P, Pdot, Pddot = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)
        self.P,self.Pdot,self.Pddot = jnp.asarray(P),jnp.asarray(Pdot), jnp.asarray(Pddot)

        self.v_min,self.v_max = 0.1,30.
        self.nvar = jnp.shape(P)[1]

        self.weight_smoothness_x = 100
        self.weight_smoothness_y = 100

        self.rho_v = 1 
        self.rho_offset = 1

        self.k_p_v = 2
        self.k_d_v = 2.0*jnp.sqrt(self.k_p_v)

        self.k_p = 2
        self.k_d = 2.0*jnp.sqrt(self.k_p)

        self.A_eq_x = jnp.vstack(( P[0], Pdot[0], Pddot[0]  ))
        self.A_eq_y = jnp.vstack(( P[0], Pdot[0], Pddot[0], Pdot[-1]  ))
            
        self.modes = jnp.asarray([1,2,3])
        self.modes_probs = jnp.asarray([0.4,0.2,0.4])
        # self.v_mu = jnp.asarray([5.,7.,3.])
        # self.v_sigma = jnp.asarray([1.5,0.1,3]) 

        self.v_mu = 6.
        self.v_sigma = 0.1 

        self._num_batch = _num_batch

        self.size_1 = jnp.array(self.modes_probs[0]*self._num_batch,int) 
        self.size_2 = jnp.array(self.modes_probs[1]*self._num_batch,int) 
        self.size_3 = jnp.array(self.modes_probs[2]*self._num_batch,int) 

        self.size_1 = jax.lax.cond(self.size_1+self.size_2+self.size_3==self._num_batch,
                                lambda _: self.size_1,lambda _: self._num_batch-(self.size_2 + self.size_3), 0)
        
    @partial(jit, static_argnums=(0,))	
    def compute_boundary_vec( self,x_init, vx_init, ax_init, y_init, vy_init, ay_init):

        x_init_vec = x_init*jnp.ones((self._num_batch, 1))
        y_init_vec = y_init*jnp.ones((self._num_batch, 1))

        vx_init_vec = vx_init*jnp.ones((self._num_batch, 1))
        vy_init_vec = vy_init*jnp.ones((self._num_batch, 1))

        ax_init_vec = ax_init*jnp.ones((self._num_batch, 1))
        ay_init_vec = ay_init*jnp.ones((self._num_batch, 1))

        b_eq_x = jnp.hstack(( x_init_vec, vx_init_vec, ax_init_vec ))
        b_eq_y = jnp.hstack(( y_init_vec, vy_init_vec, ay_init_vec, jnp.zeros((self._num_batch, 1  ))   ))

        return b_eq_x, b_eq_y

    @partial(jit, static_argnums=(0,))	
    def compute_obs_guess(self,b_eq_x,b_eq_y,y_samples,seed):

        v_des = self.sampling_param(seed)
        y_des = y_samples

        #############################
        A_vd = self.Pddot-self.k_p_v*self.Pdot
        b_vd = -self.k_p_v*jnp.ones((self._num_batch, self.num))*(v_des)[:, jnp.newaxis]
        
        A_pd = self.Pddot-self.k_p*self.P
        b_pd = -self.k_p*jnp.ones((self._num_batch, self.num ))*(y_des)[:, jnp.newaxis]

        cost_smoothness_x = self.weight_smoothness_x*jnp.dot(self.Pddot.T, self.Pddot)
        cost_smoothness_y = self.weight_smoothness_y*jnp.dot(self.Pddot.T, self.Pddot)
        
        cost_x = cost_smoothness_x+self.rho_v*jnp.dot(A_vd.T, A_vd)
        cost_y = cost_smoothness_y+self.rho_offset*jnp.dot(A_pd.T, A_pd)

        cost_mat_x = jnp.vstack((  jnp.hstack(( cost_x, self.A_eq_x.T )), jnp.hstack(( self.A_eq_x, jnp.zeros(( jnp.shape(self.A_eq_x)[0], jnp.shape(self.A_eq_x)[0] )) )) ))
        cost_mat_y = jnp.vstack((  jnp.hstack(( cost_y, self.A_eq_y.T )), jnp.hstack(( self.A_eq_y, jnp.zeros(( jnp.shape(self.A_eq_y)[0], jnp.shape(self.A_eq_y)[0] )) )) ))
        
        lincost_x = -self.rho_v*jnp.dot(A_vd.T, b_vd.T).T
        lincost_y = -self.rho_offset*jnp.dot(A_pd.T, b_pd.T).T

        sol_x = jnp.linalg.solve(cost_mat_x, jnp.hstack(( -lincost_x, b_eq_x )).T).T
        sol_y = jnp.linalg.solve(cost_mat_y, jnp.hstack(( -lincost_y, b_eq_y )).T).T

        #######################

        primal_sol_x = sol_x[:,0:self.nvar]
        primal_sol_y = sol_y[:,0:self.nvar]
    
        x = jnp.dot(self.P, primal_sol_x.T).T
        y = jnp.dot(self.P, primal_sol_y.T).T
        
        return x,y

    @partial(jit, static_argnums=(0,))	
    def sampling_param(self,seed):
        
        param_samples = jax.random.normal(jax.random.PRNGKey(seed),(self._num_batch,))

        eps_v = param_samples*self.v_sigma + self.v_mu

        # eps_v1 = param_samples*self.v_sigma[0] + self.v_mu[0]
        # eps_v2 = param_samples*self.v_sigma[1] + self.v_mu[1]
        # eps_v3 = param_samples*self.v_sigma[2] + self.v_mu[2]

        # eps_v1 = jnp.clip(eps_v1, self.v_min*jnp.ones(self._num_batch),self.v_max*jnp.ones(self._num_batch)   )
        # eps_v2 = jnp.clip(eps_v2, self.v_min*jnp.ones(self._num_batch),self.v_max*jnp.ones(self._num_batch)   )
        # eps_v3 = jnp.clip(eps_v3, self.v_min*jnp.ones(self._num_batch),self.v_max*jnp.ones(self._num_batch)   )

        # weight_samples = jax.random.choice(jax.random.PRNGKey(seed),self.modes, (self._num_batch,),p=self.modes_probs)

        # idx_1 = jnp.where(weight_samples==1,size=self.size_1)
        # idx_2 = jnp.where(weight_samples==2,size=self.size_2)
        # idx_3 = jnp.where(weight_samples==3,size=self.size_3)

        # eps_v = jnp.hstack((eps_v1[idx_1],eps_v2[idx_2],eps_v3[idx_3]))

        return eps_v

    @partial(jit, static_argnums=(0,1))	
    def compute_obs_data(self,num_obs,seed):

        x_obs_init = jax.random.choice(jax.random.PRNGKey(seed),jnp.linspace(15,45,30), (num_obs, ),replace=False)
        # y_obs_init = jax.random.choice(jax.random.PRNGKey(seed),jnp.array([-1.75,1.75]),(num_obs,))
        y_obs_init = 1.75*jnp.ones(num_obs)

        vx_obs_init = jax.random.choice(jax.random.PRNGKey(seed),jnp.linspace(0.5,5,15), (num_obs, ),replace = False)
        vy_obs_init = jnp.zeros(num_obs)

        psi_obs_init = jnp.zeros(num_obs)

        return x_obs_init,y_obs_init,vx_obs_init,vy_obs_init,psi_obs_init
