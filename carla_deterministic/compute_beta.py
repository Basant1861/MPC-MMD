import jax
import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random,lax,vmap
from kernel_computation import kernel_matrix

class beta_cem():

    def __init__(self,num_reduced,num_mother,ker_wt,P_jax):
        
        self.prob = kernel_matrix(num_reduced,ker_wt,P_jax)
        
        self.num_samples_cem = 100
        self.maxiter_beta_cem = 20

        self.num_samples = num_mother
        self.num_samples_reduced_set = num_reduced

        self.mean_beta = jnp.zeros(self.num_samples + 1)

        cov_diag = np.ones(self.num_samples + 1)

        self.cov_beta = 20*jnp.asarray(np.diag(cov_diag))
        key = random.PRNGKey(0)
        self.num_ellite_beta = np.maximum(int(0.1*self.num_samples_cem) + 1, 3)

        self.key = key
        self.sigma_clip = 0.01

        self.A_projection = jnp.identity(self.num_samples_reduced_set)
        self.A_eq_proj = jnp.ones((1,self.num_samples_reduced_set))
        self.b_eq_proj = jnp.ones(1)

        self.A_eq_beta = jnp.ones((1,self.num_samples_reduced_set))
        self.b_eq_beta = jnp.ones(1)
        self.rho_beta = 1.0

        self.compute_beta_reduced_vmap = jit(vmap(self.compute_beta_reduced,in_axes=(0,0,0)))

    @partial(jit, static_argnums=(0,))	
    def compute_beta_samples_initial(self, key,mean_beta, cov_beta):

        key, subkey = random.split(key)
        
        beta_samples = jax.random.multivariate_normal(key, mean_beta, cov_beta, (self.num_samples_cem, ))	
        beta_samples = beta_samples.at[:,-1].set(jnp.clip(beta_samples[:,-1],self.sigma_clip))		

        return beta_samples

    @partial(jit, static_argnums=(0,))	
    def compute_mean_cov_beta(self, key,cost_batch, beta):

        key, subkey = random.split(key)
        
        idx_ellite = jnp.argsort(cost_batch)

        beta_ellite = beta[idx_ellite[0:self.num_ellite_beta]]

        mean_beta = jnp.mean(beta_ellite, axis = 0)
        cov_beta = jnp.cov(beta_ellite.T)+0.05*jnp.identity(self.num_samples+1)

        beta_samples = jax.random.multivariate_normal(key, mean_beta, cov_beta, (self.num_samples_cem-self.num_ellite_beta, ))
        beta_samples = jnp.vstack(( beta_ellite, beta_samples  ))

        beta_samples = beta_samples.at[:,-1].set(jnp.clip(beta_samples[:,-1],self.sigma_clip))		

        return beta_samples
    
    @partial(jit, static_argnums=(0,))
    def compute_beta_reduced(self,ker_red,ker_mixed,ker_total):
        cost = self.rho_beta*ker_red + 0.05*jnp.identity(self.num_samples_reduced_set)

        # 11x11
        cost_mat = jnp.vstack((  jnp.hstack(( cost, self.A_eq_beta.T )), jnp.hstack(( self.A_eq_beta, jnp.zeros(( jnp.shape(self.A_eq_beta)[0], jnp.shape(self.A_eq_beta)[0] )) )) ))
        
        lincost = -self.rho_beta*(1/self.num_samples)*jnp.sum(ker_mixed,axis=1).reshape(-1)

        sol_beta = jnp.linalg.solve(cost_mat,jnp.hstack(( -lincost, self.b_eq_beta )))

        primal_sol_beta = sol_beta[0:self.num_samples_reduced_set]
        
        ##############################

        beta_const = (1/self.num_samples)*jnp.ones(self.num_samples)
        q = -2*(1/self.num_samples)*jnp.sum(ker_mixed,axis=1)
        cost_mmd = jnp.dot( primal_sol_beta.T, jnp.dot(ker_red, primal_sol_beta) )+jnp.dot(q.T, primal_sol_beta)\
                        # + (1/self.num_samples**2)*jnp.sum(ker_total)
                        # + jnp.dot(beta_const.T, jnp.dot(ker_total, beta_const) )

        return primal_sol_beta,cost_mmd
    
    @partial(jit, static_argnums=(0, ))	
    def compute_cem(self, cx_mother,cy_mother,x_mother,y_mother):
        cx_mother_stack = jnp.tile(cx_mother,(self.num_samples_cem,1,1))
        cy_mother_stack = jnp.tile(cy_mother,(self.num_samples_cem,1,1))

        x_mother_stack = jnp.tile(x_mother,(self.num_samples_cem,1,1))
        y_mother_stack = jnp.tile(y_mother,(self.num_samples_cem,1,1))

        B = jnp.dstack((cx_mother_stack,cy_mother_stack)) # num_samples_cem x num_samples x 22
        
        res_init = jnp.zeros(self.maxiter_beta_cem)

        mean_beta = self.mean_beta
        cov_beta = self.cov_beta
        
        key_init,_ = jax.random.split(self.key)

        beta_samples_init = self.compute_beta_samples_initial(key_init,mean_beta, cov_beta) # num_samples_cem x (num_samples+1)
       
        def lax_cem(carry,idx):
            res,key,beta_samples = carry
            
            sigma = beta_samples[:,-1]
            
            idx_beta = jnp.argsort(jnp.abs(beta_samples[:,0:self.num_samples]),axis=1) # num_samples_cem x num_samples
            idx_beta_top = idx_beta[:,self.num_samples-self.num_samples_reduced_set:self.num_samples] # num_samples_cem x red_set
            
            cx_red,x_red,cy_red,y_red\
            = vmap(lambda i,j:  (cx_mother_stack[i][j], x_mother_stack[i][j], cy_mother_stack[i][j], y_mother_stack[i][j]))\
                (jnp.arange(self.num_samples_cem),idx_beta_top) # num_samples_cem x red_set x 11
            
            A = jnp.dstack((cx_red,cy_red)) # num_samples_cem x red_set x 22
            
            ## ker_red - num_samples_cem x red_set x red_set; ker_mixed - num_samples_cem x red_set x num_samples
            ker_red,ker_mixed,ker_total = self.prob.compute_kernel_vmap(A,B,sigma)

            beta_top,cost_batch = self.compute_beta_reduced_vmap(ker_red,ker_mixed,ker_total) # num_samples_cem x red_set

            key,_ = jax.random.split(key)

            beta_samples = self.compute_mean_cov_beta(key,cost_batch, beta_samples)
            
            idx_min = jnp.argmin(cost_batch)

            beta_best = beta_top[idx_min]
            sigma_best = beta_samples[idx_min][-1]

            res = res.at[idx].set((jnp.min(cost_batch)))

            x_obs_red, y_obs_red = x_red[idx_min], y_red[idx_min]
            cx_obs_red, cy_obs_red = cx_red[idx_min], cy_red[idx_min]

            return (res,key,beta_samples),\
                    (sigma_best,beta_best,ker_red[idx_min],x_obs_red, y_obs_red,
                     cx_obs_red,cy_obs_red,beta_samples[idx_min][0:-1])
        
        carry_init = (res_init,key_init,beta_samples_init)
        
        carry_final,result = lax.scan(lax_cem,carry_init,jnp.arange(self.maxiter_beta_cem))
        res,_,_  = carry_final
        
        sigma_best = result[0]
        beta_best = result[1]
        ker_red_best = result[2]
        x_obs_red, y_obs_red = result[3],result[4]
        cx_obs_red, cy_obs_red = result[5],result[6]

        beta_samples = result[7]

        return beta_best,res,sigma_best,ker_red_best,\
            x_obs_red, y_obs_red,cx_obs_red, cy_obs_red,\
            beta_samples




    
