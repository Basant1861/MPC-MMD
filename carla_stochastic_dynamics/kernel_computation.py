import jax.numpy as jnp
from functools import partial
from jax import jit,vmap

class kernel_matrix():
    def __init__(self,num_reduced,ker_wt,P_jax):
        
        self.P_jax = P_jax
        self.nvar = P_jax.shape[1]
        self.num_reduced = num_reduced
  
        #### Kernel
        self.ker_wt = ker_wt

        self.compute_mmd_vmap = jit(vmap(self.compute_mmd,in_axes=(None,0,None)))
        self.compute_mmd_double_vmap = jit(vmap(self.compute_mmd_vmap,in_axes=(None,0)))
        self.compute_kernel_vmap = jit(vmap(self.compute_kernel,in_axes=(0,0,0)))
        
    @partial(jit, static_argnums=(0,))
    def compute_kernel_matrix(self,C_aa,C_ab,C_bb,sigma):
        ### Gaussian Kernel

        # dists_aa = jnp.linalg.norm(C_aa,axis=-1) # samples_A x samples_A
        # dists_ab = jnp.linalg.norm(C_ab,axis=-1) # samples_A x samples_B
        # dists_bb = jnp.linalg.norm(C_bb,axis=-1) # samples_B x samples_B

        # K_aa = jnp.exp(-dists_aa**2/(2*self.sigma**2))
        # K_ab = jnp.exp(-dists_ab**2/(2*self.sigma**2))
        # K_bb = jnp.exp(-dists_bb**2/(2*self.sigma**2))

        ### Laplace Kernel

        dists_aa = jnp.sum(jnp.absolute(C_aa), axis=-1)
        dists_ab = jnp.sum(jnp.absolute(C_ab), axis=-1)
        dists_bb = jnp.sum(jnp.absolute(C_bb), axis=-1)

        K_aa = jnp.exp(-dists_aa/sigma)
        K_ab = jnp.exp(-dists_ab/sigma)
        K_bb = jnp.exp(-dists_bb/sigma)

        #### Matern Kernel

        # dists_aa1 = jnp.sum(jnp.absolute(C_aa), axis=-1)
        # dists_ab1 = jnp.sum(jnp.absolute(C_ab), axis=-1)
        # dists_bb1 = jnp.sum(jnp.absolute(C_bb), axis=-1)

        # dists_aa2 = jnp.linalg.norm(C_aa,axis=-1)
        # dists_ab2 = jnp.linalg.norm(C_ab,axis=-1)
        # dists_bb2 = jnp.linalg.norm(C_bb,axis=-1)

        # K_aa = ( 1 + jnp.sqrt(5)*dists_aa1/sigma + 5*dists_aa2**2/(3*sigma**2) )*jnp.exp(-jnp.sqrt(5)*dists_aa1/sigma)
        # K_ab = ( 1 + jnp.sqrt(5)*dists_ab1/sigma + 5*dists_ab2**2/(3*sigma**2) )*jnp.exp(-jnp.sqrt(5)*dists_ab1/sigma)
        # K_bb = ( 1 + jnp.sqrt(5)*dists_bb1/sigma + 5*dists_bb2**2/(3*sigma**2) )*jnp.exp(-jnp.sqrt(5)*dists_bb1/sigma)

        return K_aa,K_ab,K_bb

    @partial(jit, static_argnums=(0,))
    def compute_kernel(self,A,B,sigma): # A - (samples_A x features) , B - (samples_B x features)
        C_aa = A[:, jnp.newaxis, :] - A[jnp.newaxis, :, :]  # samples_A x samples_A x features
        C_ab = A[:, jnp.newaxis, :] - B[jnp.newaxis, :, :]  # samples_A x samples_B x features 
        C_bb = B[:, jnp.newaxis, :] - B[jnp.newaxis, :, :]  # samples_B x samples_B x features

        K_aa,K_ab,K_bb = self.compute_kernel_matrix(C_aa,C_ab,C_bb,sigma)

        return K_aa,K_ab,K_bb
    
    @partial(jit, static_argnums=(0,))
    def compute_mmd(self,beta,cost,sigma):
        cost = cost.reshape(-1,1)
        y = jnp.zeros((cost.shape[0],cost.shape[1]))

        beta_del = (1/(self.num_reduced))*jnp.ones((self.num_reduced,1))
        beta = beta.reshape((self.num_reduced,1))

        x_kernel1,xy_kernel1,y_kernel1 = self.compute_kernel(cost, y,sigma)
        # x_kernel2,xy_kernel2,y_kernel2 = self.compute_discrete_kernel(cost)

        x_kernel = x_kernel1 #+ 0*x_kernel2 + 0*x_kernel1*x_kernel2
        y_kernel = y_kernel1 #+ 0*y_kernel2 + 0*y_kernel1*y_kernel2
        xy_kernel = xy_kernel1 #+ 0*xy_kernel2 + 0*xy_kernel1*xy_kernel2
 
        mmd_cost = jnp.dot(beta.T,jnp.dot(x_kernel,beta))\
                    -2*jnp.dot(beta.T,jnp.dot(xy_kernel,beta_del))\
                    # + jnp.dot(beta_del.T,jnp.dot(y_kernel,beta_del)) 
                
        # return jnp.sqrt(jnp.abs(mmd_cost.reshape(-1)))
        return self.ker_wt*mmd_cost.reshape(-1)

    