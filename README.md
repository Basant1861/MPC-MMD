# MPC-MMD
Repository associated with our RAL submission.


https://github.com/user-attachments/assets/50226986-e1ef-4133-9a12-ed1860260f7d

## Getting Started

1. Clone this repository:
```
git clone https://github.com/Basant1861/MPC-MMD.git
cd MPC-MMD
```
2. Create a conda environment and install the dependencies:

```
conda create -n mpc_mmd python=3.8
conda activate mpc_mmd
pip install -r requirements.txt
```
3. For CARLA(version 13), follow the instructions given in the official documentation https://carla.readthedocs.io/en/0.9.13/

***Step 0*** Create two folders *data* and *stats* (in the *synthetic_static_obs* as well as in the *synthetic_dynamic_obs* directories) with the following directory structure:
1. data
   - gaussian_noise
     - noise_<int(noise_level*100)>
       - ts_<num_prime>
   - beta_noise
     - noise_<int(noise_level*100)>
       - ts_<num_prime>
2. stats
   - gaussian_noise
     - noise_<int(noise_level*100)>
       - ts_<num_prime>
   - beta_noise
     - noise_<int(noise_level*100)>
       - ts_<num_prime>

Here noise_level corresponds to the control noise you want to add, typically between 0.1 and 0.5, num_prime is the rollout horizon and can be any positive integer<=100, typically 20,30,40,50,60.

## Synthetic Static Environment
Once inside the *synthetic_static_obs* directory follow the below steps:
***Step 0*** Change the following line to include the path to the optimizer module in ```main_mpc.py, validation.py, plot_traj_video.py```:
```
sys.path.insert(1, 'path/to/optimizer')
```

***Step 1*** To configure the obstacle scenarios you need to edit the function ```compute_obs_data``` in ```main_mpc.py```.

***Step 2*** Run the following command from the root directory:
```
python3 main_mpc.py --noise_levels <list_of_noise_levels(float)> --num_reduced_sets <list_of_reduced_set_sizes>
 --num_obs <int> --num_prime <int> --noises <beta,gaussian or both> --acc_const_noise <float> --steer_const_noise <float> --costs <list_of_costs> 
```
where *costs* can be one or all of <**mmd_opt**, **cvar**> and *noises* can be one or all of <**gaussian**, **beta**>. The above command will run for default 200 obstacle configurations. This number can be changed by modifying *num_configs* variable in ```main_mpc.py```

***Step 3*** Once *Step 2* is complete, there will be data files in the corresponding folders in *data-> <gaussian_noise/beta_noise> -> noise_level/ ts_<num_prime>*. Now we need to calculate the statistics for the collected data. Run the following command from the root directory:
```
python3 validation.py --noise_levels <from Step 2> --num_reduced_sets <from Step 2> --num_obs <from Step 2> --num_prime <from Step 2>
--noises <from Step 2> --acc_const_noise <from Step 2> --steer_const_noise <from Step 2>
```
This will store the statistics in the corresponding locations in the *stats* folder

***Step 4*** 

(A) Next we plot the box plots as well as visualize the trajectories. To plot the box plots run the following command:
```
python3 plot_box_plots.py --noise_levels <from Step 2> --num_reduced_sets <from Step 2>
--num_obs <from Step 2> --num_prime <from Step 2> --noises <from Step 2>
```
**Note:** 
1. The above command takes in a single integer value for *num_obs*. To plot the box plots for different number of obstacles you need to run the above command separately for each value of *num_obs*.
2. Currently, *plot_box_plots.py* generates box plots for both the costs, i.e, *mmd_opt, cvar*. In order to plot the box plots for a subset of these costs you need to uncomment/edit the lines corresponding to the costs that you do not want to plot the box plots for.

(B) To generate videos for the trajectories run the following command:
```
python3 plot_traj_video.py --num_obs <int> --num_reduced_sets <from Step 2>
--noise_levels <from Step 2> --num_prime <from Step 2> --noises <from Step 2>
--acc_const_noise <from Step 2> --steer_const_noise <from Step 2>
```

## Synthetic Dynamic Environment
Once inside the *synthetic_dynamic_obs* directory follow the below steps:

***Step 0*** Change the following line to include the path to the optimizer module in ```main_mpc.py, validation.py, plot_traj_video.py```:
```
sys.path.insert(1, 'path/to/optimizer')
```

***Step 1*** Run the following command from the root directory:
```
python3 main_mpc.py --noise_levels <list_of_noise_levels(float)> --num_reduced_sets <list_of_reduced_set_sizes>
 --num_obs <int> --num_prime <int> --noises <beta,gaussian or both> --acc_const_noise <float> --steer_const_noise <float> --costs <list_of_costs> 
```
where *costs* can be one or all of <**mmd_opt**, **cvar**> and *noises* can be one or all of <**gaussian**, **beta**>. The above command will run for default 200 obstacle configurations. This number can be changed by modifying *num_configs* variable in ```main_mpc.py```

***Step 2*** Once *Step 2* is complete, there will be data files in the corresponding folders in *data-> <gaussian_noise/beta_noise> -> noise_level/ ts_<num_prime>*. Now we need to calculate the statistics for the collected data. Run the following command from the root directory:
```
python3 validation.py --noise_levels <from Step 2> --num_reduced_sets <from Step 2> --num_obs <from Step 2> --num_prime <from Step 2>
--noises <from Step 2> --acc_const_noise <from Step 2> --steer_const_noise <from Step 2>
```
This will store the statistics in the corresponding locations in the *stats* folder

***Step 3*** 

(A) Next we plot the box plots as well as visualize the trajectories. To plot the box plots run the following command:
```
python3 plot_box_plots.py --noise_levels <from Step 2> --num_reduced_sets <from Step 2>
--num_obs <from Step 2> --num_prime <from Step 2> --noises <from Step 2>
```
**Note:** 
1. The above command takes in a single integer value for *num_obs*. To plot the box plots for different number of obstacles you need to run the above command separately for each value of *num_obs*.
2. Currently, *plot_box_plots.py* generates box plots for both the costs, i.e, *mmd_opt, cvar*. In order to plot the box plots for a subset of these costs you need to uncomment/edit the lines corresponding to the costs that you do not want to plot the box plots for.

(B) To generate videos for the trajectories run the following command:
```
python3 plot_traj_video.py --num_obs <int> --num_reduced_sets <from Step 2>
--noise_levels <from Step 2> --num_prime <from Step 2> --noises <from Step 2>
--acc_const_noise <from Step 2> --steer_const_noise <from Step 2>
```
