import numpy as np

cost = "mmd_opt"
list_town = ["Town05","Town10HD"]

list_noise = ["beta"]
list_noise_level = ["5"]
num_prime = "40"
list_num_reduced = ["2"]
obs_type = "static"

# list_noise = ["gaussian"]
# list_noise_level = ["30"]
# num_prime = "40"
# list_num_reduced = ["2"]
# obs_type = "static"

len_num_red = len(list_num_reduced)

for town in list_town:
    for noise in list_noise:
        for noise_level in list_noise_level:
            print(town, noise,noise_level)

            mmd_opt_obs,cvar_obs,mmd_random_obs = [],[],[]
            mmd_opt_lane,cvar_lane,mmd_random_lane = [],[],[]
            mmd_opt_vel,cvar_vel,mmd_random_vel = [],[],[]
            mmd_opt_vel_max = []

            # filename = "./stats/{}/{}_noise/noise_{}/static_det.npz".format(town,noise,noise_level)

            # det_obs = round(np.load(filename)["coll"][0], 2)   
            # det_lane = round(np.load(filename)["lane"][0], 2)   
            # det_vel = round(np.load(filename)["vel"][0], 2)   
            # det_vel_max = round(np.load(filename)["vel_max"][0], 2)   

            for num_reduced in list_num_reduced:

                filename = "./stats/{}/{}_noise/noise_{}/ts_{}/{}_{}_{}_samples.npz".format(town,noise,noise_level,
                                                            num_prime,cost,obs_type,
                                                            num_reduced)

                data = np.load(filename)

                mmd_opt_obs.append(round(data["coll"][0],2))
                # cvar_obs.append(round(data["coll"][0],2))
                # mmd_random_obs.append(round(data["coll"][2],2))

                mmd_opt_lane.append(round(data["lane"][0],2))
                # cvar_lane.append(round(data["lane"][0],2))
                # mmd_random_lane.append(round(data["lane"][2],2))

                mmd_opt_vel.append(round(data["vel"][0],2))
                mmd_opt_vel_max.append(round(data["vel_max"][0],2))
                # mmd_random_vel.append(round(data["vel"][2],2))

            print("obs ", mmd_opt_obs)
            print("lane ", mmd_opt_lane)
            print("vel ", mmd_opt_vel)
            print("vel_max ", mmd_opt_vel_max)

            # print("CVaR obs ", cvar_obs)
            # print("CVaR lane ", cvar_lane)
            # print("CVaR vel ", cvar_vel)
            # print("------")

            # print("MMD-R obs ", mmd_random_obs)
            # print("MMD-R lane ", mmd_random_lane)
            # print("MMD-R vel ", mmd_random_vel)

            # print("Det obs ",det_obs )
            # print("Det lane ",det_lane )
            # print("Det vel ",det_vel )
            # print("Det vel max ",det_vel_max )

            print("-------------------")
