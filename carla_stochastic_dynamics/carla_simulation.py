#!/usr/bin/env python
import numpy as np
import sys
sys.path.insert(1, '/home/ims-robotics/CARLA_0.9.13/PythonAPI/carla')
import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
import random
from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions

import cv2
import pygame
from simple_pid import PID

try:
	import queue
except ImportError:
	import Queue as queue

class CarlaSimulation():
	def __init__(self,display, font, clock, n_obs, town,fps=20):
		self.traj_values = []
		self.actor_list = []
		self.obs_list = []
		self.sensors = []
		self.delta_seconds = 1/fps
		self._queues = []
		self.sensor_data = []
		self.behind_ego = [None]*n_obs
		self.path_pixels = []
		self.vis_path = True
		self.pre_x = []
		self.pre_y = []
		self.pre_psi = []
		self.pre_sel_psi = []
		self.steps = []
		self.sel_index = 0
		self.num_goal = 0
		self.throttle = 0
		self.prev_acc = 0
		self.vel = 0
		self.prev_vel = 0.0
		self.selected_pose = []
		self.prev_waypoint = None
		self.prev_pos = None

		self.display = display
		self.font = font
		self.clock = clock
		
		self.n_obs = n_obs
		self.other_vehicles = np.zeros([self.n_obs, 6])
		
		if town=="Town10HD":
			self.lane_y = [0.,3.5]
		else:
			self.lane_y = [0.,-3.5]

		# self.other_vehicles[:,1] = np.hstack((self.lane_y,self.lane_y,self.lane_y,self.lane_y,self.lane_y,self.lane_y,self.lane_y))
		# self.other_vehicles[:,0] = np.array([ 35,70,105,140,175,185,195,210,220,230,245,255,265,280 ])

		# self.other_vehicles[:,1] = np.hstack((self.lane_y,self.lane_y,self.lane_y,self.lane_y))
		# self.other_vehicles[:,0] = np.array([ 35,70,105,140,175,210,245,280 ])
		
		# self.other_vehicles[:,1] = np.hstack((self.lane_y,self.lane_y,self.lane_y,self.lane_y,self.lane_y,self.lane_y,self.lane_y))
		# self.other_vehicles[:,0] = np.array([ -15,-10,5,15,35,55,70,105,120,140,175,210,245,280 ])

		# self.other_vehicles[:,1] = np.hstack((self.lane_y,self.lane_y,self.lane_y,self.lane_y))
		# self.other_vehicles[:,0] = np.array([ 70,105,120,140,175,210,245,280 ])

	##################### Town 5
		# self.other_vehicles[:,1] = np.hstack((self.lane_y,self.lane_y,self.lane_y,self.lane_y,self.lane_y))
		# self.other_vehicles[:,0] = np.array([ 15,25,35,45,55,65,75,85,95,105])

		self.other_vehicles[:,1] = np.hstack((self.lane_y,self.lane_y,self.lane_y,
										self.lane_y,self.lane_y,self.lane_y,
										self.lane_y,self.lane_y))
		self.other_vehicles[:,0] = np.array([ 15,25,35,45,55,65,75,85,95,105,130,140,160,180,190,200])
		
		# self.other_vehicles[:,1] = np.hstack((self.lane_y,self.lane_y,self.lane_y,self.lane_y,self.lane_y,self.lane_y))
		# self.other_vehicles[:,0] = np.array([25,35,50,70,95,105,130,150,160,170,180,190])
		
		self.client = carla.Client('localhost', 2000)
		self.client.set_timeout(10.0)
		
		self.town = town
		self.world = self.client.load_world(town)
		# # Toggle all buildings off
		# self.world.unload_map_layer(carla.MapLayer.Buildings)

		# self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)

		self.m =self.world.get_map()
		self.tm = self.client.get_trafficmanager()
		self.tm_port = self.tm.get_port()
		self.grp = GlobalRoutePlanner(self.m, 0.25)

		self.world.set_weather(carla.WeatherParameters.CloudyNoon)

		self.image_x = 800
		self.image_y = 600
		
		self.birdview_producer = BirdViewProducer(
				self.client,  # carla.Client
				target_size=PixelDimensions(width=self.image_x, height=self.image_y),
				pixels_per_meter=4,
				crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
			)
		self.bv_img_x = 300
		self.bv_img_y = 800
		self.blueprint_library = self.world.get_blueprint_library()

		self.pid = PID(0.05, 0.0, 0.05)

		self.spawn_player()
		self.spawn_sensors()
		self.spawn_vehicles(self.n_obs)
		self.init()

	def init(self):
		self._settings = self.world.get_settings()
		self.frame = self.world.apply_settings(carla.WorldSettings(
			no_rendering_mode=False,
			synchronous_mode=True,
			fixed_delta_seconds=self.delta_seconds))

		def make_queue(register_event):
			q = queue.Queue()
			register_event(q.put)
			self._queues.append(q)

		make_queue(self.world.on_tick)
		for sensor in self.sensors:
			make_queue(sensor.listen)
	
	def _retrieve_data(self, sensor_queue, timeout):
		while True:
			data = sensor_queue.get(timeout=timeout)
			if data.frame == self.frame:
				return data

	def tick(self, timeout):
		self.frame = self.world.tick()
		data = [self._retrieve_data(q, timeout) for q in self._queues]
		assert all(x.frame == self.frame for x in data)
		self.sensor_data = data

	def spawn_sensors(self):
		camera_rgb = self.blueprint_library.find('sensor.camera.rgb')
		camera_rgb.set_attribute("image_size_x", str(self.image_x))
		camera_rgb.set_attribute("image_size_y", str(self.image_y))

		camera_rgb = self.world.spawn_actor(
			camera_rgb,
			# carla.Transform(carla.Location(x=-10.0, z=6.0), carla.Rotation(pitch=0.0)),
			carla.Transform(carla.Location(x=-8.0, z=4.0), carla.Rotation(pitch=0.0)),
			attach_to=self.vehicle)
	
		self.actor_list.append(camera_rgb)
		self.sensors.append(camera_rgb)

	def spawn_player(self, start_pose=None):
		if start_pose == None:

			start_pose = carla.Transform()
			if self.town=="Town10HD":
				start_pose.location.x = -47 
				start_pose.location.y = 137 
			else: # Town05
				# start_pose.location.x =  83.27630615234375 # town03
				# start_pose.location.y = -79.50776672363281

				start_pose.location.x =  -65 # town05
				start_pose.location.y = 84.5

			self.prev_pos = start_pose.location
			self.prev_waypoint = self.m.get_waypoint(start_pose.location, project_to_road=True)
			wp = self.m.get_waypoint(start_pose.location, project_to_road=True)
			start_pose = wp.transform
			
			start_pose.location.z = 0.25

		vehicle = self.world.spawn_actor(
			random.choice(self.blueprint_library.filter('vehicle.tesla.model3')),
			start_pose)

		self.actor_list.append(vehicle)
		vehicle.set_simulate_physics(True)
		vehicle.set_autopilot(False)

		self.vehicle = vehicle

	def spawn_vehicles(self, n):
		if(n>0):
			blueprint1 = self.blueprint_library.filter('vehicle.aud*')
			blueprint2 = self.blueprint_library.filter('vehicle.lin*')
			blueprint3 = self.blueprint_library.filter('vehicle.niss*')
			blueprint4 = self.blueprint_library.filter('vehicle.bmw*')
			blueprints = []
			for i in blueprint1:
				blueprints.append(i)
			for i in blueprint2:
				blueprints.append(i)
			for i in blueprint3:
				blueprints.append(i)
			for i in blueprint4:
				blueprints.append(i)

			blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
			for i in range(n):
				blueprint = np.random.choice(blueprints)
				blueprint.set_attribute('role_name', 'autopilot')
		
				if blueprint.has_attribute('color'):
					color = np.random.choice(blueprint.get_attribute('color').recommended_values)
					blueprint.set_attribute('color', color)

				if blueprint.has_attribute('driver_id'):
					driver_id = np.random.choice(blueprint.get_attribute('driver_id').recommended_values)
					blueprint.set_attribute('driver_id', driver_id)
				
				car = None
				while car is None:
					obs_pose = carla.Transform()
					ego_wp = self.m.get_waypoint(self.vehicle.get_location(), project_to_road=True)
					if self.other_vehicles[i][0]>0:
						obs_wp = ego_wp.next(self.other_vehicles[i][0])[0]
					else:
						obs_wp = ego_wp.previous(-self.other_vehicles[i][0])[0]

					if self.town=="Town10HD":
						if self.other_vehicles[i][1]>0:
							obs_wp = obs_wp.get_right_lane()
					else:
						if self.other_vehicles[i][1]<0:
							obs_wp = obs_wp.get_left_lane()
					
				
					wp = self.m.get_waypoint(obs_pose.location, project_to_road=True)
					
					obs_pose = obs_wp.transform
					obs_pose.location.z = 0.25
					
					car = self.world.try_spawn_actor(blueprint, obs_pose)

					self.tm.ignore_lights_percentage(car,100)
					self.tm.distance_to_leading_vehicle(car, 5.0)
					speed = [40, 60, 40]
					self.tm.vehicle_percentage_speed_difference(car,random.choice(speed))

				# car.set_autopilot(False)
				self.actor_list.append(car)
				self.obs_list.append(car)
				self.move_vehicles()

	def move_vehicles(self):
		vel_list = [10.0, 15.0, 20.0]
		for obs in self.obs_list:
			# vel = random.choice(vel_list)
			# obs.set_target_velocity(carla.Vector3D(0.0,0.0,0))
			self.tm.auto_lane_change(obs,True)
			obs.set_autopilot(True,self.tm_port)
			
	def draw_image(self, surface, image, blend=False):
		array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
		array = np.reshape(array, (image.height, image.width, 4))
		array = array[:, :, :3]
		array = array[:, :, ::-1]
		image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
		if blend:
			image_surface.set_alpha(100)
		surface.blit(image_surface, (0, 0))


	def visualize(self):
		self.draw_image(self.display, self.sensor_data[1])
		self.display.blit(
			self.font.render('Location = % 5d , % 5d ' %(self.vehicle.get_location().x, self.vehicle.get_location().y), True, (255, 255, 255)),
			(8, 10))
		v = self.vehicle.get_velocity()
		vel = ((v.x)**2 + (v.y)**2 + (v.z)**2)**0.5
		self.display.blit(
			self.font.render('Velocity = % 5d ' % vel, True, (255, 255, 255)),
			(8, 28))
		pygame.display.flip()
		self.pre_sel_psi = []
		birdview = self.birdview_producer.produce(
					agent_vehicle=self.vehicle  # carla.Actor (spawned vehicle)
		)
		# Use only if you want to visualize
		# produces np.ndarray of shape (height, width, 3)
		rgb = BirdViewProducer.as_rgb(birdview)
		if self.vis_path:
			prev = 0
			for i in range(self.num_goal):
				#path_pixels = np.array(self.pre_x[prev+2:prev+100],self.pre_x[prev+2:prev+100] )
				path_x = self.pre_x[prev+0:prev+self.steps[i]]
				path_y = self.pre_y[prev+0:prev+self.steps[i]]
				path_psi = self.pre_psi[prev+0:prev+self.steps[i]]
				#poses = self.centerline_to_global(self.pre_x[prev+2:prev+100], self.pre_y[prev+2:prev+100],
				#                 self.pre_psi[prev+2:prev+100])
				poses = self.centerline_to_global(path_x, path_y, path_psi, i)
				path_pixels = self.path_to_pixel(poses)
				#print(path_pixels)
				if i == self.sel_index:
					#ind = 0
					for angle in poses:
						#print(angle.rotation.yaw, end="  ")
						"""if angle.rotation.yaw<0 and angle.rotation.yaw>-360:
							angle.rotation.yaw = 360 + angle.rotation.yaw
						if angle.rotation.yaw<-360 and angle.rotation.yaw>-720:
							angle.rotation.yaw = 720 + angle.rotation.yaw
						if angle.rotation.yaw>360:
							angle.rotation.yaw = angle.rotation.yaw-360
						if ind>0:
							if self.pre_sel_psi[-1]<90.0 and angle.rotation.yaw>270.0:
								angle.rotation.yaw = angle.rotation.yaw - 360
							elif self.pre_sel_psi[-1]>270.0 and angle.rotation.yaw<90.0:
								angle.rotation.yaw = 360 + angle.rotation.yaw"""
						self.pre_sel_psi.append(angle.rotation.yaw)
						#print(angle.rotation.yaw, end=" ")
						#ind = ind + 1
					# print(" ")
					self.pre_sel_psi = np.unwrap(self.pre_sel_psi)
					self.selected_pose = poses
					#print(self.pre_sel_psi)
					cv2.polylines(rgb,np.int32([path_pixels]),False,(0,255,0), 2)
				else:
					cv2.polylines(rgb,np.int32([path_pixels]),False,(0,0,255), 1)
				prev += self.steps[i]

		# cv2.imshow('bv', rgb)
		cv2.waitKey(10)

	def path_to_pixel(self, path_list):
		path_pixels = []
		#ego_wp = self.m.get_waypoint(self.vehicle.get_location(), project_to_road=True)
		vtf = self.vehicle.get_transform()#ego_wp.transform
		#vtf.location.x = self.vehicle.get_transform().location.x
		#vtf.location.y = self.vehicle.get_transform().location.y
		origin_rad = (((self.bv_img_x/2)**2 + (self.bv_img_y/2)**2)**0.5)/4
		origin_theta = np.arctan2(self.bv_img_x/2, self.bv_img_y/2)
		origin_x = origin_rad*np.cos(vtf.rotation.yaw*2*np.pi/360-origin_theta) + vtf.location.x
		origin_y = origin_rad*np.sin(vtf.rotation.yaw*2*np.pi/360-origin_theta) + vtf.location.y
		for i in path_list:
			pixel_rad = (((origin_x-i.location.x)**2 + (origin_y-i.location.y)**2)**0.5)*4
			pixel_theta = np.arctan2(origin_y-i.location.y, origin_x-i.location.x)
			if pixel_theta<0:
				pixel_theta = 2*np.pi + pixel_theta
			pixel_theta = pixel_theta*360/(2*np.pi)
			rgb_theta = vtf.rotation.yaw-pixel_theta
			rgb_x = pixel_rad*np.sin(rgb_theta*2*np.pi/360)
			rgb_y = pixel_rad*np.cos(rgb_theta*2*np.pi/360)
			path_pixels.append([rgb_x, rgb_y])
		return path_pixels

	def centerline_to_global(self, path_x, path_y, path_psi):
		#ego_wp = self.m.get_waypoint(self.vehicle.get_location(), project_to_road=True)
		#while abs(ego_wp.lane_id) > 1:
		#    ego_wp = ego_wp.get_left_lane()
		ego_wp = self.prev_waypoint
		path_pose_list = []
		prev_x = 0
		wp = ego_wp
		for i in range(len(path_x)):
			if path_x[i]-prev_x>0.0:
				wp = wp.next(path_x[i]-prev_x)[0]
			wp_tf = wp.transform
			#print(wp_tf, wp.lane_id)
			prev_x = path_x[i]
			y_wp = 1.75 - path_y[i]
			#wp_tf.rotation.yaw = abs(wp_tf.rotation.yaw)
			theta_wp = (wp_tf.rotation.yaw+90)*2*np.pi/360
			wp_tf.location.x = wp_tf.location.x + y_wp*np.cos(theta_wp)
			wp_tf.location.y = wp_tf.location.y + y_wp*np.sin(theta_wp)
			loc = wp_tf.location
			#n_wp = self.m.get_waypoint(loc, project_to_road=False)
			#if n_wp != None:
			#    wp_tf.rotation.yaw = n_wp.transform.rotation.yaw - path_psi[i]*360/(2*np.pi)    
			#else:
			wp_tf.rotation.yaw = wp_tf.rotation.yaw - path_psi[i]*360/(2*np.pi)
			#print(path_x[i], path_y[i], path_psi[i], wp_tf)
			path_pose_list.append(wp_tf)
		return path_pose_list

	def should_quit(self):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				return True
			elif event.type == pygame.KEYUP:
				if event.key == pygame.K_ESCAPE:
					return True
		return False

	def destroy(self):
		for actor in self.actor_list:
			actor.destroy()

def get_font():
	fonts = [x for x in pygame.font.get_fonts()]
	default_font = 'ubuntumono'
	font = default_font if default_font in fonts else fonts[0]
	font = pygame.font.match_font(font)
	return pygame.font.Font(font, 14)
