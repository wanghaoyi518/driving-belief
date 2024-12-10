import lane
import car
import math
import feature
import dynamics
import visualize
import utils
import sys
import theano as th
import theano.tensor as tt
import numpy as np
import pickle
import matplotlib.pyplot as plt  # Add at top with other imports
from car import Car
import os  # Add at top with other imports
import time  # Add at top with other imports

class Object(object):
    def __init__(self, name, x):
        self.name = name
        self.x = np.asarray(x)

class World(object):
    def __init__(self):
        self.cars = []
        self.lanes = []
        self.roads = []
        self.fences = []
        self.objects = []
        self.position_history = {0: [], 1: []}  # Initialize empty lists for human and robot positions
        print("World initialized with empty position history")

    def plot_trajectories(self):
        """Plot the trajectories of human and robot cars"""
        # Create trajectory_plots directory if it doesn't exist
        plot_dir = "trajectory_plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            
        plt.figure(figsize=(8, 8))
        
        # Plot human trajectory in red
        human_pos = np.array(self.position_history[0])
        if len(human_pos) > 0:
            plt.plot(human_pos[:, 0], human_pos[:, 1], 'r-', label='Human', linewidth=2)
            
        # Plot robot trajectory in yellow
        robot_pos = np.array(self.position_history[1]) 
        if len(robot_pos) > 0:
            plt.plot(robot_pos[:, 0], robot_pos[:, 1], 'y-', label='Robot', linewidth=2)
            
        # Plot lanes
        for lane in self.lanes:
            plt.plot([lane.p[0], lane.q[0]], [lane.p[1], lane.q[1]], 'k--', alpha=0.5)
            
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Vehicle Trajectories')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # Save plot with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"trajectory_plots/trajectory_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

    def features(self, theta, exclude_cars=[], traj='linear'):
        if isinstance(exclude_cars, Car):
            exclude_cars = [exclude_cars]
        r  = theta[0]*sum(lane.gaussian() for lane in self.lanes)
        r += theta[1]*sum(fence.gaussian() for fence in self.fences)
        r += theta[2]*sum(road.gaussian(10) for road in self.roads)
        r += theta[3]*feature.control([(-1., 1.), (-1., 1.)])
        r += theta[4]*feature.speed(1.)
        r += theta[5]*sum(getattr(car, traj).gaussian() for car in self.cars if car not in exclude_cars)
        return r

theta_normal     = [1., -50., 10., 100., 10., -50.]
theta_no_speed   = [1., -50., 10., 100., 0. , -50.]
theta_aggressive = [1., -50., 10., 100., 30., -50.]
theta_timid      = [1., -50., 10., 100., 5. , -50.]
theta_distracted = [1., -50., 10., 100., 10., -20.]
theta_attentive  = [1., -50., 10., 100., 10., -70.]
theta_distracted1= [1., -50., 10., 100., 1., 0.]
theta_distracted2= [1., -50., 10., 100., 10., -10.]
theta_intrusive  = [1., -20., 10., 100., 10., -30.]  # Allows more lane deviation
theta_passive    = [1., -70., 10., 100., 10., -50.]  # Strongly prefers lane center

theta0 = [1., -50., 10., 100., 10., -30.]
theta1 = [1., -50., 10., 100., 10., -70.]

def highway():
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes = [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads = [clane]
    world.fences = [clane.shifted(2), clane.shifted(-2)]
    return world

def world0(active=True, theta_explore=100., theta_exploit=1.):
    T = 5
    dyn = dynamics.CarDynamics(0.1)
    world = highway()
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2., 1.], color='red', T=T))
    world.cars.append(car.BeliefOptimizerCar(dyn, [0.0, 0.2, math.pi/2., .8], color='yellow', T=T))
    
    # Set world reference in cars
    for c in world.cars:
        c.world = world
        
    world.cars[1].human = world.cars[0]
    world.cars[0].reward = world.features(theta_attentive, world.cars[0], 'linear')
    world.cars[1].add_model(lambda traj: world.features(theta_attentive, world.cars[0], traj))
    world.cars[1].add_model(lambda traj: world.features(theta_distracted, world.cars[0], traj), log_p=3.)
    @feature.feature
    def left_lane(t, x, u):
        return -(x[0]+0.13)**2
    obj0 = world.cars[1].traj.total(world.features(theta_normal, world.cars[1], 'linear')+theta_exploit*left_lane)
    if active:
        world.cars[1].objective = lambda traj_h: theta_explore*world.cars[1].entropy(traj_h)+obj0
    else:
        world.cars[1].objective = obj0
    return world

def make_world0(version, explore=100., exploit=10.):
    def active():
        return world0(True, explore, exploit)
    def passive():
        return world0(False, explore, exploit)
    globals()['world0_{}_active'.format(version)]=active
    globals()['world0_{}_passive'.format(version)]=passive

make_world0(1, 100., 100.)

def world_test():
    dyn = dynamics.CarDynamics(0.1)
    world = highway()
    world.cars.append(car.UserControlledCar(dyn, [-0.13, 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.0, 0.5, math.pi/2., 0.8], color='yellow'))
    world.cars[1].reward = world.features(theta0, world.cars[1])
    return world

def perturb():
    T = 5
    dyn = dynamics.CarDynamics(0.1)
    world = highway()
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2., 1.], color='red', T=T))
    world.cars.append(car.BeliefOptimizerCar(dyn, [0.0, 0.1, math.pi/2., .8], color='yellow', T=T))
    world.cars[1].human = world.cars[0]
    world.cars[0].reward = world.features(theta_distracted1, world.cars[0], 'linear')
    world.cars[1].add_model(lambda traj: world.features(theta_attentive, world.cars[0], traj))
    world.cars[1].add_model(lambda traj: world.features(theta_distracted, world.cars[0], traj))
    obj0 = world.cars[1].traj.total(world.features(theta_normal, world.cars[1], 'linear'))
    world.cars[1].objective = lambda traj_h: 100.*world.cars[1].entropy(traj_h)+obj0
    return world

def world1(active=True, model='human'):
    T = 5
    dyn = dynamics.CarDynamics(0.1)
    world = highway()
    
    # Initialize position history
    world.position_history = {0: [], 1: []}
    
    # Create cars with different lateral positions
    if model=='human':
        world.cars.append(car.UserControlledCar(dyn, [-0.13, 0., math.pi/2., 1.], color='red', T=T))
    else:
        # Create SimpleOptimizerCar for non-human model
        human_car = car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2., 1.], color='red', T=T)
        human_car.is_probing = False  # Human car doesn't probepy
        world.cars.append(human_car)

    # Robot car starts in adjacent lane
    robot_car = car.BeliefOptimizerCar(dyn, [0.13, 0.1, math.pi/2., .8], color='yellow', T=T)
    robot_car.is_probing = active  # Only probe in active mode
    world.cars.append(robot_car)
    
    # Set world reference in cars
    for c in world.cars:
        c.world = world
        
    world.cars[1].human = world.cars[0]
    
    # Set rewards based on driver type and active/passive behavior
    if model=='attentive':
        # Attentive driver strongly avoids collisions
        world.cars[0].reward = world.features(theta_attentive, world.cars[0], 'linear')
    elif model=='distracted':
        # Distracted driver barely responds to other cars
        world.cars[0].reward = world.features(theta_distracted1, world.cars[0], 'linear')
    
    # Define robot behavior
    if active:
        # Active probing - allow lane intrusion
        obj0 = world.cars[1].traj.total(world.features(theta_intrusive, world.cars[1], 'linear'))
        
        @feature.feature
        def probe_objective(t, x, u):
            # Encourage periodic lane intrusion
            return -5.0 * ((x[0] - (-0.13))**2)  # Pull towards human's lane
        
        obj0 += probe_objective
    else:
        # Passive behavior - stay in lane
        obj0 = world.cars[1].traj.total(world.features(theta_passive, world.cars[1], 'linear'))
    
    # Set final objective
    if model=='human':
        if active:
            world.cars[1].objective = lambda traj_h: 100.*world.cars[1].entropy(traj_h)+obj0
        else:
            world.cars[1].objective = obj0
    else:
        with open('data/world1-trajectories.pickle', 'rb') as f:
            ur = pickle.load(f)[0][1]
        world.cars[1].objective = ur
        world.cars[1].dumb = True
    
    return world

def world1_active():
    return world1(True)
def world1_passive():
    return world1(False)
def world1_active_attentive():
    return world1(True, 'attentive')
def world1_active_distracted():
    return world1(True, 'distracted')
def world1_passive_attentive():
    return world1(False, 'attentive')
def world1_passive_distracted():
    return world1(False, 'distracted')

def world2(active=True, model='human'):
    T = 5
    dyn = dynamics.CarDynamics(0.1)
    world = highway()
    if model=='human':
        world.cars.append(car.UserControlledCar(dyn, [-0.13, 0., math.pi/2., 1.], color='red', T=T))
    else:
        world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2., 1.], color='red', T=T))
    world.cars.append(car.BeliefOptimizerCar(dyn, [-0.13, 0.3, math.pi/2., 1.], color='yellow', T=T))
    world.cars[1].human = world.cars[0]
    if model=='attentive':
        world.cars[0].reward = world.features(theta_attentive, world.cars[0], 'linear')
    elif model=='distracted':
        world.cars[0].reward = world.features(theta_distracted1, world.cars[0], 'linear')
    world.cars[1].add_model(lambda traj: world.features(theta_attentive, world.cars[0], traj))
    world.cars[1].add_model(lambda traj: world.features(theta_distracted, world.cars[0], traj))
    obj0 = world.cars[1].traj.total(world.features(theta_normal, world.cars[1], 'linear'))
    if model=='human':
        if active:
            world.cars[1].objective = lambda traj_h: 100.*world.cars[1].entropy(traj_h)+obj0
        else:
            world.cars[1].objective = obj0
    else:
        with open('robot/S2.{}'.format('active' if active else 'passive')) as f:
            ur = pickle.load(f)[0][1]
        world.cars[1].objective = ur
        world.cars[1].dumb = True
    return world

def world2_active():
    return world2(True)
def world2_passive():
    return world2(False)
def world2_active_attentive():
    return world2(True, 'attentive')
def world2_active_distracted():
    return world2(True, 'distracted')
def world2_passive_attentive():
    return world2(False, 'attentive')
def world2_passive_distracted():
    return world2(False, 'distracted')


def world3(active=True, model='human'):
    T = 5
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    vlane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    hlane = lane.StraightLane([-1., 0.], [1., 0.], 0.13)
    world.lanes = [vlane, hlane]
    world_r =  World()
    world_r.lanes = [hlane]
    world_r.roads = [hlane]
    world_r.fences = [hlane.shifted(-2), hlane.shifted(2)]
    world_h = World()
    world_h.lanes = [vlane]
    world_h.roads = [vlane]
    world_h.fences = [vlane.shifted(-2), vlane.shifted(2)]
    if model=='human':
        world.cars.append(car.UserControlledCar(dyn, [0., -0.4, math.pi/2., .5], color='red', T=T))
    else:
        world.cars.append(car.SimpleOptimizerCar(dyn, [0., -0.4, math.pi/2., .5], color='red', T=T))
    world.cars.append(car.BeliefOptimizerCar(dyn, [-0.15, 0., 0., 0.05], color='yellow', T=T))
    world_h.cars = world.cars
    world_r.cars = world.cars
    world.cars[1].human = world.cars[0]
    if model=='attentive':
        world.cars[0].reward = world_h.features(theta_attentive, world.cars[0], 'linear')
    elif model=='distracted':
        world.cars[0].reward = world_h.features(theta_distracted2, world.cars[0], 'linear')
    world.cars[1].add_model(lambda traj: world_h.features(theta_attentive, world.cars[0], traj))
    world.cars[1].add_model(lambda traj: world_h.features(theta_distracted, world.cars[0], traj))
    obj0 = world.cars[1].traj.total(world_r.features(theta_no_speed, world.cars[1], 'linear'))
    if model=='human':
        if active:
            world.cars[1].objective = lambda traj_h: 100.*world.cars[1].entropy(traj_h)+obj0
        else:
            world.cars[1].objective = obj0
    else:
        with open('robot/S3.{}'.format('active' if active else 'passive')) as f:
            ur = pickle.load(f)[0][1]
        world.cars[1].objective = ur
        world.cars[1].dumb = True
    return world

def world3_active():
    return world3(True)
def world3_passive():
    return world3(False)
def world3_active_attentive():
    return world3(True, 'attentive')
def world3_active_distracted():
    return world3(True, 'distracted')
def world3_passive_attentive():
    return world3(False, 'attentive')
def world3_passive_distracted():
    return world3(False, 'distracted')
