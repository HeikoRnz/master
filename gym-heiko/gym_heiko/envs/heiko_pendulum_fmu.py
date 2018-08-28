import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
# FMU related imports
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
from fmpy.util import plot_result
import shutil

#####################################################################################################
# This environment is an alternated version from the "pendulum-v0" environment by openAI
# Here, with the use of the library FMpy, the differential equations are drawn from an FMU file
# that was created using OpenModelica

#####################################################################################################

class HeikoPendulumFMUEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):

        # FMpy implementation
        # define the model name and simulation parameters
        self.show_plot = False
        # has to be the complete path on the local machine for now:
        self.fmu_filename = 'C:/Users/HLaptop/Daten/Studium/Master/6. Semester/Masterthesis/Python/Testen/gym-heiko/gym_heiko/envs/assets/Pendel_interaktiv.fmu'
        self.start_time = 0.0
        self.stop_time = 20.0
        self.step_size = 5e-2
        # read the model description
        self.model_description = read_model_description(self.fmu_filename)
        # collect the value references
        self.vrs = {}
        for variable in self.model_description.modelVariables:
            self.vrs[variable.name] = variable.valueReference
        # get the value references for the variables we want to get/set
        self.vr_inputs = self.vrs['u']  # input torque
        self.vr_output_th = self.vrs['th']  # angular displacement
        self.vr_output_thdot = self.vrs['thdot']  # angular speed
        # extract the FMU
        self.unzipdir = extract(self.fmu_filename)
        self.fmu = FMU2Slave(guid=self.model_description.guid,
                             unzipDirectory=self.unzipdir,
                             modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                             instanceName='instance1')
        # initialize
        self.fmu.instantiate()
        self.fmu.setupExperiment(startTime=self.start_time)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()
        self.time = self.start_time
        self.rows = []  # list to record the results

        # Standard pendukum-v0 stuff
        self.max_speed=8
        self.max_torque=4.
        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):

        # FMU step
        self.fmu.setReal([self.vr_inputs], [u])  # setReal is used to choose the inputs for the FMU
        # perform one step
        self.fmu.doStep(currentCommunicationPoint=self.time, communicationStepSize=self.step_size)  # doStep performs a step of certain size
        # get the values for 'inputs' and the two outputs
        self.inputs, self.output_th, self.output_thdot = self.fmu.getReal([self.vr_inputs, self.vr_output_th, self.vr_output_thdot])
        # append the results
        self.rows.append((self.time, self.inputs, self.output_th, self.output_thdot))
        # advance the time
        self.time += self.step_size

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        newth = self.output_th + np.pi  # plus pi, because it was 180° different in the pendulum example by openAI
        newthdot = self.output_thdot
        # Cost function
        costs = angle_normalize(newth+np.pi)**2 + .1*newthdot**2 + .001*(u**2)  # Cost function

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, self.time

    def reset(self):

        # # FMU reset
        self.fmu.terminate()
        self.fmu.freeInstance()
        # # clean up
        shutil.rmtree(self.unzipdir)
        # convert the results to a structured NumPy array
        result = np.array(self.rows, dtype=np.dtype([('time', np.float64), ('inputs', np.float64), ('th', np.float64), ('thdot', np.float64)]))
        # plot the results
        if self.show_plot:
            plot_result(result)
        # initialize has to be done again as it seems to reset the FMU coSimulation
        # there will be a beter way I guess, but this works for now
        self.unzipdir = extract(self.fmu_filename)

        self.fmu = FMU2Slave(guid=self.model_description.guid,
                             unzipDirectory=self.unzipdir,
                             modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                             instanceName='instance1')

        self.fmu.instantiate()
        self.fmu.setupExperiment(startTime=self.start_time)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()
        self.time = self.start_time
        self.rows = []  # list to record the results

        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(0.2, 0.2, 0.2)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()

def angle_normalize(x):
    return (((x) % (2*np.pi)) - np.pi)
