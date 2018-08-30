import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
# FMU related imports
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
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

        etaFMU_initialize(self,
                          path_to_FMU='C:/Users/HLaptop/Daten/Studium/Master/6. Semester/Masterthesis/Python/Testen/gym-heiko/gym_heiko/envs/assets/Pendel_interaktiv.fmu',
                          start_time=0.0,
                          stop_time=20.0,
                          step_size=5e-2,
                          inputs=['u'],
                          outputs=['th', 'thdot'],
                          show_plot=False)  # !! etaFMU

        # Standard pendukum-v0 stuff
        self.max_speed = 8
        self.max_torque = 4.
        self.viewer = None
        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering

        self.inputs, self.output_th, self.output_thdot = etaFMU_step(self, [u])  # !! etaFMU

        newth = self.output_th + np.pi # plus pi, because it was 180° different in the pendulum example by openAI
        newthdot = self.output_thdot
        # Cost function
        costs = angle_normalize(newth + np.pi)**2 + .1*newthdot**2 + .01*(u**2)  # Cost function

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, self.time

    def reset(self):

        etaFMU_reset(self)  # !! etaFMU

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

        etaFMU_close(self)  # !! etaFMU

        if self.viewer:
            self.viewer.close()


def angle_normalize(x):
    return (((x) % (2*np.pi)) - np.pi)


# functions for the etaFMU library


def etaFMU_initialize(self, path_to_FMU, start_time, stop_time, step_size, inputs, outputs, show_plot=False):
    # DESCRIPTION FOR THE FUNCTION
    #   inputs:
    #   path_to_FMU [string]: Absolute path to fmu file
    #   start_time [float]
    #   stop_time [float]
    #   step_size [float]
    #   inputs [list]: List of input names that correspond with the names used in the fmu file (e.g. ['u', 'p'])
    #   outputs [list]: List of output names that correspond with the names used in the fmu file (e.g. ['y', 'th', 'thdot'])
    #   show_plot [Boolean]: Toggles the visibility of a graph that shows all inputs and outputs in relation to time
    #
    #   returns:
    #   show_plot, start_time, stop_time, step_size, inputs (?), outputs (?), path_to_FMU, read_model_description
    #   input_n, output_n, all_n, fmu, time, rows - all stored in self

    # issues to solve:
    #   - find a better way to reset the environment
    #   - find a way to let the path_to_FMU be relative instead of absolute

    # read the model description and input/outputs
    self.show_plot = show_plot
    self.start_time = start_time
    self.stop_time = stop_time
    self.step_size = step_size
    self.inputs = inputs
    self.outputs = outputs
    self.path_to_FMU = path_to_FMU
    self.model_description = read_model_description(path_to_FMU)

    # collect the value references
    vrs = {}
    for variable in self.model_description.modelVariables:
        vrs[variable.name] = variable.valueReference

    # create a list of the reference values for the inputs and outputs
    self.input_n = []
    self.output_n = []
    self.all_n = []
    for i in range(len(self.inputs)):
        self.input_n.append(vrs[self.inputs[i]])
    for i in range(len(self.outputs)):
        self.output_n.append(vrs[self.outputs[i]])
    self.all_n = self.input_n + self.output_n

    # extract the FMU
    self.unzipdir = extract(path_to_FMU)
    self.fmu = FMU2Slave(guid=self.model_description.guid,
                         unzipDirectory=self.unzipdir,
                         modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                         instanceName='ETAFMU_Instance')

    # initialize
    self.fmu.instantiate()
    self.fmu.setupExperiment(startTime=start_time)
    self.fmu.enterInitializationMode()
    self.fmu.exitInitializationMode()
    self.time = start_time
    self.rows = []  # list to record the results

    return self


def etaFMU_step(self, inputvalues):
    # DESCRIPTION FOR THE FUNCTION
    #   inputs:
    #   inputvalues [list]: List of the actual values that should be pushed to the FMU.
    #                       The order has to correspond to the order of the list 'inputs' defined in the
    #                       etaFMU_initialize function (e.g. [x, g, f], when inputs was ['x', 'g', 'f'])
    #
    #   returns:
    #   in_and_outputvalues [list]: List of the in- and outputvalues from the FMU
    #                               The order corresponds to [inputs, outputs] defined in the etaFMU_initialize
    #                               function

    # set input values for the FMU
    self.fmu.setReal(self.input_n, inputvalues)
    # push input values to the FMU and do one timestep
    self.fmu.doStep(currentCommunicationPoint=self.time, communicationStepSize=self.step_size)  # doStep performs a step of certain size
    # get the values for all inputs and outputs in the order [inputs, outputs] defined in the etaFMU_initialize function
    in_and_outputvalues = self.fmu.getReal(self.all_n)
    # append the results
    self.rows.append((self.time, in_and_outputvalues))  # !originally it was a list like (self.time, bla, bla), now (self.time, [bla, bla])
    # advance the time
    self.time += self.step_size

    return in_and_outputvalues


def etaFMU_reset(self):
    # DESCRIPTION FOR THE FUNCTION
    #   resets the environment to the initial conditions
    self.fmu.reset()
    self.time = self.start_time
    self.rows = []

    return self


def etaFMU_close(self):
    self.fmu.terminate()
    self.fmu.freeInstance()
    # # clean up
    shutil.rmtree(self.unzipdir)
