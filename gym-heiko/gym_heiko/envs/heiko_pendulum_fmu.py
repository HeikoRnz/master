import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
# FMU related imports
from etaFMU import etaFMU_initialize, etaFMU_step, etaFMU_reset, etaFMU_close


# from fmpy import read_model_description, extract
# from fmpy.fmi2 import FMU2Slave
# import shutil

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
                          stop_time=10.0,
                          step_size=5e-2,
                          inputs=['u'],
                          outputs=['th', 'thdot'],
                          show_plot=False)  # !! etaFMU

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

        newth = self.output_th + np.pi  # plus pi, because it was 180° different in the pendulum example by openAI
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
            rod.set_color(160/255, 0, 0)
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
