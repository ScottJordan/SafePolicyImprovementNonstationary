from simglucose.patient.t1dpatient import Action
from simglucose.analysis.risk import risk_index
import pandas as pd
from datetime import timedelta
import logging
from collections import namedtuple
from simglucose.simulation.rendering import Viewer
import numpy as np
import numbers
from copy import deepcopy

try:
    from rllab.envs.base import Step
except ImportError:
    _Step = namedtuple("Step", ["observation", "reward", "done", "info"])

    def Step(observation, reward, done, **kwargs):
        """
        Convenience method creating a namedtuple with the results of the
        environment.step method.
        Put extra diagnostic info in the kwargs
        """
        return _Step(observation, reward, done, kwargs)


Observation = namedtuple('Observation', ['CGM'])
logger = logging.getLogger(__name__)


def risk_diff(BG_last_hour):
    if len(BG_last_hour) < 2:
        return 0
    else:
        _, _, risk_current = risk_index([BG_last_hour[-1]], 1)
        _, _, risk_prev = risk_index([BG_last_hour[-2]], 1)
        return risk_prev - risk_current


def neg_risk(BG_last_hour):
    if len(BG_last_hour) < 2:
        return 0
    else:
        _, _, risk_current = risk_index([BG_last_hour[-1]], 1)
        return - risk_current


class T1DSimEnv(object):
    def __init__(self, patient, sensor, pump, scenario):
        self.patient = patient
        self.sensor = sensor
        self.pump = pump
        self.scenario = scenario
        self._reset()

    @property
    def time(self):
        return self.scenario.start_time + timedelta(minutes=self.patient.t)

    def mini_step(self, action):
        # Note: Every mini_step runs 1 min in the patient's life

        # current action
        patient_action = self.scenario.get_action(self.time)        # Get food for current minute
        basal = self.pump.basal(action.basal)
        bolus = self.pump.bolus(action.bolus)
        insulin = basal + bolus
        CHO = patient_action.meal
        patient_mdl_act = Action(insulin=insulin, CHO=CHO)

        # State update
        self.patient.step(patient_mdl_act)

        # next observation
        BG = self.patient.observation.Gsub
        CGM = self.sensor.measure(self.patient)

        return CHO, insulin, BG, CGM

    def step(self, action, reward_fun=neg_risk):
        '''
        action is a namedtuple with keys: basal, bolus
        '''
        CHO = 0.0
        insulin = 0.0
        BG = 0.0
        CGM = 0.0

        # This loop runs for n minutes in patients life
        # n is determined by the sample time, which is in turn defined by
        # the quality of sensor and the interval between each measurement
        for _ in range(int(self.sample_time)):
            # Compute moving average as the sample measurements
            tmp_CHO, tmp_insulin, tmp_BG, tmp_CGM = self.mini_step(action)
            CHO += tmp_CHO / self.sample_time
            insulin += tmp_insulin / self.sample_time
            BG += tmp_BG / self.sample_time
            CGM += tmp_CGM / self.sample_time

        # Compute risk index
        horizon = 1
        LBGI, HBGI, risk = risk_index([BG], horizon)

        # Record current action
        self.CHO_hist.append(CHO)
        self.insulin_hist.append(insulin)

        # Record next observation
        self.time_hist.append(self.time)
        self.BG_hist.append(BG)
        self.CGM_hist.append(CGM)
        self.risk_hist.append(risk)
        self.LBGI_hist.append(LBGI)
        self.HBGI_hist.append(HBGI)

        # Compute reward, and decide whether game is over
        window_size = int(60 / self.sample_time)
        BG_last_hour = self.CGM_hist[-window_size:]
        reward = reward_fun(BG_last_hour)

        # Stopping criteria
        # done = BG < 70 or BG > 350    # If the BG level go out of control
        done = self.patient.t >= 1440   # At the end of the day (24hrs = 1440 mins)

        obs = Observation(CGM=CGM)

        return Step(
            observation=obs,
            reward=reward,
            done=done,
            sample_time=self.sample_time,
            patient_name=self.patient.name,
            meal=CHO,
            patient_state=self.patient.state)

    def _reset(self):
        self.sample_time = self.sensor.sample_time
        self.viewer = None

        BG = self.patient.observation.Gsub
        horizon = 1
        LBGI, HBGI, risk = risk_index([BG], horizon)
        CGM = self.sensor.measure(self.patient)
        self.time_hist = [self.scenario.start_time]
        self.BG_hist = [BG]
        self.CGM_hist = [CGM]
        self.risk_hist = [risk]
        self.LBGI_hist = [LBGI]
        self.HBGI_hist = [HBGI]
        self.CHO_hist = []
        self.insulin_hist = []

    def reset(self):
        self.patient.reset()
        self.sensor.reset()
        self.pump.reset()
        self.scenario.reset()
        self._reset()
        CGM = self.sensor.measure(self.patient)
        obs = Observation(CGM=CGM)
        return Step(
            observation=obs,
            reward=0,
            done=False,
            sample_time=self.sample_time,
            patient_name=self.patient.name,
            meal=0,
            patient_state=self.patient.state)

    def render(self, close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = Viewer(self.scenario.start_time, self.patient.name)

        self.viewer.render(self.show_history())

    def show_history(self):
        df = pd.DataFrame()
        df['Time'] = pd.Series(self.time_hist)
        df['BG'] = pd.Series(self.BG_hist)
        df['CGM'] = pd.Series(self.CGM_hist)
        df['CHO'] = pd.Series(self.CHO_hist)
        df['insulin'] = pd.Series(self.insulin_hist)
        df['LBGI'] = pd.Series(self.LBGI_hist)
        df['HBGI'] = pd.Series(self.HBGI_hist)
        df['Risk'] = pd.Series(self.risk_hist)
        df = df.set_index('Time')
        return df


class NS_T1DSimEnv(object):
    def __init__(self, patient_a, patient_b, sensor, pump, scenario, speed, oracle):
        self.patient_a = patient_a
        self.patient_b = patient_b

        self.patient = deepcopy(self.patient_a)   # Initialize with patient A's param values
        self.sensor = sensor
        self.pump = pump
        self.scenario = scenario

        self.episode = 0
        self.speed = speed
        self.oracle = oracle
        self.frequency = speed * 0.003

        self._reset()

    @property
    def time(self):
        return self.scenario.start_time + timedelta(minutes=self.patient.t)

    def mini_step(self, action):
        # Note: Every mini_step runs 1 min in the patient's life

        # current action
        patient_action = self.scenario.get_action(self.time)        # Get food for current minute
        basal = self.pump.basal(action.basal)
        bolus = self.pump.bolus(action.bolus)
        insulin = basal + bolus
        CHO = patient_action.meal
        patient_mdl_act = Action(insulin=insulin, CHO=CHO)

        # State update
        self.patient.step(patient_mdl_act)

        # next observation
        BG = self.patient.observation.Gsub
        CGM = self.sensor.measure(self.patient)

        return CHO, insulin, BG, CGM

    def step(self, action, reward_fun=neg_risk):
        '''
        action is a namedtuple with keys: basal, bolus
        '''
        CHO = 0.0
        insulin = 0.0
        BG = 0.0
        CGM = 0.0

        # This loop runs for n minutes in patients life
        # n is determined by the sample time, which is in turn defined by
        # the quality of sensor and the interval between each measurement
        for _ in range(int(self.sample_time)):
            # Compute moving average as the sample measurements
            tmp_CHO, tmp_insulin, tmp_BG, tmp_CGM = self.mini_step(action)
            CHO += tmp_CHO / self.sample_time
            insulin += tmp_insulin / self.sample_time
            BG += tmp_BG / self.sample_time
            CGM += tmp_CGM / self.sample_time

        # Compute risk index
        horizon = 1
        LBGI, HBGI, risk = risk_index([BG], horizon)

        # Record current action
        self.CHO_hist.append(CHO)
        self.insulin_hist.append(insulin)

        # Record next observation
        self.time_hist.append(self.time)
        self.BG_hist.append(BG)
        self.CGM_hist.append(CGM)
        self.risk_hist.append(risk)
        self.LBGI_hist.append(LBGI)
        self.HBGI_hist.append(HBGI)

        # Compute reward, and decide whether game is over
        window_size = int(60 / self.sample_time)
        BG_last_hour = self.CGM_hist[-window_size:]
        reward = reward_fun(BG_last_hour)

        # Stopping criteria
        # done = BG < 70 or BG > 350    # If the BG level go out of control
        done = self.patient.t >= 1440   # At the end of the day (24hrs = 1440 mins)

        obs = Observation(CGM=CGM)

        return Step(
            observation=obs,
            reward=reward,
            done=done,
            sample_time=self.sample_time,
            patient_name=self.patient.name,
            meal=CHO,
            patient_state=self.patient.state)

    def _reset(self):
        self.sample_time = self.sensor.sample_time
        self.viewer = None

        BG = self.patient.observation.Gsub
        horizon = 1
        LBGI, HBGI, risk = risk_index([BG], horizon)
        CGM = self.sensor.measure(self.patient)
        self.time_hist = [self.scenario.start_time]
        self.BG_hist = [BG]
        self.CGM_hist = [CGM]
        self.risk_hist = [risk]
        self.LBGI_hist = [LBGI]
        self.HBGI_hist = [HBGI]
        self.CHO_hist = []
        self.insulin_hist = []

    def reset(self):
        self.episode += 1

        if self.oracle >= 0:
            # Ensure alpha is between 0 and 1
            alpha = np.cos(self.oracle * self.frequency) * 0.5 + 0.5
        else:
            alpha = np.cos(self.episode * self.frequency) * 0.5 + 0.5

        # Slowly interpolate between the two possible parameter configurations
        for key, val in self.patient_a._params.items():
            if isinstance(val, numbers.Number):
                self.patient._params[key] = alpha * self.patient_a._params[key] \
                                            + (1 - alpha) * self.patient_b._params[key]

        self.patient.set_init_state()
        self.patient.reset()

        self.sensor.reset()
        self.pump.reset()
        self.scenario.reset()
        self._reset()

        CGM = self.sensor.measure(self.patient)
        obs = Observation(CGM=CGM)

        return Step(
            observation=obs,
            reward=0,
            done=False,
            sample_time=self.sample_time,
            patient_name=self.patient.name,
            meal=0,
            patient_state=self.patient.state)

    def render(self, close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = Viewer(self.scenario.start_time, self.patient.name)

        self.viewer.render(self.show_history())

    def show_history(self):
        df = pd.DataFrame()
        df['Time'] = pd.Series(self.time_hist)
        df['BG'] = pd.Series(self.BG_hist)
        df['CGM'] = pd.Series(self.CGM_hist)
        df['CHO'] = pd.Series(self.CHO_hist)
        df['insulin'] = pd.Series(self.insulin_hist)
        df['LBGI'] = pd.Series(self.LBGI_hist)
        df['HBGI'] = pd.Series(self.HBGI_hist)
        df['Risk'] = pd.Series(self.risk_hist)
        df = df.set_index('Time')
        return df

