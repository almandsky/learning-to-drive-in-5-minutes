import numpy as np
from donkeycar.gym.remote_controller import DonkeyRemoteContoller
from config import INPUT_DIM, ROI, THROTTLE_REWARD_WEIGHT, MAX_THROTTLE, MIN_THROTTLE, \
    REWARD_CRASH, CRASH_SPEED_WEIGHT

class VAEDonkeyRemoteController(DonkeyRemoteContoller):
    def __init__(self, *args, **kwargs):
        super(VAEDonkeyRemoteController, self).__init__(*args, **kwargs)
        self.last_throttle = 0.
        self.info = {}

    def calc_reward(self, done):
        """
        Compute reward:
        - +1 life bonus for each step + throttle bonus
        - -10 crash penalty - penalty for large throttle during a crash

        :param done: (bool)
        :return: (float)
        """
        if done:
            # penalize the agent for getting off the road fast
            norm_throttle = (self.last_throttle - MIN_THROTTLE) / (MAX_THROTTLE - MIN_THROTTLE)
            return REWARD_CRASH - CRASH_SPEED_WEIGHT * norm_throttle

        # 1 per timesteps + throttle
        throttle_reward = THROTTLE_REWARD_WEIGHT * (self.last_throttle / MAX_THROTTLE)
        return 1 + throttle_reward

    def take_action(self, action):
        self.last_throttle = action[0]
        #steering and throttle are inverted
        #steering = float(action[1]) * -1.0
        #throttle = float(action[0]) * -1.0
        #action = (steering, throttle)
        action = (float(action[0]), float(action[1]))
        super(VAEDonkeyRemoteController, self).take_action(action)

    def wait_until_loaded(self):
        pass

    def is_game_over(self):
        return False

    def observe(self):
        obs = super(VAEDonkeyRemoteController, self).observe()
        #trim top 40 pixels so we get 80, 160, 3 
        if obs is None:
            obs = np.zeros((80, 160, 3))
        else:
            obs = obs[40:, :, :]
        done = self.is_game_over()
        reward = self.calc_reward(done)
        return obs, reward, done, self.info
