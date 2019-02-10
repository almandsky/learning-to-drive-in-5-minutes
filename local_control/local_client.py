from threading import Event, Thread
from donkey_gym.envs.vae_env import DonkeyVAEEnv
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, \
    VecFrameStack
from stable_baselines.bench import Monitor
from donkeycar.parts.controller import PS3JoystickController

class LocalControlEnv(object):
    '''
    run training from the robot with joystick controls
    '''
    def __init__(self, env, model=None, is_recording=False,
                 is_training=False, deterministic=True):
        super(LocalControlEnv, self).__init__()
        self.env = env
        self.model = model
        self.need_reset = False
        self.is_manual = True
        self.is_recording = is_recording
        self.is_training = is_training
        # For keyboard trigger
        self.fill_buffer = False
        # For display
        self.is_filling = False
        self.current_obs = None
        self.exit_event = Event()
        self.done_event = Event()
        self.ready_event = Event()
        # For testing
        self.deterministic = deterministic
        self.window = None
        self.process = None
        self.action = None
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.donkey_env = None
        self.n_out_of_bound = 0
        self.current_image = None
        self.image_surface = None
        self.decoded_surface = None

        self.start_process()

    def start_process(self):
        """Start preprocessing process"""
        self.process = Thread(target=self.main_loop)
        # Make it a deamon, so it will be deleted at the same time
        # of the main process
        self.process.daemon = True
        self.process.start()

    def reset(self):
        self.n_out_of_bound = 0
        # Zero speed, neutral angle
        self.donkey_env.controller.take_action([0, 0])
        return self.current_obs

    def wait_for_teleop_reset(self):
        self.ready_event.wait()
        return self.reset()

    def exit(self):
        self.env.reset()
        self.donkey_env.exit_scene()

    def wait(self):
        self.process.join()
    
    def render(self, mode='human'):
        return self.current_obs

    def toggle_manual_control(self):
        self.is_manual = not self.is_manual
        if self.is_training:
            if self.is_manual:
                # Stop training
                self.ready_event.clear()
                self.done_event.set()
            else:
                # Start training
                self.done_event.clear()
                self.ready_event.set()

    def main_loop(self):
        end = False
        control_throttle, control_steering = 0, 0
        action = [control_steering, control_throttle]

        donkey_env = self.env
        # Unwrap env
        while isinstance(donkey_env, VecNormalize) or isinstance(donkey_env, VecFrameStack):
            donkey_env = donkey_env.venv

        if isinstance(donkey_env, DummyVecEnv):
            donkey_env = donkey_env.envs[0]
        if isinstance(donkey_env, Monitor):
            donkey_env = donkey_env.env

        assert isinstance(donkey_env, DonkeyVAEEnv), print(donkey_env)
        self.donkey_env = donkey_env

        controller = PS3JoystickController()
        controller.set_button_down_trigger("circle", self.toggle_manual_control)

        js_thread = Thread(target=controller.update)
        js_thread.daemon = True
        js_thread.start()

        last_time_pressed = {'space': 0, 'm': 0, 't': 0, 'b': 0, 'o': 0}
        self.current_obs = self.reset()

        if self.model is not None:
            # Prevent error (uninitialized value)
            self.model.n_updates = 0

        while not end:
            
            #process input
            control_steering, control_throttle, _, _ = controller.run_threaded()

            # Send Orders
            if self.model is None or self.is_manual:
                self.action = [control_steering, control_throttle]
            elif self.model is not None and not self.is_training:
                self.action, _ = self.model.predict(self.current_obs, deterministic=self.deterministic)

            self.is_filling = False
            if not (self.is_training and not self.is_manual):
                if self.is_manual and not self.fill_buffer:
                    donkey_env.controller.take_action(self.action)
                    self.current_obs, reward, done, info = donkey_env.observe()
                    self.current_obs, _, _, _ = donkey_env.postprocessing_step(self.action, self.current_obs,
                                                                               reward, done, info)

                else:
                    if self.fill_buffer:
                        old_obs = self.current_obs
                    self.current_obs, reward, done, _ = self.env.step(self.action)

                    # Store the transition in the replay buffer
                    if self.fill_buffer and hasattr(self.model, 'replay_buffer'):
                        assert old_obs is not None
                        if old_obs.shape[1] == self.current_obs.shape[1]:
                            self.is_filling = True
                            self.model.replay_buffer.add(old_obs, self.action, reward, self.current_obs, float(done))


