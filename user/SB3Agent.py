
from typing import Optional, Type
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import PPO


class Agent:
    """
    Minimal Agent base class used as a fallback when a project-wide Agent
    base class is not available. This provides the attributes and methods
    that SB3Agent expects (file_path, env, model) and simple placeholders.
    """
    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path
        self.env = None
        self.model = None

    def _initialize(self) -> None:
        # Placeholder; subclasses should implement their own initialization.
        return

    def _gdown(self) -> str:
        # Placeholder for downloading model from Google Drive or similar.
        return ""

    def predict(self, obs):
        raise NotImplementedError("Subclasses should implement predict")

    def save(self, file_path: str) -> None:
        raise NotImplementedError("Subclasses should implement save")

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        raise NotImplementedError("Subclasses should implement learn")


class SB3Agent(Agent):
    '''
    SB3Agent:
    - Defines an AI Agent that takes an SB3 class input for specific SB3 algorithm (e.g. PPO, SAC)
    Note:
    - For all SB3 classes, if you'd like to define your own neural network policy you can modify the `policy_kwargs` parameter in `self.sb3_class()` or make a custom SB3 `BaseFeaturesExtractor`
    You can refer to this for Custom Policy: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    '''
    def __init__(
            self,
            sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
            file_path: Optional[str] = None
    ):
        self.sb3_class = sb3_class
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class("MlpPolicy", self.env, verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.01)
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )