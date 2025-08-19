import torch
import math

from configs.config import load_single_config
from agents import *


class Trainer:
    """
    Trainer for the DQN Agent.
    """

    def __init__(self, config=load_single_config("trainer", "base"), **kwargs):
        pass
