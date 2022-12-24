import math
import random
import numpy as np


def get_explore_rate(episode, DECAY_CONSTANT, MIN_EXPLORE_RATE):
    if episode >= DECAY_CONSTANT-1:
        return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((episode+1)/DECAY_CONSTANT)))
    else:
        return 1.0

def select_action(actiontable, explore_rate, env):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    else:
        # Select the action with the highest q value
        action = actiontable.argmax().item()
    return action