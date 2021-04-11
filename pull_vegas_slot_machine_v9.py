import numpy as np
import pandas as pd
import random, os, datetime, math
from random import shuffle
from collections import OrderedDict
from collections import defaultdict


total_reward = 0
bandit_dict = {}


def set_seed(my_seed=42):
    os.environ["PYTHONHASHSEED"] = str(my_seed)
    random.seed(my_seed)
    np.random.seed(my_seed)

def get_next_bandit(exception=None):
    '''
    Choose best next bandit
    '''

    # init best bandit number and expectation score
    best_bandit = 0
    best_bandit_expected = 0

    # shuffle bandit_dict to not explore bandits in order 
    b = list(bandit_dict.items())
    shuffle(b)
    a = OrderedDict(b)
    for bnd in dict(a):
        expect = (
            (
                bandit_dict[bnd]["win"] # from nb of win
                - bandit_dict[bnd]["loss"] # remove nb of loss
                + (bandit_dict[bnd]["loss"] > 0)
                + bandit_dict[bnd]["opp"] # add nb of opponant pull
                - (bandit_dict[bnd]["opp"] > 0) * 1.5 # minus a bonus if opponant did pulled
                + bandit_dict[bnd]["op_continue"] # add nb of times opponant continued to pull
            )
            / ( # divided by
                bandit_dict[bnd]["win"]  # nb of win
                + bandit_dict[bnd]["loss"] # plus number of loss
                + bandit_dict[bnd]["opp"] # nb of times opponant used it 
            ) # times
            * math.pow(
                0.97, # decay to the power of
                bandit_dict[bnd]["win"] 
                + bandit_dict[bnd]["loss"]
                + bandit_dict[bnd]["opp"], # total number of pull on this bandit
            )
        )
        if expect > best_bandit_expected:
            if bnd != exception:
                best_bandit_expected = expect
                best_bandit = bnd
    return best_bandit

def get_a_virgin_bandit():
    '''
    return a bandit never explored by me or opponant
    '''
    l = list(bandit_dict.items())
    random.shuffle(l)
    d = dict(l)
    for bnd in d:
        if (d[bnd]["win"] == 1) and (
            d[bnd]["loss"] == 0) and (
            d[bnd]["opp"] == 0):
            return bnd

def is_still_virgin_bandit_present():
    '''
    return a bandit never explored by me or opponant
    '''
    count_virgin_bandit = 0
    for bnd in bandit_dict:
        if (bandit_dict[bnd]["win"] == 1) and (
            bandit_dict[bnd]["loss"] == 0) and (
            bandit_dict[bnd]["opp"] == 0):
            count_virgin_bandit += 1
    if count_virgin_bandit > 0:
        return 1
    else:
        return 0
    


my_action_list = []
op_action_list = []

op_continue_cnt_dict = defaultdict(int)


def multi_armed_probabilities(observation, configuration):
    global total_reward, bandit_dict

    # initialise randomly
    my_pull = random.randrange(configuration["banditCount"])

    # first step: initialise bandit_dict with default values
    if 0 == observation["step"]:
        set_seed()
        total_reward = 0
        bandit_dict = {}
        for i in range(configuration["banditCount"]):
            bandit_dict[i] = {
                "win": 1,
                "loss": 0,
                "opp": 0,
                "my_continue": 0,
                "op_continue": 0,
            }
    
    else:
        # update total reward (starting at 0)
        last_reward = observation["reward"] - total_reward
        total_reward = observation["reward"]

        # update (last) action lists
        my_idx = observation["agentIndex"]
        my_last_action = observation["lastActions"][my_idx]
        op_last_action = observation["lastActions"][1 - my_idx]
        my_action_list.append(my_last_action)
        op_action_list.append(op_last_action)

        # update bandit dict
        if 0 < last_reward:
            # update nb of wining if won on last used bandit
            bandit_dict[my_last_action]["win"] = bandit_dict[my_last_action]["win"] + 1
        else:
            # update nb of loss if lost on last used bandit
            bandit_dict[my_last_action]["loss"] = (
                bandit_dict[my_last_action]["loss"] + 1
            )
        # update opponant action count on bandit
        bandit_dict[op_last_action]["opp"] = bandit_dict[op_last_action]["opp"] + 1

        # if we played for more than 3 times since started
        if observation["step"] >= 3:
            if my_action_list[-1] == my_action_list[-2]:
                # update 'my_continue' since I played the same bandit two times in a row
                bandit_dict[my_last_action]["my_continue"] += 1
            else:
                bandit_dict[my_last_action]["my_continue"] = 0
            if op_action_list[-1] == op_action_list[-2]:
                # update 'op_continue' since opponant played the same bandit two times in a row
                bandit_dict[op_last_action]["op_continue"] += 1
            else:
                bandit_dict[op_last_action]["op_continue"] = 0

        # if we played less than 4 times since started
        if observation["step"] < 4:
                return get_a_virgin_bandit()

        if (observation["step"] < 100) and (op_action_list[-1] != op_action_list[-2]):
            if is_still_virgin_bandit_present() == 1:
                return get_a_virgin_bandit()

        # if opponant stays on same bandit 2 times in a row
        if (op_action_list[-1] == op_action_list[-2]):
            # if I wasn't on his bandit 
            if my_action_list[-1] != op_action_list[-1]:
                # I go there
                my_pull = op_action_list[-1]
            # else if I was there
            elif  my_action_list[-1] == op_action_list[-1]:
                # if I just won
                if last_reward > 0:
                    my_pull = my_last_action
                else:
                    my_pull = get_next_bandit()

        # else if I won
        elif last_reward > 0:
             my_pull = get_next_bandit(my_action_list[-1])
            
        else:
            # if I was winning 3 times in a row but I lost last time 
            if (my_action_list[-1] == my_action_list[-2]) and (
                my_action_list[-1] == my_action_list[-3]
            ):
                # then I choose 50/50 if I continue
                if random.random() < 0.5:
                    # random tell me to stay on the same bandit
                    my_pull = my_action_list[-1]
                else:
                    # I choose another one
                    my_pull = get_next_bandit()
            # As I wasn't on the same bandit 3 times in a row, I move
            else:
                my_pull = get_next_bandit()

    return my_pull
