# This file contains the options that you should modify to solve Question 2

def question2_1():
    #TODO: Choose options that would lead to the desired results 
    return {
        "noise": 0.01,   #make the noise very small to avoid falling in the row of -10 states
        "discount_factor": 0.1,  #I want to reach to the near terminal the the discount factor should be small
        "living_reward": -5.0
    }

def question2_2():
    #TODO: Choose options that would lead to the desired results
    return {
        "noise": 0.1,            # noise may increase a little bit because we are going far from the row of -10 states
        "discount_factor": 0.35, #I want to reach to the near terminal the the discount factor should be small
        "living_reward": -0.16   #the minimum value used to be lost by one tile so when reaching the terminal state it is compensated
    }

def question2_3():
    #TODO: Choose options that would lead to the desired results
    return {
        "noise": 0.01,          #make the noise very small to avoid falling in the row of -10 states
        "discount_factor": 1.0, #I want to reach to the far terminal the the discount factor should be big
        "living_reward": -2.0   #the minimum value used to be lost by one tile so when reaching the terminal state it is compensated
    }

def question2_4():
    #TODO: Choose options that would lead to the desired results
    return {
        "noise": 0.1,           # noise may increase a little bit because we are going far from the row of -10 states
        "discount_factor": 1.0, #I want to reach to the far terminal the the discount factor should be big
        "living_reward": -0.125  #the minimum value used to be lost by one tile so when reaching the terminal state it is compensated
    }

def question2_5():
    #TODO: Choose options that would lead to the desired results
    return {
        "noise": 0.0,
        "discount_factor": 1.0,
        "living_reward": 100   #I want to stay here forever so increase the reward alot to avoid going to terminal states
    }

def question2_6():
    #TODO: Choose options that would lead to the desired results
    return {
        "noise": 0.0,
        "discount_factor": 1.0,
        "living_reward": -100  #I want to leave as soon as possible so increase the reward in negative to be favoring going to the terminal states as soon as possible
    }