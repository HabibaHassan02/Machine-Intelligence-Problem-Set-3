from typing import Dict, Optional
from agents import Agent
from environment import Environment
from mdp import MarkovDecisionProcess, S, A
import json
import math

from helpers.utils import NotImplemented

# This is a class for a generic Value Iteration agent
class ValueIterationAgent(Agent[S, A]):
    mdp: MarkovDecisionProcess[S, A] # The MDP used by this agent for training 
    utilities: Dict[S, float] # The computed utilities
                                # The key is the string representation of the state and the value is the utility
    discount_factor: float # The discount factor (gamma)

    def __init__(self, mdp: MarkovDecisionProcess[S, A], discount_factor: float = 0.99) -> None:
        super().__init__()
        self.mdp = mdp
        self.utilities = {state:0 for state in self.mdp.get_states()} # We initialize all the utilities to be 0
        self.discount_factor = discount_factor
    
    # Given a state, compute its utility using the bellman equation
    # if the state is terminal, return 0
    def compute_bellman(self, state: S) -> float:
        #TODO: Complete this function
        if (self.mdp.is_terminal(state)): #if terminal state then return 0
            return 0
        else:                             #else compute bellman
            maxuti=-math.inf
            actions=self.mdp.get_actions(state)  #get the actions of the current state and loop over them
            for act in actions:
                successors=self.mdp.get_successor(state,act) #get the successor of the action taken on the current state
                sum=0
                for successor in successors: 
                    prob=successors[successor]    #as successor is a dictionary and its key is the state and its value is the probability of the successsor 
                    reward=self.mdp.get_reward(state,act,successor) #get the reward of taking the action from the current state to the next state
                    usprime=self.utilities[successor]             #utility of the next state 
                    newuti=prob*(reward+(self.discount_factor*usprime))  #calculate bellman
                    sum+=newuti                                          #summation of all utilities of the next states on a certain action
                if sum>maxuti:   #update the max utility reached
                    maxuti=sum
            return maxuti  #return the max utility
    
    # Applies a single utility update
    # then returns True if the utilities has converged (the maximum utility change is less or equal the tolerance)
    # and False otherwise
    def update(self, tolerance: float = 0) -> bool:
        #TODO: Complete this function
        newutidict=dict()  #create a dictionary to save in it the updates utilities to update the whole self.utilites with these value at the end of the function
        maxutichange=-math.inf
        for state in self.mdp.get_states(): #loop over stataes
            olduti=self.utilities[state]  #get the old utility before computing bellman
            newut=self.compute_bellman(state)
            newutidict[state]=newut    #get the new utility after compting bellman on the current state
            if abs(newut-olduti)>maxutichange:   #if the abs value of the change is greater than the max utility changhe reached by now then update the max utility change
                maxutichange=abs(newut-olduti)
        self.utilities=newutidict  #update the while system utilities with the new values
        if maxutichange<=tolerance:  #if the calculated max change is less than or equal the tolerance then return true, else false
            return True
        else:
            return False
            
    # This function applies value iteration starting from the current utilities stored in the agent and stores the new utilities in the agent
    # NOTE: this function does incremental update and does not clear the utilities to 0 before running
    # In other words, calling train(M) followed by train(N) is equivalent to just calling train(N+M)
    def train(self, iterations: Optional[int] = None, tolerance: float = 0) -> int:
        iteration = 0
        while iterations is None or iteration < iterations:
            iteration += 1
            if self.update(tolerance):
                break
        return iteration
    
    # Given an environment and a state, return the best action as guided by the learned utilities and the MDP
    # If the state is terminal, return None
    def act(self, env: Environment[S, A], state: S) -> A:
        #TODO: Complete this function
        if self.mdp.is_terminal(state):  #if terminal state then return None
            return None
        else:
            maxuti=-math.inf
            actions=env.actions()  #get all the environment actions and loop over them
            for act in actions:
                successors=self.mdp.get_successor(state,act) #get all the successors of the current state and current action taken 
                sum=0
                for successor in successors:  #loop over successors
                    prob=successors[successor]  #this is a part of dictionary so successor is the next state which is the key and successors[successor] is the probability
                    reward=self.mdp.get_reward(state,act,successor) #calculate the rewrad of taking the action on the current state to the next state
                    newuti=prob*(reward+self.discount_factor*self.utilities[successor])  #calculate the utility on the current state and action and next state
                    sum+=newuti #sigma part of the equation
                if sum>maxuti: #is the summation is bigger than the max utility reached by now then update the max utility and the best action with the current action
                    maxuti=sum
                    bestact=act
            return bestact  #return the best action reached
    
    # Save the utilities to a json file
    def save(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'w') as f:
            utilities = {self.mdp.format_state(state): value for state, value in self.utilities.items()}
            json.dump(utilities, f, indent=2, sort_keys=True)
    
    # loads the utilities from a json file
    def load(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'r') as f:
            utilities = json.load(f)
            self.utilities = {self.mdp.parse_state(state): value for state, value in utilities.items()}
