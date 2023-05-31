from typing import Dict, Optional
from agents import Agent
from environment import Environment
from mdp import MarkovDecisionProcess, S, A
import json
import numpy as np
import math

from helpers.utils import NotImplemented

# This is a class for a generic Policy Iteration agent
class PolicyIterationAgent(Agent[S, A]):
    mdp: MarkovDecisionProcess[S, A] # The MDP used by this agent for training
    policy: Dict[S, A]
    utilities: Dict[S, float] # The computed utilities
                                # The key is the string representation of the state and the value is the utility
    discount_factor: float # The discount factor (gamma)

    def __init__(self, mdp: MarkovDecisionProcess[S, A], discount_factor: float = 0.99) -> None:
        super().__init__()
        self.mdp = mdp
        # This initial policy will contain the first available action for each state,
        # except for terminal states where the policy should return None.
        self.policy = {
            state: (None if self.mdp.is_terminal(state) else self.mdp.get_actions(state)[0])
            for state in self.mdp.get_states()
        }
        self.utilities = {state:0 for state in self.mdp.get_states()} # We initialize all the utilities to be 0
        self.discount_factor = discount_factor
    
    # Given the utilities for the current policy, compute the new policy
    def update_policy(self):
        #TODO: Complete this function
        for state in self.mdp.get_states():
            if (self.mdp.is_terminal(state)): #if terminal state then action is none
                self.policy[state]=None
            else: 
                #calculate the best act by looping over the actions and successors of the current state and calculating the summation 
                #of utilities using bellman equation, then save the max utility reach with its action.
                #By the end of the function update the action of the policy of current state with the best action
                maxpolicy=-math.inf
                bestact=None
                for action in self.mdp.get_actions(state):
                    successors=self.mdp.get_successor(state,action)
                    sum=0
                    for successor in successors:
                        newpolicy=successors[successor]*(self.mdp.get_reward(state,action,successor)+self.discount_factor*self.utilities[successor])
                        sum+=newpolicy
                    if sum>maxpolicy:
                        maxpolicy=sum
                        bestact=action
                self.policy[state]=bestact
    
    # Given the current policy, compute the utilities for this policy
    # Hint: you can use numpy to solve the linear equations. We recommend that you use numpy.linalg.lstsq
    def update_utilities(self):
        #TODO: Complete this function
        allstates=self.mdp.get_states()
        n=len(allstates)  #number of states
        val = [[0 for _ in range(n)] for _ in range(n)] # size: n x n will save the A side of the linear equation here
        rewards=[[0] for _ in range(n)]  #size : n x 1, save the B side of the linear equation here
        indicesofstates={}  # a dictionary to sabe the indices of the states in it to be able to access them later while updating the utilities
        for index,state in enumerate(allstates):
            indicesofstates[state]=index
            val[index][index]=1  #the diagonal values of A are 1s
        for state in allstates:
            indcurrentstate=indicesofstates[state]  #row number
            action=self.policy[state]
            if action!=None:
                successors=self.mdp.get_successor(state,action)
                for successor in successors: #loop over next states and update the A matrix and B matrix with the values using policy iteration equation
                    val[indcurrentstate][indicesofstates[successor]]+=self.discount_factor*successors[successor]*-1
                    rewards[indcurrentstate][0]+=self.mdp.get_reward(state,action,successor)*successors[successor]

        #calculate the utilities : AX=B, A is the val matrix here, B is the rewards matrix here, so I need to calculate X which is the updated utilites
        rewutilist=np.linalg.lstsq(val, rewards, rcond=None)[0] 
        #loop over the states and upsate the utility of each corresponding state with its value in the rewutilist by accessing it with
        #the indices saved before of each state
        for state in allstates:
            self.utilities[state]=rewutilist[indicesofstates[state]][0]


    # Applies a single utility update followed by a single policy update
    # then returns True if the policy has converged and False otherwise
    def update(self) -> bool:
        #TODO: Complete this function
        #get the old policy then apply update utilities and update policy
        #then get the new policies after calculations, if they are the same then return true 
        #becuase it converged else false
        oldpolicy=self.policy.copy()
        self.update_utilities()
        self.update_policy()
        newpolicy=self.policy
        if oldpolicy==newpolicy: 
            return True
        return False

    # This function applies value iteration starting from the current utilities stored in the agent and stores the new utilities in the agent
    # NOTE: this function does incremental update and does not clear the utilities to 0 before running
    # In other words, calling train(M) followed by train(N) is equivalent to just calling train(N+M)
    def train(self, iterations: Optional[int] = None) -> int:
        iteration = 0
        while iterations is None or iteration < iterations:
            iteration += 1
            if self.update():
                break
        return iteration
    
    # Given an environment and a state, return the best action as guided by the learned utilities and the MDP
    # If the state is terminal, return None
    def act(self, env: Environment[S, A], state: S) -> A:
        #TODO: Complete this function
        if self.mdp.is_terminal(state): #if current state is terminal then no action can be done so return None
            return None
        else:
           return self.policy[state]  #else return the action of the policy of current state
    
    # Save the utilities to a json file
    def save(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'w') as f:
            utilities = {self.mdp.format_state(state): value for state, value in self.utilities.items()}
            policy = {
                self.mdp.format_state(state): (None if action is None else self.mdp.format_action(action)) 
                for state, action in self.policy.items()
            }
            json.dump({
                "utilities": utilities,
                "policy": policy
            }, f, indent=2, sort_keys=True)
    
    # loads the utilities from a json file
    def load(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
            self.utilities = {self.mdp.parse_state(state): value for state, value in data['utilities'].items()}
            self.policy = {
                self.mdp.parse_state(state): (None if action is None else self.mdp.parse_action(action)) 
                for state, action in data['policy'].items()
            }
