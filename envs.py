# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 18:06:57 2016

@author: aurelia
"""

import gym
from gym import spaces
import numpy as np
import random
import gym.envs.classic_control.cartpole as cartpoleGym
        
class Rooms2(gym.Env):
    def __init__(self,nb_goals=2):
        """
        the goal is random in the second room
        the position of the goal is given (=observed) only when the agent is in the second room
        """
        MAP = [
                "+---------+",
                "| : | : : |",
                "| : | : : |",
                "| : : : : |",
                "| : | : : |",
                "| : | : : |",
                "+---------+",
            ]       
        self._map = np.flipud(np.asarray(MAP,dtype='c'))        
        self.dim = (5,5)
        self.action_space = spaces.Discrete(4)
        self._seed()
        self.viewer = None
        self.steps_beyond_done = None    
        self.timestep_limit = 100
        self._configure()
        
        self.nb_goals = nb_goals
        self.observation_space = spaces.Box(0,1,shape=(25+nb_goals))
        self.y_dim = self.nb_goals
        self._reset()
        
        if self.nb_goals == 2:
            self.goals_positions = [(4,4),(4,0)]
        elif self.nb_goals == 4:
            self.goals_positions = [(4,4),(4,0),(2,4),(2,0)]
        else:
            assert self.nb_goals<=15, "too many goals, maximun is 15"
            self.goals_positions = []
            for g in range(nb_goals):
                while True:
                    gx = random.randint(2,4)
                    gy = random.randint(0,4)
                    if not ((gx,gy) in self.goals_positions):
                        self.goals_positions.append((gx,gy))
                        break
        
    def _encode(self,x,y):
        """ return number between 0 and 24 """
        return x*5+y
    
    def _reset(self, obs_init = None):
        self.state = np.zeros(25+self.nb_goals) 
        self.location = (0,4)
        self.state[self._encode(self.location[0],self.location[1])] = 1
        self.goal = random.randint(0,self.nb_goals-1)
        return self.state
    
    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        a = action
        
        new_x = self.location[0]
        new_y = self.location[1]
        reward = -1
        done = False
        if a==0 and self._map[1+new_y, 2*new_x+2]==b":" :
            new_x = min(new_x+1, 4) #right
        elif a==1 and self._map[1+new_y, 2*new_x]==b":" :
            new_x = max(new_x-1, 0) #left
        elif a==2 :
            new_y = min(new_y+1, 4) #up
        elif a==3:
            new_y = max(new_y-1, 0) #down
        if (new_x,new_y) == self.goals_positions[self.goal]:
            reward = 20
            done = True
        if (new_x,new_y) == (2,2): #the agent passed in the second room: he sees the goal
            self.state[25+self.goal] = 1
        if (new_x,new_y) == (1,2): #the agent returned in the first room: he doesn't see the goal anymore
            self.state[25+self.goal] = 0
        self.state[self._encode(self.location[0],self.location[1])] = 0
        self.location = (new_x,new_y)
        self.state[self._encode(self.location[0],self.location[1])] = 1
        
        return self.state, reward, done, {}  
        
class Rooms4(gym.Env):
    
    def __init__(self):
        """
        four 5*5 rooms       
        In each episode the position of the goal (room and location in room) is randomly choosen
        the position of the goal is given (=observed) only when the agent is in the same room as the goal
        """        
#        MAP = [
#                "+---------+---------+",
#                "| : : : : | : : : : |",
#                "| : : : : | : : : : |",
#                "| : : : :1;0: : : : |",
#                "| : : : : | : : : : |",
#                "| : :2: : | : :3: : |",
#                "+----;----+----;----+",
#                "| : :0: : | : :1: : |",
#                "| : : : : | : : : : |",
#                "| : : : :3;2: : : : |",
#                "| : : : : | : : : : |",
#                "| : : : : | : : : : |",
#                "+---------+---------+"
#            ]     
        MAP0 = [
                "+---------+",
                "| : : : : |",
                "| : : : : |",
                "| : : : :1;",
                "| : : : : |",
                "| : :2: : |",
                "+----;----+"
                ]
        MAP1 = [
                "+---------+",
                "| : : : : |",
                "| : : : : |",
                ";0: : : : |",
                "| : : : : |",
                "| : :3: : |",
                "+----;----+"
                ]
        MAP2 = [
                "+----;----+",
                "| : :0: : |",
                "| : : : : |",
                "| : : : :3;",
                "| : : : : |",
                "| : : : : |",
                "+---------+"
            ]   
        MAP3 = [
                "+----;----+",
                "| : :1: : |",
                "| : : : : |",
                ";2: : : : |",
                "| : : : : |",
                "| : : : : |",
                "+---------+"
            ]    
        self.dim = (10,10)
        self.doors = {0:[2,3],1:[0,3],2:[1,2],3:[0,1]}
        self._maps = [np.flipud(np.asarray(MAP,dtype='c')) for MAP in [MAP0,MAP1,MAP2,MAP3]] 
        
        self.action_space = spaces.Discrete(4) 
        self.observation_space = spaces.Box(0,1,shape=(25+4+25))   
        self.y_dim = 4+25
        self._seed()
        self._reset()
        self.viewer = None
        self.steps_beyond_done = None    
        self.timestep_limit = 300
        self._configure()     
    
    def _encode(self,x,y):
        """ return number between 0 and 24 """
        return x*5+y
        
    def _encode_room(self,room):
        self.state[25:] = 0
        if self.goal_room == room :
            self.state[25+4+self.goal] = 1
        for door in self.doors[room]:
            self.state[25+door] = 1
        self._map = self._maps[room]
        self.current_room = room    
        
    def _reset(self):
        self.state = np.zeros(25+4+25)
        self.location = (0,4)
        self.state[self._encode(self.location[0],self.location[1])] = 1
        self.goal = random.randint(0,24)
        self.goal_room = random.randint(0,3)
        self._encode_room(0)
        return self.state
    
    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        a = action
        
        new_x = self.location[0]
        new_y = self.location[1]
        reward = -1
        done = False
        if a==0 and self._map[1+new_y, 2*new_x+2]==b":" :
            new_x = min(new_x+1, 4) #right
        elif a==0 and self._map[1+new_y, 2*new_x+2]==b";" :
            self._encode_room(int(self._map[1+new_y,2*new_x+1]))
            new_x = 0 #right
        elif a==1 and self._map[1+new_y, 2*new_x]==b":" :
            new_x = max(new_x-1, 0) #left
        elif a==1 and self._map[1+new_y, 2*new_x]==b";"  :
            self._encode_room(int(self._map[1+new_y,2*new_x+1]))
            new_x = 4
        elif a==2 and self._map[2+new_y,2*new_x+1]==b";" :
            self._encode_room(int(self._map[1+new_y,2*new_x+1]))
            new_y = 0 #up
        elif a==2 and self._map[2+new_y,2*new_x+1]!=b"-" :
            new_y = min(new_y+1, 4) #up
        elif a==3 and self._map[new_y,2*new_x+1]==b";" :
            self._encode_room(int(self._map[1+new_y,2*new_x+1]))
            new_y = 4 #down
        elif a==3 and self._map[new_y,2*new_x+1]!=b"-" :
            new_y = max(new_y-1, 0) #down
        if self.current_room==self.goal_room and self._verif_goal(new_x,new_y,self.goal):
            reward = 20
            done = True
        self.state[self._encode(self.location[0],self.location[1])] = 0
        self.location = (new_x,new_y)
        self.state[self._encode(self.location[0],self.location[1])] = 1
        
        return self.state, reward, done, {}  
        
    def _verif_goal(self,x,y,ngoal):
        return self._encode(x,y) == ngoal
        
class Rooms9(gym.Env):
    
    def __init__(self):
        """
        nine 5*5 rooms       
        In each episode the position of the goal (room and location in room) is randomly choosen
        the position of the goal is given (=observed) only when the agent is in the same room as the goal
        """     
        self.num_rooms = 3
        self.size_room = 5
        self.dim = (self.num_rooms*self.size_room,self.num_rooms*self.size_room)
        self.dim_room = self.size_room*self.size_room
        self.rooms = [(x,y) for x in range(self.num_rooms) for y in range(self.num_rooms )]
        self.doors = {}
        for i,(x,y) in enumerate(self.rooms):
            self.doors[i] = self._get_doors_room(x,y)
            self.doors[(x,y)] = self.doors[i]
             
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0,1,shape=(self.dim_room+4+self.dim_room))
        self.y_dim = 4+self.dim_room
        self._seed()
        self._reset()
        self.viewer = None
        self.steps_beyond_done = None    
        self.timestep_limit = 300
        self._configure()  
    
    def _encode(self,x,y):
        """ return number between 0 and 24 """
        return x*self.size_room+y
        
    def _get_doors_room(self,x,y):
        doors = []
        if x!=0:
            doors.append(0)
        if y!=self.num_rooms-1:
            doors.append(1)
        if x!=self.num_rooms-1:
            doors.append(2)
        if y!=0:
            doors.append(3)
        return doors
        
    def _encode_room(self,room):
        self.state[self.dim_room:] = 0
        if self.goal_room == room :
            self.state[self.dim_room+4+self.goal] = 1
        for door in self.doors[room]:
            self.state[self.dim_room+door] = 1
        self.current_room = room    
        
    def _reset(self, obs_init = None):
        self.state = np.zeros(self.dim_room+4+self.dim_room)
        self.location = (0,4)
        self.state[self._encode(self.location[0],self.location[1])] = 1
        self.goal = random.randint(0,self.dim_room-1)
        self.goal_room = (random.randint(0,self.num_rooms-1),random.randint(0,self.num_rooms-1))
        self._encode_room((0,self.num_rooms-1))
        return self.state
    
    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        a = action
        
        new_x = self.location[0]
        new_y = self.location[1]
        reward = -1
        done = False
        if a==0 and self.location==(4,2) and 2 in self.doors[self.current_room] :
            self._encode_room((self.current_room[0]+1,self.current_room[1]))
            new_x = 0
        elif a==0:
            new_x = min(new_x+1, 4) #right
        if a==1 and self.location==(0,2) and 0 in self.doors[self.current_room] :
            self._encode_room((self.current_room[0]-1,self.current_room[1]))
            new_x = 4
        elif a==1:
            new_x = max(new_x-1, 0) #left
        if a==2 and self.location==(2,4) and 1 in self.doors[self.current_room] :
            self._encode_room((self.current_room[0],self.current_room[1]+1))
            new_y = 0
        elif a==2:
            new_y = min(new_y+1, 4)  #up
        if a==3 and self.location==(2,0) and 3 in self.doors[self.current_room] :
            self._encode_room((self.current_room[0],self.current_room[1]-1))
            new_y = 4
        elif a==3:
            new_y = max(new_y-1, 0)  #down
        if self.current_room==self.goal_room and self._verif_goal(new_x,new_y,self.goal):
            reward = 20
            done = True
        self.state[self._encode(self.location[0],self.location[1])] = 0
        self.location = (new_x,new_y)
        self.state[self._encode(self.location[0],self.location[1])] = 1
        
        return self.state, reward, done, {}  
        
    def _verif_goal(self,x,y,ngoal):
        return self._encode(x,y) == ngoal

class CartPoleStochaEnv(cartpoleGym.CartPoleEnv):
    def __init__(self,stocha):
        self.stocha = stocha  
        self.timestep_limit = 200
        super().__init__()
    
    def _step(self,action):
        if random.random()<self.stocha:
            action_taken = random.randint(0,self.action_space.n-1)
        else:
            action_taken = action
        state, reward, done, info = super()._step(action_taken)
        info["action_taken"] = action_taken
        return state, reward, done, info  
  
class Rooms2Stocha(Rooms2):
    def __init__(self,stocha):
        super().__init__()
        self.stocha = stocha  
    
    def _step(self,action):
        if random.random()<self.stocha:
            action_taken = random.randint(0,self.action_space.n-1) 
        else:
            action_taken = action
        state, reward, done, info = super()._step(action_taken)
        info["action_taken"] = action_taken
        return state, reward, done, info        

class Rooms4Stocha(Rooms4):
    def __init__(self,stocha):
        super().__init__()
        self.stocha = stocha  
    
    def _step(self,action):
        if random.random()<self.stocha:
            action_taken = random.randint(0,self.action_space.n-1) 
        else:
            action_taken = action
        state, reward, done, info = super()._step(action_taken)
        info["action_taken"] = action_taken
        return state, reward, done, info  
        

class Rooms9Stocha(Rooms9):
    def __init__(self,stocha):
        super().__init__()
        self.stocha = stocha  
    
    def _step(self,action):
        if random.random()<self.stocha:
            action_taken = random.randint(0,self.action_space.n-1) 
        else:
            action_taken = action
        state, reward, done, info = super()._step(action_taken)
        info["action_taken"] = action_taken
        return state, reward, done, info    