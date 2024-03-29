#this code is the workbench for q-learning
#it consists on a lifting particle that must reach a certain height
#it is only subjected to gravity
#Force applied to the particle might be fixed 9.9 or 9.7N

import numpy as np
import math
import random
import matplotlib.pyplot as plt


#INITIALIZE VARIABLES
######################

m=1 #1kg mass
g=9.80 #gravity
dt=0.05 #simulation time
Final_height=50 #1m
Final_vel=0


#STATES are discretized 0-1-2-3...50...-59-60 cm and speed is discretized in
n_pos=61
STATES=np.linspace(0,Final_height+10,Final_height+10+1)

#SPEEDS are discretized -10,-9,-8...0,1,2,3...,50cm/s.
n_speeds=61
SPEEDS=np.linspace(-10,50,n_speeds)

#ROWS=    States (61*61=3721 rows)
#COLUMNS= Actions (9.9 , 9.7) two actions
Rows=n_pos*n_speeds
Columns=2
Actions=([9.9, 9.7])


#time steps
n_items=302
x=np.linspace(0,301,n_items)

#Initialize Q matrix
Q=np.ones((Rows,Columns))

#Q-learning variables
alpha=0.5
gamma=0.5
epsilon=0.15
goalCounter=0
Contador=0


#function to choose the Action
def ChooseAction (Columns,Q,state):

    if np.random.uniform() < epsilon:
        rand_action=np.random.permutation(Columns)
        action=rand_action[1] #current action
        F=Actions[action]
        max_index=1
    # if not select max action in Qtable (act greedy)
    else:
        QMax=max(Q[state]) 
        max_indices=np.where(Q[state]==QMax)[0] # Identify all indexes where Q equals max
        n_hits=len(max_indices) # Number of hits
        max_index=int(max_indices[random.randint(0, n_hits-1)]) # If many hits, choose randomly
        F=Actions[max_index]

    return F, max_index

#function to apply the dynamic model
def ActionToState(F,g,m,dt,z_pos_old,z_vel_old,z_accel_old):
    z_accel=(-g + F/m)*100 
    z_vel=z_vel_old + (z_accel+z_accel_old)/2*dt
    z_pos=z_pos_old + (z_vel+z_vel_old)/2*dt
    z_accel_old=z_accel
    z_vel_old=z_vel 
    z_pos_old=z_pos 

    return z_accel,z_vel,z_pos,z_vel_old,z_pos_old


#BEGINNING of the algorithm
for episode in range(1,200000):
    # initial state
    z_pos=np.zeros(n_items)
    z_vel=np.zeros(n_items)
    z_accel=np.zeros(n_items)
    z_pos_goal=np.zeros((1000, n_items))
    z_vel_goal=np.zeros((1000, n_items))
    z_acel_goal=np.zeros((1000, n_items))
    z_accel_old=0
    z_vel_old=0
    z_pos_old=0 #initial conditions of the particle
    
    state=11 #let's choose the initial state always height 0, speed 0cm/s

    print("episode",episode) #check
  
    for i in range(1,300):

        ## Choose sometimes the Force randomly
        F,max_index = ChooseAction(Columns, Q, state)
                                
        #update the dynamic model
        z_accel[i],z_vel[i],z_pos[i],z_vel_old,z_pos_old= ActionToState (F,g,m,dt,z_pos_old,z_vel_old,z_accel_old)
                     
          
        #if negative height or velocity values, reward it very negatively.
        #If too big values, too
        if (min(z_pos)<0 or min(z_vel)<SPEEDS[1] or max(z_vel)>SPEEDS[60] or max(z_pos)>0.99*n_pos):
            Q[state,max_index]=-100 #penalty
            break

        else:    #if positive values, do the loop

            rounded_pos=round(z_pos[i])  #round the height  
            rounded_vel=round(z_vel[i])  #round the vel  

            #calculate which is my new state
            index_1=np.where(STATES==rounded_pos)
            index_2=np.where(SPEEDS==rounded_vel)
            index_1=int(index_1[0])
            index_2=int(index_2[0])

            state=n_speeds*index_1 + index_2  #new state in Q matrix
            QMax=max(Q[state])  #selects the highest value of the row
          

            #REWARD
            A1=math.exp(-abs(rounded_pos-Final_height)/(0.1*n_pos))
            A2=math.exp(-abs(rounded_vel-Final_vel)/(0.1*14))
            Reward=A1*A2*1000000  #takes into account pos and vel

            #Q VALUE update
            Q[state,max_index]=Q[state,max_index] + alpha*(Reward + gamma*(QMax - Q[state,max_index]))  #update Q value
                       

            #checking
            if (rounded_pos==Final_height or rounded_pos==Final_height-1 or rounded_pos==Final_height+1):
                print("entra")
                goalCounter=goalCounter+1
                if (rounded_vel==Final_vel or rounded_vel==Final_vel+1):
                    Contador=Contador +1  #counter of successful hits
                    
                    #saving of successful data
                    z_pos_goal[0:i,Contador]=z_pos[0:i]
                    z_vel_goal[0:i,Contador]=z_vel[0:i]
                    z_acel_goal[0:i,Contador]=z_accel[0:i]
                    state=11 #reinitialize
                    break

                else:
                    break
