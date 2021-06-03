import numpy as np
import pygame
from time import time,sleep
from random import randint 
import random
import tkinter as tk
from tkinter import *
import matplotlib.pyplot as plt


root = Tk()

def command():
    global start
    start = entry.get()
    start = int (start)
    root.destroy()
    
entry = Entry(root)
button = Button(root, text="Başlangıç değerini Kaydet", command=command)

entry.pack()
button.pack()
root.mainloop()


root2 = Tk()

def command2():
    global finish
    finish = entry2.get()
    finish = int (finish)
    root2.destroy()
    
entry2 = Entry(root2)
button2 = Button(root2, text="Bitiş değerini Kaydet", command=command2)
entry2.pack()
button2.pack()
root2.mainloop()


n = 8 #represents no. of side squares(n*n total squares)
scrx = n*100
scry = n*100
background = (51,51,51) #used to clear screen while rendering
screen = pygame.display.set_mode((scrx,scry)) #creating a screen using Pygame
colors = [(51,51,51) for i in range(n**2)] 
#reward = a = np.full((n, n), 3)
reward = a =  np.zeros((n,n))

terminals = []
penalties = 20
f = open("engel.txt", "w+")
f.close()

while penalties != 0:
    i = randint(0,n-1)
    j = randint(0,n-1)
    if reward[i,j] == 0 and [i,j] != [start-1,0] and [i,j] != [finish-1,n-1]:
        reward[i,j] = -5
        penalties-=1
        colors[n*i+j] = (255,0,0)
        terminals.append(n*i+j)
        
        f = open("engel.txt", "a")
        f.write('('+str(i)+','+str(j)+',K)')
        f.close()
        
        
reward[finish-1,n-1] = 5
colors[(n*finish)-1] = (0,255,0)
terminals.append((n*finish)-1)


Q = np.zeros((n**2,8)) #Initializing Q-Table
actions = {"up": 0,"down" : 1,"left" : 2,"right" : 3,"up-left": 4, "up-right": 5, "down-left": 6, "down-right": 7} #possible actions
states = {}
k = 0
for i in range(n):
    for j in range(n):
        states[(i,j)] = k
        k+=1
alpha = 0.8
current_pos = [start-1,0]
epsilon = 0.25



def select_action(current_state):
    global current_pos,epsilon
    possible_actions = []
    if np.random.uniform() <= epsilon:
        if current_pos[0] != 0:
            possible_actions.append("up")
        if current_pos[0] != n-1:
            possible_actions.append("down")
        if current_pos[1] != 0:
            possible_actions.append("left")
        if current_pos[1] != n-1:
            possible_actions.append("right")
        if current_pos[0] != 0 and current_pos[1] != 0:
            possible_actions.append("up-left")
        if current_pos[0] != 0 and current_pos[1] != n-1:
            possible_actions.append("up-right")
        if current_pos[0] != n-1 and current_pos[1] != 0:
            possible_actions.append("down-left")
        if current_pos[0] != n-1 and current_pos[1] != n-1:
            possible_actions.append("down-right")
        action = actions[possible_actions[randint(0,len(possible_actions) - 1)]]
    else:
        m = np.min(Q[current_state])
        if current_pos[0] != 0: #up
            possible_actions.append(Q[current_state,0])
        else:
            possible_actions.append(m - 100)
        if current_pos[0] != n-1: #down
            possible_actions.append(Q[current_state,1])
        else:
            possible_actions.append(m - 100)
        if current_pos[1] != 0: #left
            possible_actions.append(Q[current_state,2])
        else:
            possible_actions.append(m - 100)
        if current_pos[1] != n-1: #right
            possible_actions.append(Q[current_state,3])
        else:
            possible_actions.append(m - 100)
        if current_pos[0] != 0 and current_pos[1] != 0: #up-left
            possible_actions.append(Q[current_state,4])
        else:
            possible_actions.append(m - 100)
        if current_pos[0] != 0 and current_pos[1] != n-1: #up-right
            possible_actions.append(Q[current_state,5])
        else:
            possible_actions.append(m - 100)
        if current_pos[0] != n-1 and current_pos[1] != 0: #down-left
            possible_actions.append(Q[current_state,6])
        else:
            possible_actions.append(m - 100)
        if current_pos[0] != n-1 and current_pos[1] != n-1: #down-right
            possible_actions.append(Q[current_state,7])
        else:
            possible_actions.append(m - 100)
        
        action = random.choice([i for i,a in enumerate(possible_actions) if a == max(possible_actions)]) #randomly selecting one of all possible actions with maximin value
    return action
      
      
def episode(cost,i):
    global current_pos,epsilon
    current_state = states[(current_pos[0],current_pos[1])]
    action = select_action(current_state)
    if action == 0: #move up
        current_pos[0] -= 1
    elif action == 1: #move down
        current_pos[0] += 1
    elif action == 2: #move left
        current_pos[1] -= 1
    elif action == 3: #move right
        current_pos[1] += 1
    elif action == 4: #move up-left
        current_pos[0] -= 1
        current_pos[1] -= 1
    elif action == 5: #move up-right
        current_pos[0] -= 1
        current_pos[1] += 1
    elif action == 6: #move down-left
        current_pos[0] += 1
        current_pos[1] -= 1
    elif action == 7: #move down-right
        current_pos[0] += 1
        current_pos[1] += 1

    new_state = states[(current_pos[0],current_pos[1])]
    if new_state not in terminals:
        Q[current_state,action] = (reward[current_pos[0],current_pos[1]] + alpha*(np.max(Q[new_state])))
        cost += Q[current_state,action]
        i+=1
        return cost,i
    
    else:
        Q[current_state,action] = (reward[current_pos[0],current_pos[1]] + alpha*(np.max(Q[new_state])))
        current_pos = [start-1,0]
        cost = 0
        i=0
        if epsilon > 0.05:
            epsilon -= 3e-4 #reducing as time increases to satisfy Exploration & Exploitation Tradeoff
        return cost,i



def select_action2(current_state):
    global current_pos,epsilon
    epsilon = 0
    possible_actions = []
    if np.random.uniform() <= epsilon:
        if current_pos[0] != 0:
            possible_actions.append("up")
        if current_pos[0] != n-1:
            possible_actions.append("down")
        if current_pos[1] != 0:
            possible_actions.append("left")
        if current_pos[1] != n-1:
            possible_actions.append("right")
        if current_pos[0] != 0 and current_pos[1] != 0:
            possible_actions.append("up-left")
        if current_pos[0] != 0 and current_pos[1] != n-1:
            possible_actions.append("up-right")
        if current_pos[0] != n-1 and current_pos[1] != 0:
            possible_actions.append("down-left")
        if current_pos[0] != n-1 and current_pos[1] != n-1:
            possible_actions.append("down-right")
        action = actions[possible_actions[randint(0,len(possible_actions) - 1)]]
    else:
        m = np.min(Q[current_state])
        if current_pos[0] != 0: #up
            possible_actions.append(Q[current_state,0])
        else:
            possible_actions.append(m - 100)
        if current_pos[0] != n-1: #down
            possible_actions.append(Q[current_state,1])
        else:
            possible_actions.append(m - 100)
        if current_pos[1] != 0: #left
            possible_actions.append(Q[current_state,2])
        else:
            possible_actions.append(m - 100)
        if current_pos[1] != n-1: #right
            possible_actions.append(Q[current_state,3])
        else:
            possible_actions.append(m - 100)
        if current_pos[0] != 0 and current_pos[1] != 0: #up-left
            possible_actions.append(Q[current_state,4])
        else:
            possible_actions.append(m - 100)
        if current_pos[0] != 0 and current_pos[1] != n-1: #up-right
            possible_actions.append(Q[current_state,5])
        else:
            possible_actions.append(m - 100)
        if current_pos[0] != n-1 and current_pos[1] != 0: #down-left
            possible_actions.append(Q[current_state,6])
        else:
            possible_actions.append(m - 100)
        if current_pos[0] != n-1 and current_pos[1] != n-1: #down-right
            possible_actions.append(Q[current_state,7])
        else:
            possible_actions.append(m - 100)
        
        action = random.choice([i for i,a in enumerate(possible_actions) if a == max(possible_actions)]) #randomly selecting one of all possible actions with maximin value
    return action
            
def shortest_path():
    
    global current_pos,epsilon

    current_state = states[(current_pos[0],current_pos[1])]
    action = select_action2(current_state)
    if action == 0: #move up
        current_pos[0] -= 1
    elif action == 1: #move down
        current_pos[0] += 1
    elif action == 2: #move left
        current_pos[1] -= 1
    elif action == 3: #move right
        current_pos[1] += 1
    elif action == 4: #move up-left
        current_pos[0] -= 1
        current_pos[1] -= 1
    elif action == 5: #move up-right
        current_pos[0] -= 1
        current_pos[1] += 1
    elif action == 6: #move down-left
        current_pos[0] += 1
        current_pos[1] -= 1
    elif action == 7: #move down-right
        current_pos[0] += 1
        current_pos[1] += 1
    new_state = states[(current_pos[0],current_pos[1])]
    if new_state not in terminals:
        Q[current_state,action] = (reward[current_pos[0],current_pos[1]] + alpha*(np.max(Q[new_state])))
    else:
        Q[current_state,action] = (reward[current_pos[0],current_pos[1]] + alpha*(np.max(Q[new_state])))
        current_pos = [start-1,0]
        if epsilon > 0.05:
            epsilon -= 3e-4 #reducing as time increases to satisfy Exploration & Exploitation Tradeoff
        

def plot_results(steps, cost):
        #
        f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        #
        ax1.plot(np.arange(len(steps)), steps, 'b')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Steps')
        ax1.set_title('Episode via steps')

        #
        ax2.plot(np.arange(len(cost)), cost, 'r')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cost')
        ax2.set_title('Episode via cost')

        plt.tight_layout()  # Function to make distance between figures

        #
        plt.figure()
        plt.plot(np.arange(len(steps)), steps, 'b')
        plt.title('Episode via steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')

        #
        plt.figure()
        plt.plot(np.arange(len(cost)), cost, 'r')
        plt.title('Episode via cost')
        plt.xlabel('Episode')
        plt.ylabel('Cost')

        # Showing the plots
        plt.show()
    
            
            
def layout():
    c = 0
    for i in range(0,scrx,100):
        for j in range(0,scry,100):
            pygame.draw.rect(screen,(255,255,255),(j,i,j+100,i+100),0)
            pygame.draw.rect(screen,colors[c],(j+3,i+3,j+95,i+95),0)
            c+=1
            pygame.draw.circle(screen,(25,129,230),(current_pos[1]*100 + 50,current_pos[0]*100 + 50),30,0)


input_rect = pygame.Rect(250, 250, 140, 32)
color = pygame.Color('lightskyblue3')
    
run = True
run_count =0

steps = []
all_costs = []
cost = 0
i = 0
while run:
    run_count +=1
    #sleep(0.05)
    screen.fill(background)
    layout()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    pygame.display.flip()
    cost,i = episode(cost,i)
    steps += [i]
    all_costs += [cost]
    
    if run_count ==n*1000:
        run = False
#pygame.quit()
run = True
path = []
current_pos = [start-1,0]
while run:
    sleep(0.5)
    screen.fill(background)
    layout()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    pygame.display.flip()
    
    
    path.append((current_pos[0],current_pos[1]))
    colors[(states[(path[-1])])] = (24,67,0)
    shortest_path()
    
    
    if path[-1]==(finish-2,n-1) or path[-1]==(finish-1,n-2) or path[-1]==(finish-2,n-2) or path[-1]==(finish,n-1) or path[-1]==(finish,n-2) :
        sleep(0.5)
        screen.fill(background)
        layout()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        pygame.display.flip()
        
        
        path.append((current_pos[0],current_pos[1]))
        colors[(states[(path[-1])])] = (24,67,0)
        shortest_path()
        sleep(5.0)
        run = False

pygame.quit()

plot_results(steps, all_costs)





