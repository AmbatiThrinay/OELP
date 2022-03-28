from car_model import Car
from env import Env,Path
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os

# class Path:
#     def __init__(self):
#         self.x = np.linspace(10,180,500)
#         self.y = np.full(self.x.shape,30)
#         self.start = (10,30)
#         self.path_width = 20 # must be even
#         self.end_x = np.full((20,),self.x[-1])
#         self.end_y = np.linspace(30-self.path_width/2,30+self.path_width/2,20)
#         self.map_size = (200,140)
#         self.angle = 0.0

path1 = Path()
env = Env(path1)

# <-- Hyper Parameters -->
Total_esipodes = 100_000
X_samples,Y_Samples = 30,20
XTE_samples = 3
Learning_rate = 0.3
Discount = 0.85
Exploration_rate = 0.01
epsilon = 1
# decays from episode 1 to total_episodes/2
epsilon_decay_rate = Exploration_rate/(1-Total_esipodes//2)

record_every = 5_000 # multiple of these episodes are saved
render_every = 10_000 # multiple of these episode will be render
# 0 : 'pedal_gas'
# 1 : 'pedal_reverse'
# 2 : 'pedal_none'
# 3 : 'steer_right'
# 4 : 'steer_left'
# 5 : 'steer_none'
actions = {
    0 : 'pedal_gas',
    1 : 'steer_right',
    2 : 'steer_left',
    3 : 'steer_none'
}

x_max,y_max,xte_max = path1.map_size[0],path1.map_size[1],path1.path_width/2
x_min,y_min,xte_min = 0,0,0

x_interval_range = (x_max - x_min)/X_samples
y_interval_range = (y_max - y_min)/Y_Samples

def get_discrete(cords):
    x = (cords[0]-x_min)//x_interval_range
    y = (cords[1]-y_min)//y_interval_range
    if 0 <= cords[2] <= path1.path_width/4 :
        return int(x),int(y),0
    elif path1.path_width/4 < cords[2] < path1.path_width/2 :
        return int(x),int(y),1
    else :
        return int(x),int(y),2

# # for imitation learning
# input_list = '0'*500+'1'*20+'0'*300+'1'*10+'0'*50+'1'*30+'0'*100+'1'*10+'3'*100
# input_list1 = '0'*500+'1'*20+'0'*300+'1'*10+'0'*50+'1'*30
# def input_generator(lst):
#     # User input
#     for act in lst :
#         yield int(act)
# player = input_generator(input_list)


# q_table = np.load('curved_test_3\Q_tables\E30000.npy')
q_table = np.random.uniform(low=-10,high=-1,size=(X_samples,Y_Samples,XTE_samples,len(actions)))
print(f"Size of the Q table{q_table.shape}")

os.mkdir('curved') # name of the dir where the files are stored
os.mkdir('curved/Q_tables') # for storing the Q_tables

episodes_rewards = [] # total reward for each episode
# stats for every record_every episode
aggr_episodes_rewards = {
    'ep' : [],
    'avg' : [], # average reward of last record_every episode
    'min' : [], # 
    'max' : []
} 
for episode in tqdm(range(Total_esipodes+1)):

    render = False
    if episode%(render_every)== 0:
        render = True
    
    record = False
    if episode!=0 and episode%(record_every)== 0 :
        record = True

    start_time = time.time()
    # reset the envirnoment
    dis_state = get_discrete(env.reset())
    episode_reward = 0
    while not env.done :
        
        ## 1-epsilon greedy action
        if random.random() < epsilon :
            action = random.randint(0,len(actions)-1) #choose random action
        else :
            #choose best action from Q(s,a) values
            action = np.argmax(q_table[dis_state])

        # try :
        #     action = next(player)
        # except StopIteration :
        #     break

        cords,reward,done = env.step(actions[action])
        episode_reward += reward

        new_dis_state = get_discrete(cords)
        if not done :
            max_future_q = np.max(q_table[new_dis_state])
            current_q = q_table[dis_state][action]
            new_q = (1-Learning_rate)*current_q + Learning_rate * (reward + Discount * max_future_q)
            q_table[dis_state][action] = new_q
        elif reward == 5 :
            q_table[dis_state][action] = reward
        
        dis_state = new_dis_state

        if render :
            env.render_env(FPS_lock=None)
            env.record_env(f'curved/E{episode}')

        # stopping the envirnoment it eposide exceeds 3 mins
        if (time.time() - start_time)/60 > 3 :
            env.done = True
            print(f"Episode {episode} was stopped.")
        
        env.close_quit()
    
    # saving the Q table if record
    if record :
        np.save(f"curved/Q_tables/E{episode}",q_table)
        # updating the stats consider last record_every episodes
        aggr_episodes_rewards['ep'].append(episode)
        avg_reward = sum(episodes_rewards[-record_every:])/len(episodes_rewards[-record_every:])
        aggr_episodes_rewards['avg'].append(avg_reward)
        aggr_episodes_rewards['min'].append(min(episodes_rewards[-record_every:]))
        aggr_episodes_rewards['max'].append(max(episodes_rewards[-record_every:]))

    if epsilon > Exploration_rate :
        epsilon -= epsilon_decay_rate
    
    episodes_rewards.append(episode_reward)

plt.figure()
plt.subplot(211)
plt.title(f"reward metrics for every{record_every} episodes")
plt.plot(aggr_episodes_rewards['ep'],aggr_episodes_rewards['avg'],label='average reward')
plt.plot(aggr_episodes_rewards['ep'],aggr_episodes_rewards['min'],label='min reward')
plt.plot(aggr_episodes_rewards['ep'],aggr_episodes_rewards['max'],label='max reward')
plt.legend()
plt.grid()
plt.subplot(212)
plt.title("Net Reward for each Episode")
plt.xlabel("Episodes")
plt.ylabel("Net reward")
plt.plot(np.arange(Total_esipodes+1),np.array(episodes_rewards))
plt.grid()
plt.savefig('curved/metrics.png')
plt.show()

print("<< Done >>")

# def play_episode(filname):
#     X_samples,Y_Samples = 35,25
#     XTE_samples = 3
#     Learning_rate = 0.3
#     Discount = 0.85
#     Exploration_rate = 0.01
#     epsilon = 0.01

    
