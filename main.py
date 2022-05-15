# Author : Ambati Thrinay Kumar Reddy
#  Deep Q Neural network Training and Testing

from env import Env,Path
from dqn import DQN
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time, gmtime, strftime
import os, json, pickle, torch

# opening the json file
try :
    with open(os.path.join('assets','dqn_config.json'),'r') as config:
        dqn_config = json.load(config)
except FileNotFoundError :
    print("DQN config Json file is missing from assets folder.")

path1 = Path()
path_name = Path.get_path_names_from_json()[3]
path1.load_from_json(path_name)
path1.initial_velocity = 5.0
print(dqn_config)
path1.initial_velocity = dqn_config["initial_velocity"] #initial velocity of car
env = Env(path1)

# <-- Hyper Parameters -->
Total_episodes = dqn_config["total_episodes"]
X_samples = dqn_config["x_samples"]
Y_samples = dqn_config["y_samples"]
XTE_samples = dqn_config["XTE_samples"]
nn_learning_rate = dqn_config["nn_learning_rate"]
Discount = dqn_config["discount"]
Exploration_rate = dqn_config["exploration_rate"]
batch_size = dqn_config["batch_size"]
memory_size = dqn_config["memory_size"]
target_update_rate = dqn_config["target_network_update_rate"]
seed = dqn_config["seed"]
epsilon = 1
# decays from episode 1 to total_episodes/2
epsilon_decay_rate = (1-Exploration_rate)/(dqn_config["epsilon_decay_till"])

save_every = dqn_config["save_every"] # multiple of these episodes are saved
render_record_every = dqn_config["render_record_every"] # multiple of these episode will be render
# 0 : 'pedal_gas',
# 1 : 'pedal_brake',
# 2 : 'pedal_none',
# 3 : 'pedal_reverse',
# 4 : 'steer_right',
# 5 : 'steer_left',
# 6 : 'steer_none'
actions = {
    0 : 'pedal_gas',
    1 : 'pedal_reverse',
    2 : 'steer_right',
    3 : 'steer_left',
    4 : 'steer_none',
}

# DQN agent
agent = DQN(
    3, len(actions), nn_learning_rate, Discount, batch_size, seed,
    epsilon = 1,target_update_rate=target_update_rate, eps_end=Exploration_rate,
    eps_dec = epsilon_decay_rate)

# <-- Environment Discretization -->
x_max,y_max,xte_max = path1.screen_size[0]/env.ppu,path1.screen_size[1]/env.ppu,path1.path_width/2
x_min,y_min,xte_min = 0,0,0

x_interval_range = (x_max - x_min)/X_samples
y_interval_range = (y_max - y_min)/Y_samples

def get_discrete(cords):
    x_state = (cords[0]-x_min)//x_interval_range
    y_state = (cords[1]-y_min)//y_interval_range
    if 0 <= cords[2] <= path1.path_width/4 :
        return int(x_state),int(y_state),0
    elif path1.path_width/4 < cords[2] < path1.path_width/2 :
        return int(x_state),int(y_state),1
    else :
        return int(x_state),int(y_state),2

# using GMT time to create a unique folder for saving
unique_string = strftime("_%d-%b-%Y-%H-%M-%S", gmtime())
new_folder_name = path_name+unique_string
os.mkdir(new_folder_name) # name of the dir where the files are stored
os.mkdir(os.path.join(new_folder_name,'DQN_checkpoints')) # for storing the Q_tables
# writing the Q-learning config used
with open(os.path.join(new_folder_name,'dqn_config_used.json'),'w') as config:
    json.dump(dqn_config,config)


episodes_rewards = [] # total reward for each episode
# stats for every 'save_every' episode
aggr_episodes_rewards = {
    'ep' : [],
    'avg' : [], # average reward of last 'save_every' episode
    'min' : [], # 
    'max' : []
} 

# <-- Q learning Agent Training -->
for episode in tqdm(range(Total_episodes+1)):
    render = False
    if episode%(render_record_every) == 0:
        render = True
    save = False
    if episode!=0 and episode%save_every == 0 :
        save = True

    start_time = time()
    # reset the envirnoment
    dis_state = get_discrete(env.reset())
    episode_reward = 0
    while not env.done :
        
        # Agent shows action
        action = agent.choose_action(dis_state)

        # step the environment
        cords,reward,done = env.step(actions[action])
        episode_reward += reward

        # convert the observation in discrete state
        new_dis_state = get_discrete(cords)

        # store the experience and train the agent
        agent.store_transition(dis_state, action, reward, new_dis_state, done)
        agent.learn()
        dis_state = new_dis_state

        if render :
            env.render_env(FPS_lock=None,render_stats=True)
            env.record_env(f'{new_folder_name}/E{episode}')

        # stopping the environment if car moves far from the track
        if abs(env.xte >= 3*path1.path_width/4):
            env.done = True
            # print(f"Car is too far from track, Episode {episode} was stopped.")

        # stopping the environment if car takes more than 2 consecutive turns
        elif abs(env.car.angle) > 360*2 :
            env.done = True
            print(f"Car is moving in circles, Episode {episode} was stopped.")
        
        # stopping the environment if episode time exceeds 2.5 mins
        elif (time() - start_time)/60 > 2.5 :
            env.done = True
            print(f"Episode time exceeded, Episode {episode} was stopped.")

        # del this - to show env if pressed s
        # if not render and (time()-start_time)%5 :
        #     env.render_env(FPS_lock=None,render_stats=True)

        env.close_quit()
    
    # saving the DQN networks table
    if save :
        agent.save(os.path.join(new_folder_name,"DQN_checkpoints",f"E{episode}"))

    # updating the stats considering last 100 episodes
    if episode%100 :
        aggr_episodes_rewards['ep'].append(episode)
        avg_reward = sum(episodes_rewards[-100:])/len(episodes_rewards[-100:])
        aggr_episodes_rewards['avg'].append(avg_reward)
        aggr_episodes_rewards['min'].append(min(episodes_rewards[-100:]))
        aggr_episodes_rewards['max'].append(max(episodes_rewards[-100:]))

    episodes_rewards.append(episode_reward)

    # del this
    if episode_reward > 5:
        print("Got positive reward of 5")
        agent.save(os.path.join(new_folder_name,"DQN_checkpoints",f"E{episode}"))
        break

fig_handle = plt.figure(figsize=(12,7))
plt.subplot(211)
plt.title(f"reward metrics for every{save_every} episodes")
plt.plot(aggr_episodes_rewards['ep'],aggr_episodes_rewards['avg'],label='average reward')
plt.plot(aggr_episodes_rewards['ep'],aggr_episodes_rewards['min'],label='min reward')
plt.plot(aggr_episodes_rewards['ep'],aggr_episodes_rewards['max'],label='max reward')
plt.legend()
plt.grid()
plt.subplot(212)
plt.title("Net Reward for each Episode")
plt.xlabel("Episodes")
plt.ylabel("Net reward")
plt.plot(np.arange(len(episodes_rewards)),np.array(episodes_rewards))
plt.grid()

plt.savefig(os.path.join(new_folder_name,'metrics.png'))
# save matplotlib figure for later use
with open(os.path.join(new_folder_name,'metrics.pickle'), 'wb') as f:
    pickle.dump(fig_handle,f)
    print(f"matplotlib figure saved a {new_folder_name}/metrics.pickle file")

plt.show()

print("<< Done >>")

# # loading matplotlib plot
# with open(f'{new_folder_name}/metrics.pickle', 'rb') as f:
#     file_handle = pickle.load(f)
# plt.show()
