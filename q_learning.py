# Author : Ambati Thrinay Kumar Reddy
#  Q learning Training and Testing

from env import Env,Path
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time, gmtime, strftime
import os, json

# <------ for headless execution(server) --------->
# set SDL to use the dummy NULL video driver, 
# so it doesn't need a windowing system.
os.environ["SDL_VIDEODRIVER"] = "dummy"

# opening the json file
try :
    with open(os.path.join('assets','q_learning_config.json'),'r') as config:
        q_learning_config = json.load(config)
except FileNotFoundError :
    print("Q-learning config Json file is missing from assets folder.")

path1 = Path()
path_name = Path.get_path_names_from_json()[3]
path1.load_from_json(path_name)
path1.initial_velocity = 5.0
print(q_learning_config)
path1.initial_velocity = q_learning_config["initial_velocity"] #initial velocity of car
env = Env(path1)

# <-- Hyper Parameters -->
Total_episodes = q_learning_config["total_episodes"]
X_samples = q_learning_config["x_samples"]
Y_samples = q_learning_config["y_samples"]
XTE_samples = q_learning_config["XTE_samples"]
Learning_rate = q_learning_config["learning_rate"]
Discount = q_learning_config["discount"]
Exploration_rate = q_learning_config["exploration_rate"]
epsilon = 1
# decays from episode 1 to total_episodes/2
epsilon_decay_rate = (1-Exploration_rate)/(q_learning_config["epsilon_decay_till"])

save_every = q_learning_config["save_every"] # multiple of these episodes are saved
render_record_every = q_learning_config["render_record_every"] # multiple of these episode will be render
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

# q_table = np.load('curved_1\Q_tables\E12000.npy')
q_table = np.random.uniform(low=-10,high=-1,size=(X_samples,Y_samples,XTE_samples,len(actions)))
print(f"Size of the Q table{q_table.shape}")

# using GMT time to create a unique folder for saving
unique_string = strftime("_%d-%b-%Y-%H-%M-%S", gmtime())
new_folder_name = path_name+unique_string
os.mkdir(new_folder_name) # name of the dir where the files are stored
os.mkdir(os.path.join(new_folder_name,'Q_tables')) # for storing the Q_tables
# writing the Q-learning config used
with open(os.path.join(new_folder_name,'q_learning_config_used.json'),'w') as config:
    json.dump(q_learning_config,config)


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
        
        ## 1-epsilon greedy action
        if random.random() < epsilon :
            random_int = random.randint(0,len(actions)-1) #choose random action
            if random_int <= 0 : action = 0 
            else : action = random_int
        else :
            #choose best action from Q(s,a) values
            action = np.argmax(q_table[dis_state])

        cords,reward,done = env.step(actions[action])
        episode_reward += reward

        new_dis_state = get_discrete(cords)
        if not done :
            max_future_q = np.max(q_table[new_dis_state])
            current_q = q_table[dis_state][action]
            new_q = (1-Learning_rate)*current_q + Learning_rate * (reward + Discount * max_future_q)
            q_table[dis_state][action] = new_q
        elif env.goal_reached :
            q_table[dis_state][action] = reward
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
            print(q_table[dis_state])
            
        
        # stopping the environment if episode time exceeds 2.5 mins
        elif (time() - start_time)/60 > 2.5 :
            env.done = True
            print(f"Episode time exceeded, Episode {episode} was stopped.")

        env.close_quit()
    
    # saving the Q table
    if save :
        np.save(os.path.join(new_folder_name,'Q_tables',f"E{episode}"),q_table)
        # updating the stats consider last 'save_every' episodes
        aggr_episodes_rewards['ep'].append(episode)
        avg_reward = sum(episodes_rewards[-save_every:])/len(episodes_rewards[-save_every:])
        aggr_episodes_rewards['avg'].append(avg_reward)
        aggr_episodes_rewards['min'].append(min(episodes_rewards[-save_every:]))
        aggr_episodes_rewards['max'].append(max(episodes_rewards[-save_every:]))

    # annealing the exploratory rate(epsilon)
    if epsilon > Exploration_rate :
        epsilon -= epsilon_decay_rate

    episodes_rewards.append(episode_reward)

plt.figure(figsize=(12,7))
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
plt.plot(np.arange(Total_episodes+1),np.array(episodes_rewards))
plt.grid()
plt.savefig(os.path.join(new_folder_name,'metrics.png'))
plt.show()
print("<< Done >>")


def play_episode(qtable_filename,fps=60):
    '''
    Input : Relative path of file with '.npy' extension 
    FPS to render the envirnoment,default is 60 FPS
    '''

    q_table = np.load(qtable_filename)
    print(f"Size of the Q table{q_table.shape}")

    path1 = Path()
    env = Env(path1)
    episode_reward = 0
    # reset the envirnoment
    dis_state = get_discrete(env.reset())
    while not env.done :
        
        #choose best action from Q(s,a) values
        action = np.argmax(q_table[dis_state])  

        cords,reward,_ = env.step(actions[action])
        episode_reward += reward
        dis_state = get_discrete(cords)

        env.render_env(FPS_lock=fps,render_stats=True)
        env.close_quit()
    print(f"Episode total reward : {episode_reward}")
