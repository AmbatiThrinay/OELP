# OELP
## Control Designs for Autonomous Vehicles based on Reinforcement Learning

Developing control designs for autonomous vehicles
based on reinforcement learning. Control designs include discrete
and continuous action space based designs with emphasis on
learning from interaction with an environment in order to achieve
goals instead of handcrafted control strategies. Control Designs
have to be adaptable to the environment. Training and testing
simulations are to be performed in python.

---------------
`requirements.txt` can be used to create a pip virtual environment
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
## Path Editor

<p align="center">
<img src="images\path-editor-good.gif" alt="Path Editor demo png" title="path editor demo">
</p>
<p align="center">Path Editor Demo</p>

## Q-learning

<p align="center">
<img src="images\q-learning-straight.gif" alt="Q-learning agent on straight path gif" title="Q-learning agent">
</p>
<p align="center">Q-learning agent on straight track</p>

<p align="center">
<img src="images\q-learning-curved.gif" alt="Q-learning agent on straight path gif" title="Q-learning agent">
</p>
<p align="center">Q-learning agent on curved track</p>

## Deep Q-learning

<p align="center">
<img src="images\dqn-straight.gif" alt="DQN agent on straight path gif" title="DQN agent">
</p>
<p align="center">Deep Q-learning agent on straight track</p>

<p align="center">
<img src="images\dqn-curved.gif" alt="DQN agent on straight path gif" title="DQN agent">
</p>
<p align="center">Deep Q-learning agent on curved track</p>

## Results

<table>
    <tr>
        <p align="center">
        <img src="images/q_learning_straight.png" alt="Q-learning agent on straight track png" title="Q-learning agent">
        </p>
        <p align="center">
        Q-learning agent on straight track
        </p>
    </tr> 
    <tr>
        <p align="center">
        <img src="images/dqn_straight.png" alt="DQN-learning agent on straight track png" title="DQN agent">
        </p>
        <p align="center">Deep Q-learning agent on straight track</p>
    </tr>
    <tr>
        <p align="center">
        <img src="images/dqn_curved.png" alt="DQN-learning agent on straight track png" title="DQN agent">
        </p>
        <p align="center">Deep Q-learning agent on curved track</p">
    </tr>
</table>

[Final Report](report.pdf)