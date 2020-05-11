import gym
from Reinforce import PolicyGradient
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")
env.seed(1)
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(n_actions=env.action_space.n, n_features=env.observation_space.shape[0], learning_rate=0.02, reward_decay=0.99)

for i_episode in range(3000):
    observation = env.reset()
    while True:
        env.render()
        action = RL.choose_action(observation)
        
        observation_, reward, done, info = env.step(action)
        RL.store_transition(observation, action, reward)
        
        if done:
            vt = RL.learn()
            
            if i_episode == 0:
                plt.plot(vt)  # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break
            
        observation = observation_