import gym
import tensorflow as tf
from Actor_critic import Actor, Critic

OUTPUT_GRAPH = False
NUM_EPISODES = 3000
RENDER = False
DISPLAY_REWARD_THRESHOLD = 200

if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    env.seed(1)
    #env = env.unwrapped

    sess = tf.Session()
    actor = Actor(sess, env.action_space.n, env.observation_space.shape[0])
    critic = Critic(sess, env.observation_space.shape[0])
    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)


    for i_episode in range(NUM_EPISODES):
        s = env.reset()
        t = 0
        track_r = []

        while True:
            if RENDER: env.render()

            a = actor.choose_action(s)
            s_, r, done, info = env.step(a)

            if done: r = -20
            track_r.append(r)

            td_error = critic.learn(s, r, s_)
            loss = actor.learn(s, a, td_error)

            s = s_
            t += 1

            if done:
                ep_rs_sum = sum(track_r)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
                print("episode:", i_episode, "  reward:", running_reward)
                break