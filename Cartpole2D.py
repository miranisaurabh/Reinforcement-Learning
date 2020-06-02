#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import itertools
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing
from lib import plotting

from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

matplotlib.style.use('ggplot')


# In[124]:


env = gym.make('CartPole-v1')


# In[125]:


env.observation_space.sample()


# In[126]:


observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
# scaler = sklearn.preprocessing.StandardScaler()
# scaler.fit(observation_examples)

featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(observation_examples)
# np.amin(observation_examples,axis=0)


# In[ ]:





# In[138]:


class Function_Approximator():
    
    def __init__(self):
        
        self.models = []
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)
            
    
    def featurize_state(self, state):
        
#         scaled = scaler.transform([state])
        features = featurizer.transform(state.reshape(1,-1))
        print(features[0])
        return features[0]
    
    
    def predict(self, s, a=None):
        
        state_features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([state_features])[0] for m in self.models])
        else:
            return self.models[a].predict([state_features])[0]
        
    def update(self, s, a, y):
       
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])


# In[128]:


def make_epsilon_greedy_policy(estimator, epsilon, nA):
    
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        print(best_action)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


# In[130]:


def sarsa(env, estimator, num_episodes, discount_factor=0.99, epsilon=0.1, epsilon_decay=1.0):
    
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    
    for i_episode in range(num_episodes):
        
        policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
#         if action == 0:
#             action_pge = np.array([1,1])
#         elif action == 1:
#             action_pge = np.array([1,-1])
#         elif action == 2:
#             action_pge = np.array([-1,1])
#         elif action == 3:
#             action_pge = np.array([-1,-1])
#         elif action == 4:
#             action_pge = np.array([1,0])
#         elif action == 5:
#             action_pge = np.array([0,-1])
#         elif action == 6:
#             action_pge = np.array([-1,0])
#         elif action == 7:
#             action_pge = np.array([0,1])
#         action_pge = get_pge_action(action)
            
        for t in itertools.count():
            
            next_state, reward, end, _ = env.step(action)
            env.render()
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            
#             print(f'next_a = {next_action.dtype}')
            
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            q_values_next = estimator.predict(next_state)
            td_target = reward + discount_factor * q_values_next[next_action]
            
            estimator.update(state, action, td_target)
            
            if i_episode % 10 == 0:
                print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, reward))
                
            if end:
                break
                
            state = next_state
            action = next_action
    
    return stats


# In[ ]:


estimator = Function_Approximator()

stats = sarsa(env, estimator, 2000, epsilon=0.1)


# In[140]:


plotting.plot_episode_stats(stats, smoothing_window=25)


# In[30]:


# state = env.observation_space.sample()
state = env.reset()
# plt.figure()
# plt.imshow(env.render(mode='rgb_array'))
while True:
    q_values = estimator.predict(state)
    best_action = np.argmax(q_values)
#     plt.figure()
#     plt.imshow(env.render(mode='rgb_array'))

#     action_pge = get_pge_action(best_action)
    
    next_state, reward, end, _ = env.step(best_action)
    if end:
        break
        
    state = next_state
    env.render()
env.close()


# In[27]:





# In[63]:


env.observation_space.sample()


# In[31]:


env = gym.make('CartPole-v1')


# In[61]:


env.reset()
env.step(env.action_space.sample())


# In[123]:


env.step(env.action_space.sample())


# In[ ]:





# In[ ]:




