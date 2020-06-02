#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pypge
import gym
import itertools
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing
from lib import plotting
import math

from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

matplotlib.style.use('ggplot')


# In[2]:


env = gym.make('CartPole3D-v0')


# In[3]:


env.observation_space.sample()


# In[4]:


n_x = 31
n_z = 31
n_A = n_x*n_z


# In[5]:


observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))


# In[6]:


class Function_Approximator():
    
    def __init__(self):
        
        self.models = []
        for i in range(n_A):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)
            
    
    def featurize_state(self, state):
        
        scaled = scaler.transform([state])
        features = featurizer.transform(scaled)
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


# def make_epsilon_greedy_policy(estimator, epsilon, nA):
#     
#     def policy_fn(observation):
#         A = np.ones(nA, dtype=float) * epsilon / nA
#         q_values = estimator.predict(observation)
#         best_action = np.argmax(q_values)
#         print(best_action)
#         A[best_action] += (1.0 - epsilon)
#         return A
#     return policy_fn

# In[7]:


def get_action(observation,t):
    
    if np.random.random()<max(0.05, min(0.5, 1.0 - math.log10((t+1)/150.))):
        return np.random.randint(0,n_A)
    
    q_values = estimator.predict(observation)
    best_action = np.argmax(q_values)
    
    return best_action
    
    


# In[8]:


a_x = np.linspace(-1,1,n_x)
a_z = np.linspace(-1,1,n_z)
x,z = np.meshgrid(a_x,a_z)
x = x.reshape(-1,1)
z = z.reshape(-1,1)
xz = np.hstack([x,z])


# In[9]:


def get_pge_action(action):
    
    return xz[action]    


# In[10]:


def sarsa(env, estimator, num_episodes, discount_factor=0.95, epsilon=0.1, epsilon_decay=1.0):
    
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    
    for i_episode in range(num_episodes):
        
        state = env.reset()

        for t in itertools.count():
            
            action = get_action(state,i_episode)
            action_pge = get_pge_action(action)
            
            next_state, reward, end, _ = env.step(action_pge)
            q_values_this = estimator.predict(state)
            
            if end:
                reward = -1


            next_action = get_action(next_state,i_episode)
            
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            q_values_next = estimator.predict(next_state)
            td_target = reward + discount_factor * q_values_next[next_action]
            y_value = q_values_this[action] + epsilon*(td_target-q_values_this[action])
            estimator.update(state, action, y_value)
            
            if i_episode % 10 == 0:
                print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, reward))
                
            if end:
                break
                
            state = next_state
            action = next_action
            
#         print('----------')
#         print(q_values_this)
    return stats


# In[ ]:


estimator = Function_Approximator()

stats = sarsa(env, estimator, 6000, epsilon=0.1)


# In[11]:


plotting.plot_episode_stats(stats, smoothing_window=25)


# In[26]:


state = env.observation_space.sample()
# state = env.reset()
# plt.figure()
# plt.imshow(env.render(mode='rgb_array'))
for count in range(100):
    q_values = estimator.predict(state)
    best_action = np.argmax(q_values)
#     plt.figure()
#     plt.imshow(env.render(mode='rgb_array'))

    action_pge = get_pge_action(best_action)
    
    next_state, reward, end, _ = env.step(action_pge)
    if end:
        break
        
    state = next_state
    env.render(close=True)
env.render(close=True)


# In[25]:


count


# In[63]:


env.observation_space.sample()


# In[13]:


observation = env.observation_space.sample()


# In[15]:



estimator.predict(observation).shape


# In[ ]:




