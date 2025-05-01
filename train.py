import re
from collections import defaultdict, deque, Counter
import math
import string
import gymnasium as gym
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import wandb
import csv


def readability_sld(sld: str) -> float:
    """
    A simple readability function for a second-level domain.
    Starts from 100 and deducts penalties for deviation from an optimal length
    and ideal vowel ratio.
    """
    s = sld.lower()
    if len(s) == 0:
        return 0
    score = 100.0
    # Length penalty: deduct 2 points for each character away from 8.
    score -= 2 * abs(len(s) - 8)
    # Vowel ratio penalty.
    vowels = sum(ch in "aeiou" for ch in s)
    ratio = vowels / len(s)
    score -= 50 * abs(ratio - 0.4)
    return max(0, min(100, score))

###############################
# Markov Model (for entropy estimation)
###############################

def tokenize(file_path, tokenizer):
    with open(file_path, mode="r", encoding="utf-8") as file:
        for line in file:
            for token in tokenizer(line.lower().strip()):
                yield token

def chars(file_path):
    return tokenize(file_path, lambda s: s + " ")

def words(file_path):
    return tokenize(file_path, lambda s: re.findall(r"[a-zA-Z']+", s))

def markov_model(stream, model_order):
    model, stats = defaultdict(Counter), Counter()
    circular_buffer = deque(maxlen=model_order)
    for token in stream:
        prefix = tuple(circular_buffer)
        circular_buffer.append(token)
        if len(prefix) == model_order:
            stats[prefix] += 1
            model[prefix][token] += 1
    return model, stats

def entropy(stats, normalization_factor):
    return -sum(proba / normalization_factor * math.log2(proba / normalization_factor)
                for proba in stats.values())

def entropy_rate(model, stats):
    return sum(stats[prefix] * entropy(model[prefix], stats[prefix]) for prefix in stats) / sum(stats.values())

###############################
# Global Alphabet and Tokens
###############################
#ALPHABET = ['#'] + list(string.ascii_lowercase) + list(string.digits) + ['_']
ALPHABET = ['#'] + list(string.ascii_lowercase) + ['_']
TERMINATION_TOKEN_IDX = len(ALPHABET) - 1

###############################
# NGramScorer: A Simple n-gram Markov Model for Name Scoring
###############################
class NGramScorer:
    def __init__(self, n=4):
        self.n = n
        self.ngram_counts = {}     # mapping: context tuple -> dict(next_token -> count)
        self.context_counts = {}   # mapping: context tuple -> total count
        self.vocab_size = len(ALPHABET)
    
    def build_model(self, corpus):
        for sld in corpus:
            sld_str = str(sld)
            if sld_str.lower() == 'nan' or not sld_str:
                continue
            tokens = [ALPHABET.index(ch) for ch in sld_str if ch in ALPHABET]
            padded = [0] * (self.n - 1) + tokens
            for i in range(self.n - 1, len(padded)):
                context = tuple(padded[i - self.n + 1 : i])
                token = padded[i]
                if context not in self.ngram_counts:
                    self.ngram_counts[context] = {}
                self.ngram_counts[context][token] = self.ngram_counts[context].get(token, 0) + 1
                self.context_counts[context] = self.context_counts.get(context, 0) + 1

    def score(self, sequence):
        padded = [0] * (self.n - 1) + sequence
        log_prob = 0.0
        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[i - self.n + 1 : i])
            token = padded[i]
            count = self.ngram_counts.get(context, {}).get(token, 0)
            total = self.context_counts.get(context, 0)
            prob = (count + 1) / (total + self.vocab_size)
            log_prob += np.log(prob)
        return log_prob

###############################
# Trie and Gym Environment for Name Generation
###############################
class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_end_of_word = False
        self.end_count = 0
        self.count = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, count=1):
        node = self.root
        for char in word:
            node = node.children[char]
            node.count += 1
        node.is_end_of_word = True
        node.end_count += count

    def reward(self, sequence):
        node = self.root
        value = 0
        for depth, token in enumerate(sequence):
            if depth == 0 and token == 0:
                continue
            c = ALPHABET[token]
            if c not in node.children:
                return False, 0, depth + 1
            node = node.children[c]
            value += np.power(0.5, len(sequence) - (depth + 1)) * node.count
        if node.is_end_of_word:
            value += 100
        return True, value, len(sequence)

class Environment(gym.Env):
    def __init__(self, max_length=13):
        commonsld_df = pd.read_csv('clean_data/letters_training.csv')
        real_sld_data = dict(zip(commonsld_df['sld'], commonsld_df['occurrence']))

        self.max_length = max_length
        self.observation_space = gym.spaces.Box(
            low=np.array([0] * (self.max_length + 1)),
            high=np.array([27] * self.max_length + [self.max_length]),
            shape=(self.max_length + 1,), dtype=np.int32
        )
        self.action_space = gym.spaces.Discrete(len(ALPHABET))

        self.trie = Trie()
        for sld, count in real_sld_data.items():
            self.trie.insert(str(sld), count)

    def reset(self):
        self.state = [0] * self.max_length + [0]
        return np.array(self.state), {}

    def reward_function(self, state=None, next_state=None, episode=0):
        pointer = state[-1]
        sequence = state[:pointer]
        found, value, depth = self.trie.reward(sequence)
        base_reward = value
        length_factor = np.power(0.8, self.max_length - pointer)
        diversity_bonus = len(set(sequence)) / (len(sequence) + 1)
        curriculum_factor = min(1.0, episode / 5000)
        shaped_reward = length_factor * (base_reward + 10 * curriculum_factor * diversity_bonus)
        name = ''.join([ALPHABET[token] for token in sequence if token != 0])
        readability = readability_sld(name)
        if readability >= 80:
            readability_bonus = 10
        elif readability < 40:
            readability_bonus = -10
        else:
            readability_bonus = 0
        total_reward = shaped_reward + readability_bonus
        wandb.log({"readability": readability})
        return total_reward

    def step(self, action):
        prev_state = list(self.state)
        pointer = self.state[-1]
        if pointer < self.max_length:
            self.state[pointer] = int(action.item()) if hasattr(action, "item") else int(action)
        done = False
        if pointer < self.max_length:
            if self.state[pointer] == TERMINATION_TOKEN_IDX:
                done = True
            else:
                self.state[-1] = pointer + 1
        if self.state[-1] >= self.max_length:
            done = True
        reward = self.reward_function(prev_state)
        return np.array(self.state), reward, done, False, {}

###############################
# Agent and PPO for Name Generation
###############################
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        input_dim = np.array(env.observation_space.shape).prod()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, env.action_space.n), std=0.01),
        )

    def get_value(self, x): return self.critic(x)
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        logits = torch.nan_to_num(logits, nan=-1e9)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

class PPO:
    def __init__(self, env, num_episodes=5000, gamma=0.99, gae_lambda=0.95,
                 num_minibatches=4, clip_coef=0.2, vf_coef=0.5, ent_coef=0.05,
                 norm_adv=True, max_grad_norm=0.5, clip_vloss=True, update_epochs=4,
                 target_kl=None):
        self.env = env
        self.agent = Agent(env)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=3e-4, eps=1e-5)
        self.num_episodes = num_episodes
        self.num_steps = env.max_length
        self.update_epochs = update_epochs
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.batch_size = self.num_steps
        self.minibatch_size = self.batch_size // num_minibatches
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.target_kl = target_kl
        self.norm_adv = norm_adv
        self.clip_vloss = clip_vloss
        self.max_grad_norm = max_grad_norm

    def train(self):
        obs = torch.zeros((self.num_steps, self.env.observation_space.shape[0]))
        actions = torch.zeros((self.num_steps,))
        logprobs = torch.zeros((self.num_steps,))
        rewards = torch.zeros((self.num_steps,))
        dones = torch.zeros((self.num_steps,))
        values = torch.zeros((self.num_steps,))
        avg_backlog = []

        wandb.init(project="PPO-DomainName-Generation", config={
            "num_episodes": self.num_episodes,
            "max_length": self.env.max_length,
            "learning_rate": 3e-4,
            "gamma": self.gamma,
            "clip_coef": self.clip_coef,
            "ent_coef": self.ent_coef,
        })

        for episode in range(1, self.num_episodes + 1):
            next_obs, _ = self.env.reset()
            next_obs = torch.tensor(next_obs, dtype=torch.float32)
            next_done = 0.0
            for step in range(self.num_steps):
                obs[step] = next_obs
                dones[step] = next_done
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                actions[step] = action
                logprobs[step] = logprob
                values[step] = value.flatten()
                next_obs, reward, done, _, _ = self.env.step(action.item())
                rewards[step] = reward
                next_obs = torch.tensor(next_obs, dtype=torch.float32)
                next_done = float(done)

            # Compute advantages & returns
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).flatten()
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values

            # Optimize policy & value network
            b_obs = obs
            b_logprobs = logprobs
            b_actions = actions
            b_advantages = advantages
            b_returns = returns
            b_values = values

            var_y = torch.var(b_returns)
            explained_var = float('nan') if var_y == 0 else 1 - torch.var(b_returns - b_values) / var_y

            inds = np.arange(self.batch_size)
            for epoch in range(self.update_epochs):
                np.random.shuffle(inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    mb_inds = inds[start:start + self.minibatch_size]
                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds].long()
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std(unbiased=False) + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue.view(-1) - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue.view(-1) - b_values[mb_inds], -self.clip_coef, self.clip_coef)
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

            avg_reward = rewards.mean().item()
            wandb.log({"episode": episode, "avg_reward": avg_reward, "explained_variance": explained_var})
            print(f"Episode {episode}/{self.num_episodes}, Avg Reward: {avg_reward:.2f}, Explained Var: {explained_var:.2f}")

        np.save("backlog.npy", np.array(avg_backlog))
        return np.array(avg_backlog)



def load_corpus(csv_filename='clean_data/letters_training.csv'):
    df = pd.read_csv(csv_filename)
    return df['sld'].tolist()

if __name__ == '__main__':
    NUM_EPISODES = 10000
    env = Environment(max_length=13)
    ppo_agent = PPO(env, num_episodes=NUM_EPISODES)
    _ = ppo_agent.train()

    # Save trained agent weights
    torch.save(ppo_agent.agent.state_dict(), 'ppo_agent.pth')
    print("Saved trained agent to ppo_agent.pth")
