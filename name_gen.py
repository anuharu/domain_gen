import string
from collections import defaultdict
from g2p_en import G2p
import nltk
# Download and prepare NLTK data for G2P POS tagging
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
import os, shutil
from nltk.data import find
# Ensure resource directory matches g2p_en expectation (suffix "_eng")
try:
    find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    src_dir = find('taggers/averaged_perceptron_tagger')
    parent = os.path.dirname(src_dir)
    eng_dir = os.path.join(parent, 'averaged_perceptron_tagger_eng')
    if not os.path.exists(eng_dir):
        shutil.copytree(src_dir, eng_dir)

import gymnasium as gym
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
import wandb
import csv

# Initialize G2P model
g2p = G2p()

def pronounceability_sld(sld: str) -> float:
    """
    Score a second-level domain by how cleanly it maps to phonemes.
    (# phonemes / # characters) * 100, clipped to [0, 100].
    Falls back to vowel ratio if G2P fails.
    """
    # compute phoneme-based ratio, with fallback
    if len(sld) == 0:
        return 0.0
    try:
        phoneme_seq = g2p(sld.lower())
        phonemes = [ph for ph in phoneme_seq if ph.strip()]
        ratio = len(phonemes) / len(sld)
    except Exception:
        vowels = sum(ch in "aeiou" for ch in sld.lower())
        ratio = vowels / len(sld)
    score = 100 * ratio
    # clip to [0, 100]
    if score < 0:
        score = 0.0
    elif score > 100:
        score = 100.0
    return score


# Global alphabet and termination token
ALPHABET = ['#'] + list(string.ascii_lowercase) + ['_']
TERMINATION_TOKEN_IDX = len(ALPHABET) - 1

class NGramScorer:
    def __init__(self, n=4):
        self.n = n
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocab_size = len(ALPHABET)

    def build_model(self, corpus):
        for sld in corpus:
            s = str(sld)
            if not s or s.lower() == 'nan': continue
            tokens = [ALPHABET.index(ch) for ch in s if ch in ALPHABET]
            padded = [0] * (self.n-1) + tokens
            for i in range(self.n-1, len(padded)):
                ctx = tuple(padded[i-self.n+1:i])
                tok = padded[i]
                self.ngram_counts.setdefault(ctx, {})[tok] = \
                    self.ngram_counts.get(ctx, {}).get(tok, 0) + 1
                self.context_counts[ctx] = self.context_counts.get(ctx, 0) + 1

    def score(self, sequence):
        padded = [0] * (self.n-1) + sequence
        lp = 0.0
        for i in range(self.n-1, len(padded)):
            ctx = tuple(padded[i-self.n+1:i])
            tok = padded[i]
            cnt = self.ngram_counts.get(ctx, {}).get(tok, 0)
            total = self.context_counts.get(ctx, 0)
            prob = (cnt+1)/(total+self.vocab_size)
            lp += np.log(prob)
        return lp

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_end_of_word = False
        self.count = 0

class Trie:
    def __init__(self): self.root = TrieNode()
    def insert(self, w, c=1):
        node = self.root
        for ch in w:
            node = node.children[ch]; node.count += c
        node.is_end_of_word = True
    def reward(self, seq):
        node = self.root; val = 0
        for d,t in enumerate(seq):
            if d==0 and t==0: continue
            ch = ALPHABET[t]
            if ch not in node.children: return False,0,d+1
            node = node.children[ch]
            val += (0.5**(len(seq)-(d+1))) * node.count
        if node.is_end_of_word: val += 100
        return True,val,len(seq)

class Environment(gym.Env):
    def __init__(self, max_length=12):
        df = pd.read_csv('clean_data/letters_training.csv')
        real = dict(zip(df['sld'],df['occurrence']))
        self.max_length = max_length
        self.observation_space = gym.spaces.Box(
            low=np.zeros(self.max_length+1,dtype=np.int32),
            high=np.array([len(ALPHABET)]*self.max_length+[self.max_length],dtype=np.int32),
            shape=(self.max_length+1,),dtype=np.int32)
        self.action_space = gym.spaces.Discrete(len(ALPHABET))
        self.trie = Trie()
        for w,c in real.items(): self.trie.insert(str(w),c)

    def reset(self): self.state=[0]*self.max_length+[0]; return self.state,{}

    def reward_function(self, state=None, next_state=None, episode=0):
        ptr=state[-1]; seq=state[:ptr]
        _,base,_ = self.trie.reward(seq)
        lf=(0.8**(self.max_length-ptr))
        db=len(set(seq))/(len(seq)+1)
        cf=min(1.0,episode/5000)
        shaped = lf*(base+10*cf*db)
        name=''.join(ALPHABET[t] for t in seq if t)
        pron=pronounceability_sld(name)
        bonus=10 if pron>=80 else -10 if pron<40 else 0
        wandb.log({'pronounceability':pron})
        return shaped+bonus

    def step(self, action):
        prev=self.state.copy(); ptr=prev[-1];
        if ptr<self.max_length:
            self.state[ptr]=action.item() if hasattr(action,'item') else action
        done=False
        if ptr<self.max_length:
            if self.state[ptr]==TERMINATION_TOKEN_IDX: done=True
            else: self.state[-1]=ptr+1
        if self.state[-1]>=self.max_length: done=True
        r=self.reward_function(prev)
        return self.state,r,done,False,{}

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
            layer_init(nn.Linear(input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        logits = torch.nan_to_num(logits, nan=-1e9)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

class PPO:
    def __init__(self,
                 env,
                 num_episodes=5000,
                 gamma=0.99,
                 gae_lambda=0.95,
                 num_minibatches=4,
                 clip_coef=0.2,
                 vf_coef=0.5,
                 ent_coef=0.05,  # Increased to encourage exploration.
                 norm_adv=True,
                 max_grad_norm=0.5,
                 clip_vloss=True,
                 update_epochs=4,
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
        self.num_envs = 1
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.target_kl = target_kl
        self.norm_adv = norm_adv
        self.clip_vloss = clip_vloss
        self.max_grad_norm = max_grad_norm

    def train(self):
        obs = torch.zeros((self.num_steps, self.num_envs) + self.env.observation_space.shape)
        actions = torch.zeros((self.num_steps, self.num_envs) + self.env.action_space.shape)
        logprobs = torch.zeros((self.num_steps, self.num_envs))
        rewards = torch.zeros((self.num_steps, self.num_envs))
        dones = torch.zeros((self.num_steps, self.num_envs))
        values = torch.zeros((self.num_steps, self.num_envs))

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
            next_done = torch.tensor([0.0], dtype=torch.float32)

            for step in range(self.num_steps):
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, terminations, truncations, infos = self.env.step(action.cpu().numpy())
                next_done_np = np.logical_or(terminations, truncations)
                rewards[step] = reward
                next_obs = torch.tensor(next_obs, dtype=torch.float32)
                next_done = torch.tensor([float(next_done_np)], dtype=torch.float32)

            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
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

            b_obs = obs.reshape((-1,) + self.env.observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.env.action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            y_pred = b_values.cpu().numpy()
            y_true = b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            b_inds = np.arange(self.batch_size)
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
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

                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

            avg_reward = rewards.mean().item()
            avg_backlog.append(avg_reward)
            print(f"Episode {episode}/{self.num_episodes}, Avg Reward: {avg_reward:.2f}, Explained Var: {explained_var:.2f}")
            
            wandb.log({
                "episode": episode,
                "avg_reward": avg_reward,
                "explained_variance": explained_var,
            })

        np.save("backlog.npy", np.array(avg_backlog))
        return np.array(avg_backlog)

    # Beam search evaluation with variable-length outputs.
    # New parameters:
    #   min_length: minimum length before termination can be forced.
    #   termination_prob: probability of forcing termination once min_length is reached.
    def beam_search_eval(self, test_env, beam_size=25, top_k=20, ngram_scorer=None, markov_weight=0.35,
                         temperature=2.3, diversity_weight=1.2, repetition_penalty_weight=1.3, 
                         samples_per_beam=3, min_length=3, termination_prob=0.25):
        init_state, _ = test_env.reset()
        beam = [(init_state.copy(), 0.0)]
        for t in range(test_env.max_length):
            new_beam = []
            for state, cum_score in beam:
                pointer = state[-1]
                if pointer >= test_env.max_length:
                    new_beam.append((state, cum_score))
                    continue
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                logits = self.agent.actor(state_tensor) / temperature
                logits = logits - logits.max(dim=-1, keepdim=True)[0]
                logits = torch.nan_to_num(logits, nan=-1e9)
                log_probs = torch.log_softmax(logits, dim=-1)[0]
                topk_log_probs, topk_indices = torch.topk(log_probs, top_k)
                for _ in range(samples_per_beam):
                    # If sequence is long enough, force termination with some probability.
                    if pointer >= min_length and np.random.rand() < termination_prob:
                        action = TERMINATION_TOKEN_IDX
                        chosen_log_prob = 0.0  # Adjust bonus as needed.
                    else:
                        sampled_idx = torch.multinomial(torch.softmax(topk_log_probs, dim=0), num_samples=1).item()
                        action = topk_indices[sampled_idx].item()
                        chosen_log_prob = topk_log_probs[sampled_idx].item()
                    new_state = state.copy()
                    # If termination token is chosen, do not increment pointer.
                    if action == TERMINATION_TOKEN_IDX:
                        new_state[-1] = pointer
                    else:
                        new_state[-1] = pointer + 1
                    new_state[pointer] = action
                    prefix = new_state[:new_state[-1]]
                    diversity_bonus = diversity_weight * len(set(prefix))
                    rep_penalty = sum([repetition_penalty_weight for j in range(1, len(prefix)) if prefix[j] == prefix[j-1]])
                    score = cum_score + chosen_log_prob + diversity_bonus - rep_penalty
                    if ngram_scorer is not None:
                        candidate_sequence = new_state[:new_state[-1]]
                        raw_markov_score = ngram_scorer.score(candidate_sequence)
                        norm_markov = raw_markov_score / (len(candidate_sequence) if len(candidate_sequence) > 0 else 1)
                        score += markov_weight * norm_markov
                    new_beam.append((new_state, score))
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_size]
        beam.sort(key=lambda x: x[1], reverse=True)
        return beam

    def eval(self, test_env):
        done = False
        next_obs, _ = test_env.reset()
        while not done:
            with torch.no_grad():
                action, _, _, _ = self.agent.get_action_and_value(torch.tensor(next_obs, dtype=torch.float32))
            next_obs, reward, done, _, info = self.env.step(action)
        return next_obs

###############################
# Main Script: Training and Generating Names
###############################
def load_corpus(csv_filename='clean_data/letters_training.csv'):
    df = pd.read_csv(csv_filename)
    return df['sld'].tolist()

if __name__ == '__main__':
    NUM_EPISODES = 10000
    env = Environment(max_length=12)  
    ppo_agent = PPO(env, num_episodes=NUM_EPISODES)
    avg_backlog = ppo_agent.train()

    corpus = load_corpus()
    ngram_scorer = NGramScorer(n=4)
    ngram_scorer.build_model(corpus)

    test_env = Environment(max_length=12)
    print("\nBeam Search Evaluations (with n-gram Markov scoring):")
    beams = ppo_agent.beam_search_eval(
        test_env,
        beam_size=25,
        top_k=20,
        ngram_scorer=ngram_scorer,
        markov_weight=0.35,
        temperature=2.3,
        diversity_weight=1.2,
        repetition_penalty_weight=1.5,
        samples_per_beam=3,
        min_length=3,
        termination_prob=0.25
    )
    best_state, best_score = beams[0]
    # Use the pointer (best_state[-1]) to slice the sequence for variable-length names.
    name = ''.join([ALPHABET[token] for token in best_state[:best_state[-1]] if token != 0])
    print(f"Best beam: {name} (score: {best_score:.2f})")
    
    print("\nGenerated Names:")
    generated_names = []
    csv_filename = "generated_names_4.csv"
    
    for i in range(100000):
        beams = ppo_agent.beam_search_eval(
            test_env,
            beam_size=25,
            top_k=20,
            ngram_scorer=ngram_scorer,
            markov_weight=0.35,
            temperature=2.3,
            diversity_weight=1.2,
            repetition_penalty_weight=1.5,
            samples_per_beam=3,
            min_length=3,
            termination_prob=0.25
        )
        best_state, best_score = beams[0]
        name = ''.join([ALPHABET[token] for token in best_state[:best_state[-1]] if token != 0])
        generated_names.append(name)
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["SLD"])  # Write header.
            for name in generated_names:
                writer.writerow([name])