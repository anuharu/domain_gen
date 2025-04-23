import torch
import pandas as pd
from train import Environment, Agent, PPO, NGramScorer, ALPHABET, TERMINATION_TOKEN_IDX

# 1. Reconstruct the environment
env = Environment(max_length=13)

# 2. Rebuild the Agent architecture and load weights
agent = Agent(env)
state_dict = torch.load('ppo_agent.pth', map_location='cpu')
agent.load_state_dict(state_dict)
agent.eval()

# 3. Build n-gram scorer
commonsld = pd.read_csv('clean_data/popularsld.csv')['sld'].tolist()
ngram_scorer = NGramScorer(n=4)
ngram_scorer.build_model(commonsld)

# 4. Simple wrapper for beam search
class PPOBeamWrapper:
    def __init__(self, agent):
        self.agent = agent

    def beam_search_eval(self, test_env, **kwargs):
        # reuse PPO's implementation
        ppo = PPO(env=None)  # env not used here
        ppo.agent = self.agent
        return ppo.beam_search_eval(test_env, **kwargs)

wrapper = PPOBeamWrapper(agent)

test_env = Environment(max_length=13)
beams = wrapper.beam_search_eval(
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
    termination_prob=0.25,
)

# Print top 10
for i, (state, score) in enumerate(beams[:10], 1):
    name = ''.join([ALPHABET[t] for t in state[:state[-1]] if t != 0])
    print(f"{i:2d}. {name:<12} (score={score:.2f})")
