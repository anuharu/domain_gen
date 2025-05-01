import torch
import pandas as pd
from train import Environment, Agent, PPO, NGramScorer, ALPHABET, TERMINATION_TOKEN_IDX

# 1. Reconstruct the environment
env = Environment(max_length=13)

# 2. Rebuild the Agent architecture and load weights
agent = Agent(env)
state_dict = torch.load('ppo_agent.pth', map_location='cpu', weights_only=True)
agent.load_state_dict(state_dict)
agent.eval()

# 3. Build n-gram scorer
commonsld = pd.read_csv('clean_data/letters_training.csv')['sld'].tolist()
ngram_scorer = NGramScorer(n=4)
ngram_scorer.build_model(commonsld)

# 4. Modified wrapper for beam search
class PPOBeamWrapper:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        # Initialize PPO with the environment
        self.ppo = PPO(env)
        self.ppo.agent = agent

    def beam_search_eval(self, **kwargs):
        return self.ppo.beam_search_eval(self.env, **kwargs)

# Create wrapper with proper environment
wrapper = PPOBeamWrapper(agent, env)

# Parameters for generation
total_names = 100000
batch_size = 1000  # Generate in batches to manage memory
names = []
scores = []

while len(names) < total_names:
    beams = wrapper.beam_search_eval(
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
    
    for state, score in beams:
        name = ''.join([ALPHABET[t] for t in state[:state[-1]] if t != 0])
        names.append(name)
        scores.append(score)
        
    print(f"Generated {len(names)} names so far...")
    
    # Clear memory
    torch.cuda.empty_cache()

# Trim to exactly 100,000 if we went over
names = names[:total_names]
scores = scores[:total_names]

# Create DataFrame and save to CSV
df = pd.DataFrame({'name': names, 'score': scores})
df.to_csv('generated_names_100k.csv', index=False)

print(f"Successfully generated and saved {len(df)} names to 'generated_names_100k.csv'")