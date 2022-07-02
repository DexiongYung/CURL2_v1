import torch

def load_actor(agent, ckpt_path: str):
    agent.actor.load_state_dict(torch.load(ckpt_path))
    return agent
