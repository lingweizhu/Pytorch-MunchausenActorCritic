import os
import yaml
import argparse
from Munchausen_actorcritic_discrete.env import make_pytorch_env
from Munchausen_actorcritic_discrete.agent import MunchausenACAgent

def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = make_pytorch_env(args.env_id, clip_rewards=False)
    test_env = make_pytorch_env(
        args.env_id, episode_life=False, clip_rewards=False)

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    log_dir = os.path.join('logs', args.env_id, name, str(args.seed))

    # create the agent
    Agent = MunchausenACAgent
    agent = Agent(
        env=env, test_env=test_env, log_dir=log_dir, device=args.device,
        seed=args.seed, **config)
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'sacd.yaml'))
    parser.add_argument('--env_id', type=str, default='MsPacmanNoFrameskip-v4')
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args)
