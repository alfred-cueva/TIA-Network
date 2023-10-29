from agent import Agent
from deployment import Deployment
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train', help='train or test')

    args = parser.parse_args()

    # Create the agent with the provided arguments
    if args.phase == 'train':
        agent = Agent()
        agent.train()
    elif args.phase == 'test':
        deployment = Deployment()
        deployment.run()
        

if __name__ == '__main__':
    main()
    