import argparse

from mlo import Experiment

parser = argparse.ArgumentParser(description='Execute mlo experiment')
parser.add_argument('--config_path', type=str, help='path of the yaml file')
args = parser.parse_args()

experiment = Experiment(config_path=args.config_path)
experiment.run()