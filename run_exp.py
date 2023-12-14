import argparse
import yaml
import tensorflow as tf
from CfC.tf_cfc import CfcCell
from py_bridge_designer.bridge_env import BridgeEnv

parser = argparse.ArgumentParser()
parser.add_argument('--params_file', type=str, default='params.yml')
parser.add_argument('--params', type=str, default='default')
args = parser.parse_args()

params_path = args.params_file
print(f"Loading hyperparameters from: {params_path}")

with open(params_path, mode='r') as f:
    params_dict = yaml.load(f, Loader=yaml.FullLoader)
    params = params_dict[args.params]
    cfc_params = params['cfc_params']
print(cfc_params["decay_lr"])
