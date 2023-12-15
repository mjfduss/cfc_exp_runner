import argparse
import yaml
import numpy as np
import tensorflow as tf
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from CfC.tf_cfc import CfcCell, MixedCfcCell
from py_bridge_designer.bridge_env import BridgeEnv


"""
Get Command Line Arguments and Parameters
"""

parser = argparse.ArgumentParser()
parser.add_argument('--params_file', type=str, default='params.yml')
parser.add_argument('--params', type=str, default='default')
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--load_scenario', type=int, default=None)
parser.add_argument('--test_print', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()

params_path = args.params_file
print(f"Loading hyperparameters from: {params_path}")

with open(params_path, mode='r') as f:
    params_dict = yaml.load(f, Loader=yaml.FullLoader)
    params = params_dict[args.params]
    cfc_params = params['cfc_params']


"""
Setup the Environment
"""
render_mode = "rgb_array" if args.render else None
load_scenario_index = args.load_scenario
test_print = args.test_print
seed = args.seed if args.seed is not None else np.random.randint(2**32 - 1, dtype="int64").item()
tf.keras.utils.set_random_seed(seed)
print(f"Seed: {seed}")
env = BridgeEnv(render_mode=render_mode, load_scenario_index=load_scenario_index, test_print=test_print)

"""
---TODO: Maybe instead of all this mess, have a gausian distribution network
for the action and then use the CfC as the critic network?

Obs buffer
"""
class ObsBuffer:
    def __init__(self, seq_len: int, shape, dtype):
        self.seq_len = seq_len
        self.shape = shape
        self.dtype = dtype
        self.buffer = []
        self.time = [[i + 1] for i in range(seq_len)]
    def add(self, obs):
        self.buffer = self.buffer[-self.seq_len - 1:] + [obs]
    def add_partition(self):
        partition = np.full(shape=self.shape, fill_value=-1, dtype=self.dtype)
        self.add(partition)
    def get(self):
        return tf.stack(self.buffer), tf.stack(self.time)


obs_buffer = ObsBuffer(params['seq_len'], shape=env.observation_space.shape, dtype=env.observation_space.dtype)

"""     
Fill the initial obs buffer
"""
observation, _ = env.reset(seed=seed, load_scenario_index=load_scenario_index)
for step in range(obs_buffer.seq_len):
    action = env.action_space.sample()
    next_observation, _, terminated, _, _ = env.step(action)
    if terminated:
        obs_buffer.add(next_observation)
        obs_buffer.add_partition()
        observation, _ = env.reset(seed=seed, load_scenario_index=load_scenario_index)
        obs_buffer.add(observation)
    else:
        obs_buffer.add(next_observation)
        observation = next_observation
assert len(obs_buffer.buffer) == len(obs_buffer.time)
"""
Setup the Model
"""
if cfc_params['use_mixed']:
    cell = MixedCfcCell(units=cfc_params['size'], hparams=cfc_params)
else:
    cell = CfcCell(units=cfc_params['size'], hparams=cfc_params)
observations, time = obs_buffer.get()
replay_input = tf.keras.Input(shape=observations.shape, name="replay_input")
time_input = tf.keras.Input(shape=time.shape, name="time_input")
rnn = tf.keras.layers.RNN(cell, time_major=False, return_sequences=True)
output_states = rnn((replay_input, time_input))
y = tf.keras.layers.Dense(env.action_space.shape)(output_states)

model = tf.keras.Model(inputs=[replay_input, time_input], outputs=[y])
"""
Run the experiment
observations = []
actions = []
rewards = []
step_count = 0
episode_count = 0
episode_rewards = []
observation, info = env.reset(seed=seed, load_scenario_index=load_scenario_index)

for step in range(params['num_steps']): 
    action = env.action_space.sample()
    next_observation, reward, terminated, _, info = env.step(action)
    if terminated:
        episode_count += 1
        observation, info = env.reset(seed=seed, load_scenario_index=load_scenario_index)
"""

