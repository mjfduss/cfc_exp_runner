"""
test_bridge.py

Py Bridge Designer
by Nathan Hartzler
"""
import argparse
import statistics
import cv2
from py_bridge_designer import bridge_env


parser = argparse.ArgumentParser()
parser.add_argument("--render", action="store_true", default=False,
                    help="if toggled, display a rendering of the bridge progress",)
args = parser.parse_args()
render_mode = "rgb_array" if args.render else None

env = bridge_env.BridgeEnv(load_scenario_index=6, render_mode=render_mode)

valid_actions = [
    [0, 0, 8, 16, 0, 0, 18],
    [0, 0, 16, 0, 0, 0, 18],
    [16, 0, 8, 16, 0, 0, 18],
    [16, 0, 24, 16, 0, 0, 18],
    [32, 0, 24, 16, 0, 0, 18],
    [16, 0, 32, 0, 0, 0, 18],
    [32, 0, 48, 0, 0, 0, 18],
    [32, 0, 40, 16, 0, 0, 18],
    [48, 0, 40, 16, 0, 0, 18],
    [48, 0, 64, 0, 0, 0, 18],
    [48, 0, 56, 16, 0, 0, 18],
    [64, 0, 56, 16, 0, 0, 18],
    [64, 0, 80, 0, 0, 0, 18],
    [64, 0, 72, 16, 0, 0, 18],
    [80, 0, 72, 16, 0, 0, 18],
    [56, 16, 72, 16, 0, 0, 18],
    [56, 16, 40, 16, 0, 0, 18],
    [24, 16, 40, 16, 0, 0, 18],
    [24, 16, 8, 16, 0, 0, 18]
]
# Pad the actions like the Observation space
for i in range(len(valid_actions)):
    valid_actions[i][0] += env.bridge.pad_x_action
    valid_actions[i][2] += env.bridge.pad_x_action
    valid_actions[i][1] += env.bridge.pad_y_action
    valid_actions[i][3] += env.bridge.pad_y_action


rewards = []
terminal_reward = 0
terminal_bridge_valid = False
obs = env.reset(load_scenario_index=6)


for action in valid_actions:
    obs, reward, terminated, _, info = env.step(action)
    if args.render:
        cv2.imshow("Bridge Env Image", env.render())
        cv2.waitKey(30)
    rewards.append(reward)
    if terminated:
        terminal_reward = reward
        terminal_bridge_valid = info['bridge_valid']
print(f"~~~~~~~~ Total Steps: {len(rewards)}")
print(f"~~~~~~~~ Mean Step Rewards: {statistics.mean(rewards[:-1])}")
print(f"~~~~~~~~ Terminal Reward: {terminal_reward}")
print(f"~~~~~~~~ Total Rewards: {sum(rewards)}")
print(f"~~~~~~~~ Terminal Bridge Valid: {terminal_bridge_valid}")
if args.render:
    cv2.waitKey(0)
    cv2.destroyAllWindows()
