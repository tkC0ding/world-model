import gymnasium as gym
import numpy as np
import cv2
import os
import json
import argparse

parser = argparse.ArgumentParser(description="Data Generation for CarRacing-v3")
parser.add_argument("--num_episodes", type=int, default=60, help="Number of episodes to generate")
parser.add_argument("--num_time_steps", type=int, default=1000, help="Number of time steps per episode")
parser.add_argument("--data_dir", type=str, default="data", help="Directory to save the generated data")
parser.add_argument("--gym_env", type=str, default="CarRacing-v3", help="Gym environment to use")

args = parser.parse_args()

GYM_ENV = args.gym_env
DATA_DIR = args.data_dir
NUM_EPISODES = args.num_episodes
NUM_TIME_STEPS = args.num_time_steps

os.mkdir(DATA_DIR)

env = gym.make(GYM_ENV, render_mode="rgb_array", max_episode_steps=NUM_TIME_STEPS)

obs, _ = env.reset()

for i in range(NUM_EPISODES):
    data = []
    j = 0
    os.mkdir(os.path.join(DATA_DIR, f"episode_{i}"))
    done = False
    obs, _ = env.reset()
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        frame = obs.copy()
        frame = cv2.resize(frame, (64, 64))

        IMG_PATH = os.path.join(DATA_DIR, f"episode_{i}", f"frame_{j}.jpg")

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(IMG_PATH, frame_bgr)

        data.append({
            "image":IMG_PATH,
            "action":action.tolist()
        })

        if j%250 == 0:
            print(f"Episode: {i}, Time Step: {j}")

        j += 1
        cv2.imshow("CarRacing", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    with open(os.path.join(DATA_DIR, f"episode_{i}", "data.json"), "w") as f:
        json.dump(data, f, indent=4)
    print(f"\nEpisode {i} completed. JSON file saved.\n")

env.close()
cv2.destroyAllWindows()