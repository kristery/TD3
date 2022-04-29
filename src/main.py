import numpy as np
import torch
#import gym
import argparse
import os

import utils
import TD3
import OurDDPG
import DDPG
from env.wrappers import make_feat_env


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(env, policy, env_name, seed, eval_episodes=10, mean=0, std=1, offline=False):
    #eval_env = gym.make(env_name)
    eval_env = env
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state), offline=offline)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    if offline:
        print(f"Offline Policy Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    else:
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name


    # dmc parameters
    parser.add_argument("--domain_name", default='quadruped')
    parser.add_argument("--task_name", default='run')
    parser.add_argument("--episode_length", default=800, type=int)
    parser.add_argument("--action_repeat", default=4, type=int)
    parser.add_argument("--image_size", default=84)
    parser.add_argument("--job_name", default="", type=str)
    parser.add_argument("--offline_iters", default=10, type=int)
    parser.add_argument("--full_samples", action="store_true")
    # parser.add_argument("--priority_samples", action="store_true")
    parser.add_argument("--self_imitation", action="store_true")

    args = parser.parse_args()

    file_name = f"{args.policy}_{args.domain_name}_{args.task_name}{utils.add_tag(args)}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.domain_name}_{args.task_name}, Seed: {args.seed}")
    print(f"Filename: {file_name}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    #env = gym.make(args.env)

    env = make_feat_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode="train",
    )

    # Set seeds
    #env.seed(args.seed)
    #env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
    print(f"dimensionality: {state_dim}, {action_dim}")

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    
    # Evaluate untrained policy
    evaluations = [eval_policy(env, policy, args.env, args.seed)]
    offline_evaluations = [evaluations[0]]
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    state_mean = 0.
    state_std = 1.

    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action).astype(np.float32)

        # Perform action
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            if t == args.start_timesteps:
                state_mean, state_std = replay_buffer.normalize_states()
            if args.self_imitation and t % 2 == 1:
                policy.train(replay_buffer, args.batch_size, self_imitation=args.self_imitation)
            else:
                policy.train(replay_buffer, args.batch_size)

        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(env, policy, args.env, args.seed, mean=state_mean, std=state_std))
            # offline training and evaluation
            if t >= args.start_timesteps:
                policy.offline_train(replay_buffer, args.batch_size, full_samples=args.full_samples, iters=args.offline_iters)
                offline_evaluations.append(eval_policy(env, policy, args.env, args.seed, mean=state_mean, std=state_std, offline=True))
            else:
                offline_evaluations.append(evaluations[-1])
            np.save(f"./results/{file_name}", evaluations)
            np.save(f"./results/{file_name}_offline", offline_evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")
