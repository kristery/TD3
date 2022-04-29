import glob
import json
import os
import random
import subprocess
from datetime import datetime

import numpy as np
import torch

import augmentations


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def cat(x, y, axis=0):
    return torch.cat([x, y], axis=0)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def write_info(args, fp):
    data = {
        "timestamp": str(datetime.now()),
        "git": subprocess.check_output(["git", "describe", "--always"])
        .strip()
        .decode(),
        "args": vars(args),
    }
    with open(fp, "w") as f:
        json.dump(data, f, indent=4, separators=(",", ": "))


def load_config(key=None):
    path = os.path.join("setup", "config.cfg")
    with open(path) as f:
        data = json.load(f)
    if key is not None:
        return data[key]
    return data


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def listdir(dir_path, filetype="jpg", sort=True):
    fpath = os.path.join(dir_path, f"*.{filetype}")
    fpaths = glob.glob(fpath, recursive=True)
    if sort:
        return sorted(fpaths)
    return fpaths


def prefill_memory(obses, capacity, obs_shape):
    """Reserves memory for replay buffer"""
    c, h, w = obs_shape
    for _ in range(capacity):
        frame = np.ones((3, h, w), dtype=np.uint8)
        obses.append(frame)
    return obses


def prefill_memory_feat(obses, capacity, obs_shape):
    l = obs_shape[0]
    for _ in range(capacity):
        frame = np.ones((l,), dtype=np.uint8)
        obses.append(frame)
    return obses


class ReplayFeatBuffer(object):
    """Buffer to store environment transitions"""

    def __init__(
        self, obs_shape, action_shape, capacity, batch_size, prefill=True
    ):
        self.capacity = capacity
        self.batch_size = batch_size

        self._obses = []
        if prefill:
            self._obses = prefill_memory_feat(self._obses, capacity, obs_shape)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        obses = (obs, next_obs)
        if self.idx >= len(self._obses):
            self._obses.append(obses)
        else:
            self._obses[self.idx] = obses
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def _get_idxs(self, n=None):
        if n is None:
            n = self.batch_size
        return np.random.randint(
            0, self.capacity if self.full else self.idx, size=n
        )

    def _get_latest_idxs(self, n=None, ratio=0.1):
        if n is None:
            n = self.batch_size
        low = (
            self.capacity * (1 - ratio)
            if self.full
            else self.idx * (1 - ratio)
        )
        high = self.capacity if self.full else self.idx
        return np.random.randint(int(low), high, size=n)

    def _encode_obses(self, idxs):
        obses, next_obses = [], []
        for i in idxs:
            obs, next_obs = self._obses[i]
            obses.append(np.array(obs, copy=False))
            next_obses.append(np.array(next_obs, copy=False))
        return np.array(obses), np.array(next_obses)

    def sample(self, n=None):
        idxs = self._get_idxs(n)

        obs, next_obs = self._encode_obses(idxs)
        obs = torch.as_tensor(obs).cuda().float()
        next_obs = torch.as_tensor(next_obs).cuda().float()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

        # obs = augmentations.random_crop(obs)
        # next_obs = augmentations.random_crop(next_obs)

        return obs, actions, rewards, next_obs, not_dones

    def sample_latest(self, n=None):
        idxs = self._get_latest_idxs(n)

        obs, next_obs = self._encode_obses(idxs)
        obs = torch.as_tensor(obs).cuda().float()
        next_obs = torch.as_tensor(next_obs).cuda().float()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

        return obs, actions, rewards, next_obs, not_dones


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mean = 0.
        self.std = 1.

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        state = torch.FloatTensor((self.state[ind]-self.mean)/self.std).to(self.device)
        next_state = torch.FloatTensor((self.next_state[ind]-self.mean)/self.std).to(self.device)
        return (
            state,
            torch.FloatTensor(self.action[ind]).to(self.device),
            next_state,
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


    def priority_sample(self, batch_size, ratio=0.1):
        priority_size = int(ratio*self.size)
        ind = np.random.randint(0, priority_size, size=batch_size)
        ind = np.ones(ind.shape) * self.ptr - ind
        ind[ind < 0] = ind[ind < 0] + self.size

        ind = np.random.randint(0, self.size, size=batch_size)

        state = torch.FloatTensor((self.state[ind]-self.mean)/self.std).to(self.device)
        next_state = torch.FloatTensor((self.next_state[ind]-self.mean)/self.std).to(self.device)
        return (
            state,
            torch.FloatTensor(self.action[ind]).to(self.device),
            next_state,
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


    def normalize_states(self, eps=1e-3):
        self.mean = self.state.mean(0, keepdims=True)
        self.std = self.state.std(0, keepdims=True) + eps
        return self.mean, self.std


class LazyFrames(object):
    def __init__(self, frames, extremely_lazy=True):
        self._frames = frames
        self._extremely_lazy = extremely_lazy
        self._out = None

    @property
    def frames(self):
        return self._frames

    def _force(self):
        if self._extremely_lazy:
            return np.concatenate(self._frames, axis=0)
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        if self._extremely_lazy:
            return len(self._frames)
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        if self.extremely_lazy:
            return len(self._frames)
        frames = self._force()
        return frames.shape[0] // 3

    def frame(self, i):
        return self._force()[i * 3 : (i + 1) * 3]


def count_parameters(net, as_int=False):
    """Returns total number of params in a network"""
    count = sum(p.numel() for p in net.parameters())
    if as_int:
        return count
    return f"{count:,}"



def add_tag(args):
    job_tag = f"_{args.job_name}" if args.job_name != '' else ""
    t = f"_fullsamples{args.full_samples}_offlineiters{args.offline_iters}_SI{args.self_imitation}{job_tag}"
    return t