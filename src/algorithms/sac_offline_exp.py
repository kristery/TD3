# method description: it is basically a normal sac but at the same time use the buffer to train offline algorithm

from copy import deepcopy
from tkinter import E

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

import algorithms.modules as m


class SAC_OFFLINE_EXP(object):
    def __init__(self, obs_shape, action_shape, args):
        self.discount = args.discount
        self.critic_tau = args.critic_tau
        self.encoder_tau = args.encoder_tau
        self.actor_update_freq = args.actor_update_freq
        self.critic_target_update_freq = args.critic_target_update_freq

        self.full_sampling = args.full_sampling

        self.lam = args.lam
        """
                shared_cnn = m.SharedCNN(obs_shape, args.num_shared_layers, args.num_filters).cuda()
                head_cnn = m.HeadCNN(shared_cnn.out_shape, args.num_head_layers, args.num_filters).cuda()
                actor_encoder = m.Encoder(
                        shared_cnn,
                        head_cnn,
                        m.RLProjection(head_cnn.out_shape, args.projection_dim)
                )
                critic_encoder = m.Encoder(
                        shared_cnn,
                        head_cnn,
                        m.RLProjection(head_cnn.out_shape, args.projection_dim)
                )
                """

        actor_encoder = m.featEncoder(
            m.RLProjection(obs_shape, args.projection_dim)
        )
        critic_encoder = m.featEncoder(
            m.RLProjection(obs_shape, args.projection_dim)
        )

        self.actor = m.Actor(
            actor_encoder,
            action_shape,
            args.hidden_dim,
            args.actor_log_std_min,
            args.actor_log_std_max,
        ).cuda()
        self.critic = m.Critic(
            critic_encoder, action_shape, args.hidden_dim
        ).cuda()

        self.exp_actor = deepcopy(self.actor)
        # self.exp_critic = deepcopy(self.critic)
        # self.exp_critic_target = deepcopy(self.critic)

        self.critic_target = deepcopy(self.critic)

        self.log_alpha = torch.tensor(np.log(args.init_temperature)).cuda()
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape)

        # self.exp_log_alpha = deepcopy(self.log_alpha)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=args.actor_lr,
            betas=(args.actor_beta, 0.999),
        )

        # self.exp_actor_optimizer = torch.optim.Adam(
        #     self.exp_actor.parameters(),
        #     lr=args.actor_lr,
        #     betas=(args.actor_beta, 0.999),
        # )

        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.alpha_lr = args.alpha_lr
        self.actor_beta = args.actor_beta
        self.critic_beta = args.critic_beta
        self.alpha_beta = args.alpha_beta
        self.iters = args.iters

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=args.critic_lr,
            betas=(args.critic_beta, 0.999),
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999)
        )

        # self.exp_critic_optimizer = torch.optim.Adam(
        #     self.exp_critic.parameters(),
        #     lr=args.critic_lr,
        #     betas=(args.critic_beta, 0.999),
        # )
        # self.exp_log_alpha_optimizer = torch.optim.Adam(
        #     [self.exp_log_alpha],
        #     lr=args.alpha_lr,
        #     betas=(args.alpha_beta, 0.999),
        # )

        self.train()
        self.critic_target.train()

    def copy_params(self):
        self.exp_actor = deepcopy(self.actor)
        self.exp_critic = deepcopy(self.critic)
        self.exp_critic_target = deepcopy(self.critic)
        self.exp_log_alpha = deepcopy(self.log_alpha)

        self.exp_actor_optimizer = torch.optim.Adam(
            self.exp_actor.parameters(),
            lr=self.actor_lr,
            betas=(self.actor_beta, 0.999),
        )
        self.exp_critic_optimizer = torch.optim.Adam(
            self.exp_critic.parameters(),
            lr=self.critic_lr,
            betas=(self.critic_beta, 0.999),
        )
        self.exp_log_alpha_optimizer = torch.optim.Adam(
            [self.exp_log_alpha],
            lr=self.alpha_lr,
            betas=(self.alpha_beta, 0.999),
        )

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        # self.exp_actor_optimizer = torch.optim.Adam(
        #     self.exp_actor.parameters(),
        #     lr=args.actor_lr,
        #     betas=(args.actor_beta, 0.999),
        # )

    def eval(self):
        self.train(False)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def exp_alpha(self):
        return self.exp_log_alpha.exp()

    def _obs_to_input(self, obs):
        if isinstance(obs, utils.LazyFrames):
            _obs = np.array(obs)
        else:
            _obs = obs
        _obs = torch.FloatTensor(_obs).cuda()
        _obs = _obs.unsqueeze(0)
        return _obs

    def select_action(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            mu, _, _, _ = self.actor(
                _obs, compute_pi=False, compute_log_pi=False
            )
        return mu.cpu().data.numpy().flatten()

    def exp_select_action(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            mu, _, _, _ = self.exp_actor(
                _obs, compute_pi=False, compute_log_pi=False
            )
        return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            mu, pi, _, _ = self.actor(_obs, compute_log_pi=False)
            # mu, pi, _, _ = self.exp_actor(_obs, compute_log_pi=False)
        return pi.cpu().data.numpy().flatten()

    def update_critic(
        self, obs, action, reward, next_obs, not_done, L=None, step=None
    ):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = (
                torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            )
            target_Q = reward + (not_done * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        if L is not None:
            L.log("train_critic/loss", critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_exp_critic(
        self, obs, action, reward, next_obs, not_done, L=None, step=None
    ):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.exp_actor(next_obs)
            target_Q1, target_Q2 = self.exp_critic_target(
                next_obs, policy_action
            )
            target_V = (
                torch.min(target_Q1, target_Q2)
                - self.exp_alpha.detach() * log_pi
            )
            target_Q = reward + (not_done * self.discount * target_V)

        current_Q1, current_Q2 = self.exp_critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        self.exp_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.exp_critic_optimizer.step()

    def update_actor_and_alpha(
        self, obs, L=None, step=None, update_alpha=True
    ):
        _, pi, log_pi, log_std = self.actor(obs, detach=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if L is not None:
            L.log("train_actor/loss", actor_loss, step)
            entropy = 0.5 * log_std.shape[1] * (
                1.0 + np.log(2 * np.pi)
            ) + log_std.sum(dim=-1)

        if update_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (
                self.alpha * (-log_pi - self.target_entropy).detach()
            ).mean()

            if L is not None:
                L.log("train_alpha/loss", alpha_loss, step)
                L.log("train_alpha/value", self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update_exp_actor_and_alpha(
        self, obs, latest_samples, L=None, step=None, update_alpha=True
    ):
        latest_obs = latest_samples[0]
        latest_action = latest_samples[1]

        for _ in range(self.iters):
            _, pi, log_pi, log_std = self.exp_actor(obs, detach=True)
            actor_Q1, actor_Q2 = self.exp_critic(obs, pi, detach=True)

            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (self.exp_alpha.detach() * log_pi - actor_Q).mean()

            _, latest_pi, latest_log_pi, latest_log_std = self.exp_actor(
                latest_obs, detach=True
            )
            bc_loss = ((latest_pi - latest_action) ** 2).mean()

            weight = self.lam / actor_Q.detach().abs().mean()
            loss = weight * actor_loss + bc_loss

            self.exp_actor_optimizer.zero_grad()
            loss.backward()
            self.exp_actor_optimizer.step()

        if update_alpha:
            self.exp_log_alpha_optimizer.zero_grad()
            alpha_loss = (
                self.exp_alpha * (-log_pi - self.target_entropy).detach()
            ).mean()

            alpha_loss.backward()
            self.exp_log_alpha_optimizer.step()

    def update_exp_actor(self, obs):
        self.exp_actor = deepcopy(self.actor)
        self.exp_actor_optimizer = torch.optim.Adam(
            self.exp_actor.parameters(),
            lr=self.actor_lr,
            betas=(self.actor_beta, 0.999),
        )
        self.exp_actor_optimizer.load_state_dict(
            self.actor_optimizer.state_dict()
        )
        _, pi, log_pi, log_std = self.exp_actor(obs, detach=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        self.exp_actor_optimizer.zero_grad()
        actor_loss.backward()
        self.exp_actor_optimizer.step()

    def soft_update_critic_target(self):
        utils.soft_update_params(
            self.critic.Q1, self.critic_target.Q1, self.critic_tau
        )
        utils.soft_update_params(
            self.critic.Q2, self.critic_target.Q2, self.critic_tau
        )
        utils.soft_update_params(
            self.critic.encoder, self.critic_target.encoder, self.encoder_tau
        )

    def soft_update_exp_critic_target(self):
        utils.soft_update_params(
            self.exp_critic.Q1, self.exp_critic_target.Q1, self.critic_tau
        )
        utils.soft_update_params(
            self.exp_critic.Q2, self.exp_critic_target.Q2, self.critic_tau
        )
        utils.soft_update_params(
            self.exp_critic.encoder,
            self.exp_critic_target.encoder,
            self.encoder_tau,
        )

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            latest_samples = replay_buffer.sample_latest()
            # self.update_exp_actor(obs)
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()

    def offline_update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        self.copy_params()

        print(f"offline training for {self.iters} iterations")
        for _ in range(self.iters):
            self.update_exp_critic(
                obs, action, reward, next_obs, not_done, L, step
            )

            if step % self.actor_update_freq == 0:
                if self.full_sampling:
                    latest_samples = replay_buffer.sample()
                else:
                    latest_samples = replay_buffer.sample_latest()
                self.update_exp_actor_and_alpha(obs, latest_samples, L, step)

            if step % self.critic_target_update_freq == 0:
                self.soft_update_exp_critic_target()
