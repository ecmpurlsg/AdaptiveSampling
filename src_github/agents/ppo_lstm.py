import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical

################################## set device ##################################
print("============================================================================================")
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


class RolloutBuffer:
    """
    Matches the API of `src/agents/ppo.py`'s buffer so existing training code can be reused.
    The LSTM hidden state is kept inside the agent and reset via `PPO_LSTM.reset_hidden()`.
    """

    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCriticLSTM(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, has_continuous_action_space: bool, action_std_init: float):
        super().__init__()

        self.has_continuous_action_space = bool(has_continuous_action_space)
        self.action_dim = int(action_dim)
        self.lstm_hidden_dim = 64

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), float(action_std_init) ** 2).to(device)

        self.fc1 = nn.Linear(int(state_dim), 64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=self.lstm_hidden_dim, num_layers=1, batch_first=True)

        if self.has_continuous_action_space:
            self.actor_head = nn.Sequential(
                nn.Linear(self.lstm_hidden_dim, self.action_dim),
                nn.Tanh(),
            )
        else:
            self.actor_head = nn.Sequential(
                nn.Linear(self.lstm_hidden_dim, self.action_dim),
                nn.Softmax(dim=-1),
            )

        self.critic_head = nn.Linear(self.lstm_hidden_dim, 1)

    def set_action_std(self, new_action_std: float):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), float(new_action_std) ** 2).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCriticLSTM::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def init_hidden(self, batch_size: int = 1):
        h0 = torch.zeros((1, batch_size, self.lstm_hidden_dim), device=device)
        c0 = torch.zeros((1, batch_size, self.lstm_hidden_dim), device=device)
        return (h0, c0)

    def _forward_seq(self, states_seq: torch.Tensor, hidden):
        """
        states_seq: [T, state_dim] or [B, T, state_dim]
        Returns:
          lstm_out_seq: [B, T, H]
          hidden_out
        """
        if states_seq.dim() == 2:
            states_seq = states_seq.unsqueeze(0)  # [1, T, state_dim]

        x = torch.tanh(self.fc1(states_seq))  # [B, T, 64]
        lstm_out, hidden_out = self.lstm(x, hidden)  # [B, T, H]
        return lstm_out, hidden_out

    def act(self, state: torch.Tensor, hidden):
        """
        state: [state_dim]
        hidden: (h, c) each [1, 1, H]
        """
        lstm_out, hidden_out = self._forward_seq(state.view(1, -1), hidden)  # [1, 1, H]
        feat = lstm_out[:, -1, :].squeeze(0)  # [H]

        if self.has_continuous_action_space:
            action_mean = self.actor_head(feat)  # [action_dim]
            # Use a non-batched covariance so we always sample an action of shape [A]
            cov_mat = torch.diag(self.action_var)  # [A, A]
            dist = MultivariateNormal(action_mean, cov_mat)
            action = dist.sample().view(-1)  # [A]
            action_logprob = dist.log_prob(action).view(1)  # [1] (consistent shape)
        else:
            action_probs = self.actor_head(feat)  # [action_dim]
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action).view(1)  # [1]

        state_val = self.critic_head(feat).view(1)  # [1]
        return action.detach(), action_logprob.detach(), state_val.detach(), hidden_out

    def evaluate_seq(self, states_seq: torch.Tensor, actions_seq: torch.Tensor, hidden):
        """
        Evaluate a *contiguous* sequence (no terminal inside) with a given initial hidden state.
        states_seq:  [T, state_dim]
        actions_seq: [T, action_dim] (continuous) OR [T] (discrete)
        """
        lstm_out, _ = self._forward_seq(states_seq, hidden)  # [1, T, H]
        feats = lstm_out.squeeze(0)  # [T, H]

        if self.has_continuous_action_space:
            action_mean = self.actor_head(feats)  # [T, A]
            action_var = self.action_var.expand_as(action_mean)  # [T, A]
            cov_mat = torch.diag_embed(action_var).to(device)  # [T, A, A]
            dist = MultivariateNormal(action_mean, cov_mat)

            if self.action_dim == 1:
                actions_seq = actions_seq.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor_head(feats)  # [T, A]
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(actions_seq).view(-1)  # [T]
        dist_entropy = dist.entropy().view(-1)  # [T]
        state_values = self.critic_head(feats).view(-1)  # [T]
        return action_logprobs, state_values, dist_entropy


class PPO_LSTM:
    """
    PPO with an LSTM policy head.

    Drop-in-ish replacement for `src/agents/ppo.PPO`:
    - select_action(state) -> action (numpy)
    - update() -> dict or None
    - save/load

    Important: call `reset_hidden()` at the start of each episode to avoid leaking memory between episodes.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        action_std_init=0.6,
    ):
        self.has_continuous_action_space = bool(has_continuous_action_space)
        self.gamma = float(gamma)
        self.eps_clip = float(eps_clip)
        self.K_epochs = int(K_epochs)

        if self.has_continuous_action_space:
            self.action_std = float(action_std_init)

        self.buffer = RolloutBuffer()

        self.policy = ActorCriticLSTM(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.parameters(), "lr": float(lr_actor)},  # single optimizer is fine for LSTM
            ]
        )

        self.policy_old = ActorCriticLSTM(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self._hidden = None

    def reset_hidden(self):
        self._hidden = self.policy_old.init_hidden(batch_size=1)

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = float(new_action_std)
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO_LSTM::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = round(float(self.action_std) - float(action_std_decay_rate), 4)
            if self.action_std <= float(min_action_std):
                self.action_std = float(min_action_std)
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)
        else:
            print("WARNING : Calling PPO_LSTM::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):
        return self.select_action_clamped(state, clamp=None)

    def select_action_clamped(self, state, clamp=None):
        """
        Like PPO.select_action(), but optionally clamps continuous actions before storing/logprob.
        clamp: None OR (min_val, max_val)
        """
        if self._hidden is None:
            self.reset_hidden()

        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
            action, action_logprob, state_val, hidden_out = self.policy_old.act(state_t, self._hidden)

            if self.has_continuous_action_space and clamp is not None:
                lo, hi = float(clamp[0]), float(clamp[1])
                action = torch.clamp(action.view(-1), lo, hi)  # ensure [A]
                # Recompute logprob for the executed (clamped) action under the same distribution.
                # Note: still a "clipped action" approximation, but consistent with what we execute.
                # Build dist from current state feature (same as in act()).
                lstm_out, _ = self.policy_old._forward_seq(state_t.view(1, -1), self._hidden)
                feat = lstm_out[:, -1, :].squeeze(0)
                action_mean = self.policy_old.actor_head(feat)
                cov_mat = torch.diag(self.policy_old.action_var)
                dist = MultivariateNormal(action_mean, cov_mat)
                action_logprob = dist.log_prob(action).view(1)

            self._hidden = (hidden_out[0].detach(), hidden_out[1].detach())

        self.buffer.states.append(state_t)
        self.buffer.actions.append(action.detach().view(-1))
        self.buffer.logprobs.append(action_logprob.detach())
        self.buffer.state_values.append(state_val.detach())

        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten()
        return int(action.item())

    def record_step(self, state, action_value, clamp=None):
        """
        Record a transition for a *given* action (used for action repetition intervals).
        This advances the LSTM hidden state exactly once.
        action_value: float (continuous) or int (discrete)
        """
        if self._hidden is None:
            self.reset_hidden()

        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            lstm_out, hidden_out = self.policy_old._forward_seq(state_t.view(1, -1), self._hidden)
            feat = lstm_out[:, -1, :].squeeze(0)

            if self.has_continuous_action_space:
                a = float(action_value)
                if clamp is not None:
                    lo, hi = float(clamp[0]), float(clamp[1])
                    a = max(lo, min(hi, a))
                action_t = torch.tensor([a], dtype=torch.float32, device=device).view(-1)  # [A] (A=1)

                action_mean = self.policy_old.actor_head(feat)
                cov_mat = torch.diag(self.policy_old.action_var)
                dist = MultivariateNormal(action_mean, cov_mat)
                logprob = dist.log_prob(action_t).view(1)
            else:
                a = int(action_value)
                action_probs = self.policy_old.actor_head(feat)
                dist = Categorical(action_probs)
                action_t = torch.tensor(a, dtype=torch.long, device=device)
                logprob = dist.log_prob(action_t).view(1)

            state_val = self.policy_old.critic_head(feat).view(1)

            self._hidden = (hidden_out[0].detach(), hidden_out[1].detach())

        self.buffer.states.append(state_t.detach())
        self.buffer.actions.append(action_t.detach())
        self.buffer.logprobs.append(logprob.detach())
        self.buffer.state_values.append(state_val.detach())

    def _iter_segments(self):
        """
        Yield (start, end_exclusive) segments split by terminals.
        """
        n = len(self.buffer.states)
        if n == 0:
            return
        start = 0
        for i, term in enumerate(self.buffer.is_terminals):
            if term:
                yield (start, i + 1)
                start = i + 1
        if start < n:
            yield (start, n)

    def update(self):
        if len(self.buffer.rewards) == 0:
            return None

        # Monte Carlo estimate of returns with terminal resets
        rewards = []
        discounted_reward = 0.0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0.0
            discounted_reward = float(reward) + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        if len(rewards_t) > 1:
            std = rewards_t.std(unbiased=False)
            rewards_t = (rewards_t - rewards_t.mean()) / (std + 1e-7)
        else:
            rewards_t = rewards_t - rewards_t.mean()

        # convert list to tensor
        old_states = torch.stack(self.buffer.states, dim=0).detach().to(device)          # [T, S]
        # Be robust to stored action shapes like [1] vs [1,1] by squeezing after stacking.
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).view(-1).detach().to(device)  # [T]
        old_state_values = torch.stack(self.buffer.state_values, dim=0).view(-1).detach().to(device)  # [T]

        advantages = rewards_t.detach() - old_state_values.detach()

        policy_losses = []
        value_losses = []
        entropies = []
        total_losses = []

        for _ in range(self.K_epochs):
            # Evaluate in segments so the LSTM hidden resets at episode boundaries.
            logprobs_all = []
            values_all = []
            entropy_all = []

            for s0, s1 in self._iter_segments():
                seg_states = old_states[s0:s1]
                seg_actions = old_actions[s0:s1]
                h0 = self.policy.init_hidden(batch_size=1)
                lp, v, ent = self.policy.evaluate_seq(seg_states, seg_actions, h0)
                logprobs_all.append(lp)
                values_all.append(v)
                entropy_all.append(ent)

            logprobs = torch.cat(logprobs_all, dim=0)  # [T]
            state_values = torch.cat(values_all, dim=0).view(-1)  # [T]
            dist_entropy = torch.cat(entropy_all, dim=0)  # [T]

            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.MseLoss(state_values, rewards_t.view(-1))
            entropy_mean = dist_entropy.mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_mean

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy_mean.item()))
            total_losses.append(float(loss.item()))

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

        # Hidden state should not carry across updates by default
        self.reset_hidden()

        n = max(1, len(total_losses))
        return {
            "loss": sum(total_losses) / n,
            "policy_loss": sum(policy_losses) / n,
            "value_loss": sum(value_losses) / n,
            "entropy": sum(entropies) / n,
        }

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.reset_hidden()


