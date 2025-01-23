from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import torch
import torch as th
from torch import nn
from torch.nn.functional import gumbel_softmax
from functorch import combine_state_for_ensemble
from functorch import vmap

from stable_baselines3.common.distributions import (
    MultiOneHotCategoricalDistribution, MultiCategoricalDistribution)
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import BaseModel

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class FactoredValueHead(nn.Module):
    def __init__(self, input_size, output_size, num_Qs, use_layer_norm, hidden_size=1024, num_layers=2):
        super().__init__()

        self.output_size = output_size
        self.num_Qs = num_Qs

        Q_list = [] # [num_Qs * output_size (reward terms)]
        sizes = [hidden_size] * (num_layers + 1) + [1]
        for _ in range(self.num_Qs):
            for _ in range(self.output_size):
                Q1_layers = []
                # Add a trunk to the Q network
                Q1_layers += [nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh()]
                for i in range(num_layers):
                    if use_layer_norm:
                        Q1_layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.LayerNorm(sizes[i + 1]), nn.ReLU()]
                    else:
                        Q1_layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
                Q1_layers += [nn.Linear(sizes[-2], sizes[-1])]
                Q_list.append(nn.Sequential(*Q1_layers))

        # Parameter Stacking for speedup
        fmodel_Q1, params_Q1, buffers_Q1 = combine_state_for_ensemble(Q_list)
        self.Q1_params = [nn.Parameter(p) for p in params_Q1]
        self.Q1_buffers = [nn.Buffer(b) for b in buffers_Q1]

        for i, param in enumerate(self.Q1_params):
            self.register_parameter('Q1_param_' + str(i), param)

        for i, buffer in enumerate(self.Q1_buffers):
            self.register_buffer('Q1_buffer_' + str(i), buffer)

        self.Q_model = vmap(fmodel_Q1)

    def forward(self, x):
        batch_size = x.shape[0]
        # X_shape: (batch_size, output_size, embedding_size)
        x = th.swapaxes(x, 0, 1)  # (output_size, batch_size, embedding_size)
        Q = self.Q_model(self.Q1_params, self.Q1_buffers, x)  # (output_size, batch_size, 1)

        Q_out = Q.view(self.num_Qs, self.output_size, batch_size)
        Q_out = th.swapaxes(Q_out, 0, 2) # (batch_size, output_size, n_q)

        return Q_out

class Actor(BasePolicy):
    """
    Actor network (policy) for SAC.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean
        self.use_layer_norm = use_layer_norm

        latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn, use_layer_norm=use_layer_norm)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        # TODO: add these back for continous action
        # action_dim = get_action_dim(self.action_space)
        # self.action_dist = SquashedDiagGaussianDistribution(action_dim)
        # self.mu = nn.Linear(last_layer_dim, action_dim)
        # self.log_std = nn.Linear(last_layer_dim, action_dim)

        # We stick with the integer version for now - will manually convert to one-hot
        self.action_dist = MultiCategoricalDistribution(self.action_space.nvec)

        self.action_channel = len(self.action_space.nvec)
        self.action_dim = self.action_space.nvec[0]

        self.action_net = self.action_dist.proba_distribution_net(latent_dim=last_layer_dim)
        assert isinstance(self.action_dist, MultiCategoricalDistribution)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs)
        latent_pi = self.latent_pi(features)
        mean_actions = self.action_net(latent_pi) # THis is the action logit

        return mean_actions, {}

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.actions_from_params(mean_actions, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, **kwargs)

    # Get differentiable samples using gumbel softmax
    def action_differentiable_log_prob(self, obs: th.Tensor, hard=True, tau=1) -> Tuple[th.Tensor, th.Tensor]:
        logits, kwargs = self.get_action_dist_params(obs)
        gb_logits = logits.reshape(logits.shape[0], self.action_channel, self.action_dim)
        actions = gumbel_softmax(gb_logits, tau=tau, hard=hard)

        # calculate the log prob
        self.action_dist.proba_distribution(logits)
        integer_actions = th.argmax(actions, dim=-1)
        log_prob = self.action_dist.log_prob(integer_actions)
        return actions, log_prob


    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)


class DSACCritic(BaseModel):
    """
    Critic network(s) for Factored SAC
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        reward_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        use_layer_norm: bool = False,
        causal_matrix: Optional[th.Tensor] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.action_dim = action_dim = int(len(self.action_space.nvec) * self.action_space.nvec[0])

        self.output_dim = reward_dim + 1 # extra for SAC entropy

        if causal_matrix is not None:
            # add a row to the matrix
            ent_row = torch.ones(1, self.action_dim)
            causal_matrix = causal_matrix.repeat_interleave(self.action_space.nvec[0], 1)
            causal_matrix = torch.cat([causal_matrix, ent_row], dim=0)
            attn_logit = nn.Parameter(causal_matrix, requires_grad=False)
        else:
            attn_logit = nn.Parameter(th.ones([self.output_dim, self.action_dim]), requires_grad=False)

        self.register_parameter("attn_logit", attn_logit)

        self.share_features_extractor = share_features_extractor

        self.n_critics = n_critics

        self.q_nets = FactoredValueHead(features_dim + action_dim, self.output_dim, n_critics,
                                        use_layer_norm=use_layer_norm,
                                        hidden_size=net_arch[0], # we use the same hidden size for all layers
                                        num_layers=len(net_arch)-1)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)

        # extend features to self.n_critics * self.output_dim
        features = features.unsqueeze(1).repeat(1, self.n_critics * self.output_dim, 1)

        # flatten the action
        actions = actions.view(actions.shape[0], self.action_dim)
        actions = self.attention_forward(actions, self.attn_logit)
        actions = actions.repeat(1, self.n_critics, 1)

        qvalue_input = th.cat([features, actions], dim=-1)

        values = self.q_nets(qvalue_input) # shape: (batch_size, output_dim, n_critic)
        return values

    def attention_forward(self, x, attn_logits):
        x = x.unsqueeze(1)
        features = x * attn_logits  # (B, num_skills, obs_dim)
        return features


class DSACPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    :param use_layer_norm: Whether to use layer normalization or not
    :param reward_dim: Number of reward dimensions
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        use_layer_norm: bool = False,
        reward_dim: int = 1,
        causal_matrix: Optional[th.Tensor] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        if net_arch is None:
            net_arch = [256, 256]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        if use_layer_norm:
            assert not share_features_extractor, "Layer norm is not compatible with shared features extractor"

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            "use_layer_norm": use_layer_norm,
        }
        self.actor_kwargs = self.net_args.copy()

        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        self.reward_dim = reward_dim
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
                "reward_dim": reward_dim,
                "causal_matrix": causal_matrix,
            }
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        assert not self.share_features_extractor
        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [param for name, param in self.critic.named_parameters() if "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = self.critic.parameters()

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(critic_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        self.actor.reset_noise(batch_size=batch_size)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> DSACCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return DSACCritic(**critic_kwargs).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


MlpPolicy = DSACPolicy


class CnnPolicy(DSACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        use_layer_norm: bool = False,
        reward_dim: int = 1,
        causal_matrix: Optional[th.Tensor] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
            use_layer_norm,
            reward_dim,
            causal_matrix,
        )


class MultiInputPolicy(DSACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        use_layer_norm: bool = False,
        reward_dim: int = 1,
        causal_matrix: Optional[th.Tensor] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
            use_layer_norm,
            reward_dim,
            causal_matrix,
        )
