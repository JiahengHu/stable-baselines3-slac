# This file is here just to define MlpPolicy/CnnPolicy
# that work for FPPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, MultiInputActorCriticPolicy, \
    FactoredActorCriticPolicy, FactoredActorCriticCnnPolicy, FactoredMultiInputActorCriticPolicy

# MlpPolicy = ActorCriticPolicy
# CnnPolicy = ActorCriticCnnPolicy
# MultiInputPolicy = MultiInputActorCriticPolicy

FactoredMlpPolicy = FactoredActorCriticPolicy
FactoredCnnPolicy = FactoredActorCriticCnnPolicy
FactoredMultiInputPolicy = FactoredMultiInputActorCriticPolicy
