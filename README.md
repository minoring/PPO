# Proximal Policy Optimization Algorithms

## Training
You can check [`parse_utils.py`](parse_utils.py) to examine available flags and [`config_ppo.yaml`](config_ppo.yaml), [`config_game.yaml`](config_gae.yaml) for hyperparameters.
### HalfCheetah-v2
- Clipping surrogate objective
```python
python train.py --env HalfCheetah-v2 --hyperparams mujoco --seed 1 --log-interval 1 --surrogate-objective clipping
```
- Without clipping or penalty
```python
python train.py --env HalfCheetah-v2 --hyperparams mujoco --seed 1 --log-interval 1 --surrogate-objective no-clipping-penalty
```
## Testing
Run `python test.py --env <Gym env> --trained-model <path/to/trained/model> --record-video`
e.g.
```python
python test.py --env HalfCheetah-v2 --trained-model HalfCheetah-v2.pt --record-video
```
or without recording an episode,
```python
python test.py --env HalfCheetah-v2 --trained-model HalfCheetah-v2.pt
```

## Results
**HalfCheetah-v2** after training 1M timesteps. (Average reward is about 2000 for 10 test episodes)

![](assets/HalfCheetah-v2.gif)

## References
### Paper
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), Schulman et al. 2017
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), Schulman et al. 2016
### Docs
#### OpenAI Spinning Up
- [Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html#pseudocode)
- [Trust Region Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/trpo.html)
- [Vanilla Policy Gradient Algorithm](https://spinningup.openai.com/en/latest/algorithms/vpg.html)
