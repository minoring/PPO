from gym.spaces import Box

from parallel_env import make_vec_envs


def get_env(args):
    env = make_vec_envs(
        env_name=args.env,
        seed=args.seed,
        num_processes=args.n_actors,
        gamma=None,
        log_dir=None,
        device='cpu',  # Always work with CPU if it is related to environment.
        allow_early_resets=False)
    assert isinstance(env.action_space,
                      Box), "This project works for envs with continuous action spaces."
    return env
