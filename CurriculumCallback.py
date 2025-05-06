from numpy import exp
from stable_baselines3.common.callbacks import BaseCallback


def make_difficulty_fn(schedule: str, grad: float, offset: float):
    if schedule == "lin":
        print("Using linear difficulty with gradient {} and x_offset {}".format(grad, offset))
        return lambda p: min(1.0, max(0, grad * p + offset))

    elif schedule == "cub":
        print("Using cubic difficulty with gradient {} and x_offset {}".format(grad, offset))
        return lambda p: min(1.0, max(0, grad * (p + offset) ** 3))

    elif schedule == "exp":
        print("Using exponential difficulty with gradient {} and x_offset {}".format(grad, offset))
        return lambda p: min(1.0, max(0, exp(grad * (p + offset))))

    elif schedule == "sig":
        print("Using sigmoid shaped difficulty with gradient {} and x_offset {}".format(grad, offset))
        return lambda p: min(1.0, max(0, 1 / (1 + exp(-grad * (p + offset)))))

    else:
        raise ValueError(f"Unknown schedule: {schedule}")

class CurriculumCallback(BaseCallback):
    def __init__(self, env, config, verbose=0):
        super().__init__(verbose)
        self.env = env
        self._difficulty_schedule = config["difficulty_schedule"]
        self._difficulty_grad = config['difficulty_grad']
        self._difficulty_x_offset = config['difficulty_x_offset']
        self.total_timesteps = config["total_timesteps"]

        self._difficulty_fn = make_difficulty_fn(self._difficulty_schedule, self._difficulty_grad, self._difficulty_x_offset)

    def _on_step(self) -> bool:
        # Schedule to let agent learn gradually, not exact because of rollout buffer filling, but should be close enough
        # for large enough total_timestep values.
        progress = self.num_timesteps / self.total_timesteps

        difficulty = self._difficulty_fn(progress)

        # Update all sub-envs if using VecEnv
        self.env.env_method("set_difficulty", difficulty=difficulty)
        return True