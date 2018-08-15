"""Try Craft environment with random agent."""

from __future__ import division
from __future__ import print_function

import numpy as np
import time

import env_factory


def run_loop(env, n_steps, visualise=False):
  possible_actions = env.action_specs()

  observations = env.reset()
  for t in xrange(n_steps):
    # Random action
    action = np.random.choice(possible_actions.values())

    # Step (this will plot if visualise is True)
    reward, done, observations = env.step(action)
    if visualise:
      env.render_matplotlib(frame=observations['image'])
    else:
      print("[{}] reward={} done={} \n observations: {}".format(
          t, reward, done, observations))

    if reward:
      rewarding_frame = observations['image'].copy()
      rewarding_frame[:40] *= np.array([0, 1, 0])
      env.render_matplotlib(frame=rewarding_frame, delta_time=0.7)
      print("[{}] Got a rewaaaard! {:.1f}".format(t, reward))
    elif done:
      env.render_matplotlib(
          frame=np.zeros_like(observations['image']), delta_time=0.3)
      print("[{}] Finished with nothing... Reset".format(t))


def main():
  visualise = True
  recipes_path = "resources/recipes.yaml"
  hints_path = "resources/hints.yaml"
  env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, max_steps=100, seed=1, reuse_environments=True,
      visualise=visualise)

  # env = env_sampler.sample_environment()
  # print("Environment: task {}: {}".format(env.task_name, env.task))
  # run_loop(env, 100 * 3, visualise=visualise)

  env = env_sampler.sample_environment(task_name='get[grass]')
  print("Environment: task {}: {}".format(env.task_name, env.task))
  run_loop(env, 100 * 3, visualise=visualise)


if __name__ == '__main__':
  main()
