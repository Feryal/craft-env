"""Try Craft environment with random agent."""

from __future__ import division
from __future__ import print_function

import time

import numpy as np
import matplotlib.pyplot as plt

import env_factory


def run_loop(env, n_steps, visualise=False):
  possible_actions = env.action_specs()

  observations = env.reset()

  if visualise:
    plt.ion()
    f, ax = plt.subplots()
    im = ax.imshow(observations['render'])
    f.canvas.draw()

  for t in xrange(n_steps):
    # Random action
    action = np.random.choice(possible_actions.values())

    reward, done, observations = env.step(action)

    if visualise:
      im.set_data(observations['render'])
      f.canvas.draw()
      f.canvas.flush_events()
      time.sleep(0.1)
    else:
      print("[{}] reward={} done={} \n observations: {}".format(
        t, reward, done, observations))

    if reward:
      im.set_data(np.ones_like(observations['render']) * np.array([0, 1, 0]))
      f.canvas.draw()
      f.canvas.flush_events()
      time.sleep(0.3)
      print("Got a rewaaaard!")
    elif done:
      im.set_data(np.zeros_like(observations['render']))
      f.canvas.draw()
      f.canvas.flush_events()
      time.sleep(0.2)
      print("... Reset")

def main():
  visualise = True
  recipes_path = "resources/recipes.yaml"
  hints_path = "resources/hints.yaml"
  envSampler = env_factory.EnvironmentFactory(
       recipes_path, hints_path, max_steps=100, seed=2, visualise=visualise)

  env = envSampler.sample_environment()
  print("Environment: task {}: {}".format(env.task_name, env.task))
  run_loop(env, 100*3, visualise=visualise)

  # env = envSampler.sample_environment_by_name('make[plank]')
  # print("Environment: task {}: {}".format(env.task_name, env.task))
  # run_loop(env, 10)


if __name__ == '__main__':
  main()
