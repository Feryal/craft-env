"""Try Craft environment with random agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses
import numpy as np

import env_factory


def run_loop(env, n_steps, render=False):
    possible_actions = env.action_specs()

    env.reset()

    for t in xrange(n_steps):
        # Random action
        action = np.random.choice(possible_actions.values())

        reward, done, observations = env.step(action)

        if render:
            curses.wrapper(env.render(fps=15))
        else:
            print("[{}] reward={} done={} \n observations: {}".format(
                t, reward, done, observations))

        if done:
            print("Woohooo did something!")


def main():
    recipes_path = "resources/recipes.yaml"
    hints_path = "resources/hints.yaml"
    envSampler = env_factory.EnvironmentFactory(
        recipes_path, hints_path, seed=1)

    env = envSampler.sample_environment()
    print("Environment: task {}: {}".format(env.task_name, env.task))
    run_loop(env, 1000, render=True)

    # env = envSampler.sample_environment_by_name('make[plank]')
    # print("Environment: task {}: {}".format(env.task_name, env.task))
    # run_loop(env, 10)


if __name__ == '__main__':
    main()
