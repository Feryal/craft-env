"""DMLab-like wrapper for a Craft environment."""

from __future__ import division
from __future__ import print_function

import collections
import curses
import numpy as np
import time

Task = collections.namedtuple("Task", ["goal", "steps"])


class CraftLab(object):
  """DMLab-like wrapper for a Craft state."""

  def __init__(self, scenario, task_name, task, max_steps=100, visualise=False):
    """DMLab-like interface for a Craft environment.

    Given a `scenario` (basically holding an initial world state), will provide
    the usual DMLab API for use in RL.
    """

    self.world = scenario.world
    self.scenario = scenario
    self.task_name = task_name
    self.task = task
    self.max_steps = max_steps
    self._visualise = visualise
    self.steps = 0
    self._current_state = self.scenario.init()

  def reset(self, seed=0):
    """Reset the environment.

    Agent will loop in the same world, from the same starting position.
    """
    del seed
    self._current_state = self.scenario.init()
    self.steps = 0
    return self.observations()

  def observations(self):
    """Return observation dict."""
    return {
        'features': self._current_state.features().astype(np.float32),
        'task_name': self.task_name
    }

  def step(self, action, num_steps=1):
    """Step the environment, getting reward, done and observation."""
    assert num_steps == 1, "No action repeat in this environment"

    # Step environment
    # (state_reward is 0 for all existing Craft environments)
    state_reward, self._current_state = self._current_state.step(action)
    self.steps += 1

    done = self._is_done()
    reward = np.float32(self._get_reward() + state_reward)

    if self._visualise:
      curses.wrapper(self.render(fps=15))

    if done:
      self.reset()
    observations = self.observations()
    return reward, done, observations

  def _is_done(self):
    goal_name, goal_arg = self.task.goal
    done = (self._current_state.satisfies(goal_name, goal_arg)
            or self.steps >= self.max_steps)
    return done

  def _get_reward(self):
    goal_name, goal_arg = self.task.goal
    reward = self._current_state.satisfies(goal_name, goal_arg)
    return reward

  def obs_specs(self):
    return {'features': (self.world.n_features, ), 'task_name': tuple()}

  def action_specs(self):
    # last action is termination of current option, we don't use it.
    return {
        'DOWN': 0,
        'UP': 1,
        'LEFT': 2,
        'RIGHT': 3,
        'USE': 4,
        # 'TERMINATE': 5
    }

  def close(self):
    """Not used."""
    pass

  def render(self, fps=60):
    """Render the current state in curses."""
    width, height, _ = self._current_state.grid.shape
    action_spec = self.action_specs()

    def _visualize(win):
      state = self._current_state
      goal_name, _ = self.task.goal

      if state is None:
        return

      curses.start_color()
      for i in range(1, 8):
        curses.init_pair(i, i, curses.COLOR_BLACK)
        curses.init_pair(i + 10, curses.COLOR_BLACK, i)
      win.clear()
      for y in range(height):
        for x in range(width):
          if not (state.grid[x, y, :].any() or (x, y) == state.pos):
            continue
          thing = state.grid[x, y, :].argmax()
          if (x, y) == state.pos:
            if state.dir == action_spec['LEFT']:
              ch1 = "<"
              ch2 = "@"
            elif state.dir == action_spec['RIGHT']:
              ch1 = "@"
              ch2 = ">"
            elif state.dir == action_spec['UP']:
              ch1 = "^"
              ch2 = "@"
            elif state.dir == action_spec['DOWN']:
              ch1 = "@"
              ch2 = "v"
            color = curses.color_pair(goal_name or 0)
          elif thing == state.world.cookbook.index["boundary"]:
            ch1 = ch2 = curses.ACS_BOARD
            color = curses.color_pair(10 + thing)
          else:
            name = state.world.cookbook.index.get(thing)
            ch1 = name[0]
            ch2 = name[-1]
            color = curses.color_pair(10 + thing)

          win.addch(height - y, x * 2, ch1, color)
          win.addch(height - y, x * 2 + 1, ch2, color)
      win.refresh()
      time.sleep(1 / fps)

    return _visualize
