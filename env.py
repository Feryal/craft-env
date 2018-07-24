"""DMLab-like wrapper for a Craft environment."""

from __future__ import division
from __future__ import print_function

import collections
import curses
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import seaborn as sns
import time

Task = collections.namedtuple("Task", ["goal", "steps"])


class CraftLab(object):
  """DMLab-like wrapper for a Craft state."""

  def __init__(self,
               scenario,
               task_name,
               task,
               max_steps=100,
               visualise=False,
               render_scale=10,
               extra_pickup_penalty=0.3):
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
    self._extra_pickup_penalty = extra_pickup_penalty
    self._current_state = self.scenario.init()

    # Rendering options
    self._render_state = {}
    self._width, self._height, _ = self._current_state.grid.shape
    self._render_scale = render_scale
    self._inventory_bar_height = 10
    self._goal_bar_height = 30
    self._render_width = self._width * self._render_scale
    self._render_height = (self._height * self._render_scale +
                           self._goal_bar_height + self._inventory_bar_height)
    # Colors of entities for rendering
    self._colors = {
        'player': sns.xkcd_palette(('red', ))[0],
        'background': sns.xkcd_palette(('white', ))[0],
        'boundary': sns.xkcd_palette(('black', ))[0],
        'workshop0': sns.xkcd_palette(('blue', ))[0],
        'workshop1': sns.xkcd_palette(('pink', ))[0],
        'workshop2': sns.xkcd_palette(('violet', ))[0],
        'water': sns.xkcd_palette(('water blue', ))[0],
        'wood': sns.xkcd_palette(('sienna', ))[0],
        'cloth': sns.xkcd_palette(('off white', ))[0],
        'grass': sns.xkcd_palette(('grass', ))[0],
        'iron': sns.xkcd_palette(('gunmetal', ))[0],
        'stone': sns.xkcd_palette(('stone', ))[0],
        'gold': sns.xkcd_palette(('gold', ))[0],
        'gem': sns.xkcd_palette(('bright purple', ))[0],
        'bridge': sns.xkcd_palette(('grey', ))[0],
        'stick': sns.xkcd_palette(('sandy brown', ))[0],
        'shears': sns.xkcd_palette(('cherry', ))[0],
        'plank': sns.xkcd_palette(('brown', ))[0],
        'ladder': sns.xkcd_palette(('metallic blue', ))[0],
        'bed': sns.xkcd_palette(('fawn', ))[0],
        'rope': sns.xkcd_palette(('beige', ))[0],
        'axe': sns.xkcd_palette(('charcoal', ))[0]
    }

  def obs_specs(self):
    obs_specs = collections.OrderedDict()
    obs_specs['features'] = {
        'dtype': 'float32',
        'shape': (self.world.n_features, )
    }
    obs_specs['task_name'] = {'dtype': 'string', 'shape': tuple()}

    if self._visualise:
      obs_specs['image'] = {
          'dtype': 'float32',
          'shape': (self._render_height, self._render_width, 3)
      }
    return obs_specs

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
    obs = {
        'features': self._current_state.features().astype(np.float32),
        'task_name': self.task_name
    }
    if self._visualise:
      obs['image'] = self.render_frame().astype(np.float32)
    return obs

  def step(self, action, num_steps=1):
    """Step the environment, getting reward, done and observation."""
    assert num_steps == 1, "No action repeat in this environment"

    # Step environment
    # (state_reward is 0 for all existing Craft environments)
    state_reward, self._current_state = self._current_state.step(action)
    self.steps += 1

    done = self._is_done()
    reward = np.float32(self._get_reward() + state_reward)

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

    # We want the correct pickup to be in inventory.
    # But we will penalise the agent for picking up extra stuff.
    items_index = np.arange(self._current_state.inventory.size)
    reward = float(self._current_state.inventory[goal_arg] > 0)
    reward -= self._extra_pickup_penalty * np.sum(
        self._current_state.inventory[items_index != goal_arg])
    reward = np.maximum(reward, 0)
    return reward

  def close(self):
    """Not used."""
    pass

  def render_matplotlib(self, frame=None, delta_time=0.1):
    """Render the environment with matplotlib, updating itself."""

    # Get current frame of environment or draw whatever we're given.
    if frame is None:
      frame = self.render_frame()

    # Setup if needed
    if not self._render_state:
      plt.ion()
      f, ax = plt.subplots()
      im = ax.imshow(frame)
      self._render_state['fig'] = f
      self._render_state['im'] = im
      ax.set_yticklabels([])
      ax.set_xticklabels([])
    # Update current frame
    self._render_state['im'].set_data(frame)
    self._render_state['fig'].canvas.draw()
    self._render_state['fig'].canvas.flush_events()
    time.sleep(delta_time)

    return frame

  def render_frame(self):
    """Render the current state as a 2D observation."""
    state = self._current_state

    ### Environment canvas
    env_canvas = np.zeros((self._width, self._height, 3))
    env_canvas[..., :] = self._colors['background']

    # Place all components
    for name, component_i in state.world.cookbook.index.contents.iteritems():
      # Check if the component is there, if so, color env_canvas accordingly.
      x_i, y_i = np.nonzero(state.grid[..., component_i])
      env_canvas[x_i, y_i] = self._colors[name]

    # Place self
    env_canvas[state.pos] = self._colors['player']
    # Upscale to render at higher resolution
    env_img = Image.fromarray(
        (env_canvas.transpose(1, 0, 2) * 255).astype(np.uint8), mode='RGB')
    env_large = np.array(
        env_img.resize(
            (self._render_width,
             self._height * self._render_scale), Image.NEAREST)) / 255.

    ### Inventory
    # two rows: first shows color of component, second how many are there
    inventory_canvas = np.zeros((2, state.inventory.shape[0], 3))
    for name, component_i in state.world.cookbook.index.contents.iteritems():
      inventory_canvas[0, component_i] = self._colors[name]
    for c in xrange(3):
      inventory_canvas[1, :, c] = np.minimum(state.inventory, 1)
    inventory_img = Image.fromarray(
        (inventory_canvas * 255).astype(np.uint8), mode='RGB')
    inventory_large = np.array(
        inventory_img.resize(
            (self._render_width,
             self._inventory_bar_height), Image.NEAREST)) / 255.

    # Show goal text
    goal_bar = Image.new("RGB", (self._render_width, self._goal_bar_height),
                         (255, 255, 255))
    goal_canvas = ImageDraw.Draw(goal_bar)
    goal_canvas.text((10, 10), self.task_name, fill=(0, 0, 0))
    goal_bar = np.array(goal_bar)
    goal_bar = goal_bar.astype(np.float64)
    goal_bar /= 255.0

    # Combine into single window
    canvas_full = np.concatenate([goal_bar, env_large, inventory_large])

    return canvas_full

  def render_curses(self, fps=60):
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
