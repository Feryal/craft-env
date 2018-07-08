"""Factory to sample new Craft environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import yaml

import craft
import env
from misc import util

Task = collections.namedtuple("Task", ["goal", "steps"])


class EnvironmentFactory(object):
    """Factory instantiating Craft environments."""

    def __init__(self, recipes_path, hints_path, seed=0):
        self.subtask_index = util.Index()
        self.task_index = util.Index()

        # create World
        self.world = craft.CraftWorld(recipes_path, seed)

        # Load the tasks with sub-steps (== hints)
        with open(hints_path) as hints_f:
            self.hints = yaml.load(hints_f)

        # Setup all possible tasks
        self._init_tasks()

    def _init_tasks(self):
        """Build the list of tasks and subtasks."""
        # organize task and subtask indices
        self.tasks_by_subtask = collections.defaultdict(list)
        self.tasks = {}
        for hint_key, hint in self.hints.items():
            # hint_key: make[plank], hint/steps: get_wood, makeAtToolshed
            goal = util.parse_fexp(hint_key)
            # goal: (make, plank)
            goal = (self.subtask_index.index(goal[0]),
                    self.world.cookbook.index[goal[1]])
            steps = tuple(self.subtask_index.index(s) for s in hint)
            task = Task(goal, steps)
            for subtask in steps:
                self.tasks_by_subtask[subtask].append(task)

            self.tasks[hint_key] = task
            self.task_index.index(task)

    def sample_environment(self):
        return self.sample_environment_by_name(
            np.random.choice(self.tasks.keys()))

    def sample_environment_by_name(self, task_name):
        # Get the task
        task = self.tasks[task_name]
        goal_arg = task.goal[1]

        # Sample a world (== scenario for them...)
        scenario = self.world.sample_scenario_with_goal(goal_arg)

        # Wrap it into an environment
        environment = env.CraftLab(scenario, task_name, task)

        return environment
