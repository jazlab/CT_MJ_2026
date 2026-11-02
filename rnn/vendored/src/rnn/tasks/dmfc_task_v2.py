"""DMFC task v2: adds auxiliary ramp target for integrator.

Same as dmfc_task.py but _make_trial returns a third array: ramp target.

Inputs (7-dim):
  [startx, starty, dir_onehot(4), fix_on(1)]

Behavioral target (2-dim):
  [target_x, target_y] — 0 during delay, value during report, NaN elsewhere

Ramp target (1-dim):
  Linear ramp 0→1 during flow period, identical for all directions.
  NaN outside flow. Forces common integration axis.

Timeline:
  |--fixation--|--flow (len/speed)--|--delay--|--report--|
  ramp:              0==========1
  target:        NaN      NaN       0        value
"""
import numpy as np
import itertools
import torch
from torch.utils.data import Dataset


class DMFCDatasetV2(Dataset):
    def __init__(self, condition_labels, inputs, targets, ramp_targets, is_grounded):
        self.condition_labels = condition_labels
        self.inputs = inputs
        self.targets = targets
        self.ramp_targets = ramp_targets
        self.is_grounded = is_grounded

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (self.inputs[idx], self.targets[idx],
                self.ramp_targets[idx], self.is_grounded[idx])


class DMFCTaskV2:
    def __init__(self, **kwargs):
        self.tau = kwargs.get('tau', 0.1)
        self.dt = kwargs.get('dt', 1/30)
        self.alpha = self.dt / self.tau

        self.x_range = kwargs.get('x_range', [-3, -2, -1, 0, 1, 2, 3])
        self.y_range = kwargs.get('y_range', [-3, -2, -1, 0, 1, 2, 3])
        self.scale = kwargs.get('scale', 1)

        self.directions = kwargs.get('directions', [(1, 0), (0, 1), (-1, 0), (0, -1)])
        self.vector_lengths = kwargs.get('vector_lengths', [1, 2, 3, 4, 5, 6])
        self.BALL_SPEED = kwargs.get('BALL_SPEED', 3)

        self.FLOW_ONSET_TIME = kwargs.get('start_time', 0.3)
        self.delay = kwargs.get('delay', 0.2)
        self.report_length = kwargs.get('report_length', 0.2)

        self.training_set = kwargs.get('training_set', 'circular')

        max_flow_dur = max(self.vector_lengths) / self.BALL_SPEED
        self.total_time = self.FLOW_ONSET_TIME + max_flow_dur + self.delay + self.report_length
        self.n_timesteps = int(np.round(self.total_time / self.dt))

        self.flow_on = int(np.round(self.FLOW_ONSET_TIME / self.dt))
        self.report_duration = int(np.round(self.report_length / self.dt))

        self.direction_dict = {
            (0, 1): 0,
            (0, -1): 1,
            (1, 0): 2,
            (-1, 0): 3,
        }

        self._build_conditions()

    def _path_in_region(self, start, direction, length):
        x_lo, x_hi = self.x_range[0], self.x_range[-1]
        y_lo, y_hi = self.y_range[0], self.y_range[-1]
        for k in range(int(round(length)) + 1):
            px = start[0] + direction[0] * k
            py = start[1] + direction[1] * k
            if px < x_lo or px > x_hi or py < y_lo or py > y_hi:
                return False
        return True

    def _build_conditions(self):
        self.conditions_joint = []
        for x, y, d, length in itertools.product(
                self.x_range, self.y_range, self.directions, self.vector_lengths):
            start = np.array((x, y))
            direction = np.array(d)
            vector = direction * length
            target = start + vector
            if not self._path_in_region(start, direction, length):
                continue
            self.conditions_joint.append((tuple(start), tuple(vector), tuple(target)))

        self.conditions_grounded = set(self._generate_grounded(self.training_set))

    def _generate_grounded(self, preset='circular'):
        if preset == 'circular':
            conditions = []
            direction_dict = {
                (1, 0): np.array([0, 1], dtype=np.int32),
                (0, 1): np.array([-1, 0], dtype=np.int32),
                (-1, 0): np.array([0, -1], dtype=np.int32),
                (0, -1): np.array([1, 0], dtype=np.int32),
            }
            for direction in self.directions:
                direction_array = np.array(direction, dtype=np.int32)
                for length in self.vector_lengths:
                    for r in range(1, int(np.max(self.x_range)) + 1):
                        start_x = direction_dict[direction] * r
                        for step in range(self.x_range[0], self.x_range[-1] + 1):
                            start_y = direction_array * step
                            start = start_x + start_y
                            if not self._path_in_region(start, direction_array, length):
                                continue
                            vector = direction_array * length
                            target = start + vector
                            conditions.append((tuple(start), tuple(vector), tuple(target)))
            return conditions
        elif preset == 'monkey':
            import json, os
            json_path = os.path.join(os.path.dirname(__file__), 'monkey_training_conditions.json')
            with open(json_path) as f:
                data = json.load(f)
            conditions = []
            for c in data['conditions']:
                start = tuple(c[0])
                vector = tuple(c[1])
                target = tuple(c[2])
                conditions.append((start, vector, target))
            return conditions
        raise ValueError(f"Unknown preset: {preset}")

    def _make_trial(self, condition):
        """Build input, behavioral target, and ramp target for one trial."""
        start_pos = condition[0]
        vector = condition[1]
        target = condition[2]
        scale = self.scale
        n = self.n_timesteps

        flow_duration = np.linalg.norm(vector) / self.BALL_SPEED
        flow_off = int(np.round((self.FLOW_ONSET_TIME + flow_duration) / self.dt))
        fixation_off = int(np.round((self.FLOW_ONSET_TIME + flow_duration + self.delay) / self.dt))

        # Input: [startx, starty, dir_onehot(4), fix_on(1)] = 7 dims
        inp = np.zeros((n, 7), dtype=np.float32)
        inp[:, 0] = start_pos[0] * scale
        inp[:, 1] = start_pos[1] * scale

        if np.linalg.norm(vector) > 0:
            direction = tuple((np.array(vector) / np.linalg.norm(vector)).astype(int))
            inp[self.flow_on:flow_off, 2 + self.direction_dict[direction]] = scale

        inp[:fixation_off, 6] = scale

        # Behavioral target
        tgt = np.full((n, 2), np.nan, dtype=np.float32)
        tgt[flow_off:fixation_off, :] = 0
        tgt[fixation_off:fixation_off + self.report_duration, 0] = target[0] * scale
        tgt[fixation_off:fixation_off + self.report_duration, 1] = target[1] * scale

        # Ramp target: integrated time, 0 → flow_duration (seconds)
        ramp = np.full((n, 1), np.nan, dtype=np.float32)
        n_flow = flow_off - self.flow_on
        if n_flow > 0:
            flow_duration = np.linalg.norm(vector) / self.BALL_SPEED
            ramp[self.flow_on:flow_off, 0] = np.linspace(0, flow_duration, n_flow)

        return inp, tgt, ramp

    def generate_data(self, mode='train'):
        condition_labels, inputs, targets, ramp_targets, is_grounded = [], [], [], [], []

        for cond in self.conditions_joint:
            if mode == 'val' and cond not in self.conditions_grounded:
                continue

            inp, tgt, ramp = self._make_trial(cond)
            condition_labels.append(cond)
            inputs.append(inp)
            targets.append(tgt)
            ramp_targets.append(ramp)
            is_grounded.append(1 if cond in self.conditions_grounded else 0)

        return condition_labels, inputs, targets, ramp_targets, is_grounded

    def get_datasets(self):
        labels_train, inp_train, tgt_train, ramp_train, g_train = self.generate_data('train')
        labels_val, inp_val, tgt_val, ramp_val, g_val = self.generate_data('val')

        ds_train = DMFCDatasetV2(labels_train, inp_train, tgt_train, ramp_train, g_train)
        ds_val = DMFCDatasetV2(labels_val, inp_val, tgt_val, ramp_val, g_val)
        return ds_train, ds_val

    def summary(self):
        n_grounded = sum(1 for c in self.conditions_joint if c in self.conditions_grounded)
        n_holdout = len(self.conditions_joint) - n_grounded
        print(f'Task: {len(self.conditions_joint)} conditions '
              f'({n_grounded} train, {n_holdout} holdout)')
        print(f'  lengths={self.vector_lengths}, delay={self.delay}s')
        print(f'  n_timesteps={self.n_timesteps} ({self.total_time:.2f}s at dt={self.dt:.4f})')
        print(f'  input_dim=7 [start(2), dir(4), fix_on(1)]')
        print(f'  output_dim=2 [target_x, target_y]')
        print(f'  ramp: 0->1 during flow (aux target)')


if __name__ == '__main__':
    task = DMFCTaskV2(delay=0.2)
    task.summary()

    cond = task.conditions_joint[0]
    inp, tgt, ramp = task._make_trial(cond)
    print(f'\nExample: {cond}')
    print(f'  input: {inp.shape}, target: {tgt.shape}, ramp: {ramp.shape}')
    print(f'  ramp non-NaN bins: {int((~np.isnan(ramp[:, 0])).sum())}')
    print(f'  ramp range: {np.nanmin(ramp):.2f} -> {np.nanmax(ramp):.2f}')
