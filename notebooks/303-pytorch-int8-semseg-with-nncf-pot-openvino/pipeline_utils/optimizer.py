"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import re

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, StepLR, LambdaLR, ExponentialLR


def get_parameter_groups(model, config):
    optim_config = config.get('optimizer', {})
    base_lr = optim_config.get('base_lr', 1e-4)
    weight_decay = optim_config.get('weight_decay', get_default_weight_decay(config))

    if 'parameter_groups' not in optim_config:
        return [
            {'lr': base_lr, 'weight_decay': weight_decay, 'params': model.parameters()}
        ]

    param_groups = optim_config.parameter_groups
    for param_name, param in model.named_parameters():
        group = None
        found = False
        for group in param_groups:
            # find first matched group for a given param
            if re.search(group.get('re', ''), param_name):
                found = True
                break
        if found:
            group.setdefault('params', []).append(param)
    return param_groups


def make_optimizer(params_to_optimize, config):
    optim_config = config.get('optimizer', {})

    optim_type = optim_config.get('type', 'adam').lower()
    optim_params = optim_config.get("optimizer_params", {})
    if optim_type == 'adam':
        optim = Adam(params_to_optimize, **optim_params)
    elif optim_type == 'sgd':
        optim_params = optim_config.get("optimizer_params", {"momentum": 0.9})
        optim = SGD(params_to_optimize, **optim_params)
    else:
        raise KeyError("Unknown optimizer type: {}".format(optim_type))

    scheduler_type = optim_config.get('schedule_type', 'step').lower()
    scheduler_params = optim_config.get("schedule_params", optim_config.get("scheduler_params", {}))

    gamma = optim_config.get('gamma', 0.1)
    if scheduler_type == 'multistep':
        scheduler = MultiStepLR(optim, optim_config.get('steps'), gamma=gamma,
                                **scheduler_params)
    elif scheduler_type == 'step':
        scheduler = StepLR(optim, step_size=optim_config.get('step', 30), gamma=gamma,
                           **scheduler_params)
    elif scheduler_type == 'plateau':
        if not scheduler_params:
            scheduler_params = {'threshold': 0.1}
        scheduler = ReduceLROnPlateau(optim, factor=gamma, mode='max', threshold_mode='abs',
                                      **scheduler_params)
    elif scheduler_type == 'poly':
        if not scheduler_params:
            scheduler_params = {'power': 0.9}
        power = scheduler_params.power
        poly_lambda = lambda epoch: (1 - epoch / config.epochs) ** power
        scheduler = LambdaLR(optim, poly_lambda)
    elif scheduler_type == 'exponential':
        scheduler = ExponentialLR(optim, gamma)
    else:
        raise KeyError("Unknown scheduler type: {}".format(scheduler_type))

    return optim, scheduler


def get_default_weight_decay(config):
    compression_configs = config.get('compression', {})
    if isinstance(compression_configs, dict):
        compression_configs = [compression_configs]

    weight_decay = 1e-4
    for compression_config in compression_configs:
        if compression_config.get('algorithm') == 'rb_sparsity':
            weight_decay = 0

    return weight_decay
