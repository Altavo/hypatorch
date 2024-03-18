import torch
import logging

from .assessments import HypaAssessment

from typing import Optional
from typing import Union
from typing import Dict
from typing import List

class MAE_Loss( HypaAssessment ):
    def __init__(
            self,
            inputs,
            weight = 1.0,
            ):
        super().__init__(
            assessment = torch.nn.L1Loss(reduction='mean'),
            name = f'MAE_{inputs[ "input" ]}_{inputs[ "target" ]}',
            inputs = inputs,
            harmonize_inputs = [
                'input',
                'target',
                ],
            masking = None,
            weight = weight,
            )
        return
    
class MMAE_Loss( HypaAssessment ):
    def __init__(
            self,
            inputs,
            len_key,
            weight = 1.0,
            ):
        super().__init__(
            assessment = torch.nn.L1Loss(reduction='sum'),
            name = f'MMAE_{inputs["input"]}_{inputs["target"]}',
            inputs = inputs,
            harmonize_inputs = [
                'input',
                'target',
                ],
            masking = dict(
                apply = [
                    'input',
                    'target',
                    ],
                len_key = len_key,
            ),
            weight = weight,
            )
        return

class MSE_Loss( HypaAssessment ):
    def __init__(
            self,
            inputs,
            weight = 1.0,
            ):
        super().__init__(
            assessment = torch.nn.MSELoss(reduction='mean'),
            name = f'MSE_{inputs[ "input" ]}_{inputs[ "target" ]}',
            inputs = inputs,
            harmonize_inputs = [
                'input',
                'target',
                ],
            masking = None,
            weight = weight,
            )
        return

class MMSE_Loss( HypaAssessment ):
    def __init__(
            self,
            inputs,
            len_key,
            weight = 1.0,
            ):
        super().__init__(
            assessment = torch.nn.MSELoss(reduction='sum'),
            name = f'MMSE_{inputs["input"]}_{inputs["target"]}',
            inputs = inputs,
            harmonize_inputs = [
                'input',
                'target',
                ],
            masking = dict(
                apply = [
                    'input',
                    'target',
                    ],
                len_key = len_key,
            ),
            weight = weight,
            )
        return