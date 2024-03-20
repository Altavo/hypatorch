import torch
import logging

from .utils import create_mask, create_mask_same_shape
from .utils import LengthHarmonizer
from .utils import get_module_input

from typing import Optional
from typing import Union
from typing import Dict
from typing import List


class MaskedAssessment(torch.nn.Module):
    def __init__(
            self, 
            unmasked_assessment=torch.nn.MSELoss(reduction='sum'),
            reduction='mean',
            ):
        super(MaskedAssessment, self).__init__()
        self._assessment = unmasked_assessment
        self.reduction = reduction

        if hasattr(self._assessment, 'reduction'):
            if self._assessment.reduction != 'sum':
                raise ValueError(
                    f"""
                    The module passed to 'unmasked_assessment' must
                    have an input argument called 'reduction=sum'.
                    Otherwise, the module is not compatible with the
                    masking mechanism.
                    """
                    )
        else:
            raise ValueError(
                    f"""
                    The module passed to 'unmasked_assessment' must
                    have an input argument called 'reduction=sum'.
                    Otherwise, the module is not compatible with the
                    masking mechanism.
                    """
                    )

        # Masked reduction must be either "sum" or "mean"
        if self.reduction not in ['sum', 'mean']:
            raise ValueError(
                f"""
                Input argument 'reduction' must be either 'sum' or 'mean',
                but got {self.reduction}.
                """
                )
        
        return

    def forward(
            self,
            inputs,
            mask,
            apply,
            ):
        if not apply:
            raise ValueError(
                f"""
                Input argument 'apply' must be a list of keys
                of the inputs to be masked, but got {apply}.
                """
                )
        for k in apply:
            inputs[k] = inputs[k] * mask
        
        sum_assessment = self._assessment( **inputs )
                
        if self.reduction == 'mean':
            # multiply all elements of x.shape[:-1]
            normalize = 1
            for i in inputs[ apply[0] ].shape[:-1]:
                normalize *= i

            return sum_assessment / (normalize * mask.flatten().sum())
        else:
            return sum_assessment
    
class HypaAssessment( torch.nn.Module ):

    def __init__(
            self,
            assessment: torch.nn.Module,
            name: str,
            inputs: List[
                Union[
                    Dict[ str, str ],
                    Dict[ str, Dict ],
                    ],
                ],
            harmonize_inputs: List[ str ],
            masking: Optional[Dict] = None,
            weight = 1.0,
            apply: Optional[List[str]] = None,
            ):
        super().__init__()
        self.apply = apply
        self.weight = weight
        self.name = name
        self.inputs = inputs
        self.harmonize_inputs = harmonize_inputs
        self.masking = masking
        if masking is not None:
            self.assessment = MaskedAssessment(
                unmasked_assessment=assessment,
                )
        else:
            self.assessment = assessment
        self.harmonize_lengths =  LengthHarmonizer()
        return

    def forward(
            self,
            data_dict,
            ):
        
        # Get inputs
        assessment_in = get_module_input(
            inputs = self.inputs,
            data_dict = data_dict,
            )
        
        # Masking
        if self.masking is not None:
            mask = create_mask(
                data_dict[ self.inputs[ self.masking[ 'apply' ][0] ] ],
                data_dict[ self.masking[ 'len_key'] ],
                )

        # Harmonize lengths
        if self.harmonize_inputs:
            h_in = [ assessment_in[ k ] for k in self.harmonize_inputs ]
            if self.masking is not None:
                h_in.append( mask )
            h_out = self.harmonize_lengths( h_in )
            for k, v in zip(self.harmonize_inputs, h_out):
                assessment_in[ k ] = v

            if self.masking is not None:
                mask = h_out[-1]
        
        if self.masking is not None:
            x = self.assessment(
                inputs = assessment_in,
                mask = mask,
                apply = self.masking[ 'apply' ],
                )
        else:
            x = self.assessment( **assessment_in )

        return x * self.weight