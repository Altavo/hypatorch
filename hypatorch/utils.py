import torch
import inspect
from omegaconf import DictConfig

from typing import Iterable
from typing import Union
from typing import Dict
from typing import List


def get_input_variable_names(func):
    full_args = inspect.getfullargspec(func)
    input_variables = full_args.args

    # Check if 'self' is in the list, if so remove it
    if 'self' in input_variables:
        input_variables.remove('self')

    # Get the default values for the optional arguments
    defaults = full_args.defaults if full_args.defaults else []
    optional_variables = input_variables[-len(defaults):]

    # The required arguments are all the input arguments minus the optional ones
    required_variables = input_variables[:-len(defaults)] if defaults else input_variables

    return required_variables, optional_variables

def get_output_variable_names(func):
    # Get the source code of the function
    source_lines = inspect.getsource(func)
    
    # Get the part after the return statement
    return_part = source_lines.split("return ")[1]

    # If no return statement or multiple return statements,
    # are found, raise an error
    if "return " not in source_lines or return_part.count("return ") > 1:
        raise ValueError(
            f"""
            The function must have a single return statement,
            but passed function has {return_part.count("return ")}
            return statements.
            """
            )
    
    # Split by commas and clean up the code
    returned_variables = return_part.split(",")
    returned_variables = [var.strip() for var in returned_variables]
    
    return returned_variables

def make_iterable(
        x,
        make_none_iterable = False,
        ):
    if x is None and not make_none_iterable:
        return x

    if isinstance( x, str ) or not isinstance( x, Iterable ):
        return [ x ]
    
    return x

def shared_dict(
        input_dict,
        output_dict,
        keys = None,
        ):
    # Check if keys overlap:
    if any( x in input_dict.keys() for x in output_dict.keys() ):
        raise ValueError(
            'Keys overlap between input and output dict. '
            'Please use different keys.'
            )
    x = dict( input_dict, **output_dict )
    #if keys is not None:
    #    x = { key: x[key] for key in keys }
    
    return x

def validate_io_keys(
        module_name,
        module_object_name,
        input_key_map,
        output_key_map,
        expected_inputs,
        expected_outputs,
        ):
    required_inputs, optional_inputs = expected_inputs
    input_keys = list( input_key_map.keys() )
    output_keys = list( output_key_map.keys() )
    # check if set( input_keys ) is a subset of set( expected_inputs )
    if not set( required_inputs ).issubset( set( input_keys ) ):
        raise ValueError(
            f"""
            {module_name} ({module_object_name}) expects {required_inputs}
            as required input_keys, but {input_keys} were given.
            """
            )
    if not set( output_keys ).issubset( set( expected_outputs ) ):
        raise ValueError(
            f"""
            {module_name} ({module_object_name}) expects {expected_outputs}
            as output_keys, but {output_keys} were given.
            """
            )
    return

def _get_module_input(
        k,
        v,
        data_dict,
        ):
    if isinstance( v, str ):
        try:
            submodule_in = data_dict[ v ]
        except KeyError:
            raise ValueError(
                f"""
                Error with input '{k}: {v}'.
                Input not found in data_dict. If you intended to use a string
                as an input, it should be passed as a dictionary like this:
                {k}: {{ 'value': '{v}', 'key_map': false }}
                """
                )
    elif isinstance( v, dict ) or isinstance(v, DictConfig):
        if v[ 'key_map' ]:
            try:
                submodule_in = data_dict[ v[ 'value' ] ]
            except KeyError:
                raise ValueError(
                    f"""
                    Error with input '{k}: {v}'.
                    Input not found in data_dict. If you intended to use a string
                    as an input, it should be passed as a dictionary like this:
                    {k}: {{ 'value': '{v['value']}', 'key_map': false }}
                    """
                    )
        else:
            submodule_in = v[ 'value' ]
    else:
        raise ValueError(
            f"""
            Error with input '{k}: {v}'.
            Input must be a string or a dictionary,
            bot got {type(v)}.
            """
            )
    return submodule_in

def get_module_input(
        inputs: Union[ Dict, DictConfig ],
        data_dict: Dict,
        ):
    submodule_in = {}
    for k, v in inputs.items():
        if isinstance( v, list ):
            submodule_in[ k ] = [
                _get_module_input(
                    k = k,
                    v = x,
                    data_dict = data_dict
                    ) for x in v
                ]
        else:
            submodule_in[ k ] = _get_module_input(
                k = k,
                v = v,
                data_dict = data_dict
                )
    return submodule_in

def create_mask(x, x_length):
    with torch.no_grad():
        S = []
        N = x.shape[-1]
        for n in x_length:
            s = torch.zeros([1, N])
            s[..., :n] = 1.0

            S.append(s)

        mask = torch.stack(S).to(x.device)
        ## make mask the same dtype as x
        mask = mask.to(x.dtype)

        return mask
    
def create_mask_same_shape( x, x_length ):
    if len(x.shape) < 2:
        raise ValueError( 'x must have at least 2 dimensions (B, L)' )
    
    if x.shape[0] != len(x_length):
        raise ValueError(
            'len(x_len) must be same as first dimension of x (batch size)'
            )
    
    with torch.no_grad():
        S = torch.zeros_like( x )
        for index, length in enumerate( x_length ):
            S[ index, ..., :length ] = 1.0

    S = S.to( x.device )
    S = S.to( x.dtype )

    return S

class LengthHarmonizer( torch.nn.Module ):
    def __init__(
            self,
            mismatch_treshold = 5,
            ):
        super().__init__()
        self.mismatch_treshold = mismatch_treshold
        return
    
    def forward(
            self,
            data,
            ):
        # data is a list of tensors
        # find the minimum length
        lengths = [ x.shape[-1] for x in data ]

        # check if lengths differ by a threshold, if so raise error
        if self.mismatch_treshold is not None:
            if max( lengths ) - min( lengths ) > self.mismatch_treshold:
                raise ValueError(
                    f"""
                    losses.LengthHarmonizer: lengths are {lengths}.
                    Lengths differ by more than {self.mismatch_treshold} samples.
                    A bug is likely.
                    """
                    )
            
        min_length = min( lengths )
        # truncate all tensors to minimum length
        # NOTE: assumes that last dimension is the length dimension
        # TODO: add support for other dimensions
        data = [ x[ ..., :min_length ] for x in data ]

        return data