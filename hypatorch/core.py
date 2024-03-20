import os
import torch
import lightning as L
from contextlib import nullcontext
from omegaconf import DictConfig, ListConfig
from hydra import compose, initialize
from hydra.utils import instantiate

import logging
from collections.abc import Iterable
import yaml

from typing import List, Dict, Optional, Union, Callable, Any, Tuple

from .utils import get_input_variable_names
from .utils import get_output_variable_names
from .utils import shared_dict
from .utils import validate_io_keys
from .utils import get_module_input



class Model( L.LightningModule ):
    def __init__(
            self,
            # core functionality
            submodules: Union[
                Dict[ str, str ],
                Dict[ str, torch.nn.Module ],
                ],
            operations: List[ List[ Dict[ str, Dict ] ] ],

            # training related
            exclude_from_checkpoint: Optional[ List[str] ] = None,
            accumulate_grad_batches: Optional[ int ] = None,
            gradient_clip_val: Optional[ float ] = 5.0,
            checkpoints: Optional[ List[ Dict[ str, str ] ] ] = None,
            ):
        
        super().__init__()
        
        # Processing modules and core functionality
        self.submodule_names = list( submodules.keys() )

        for sm_name, sm in submodules.items():
            if isinstance( sm, str ):
                # load yaml file from string and instantiate its content
                # TODO: this may only work with absolute paths, make it work with relative paths
                y = yaml.load( sm, safe_load = True )
                sm = instantiate( y )

            if not isinstance( sm, torch.nn.Module ):
                raise ValueError(
                    f"""
                    The submodule {sm_name} must be a torch.nn.Module 
                    or a file path to a yaml file that can be instantiated
                    into a torch.nn.Module. However, the passed submodule
                    is of type {type(sm)}.
                    """
                    )

            setattr( self, sm_name, sm )

        logging.info(
            f"""
            Created a hypaTorch model from the following submodules:
            {self.submodule_names}
            """
            )

        self.operations = operations
        self.mappings = self._get_content( 'mappings' )

        self.logging = self._get_content( 'logging' )


        ## Losses and metrics
        self.losses = self._get_content( 'losses' )
        self.metrics = self._get_content( 'metrics' )


        # Lightning does not accept gradient accumulation or clipping
        # as arguments if optimization is manual. Therefore, we need to
        # implement these functionalities manually.
        self.accumulate_grad_batches = accumulate_grad_batches
        self.gradient_clip_val = gradient_clip_val


        # Checkpoints
        self.checkpoints = checkpoints if checkpoints is not None else []

        # Freeze submodules if specified
        optimized_sm_list = []
        for x in self._get_content(
            'optimize_submodules',
            return_on_failure = {},
            ).values():
            optimized_sm_list.extend( x )
        optimized_sm_set = set( optimized_sm_list )
        logging.info(
            f"""
            All submodules that get optimized by at least
            one optimizer: {optimized_sm_set}
            """
            )

        # self frozen is the disjoin of the optimized submodules and submodule_names
        self.frozen = list( set( self.submodule_names ) - optimized_sm_set )
        logging.info(
            f"""
            All submodules that get optimized by at least
            one optimizer: {optimized_sm_set}
            """
            )
        self._freeze_submodules( self )

        # Exclude from checkpoint
        if exclude_from_checkpoint is None:
            exclude_from_checkpoint = []
        self.exclude_from_checkpoint = exclude_from_checkpoint

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        return
    
    @classmethod
    def from_config(
            cls,
            config_path: str,
            ):
        with initialize(
            version_base=None,
            config_path="conf",
            job_name="test_app",
            ):
            cfg = compose( config_name=config_path )
        
        # Instantiate the model
        if 'model' in cfg:
            model = instantiate( cfg.model )
        else:
            model = instantiate( cfg )

        model._load_checkpoints(
            checkpoints_dict = cfg.checkpoints,
            )
        
        return model
    
    def _load_checkpoints( self ):
        for ckpt_dict in self.checkpoints:

            ckpt =torch.load( ckpt_dict[ 'path' ] )[ 'state_dict' ]
            if 'prefix_rm' in ckpt_dict:
                prefix_rm = ckpt_dict[ 'prefix_rm' ]
                if prefix_rm.endswith( '.' ):
                    # delete all trailing dots
                    prefix_rm = prefix_rm.rstrip( '.' )
                ckpt = {
                    k[ len( prefix_rm ) + 1: ]: v 
                    for k, v in ckpt.items()
                    if k.startswith( prefix_rm )
                    }
            if 'prefix_add' in ckpt_dict:
                prefix_add = ckpt_dict[ 'prefix_add' ]
                if prefix_add.endswith( '.' ):
                    prefix_add = prefix_add.rstrip( '.' )
                prefix_add = prefix_add + '.'
                ckpt = {
                    prefix_add + k: v 
                    for k, v in ckpt.items()
                    }
            try:
                self.load_state_dict(
                    ckpt,
                    strict = True,
                    )
            except Exception as e:
                logging.warning(
                    f"""
                    Could not load the checkpoint {ckpt_dict[ 'path' ]}
                    in 'strict' mode. Trying again with 'strict = False'.
                    Enable logging.DEBUG level for more information.
                    """
                    ) 
                logging.debug( f'Due to the following error: {e}.' )
                self.load_state_dict(
                    ckpt,
                    strict = False,
                    )
        return
    
    def _get_content(
            self,
            key,
            return_on_failure = [],
            ):
        x = {}

        for op_name, op in self.operations.items():
            if key in op:
                x[ op_name ] = op[ key ]
            else:
                x = return_on_failure
        return x
    
    def _collect_trainable_parameters(
            self,
            submodule_names,
            ):
        parameters = []
        
        for submodule_name in submodule_names:
            submodule = getattr(self, submodule_name)
            if submodule_name not in self.frozen:
                parameters.extend( submodule.parameters() )

        return parameters
    
    def _compute_assessments(
            self,
            data_dict,
            assessments,
            mode,
            ):
        x = {}
        
        for fn in assessments:
            if fn.apply is None or mode in fn.apply:
                y = fn(
                    data_dict = data_dict,
                    )
                x[ fn.name ] = y

        return x
    
    def _check_if_submodule_gets_applied(
            self,
            mode,
            mapping_dict,
            ):
        if 'apply' in mapping_dict:
            if mode in mapping_dict[ 'apply' ]:
                return True
            else:
                return False
        else:
            return True
    
    def _freeze_submodules(
            self,
            module,
            ):
        for x in self.frozen:
            module = getattr(self, x)
            for param in module.parameters():
                param.requires_grad = False
        return
    
    def configure_optimizers(self):

        optimizers = []

        for op_name, opt in self.operations.items():
            # List of parameters to optimize
            parameters = self._collect_trainable_parameters(
                submodule_names = opt[ 'optimize_submodules' ],
            )

            optimizer = opt[ 'optimizer' ](parameters)


            d = {'optimizer': optimizer}
            if opt[ 'lr_scheduler' ] is not None:
                # Instantiate the partially instantiated LR scheduler
                sch = {
                    'scheduler': opt[ 'lr_scheduler' ]['scheduler'](optimizer),
                }

                # add the other keys to the dict
                for k, v in opt[ 'lr_scheduler' ].items():
                    if k != 'scheduler':
                        sch[k] = v

                d['lr_scheduler'] = sch
            
            optimizers.append( d )

        return optimizers

    def _run_submodule(
            self,
            submodule,
            submodule_name,
            mapping,
            data_dict,
            ):
        
        frozen = submodule_name in self.frozen
        calculate_grad = mapping[ submodule_name ][ 'calculate_grad' ]
        if not calculate_grad or ( frozen and not calculate_grad ):
            context = torch.no_grad()
        else:
            context = nullcontext()
        with context:
            if 'fn' in mapping[ submodule_name ]:
                fn = getattr( submodule, mapping[ submodule_name ][ 'fn' ] )
                fn_to_inspect = fn
            else:
                fn_to_inspect = submodule.forward
                fn = submodule.__call__

            output_key_map = mapping[ submodule_name ][ 'outputs' ]
            inputs = mapping[ submodule_name][ 'inputs' ]

            expected_inputs = get_input_variable_names( fn_to_inspect )
            expected_outputs = get_output_variable_names( fn_to_inspect )

            validate_io_keys(
                module_name = submodule_name,
                module_object_name = submodule.__class__.__name__,
                input_key_map = inputs,
                output_key_map = output_key_map,
                expected_inputs = expected_inputs,
                expected_outputs = expected_outputs,
                )
            
            submodule_in = get_module_input(
                inputs = inputs,
                data_dict = data_dict,
                )

            if 'fn' in mapping[ submodule_name ]:
                submodule_out = fn( **submodule_in )
            else:
                submodule_out = submodule( **submodule_in )

            # if submodule is not a tuple, make it a tuple of length 1
            if not isinstance( submodule_out, tuple ):
                submodule_out = ( submodule_out, )

            if len( submodule_out ) != len( expected_outputs ):
                raise ValueError(
                    f"""
                    Error with output of {submodule_name}.
                    Expected {len( expected_outputs )} outputs,
                    but got {len( submodule_out )} outputs.
                    """
                    )
            
            sm_out_dict = { key: value for key, value in zip( expected_outputs, submodule_out ) }
            x = { key_map: sm_out_dict[ key ] for key, key_map in output_key_map.items() }

        return x
    
    def forward(
            self,
            input_dict,
            mappings,
            mode,
            ):

        output_dict = {}

        for mapping in mappings:
            for submodule_name in mapping.keys():
                submodule = getattr(self, submodule_name)
                if submodule is not None:
                    apply_submodule = self._check_if_submodule_gets_applied(
                        mode = mode,
                        mapping_dict = mapping[ submodule_name ],
                        )
                    if apply_submodule:
                        data_dict = shared_dict(
                            input_dict,
                            output_dict,
                            )
                        x = self._run_submodule(
                            submodule = submodule,
                            submodule_name = submodule_name,
                            mapping = mapping,
                            data_dict = data_dict,
                        )
                        # check that x.keys do not overlap with output_dict.keys
                        if not any( x in output_dict.keys() for x in x.keys() ):
                            output_dict.update( x )
                        else:
                            raise ValueError(
                                f"""
                                Error with output_dict of {submodule_name}.
                                Submodules are not allowed to overwrite existing keys.
                                However, {submodule_name} has the following keys: {x.keys()}
                                and the output_dict has the following keys: {output_dict.keys()}.
                                """
                                )
                    #else:
                    #    print( f'module {submodule_name} not applied')
                else:
                    raise ValueError(
                        f'No submodule {submodule_name} found. Forgot to define it?'
                        )

        return output_dict

    def training_step(self, batch, batch_idx):

        mode = 'train'
        
        input_dict = batch

        opts = self.optimizers()

        # check if opts is iterable, if not, make it iterable
        if not isinstance( opts, list ):
            opts = [ opts ]

        for operation_idx, _ in enumerate( self.operations ):
            operation_name = list( self.operations.keys() )[ operation_idx ]

            # Forward Pass
            output_dict, loss = self._forward_pass(
                input_dict = input_dict,
                operation_name = operation_name,
                mode = mode,
                )

            # Backward Pass if self.losses is not empty list
            opt = opts[ operation_idx ]
            if self.losses:
                #opt = opts[ operation_idx ]
                self._backward_pass(
                    opt = opt,
                    loss = loss,
                    batch_idx = batch_idx,
                    )
            else:
                opt.step()
            
            self._handle_assessments(
                assessments = self.metrics,
                data_dict = shared_dict(
                    input_dict,
                    output_dict,
                    ),
                operation_name = operation_name,
                mode = mode,
                )

        return loss

    def validation_step(self, batch, batch_idx):

        mode = 'val'
        
        input_dict = batch

        for operation_idx, _ in enumerate( self.operations ):
            operation_name = list( self.operations.keys() )[ operation_idx ]
            # Forward Pass
            with torch.no_grad():
                output_dict, loss = self._forward_pass(
                    input_dict = input_dict,
                    operation_name = operation_name,
                    mode = mode,
                    )
            
            # handle metrics
            self._handle_assessments(
                assessments = self.metrics,
                data_dict = shared_dict(
                    input_dict,
                    output_dict,
                    ),
                operation_name = operation_name,
                mode = mode,
                )
            
            # Once per epoch, log the first batch
            if batch_idx == 0 and self.logging:
                loggings = self.logging[ operation_name ]
                if loggings:
                    if not isinstance( loggings, list ) and not isinstance( loggings, ListConfig ):
                        loggings = [ loggings ]
                    for logging in loggings:

                        fn_name = logging[ 'fn' ]
                        fn_args = { k: v for k, v in logging.items() if k != 'fn' }

                        log_fn = getattr(
                            self.logger,
                            fn_name,
                        )
                        log_fn(
                            data_dict = shared_dict(
                                input_dict,
                                output_dict,
                                ),
                            global_step = self.global_step,
                            **fn_args,
                            )
        
        return loss
    
    def _backward_pass(
            self,
            opt,
            loss,
            batch_idx,
            ):
        if self.accumulate_grad_batches is None:
            opt.zero_grad()
            self.manual_backward( loss )
            #Implement gradient clipping
            self.clip_gradients(
                opt,
                gradient_clip_val = self.gradient_clip_val,
                gradient_clip_algorithm='value',
                )
            opt.step()
        else:
            N = self.accumulate_grad_batches
            loss = loss / N
            self.manual_backward( loss )
            # accumulate gradients
            if ( batch_idx + 1 ) % N == 0:
                #Implement gradient clipping
                self.clip_gradients(
                    opt,
                    gradient_clip_val = self.gradient_clip_val,
                    gradient_clip_algorithm='value',
                    )
                opt.step()
                opt.zero_grad()
        return
    
    def _forward_pass(
            self,
            input_dict,
            operation_name,
            mode,
            ):
        output_dict = self(
            input_dict,
            self.mappings[ operation_name ],
            mode = mode,
            )
        loss_dict = self._handle_assessments(
            assessments = self.losses,
            data_dict = shared_dict(
                input_dict,
                output_dict,
                ),
            operation_name = operation_name,
            mode = mode,
            )
        if loss_dict:
            loss = sum( loss_dict.values() )
        else:
            loss = None
        return output_dict, loss
    
    def _handle_assessments(
            self,
            assessments,
            data_dict,
            operation_name,
            mode,
            ):
        if assessments:
            assessments_dict = self._compute_assessments(
                data_dict = data_dict,
                assessments = assessments[ operation_name ],
                mode = mode,
                )
            for k, v in assessments_dict.items():
                self.log(
                    f'{mode}_{k}',
                    v,
                    on_epoch=True,
                    on_step=True,
                    prog_bar=True,
                    logger=True,
                    #sync_dist=False, #TODO: check if this is necessary
                )
        else:
            assessments_dict = None
        return assessments_dict

    def predict_step(self, batch, batch_idx):

        mode = 'test'

        input_dict = batch

        for operation_idx, _ in enumerate( self.operations ):
            operation_name = list( self.operations.keys() )[ operation_idx ]
            # Forward Pass
            with torch.no_grad():
                output_dict, loss = self._forward_pass(
                    input_dict = input_dict,
                    operation_name = operation_name,
                    mode = mode,
                    )
            
        data_dict = shared_dict(
            input_dict,
            output_dict,
            )
            
        return data_dict
    
    def on_save_checkpoint(self, checkpoint):
        for submodule in self.exclude_from_checkpoint:
            keys_to_remove = [
                k for k in checkpoint['state_dict'].keys() if k.startswith(submodule + '.')
                ]
            for key in keys_to_remove:
                checkpoint['state_dict'].pop(key, None)