import torch
from contextlib import ExitStack, nullcontext

from .core import Model
from .utils import shared_dict, update_output

class Trainer:
    def __init__(self,
                 device = None, 
                 log_every_n_steps = 25, 
                 logger=None,
                 seed=1234,
                 float32_matmul_precision='high',
                 compile_model=False,
                 autocast_dtype=None,
                 grad_accum_steps=1,
                 **kwargs):

        # Device setup
        self.device = device       
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Computation setup
        self.float32_matmul_precision = float32_matmul_precision
        self.seed = seed
        self.autocast_dtype = autocast_dtype
        self.compile_model = compile_model

        # Logging setup
        self.log_every_n_steps = log_every_n_steps
        self.logger = logger

        # Optimizer setup
        self.grad_accum_steps = grad_accum_steps


    def _reset_random_seed(self):
        torch.manual_seed(self.seed)        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _forward_context(self, mode):
        forward_context = ExitStack()

        if self.autocast_dtype is not None:
            forward_context.enter_context(torch.autocast(device_type=self.device, dtype=self.autocast_dtype))

        return forward_context
        

    def _input_to_device(self, input_dict):
        # Move input_dict to device
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].to(self.device)

        return input_dict

    def _is_optimizer_step_complete(self, epoch_step):
        return (epoch_step + 1) % self.grad_accum_steps == 0

    def _backward_context(self, model, epoch_step):
        # In DDP we only want to sync gradients every grad_accum_steps
        if hasattr(model, 'no_sync') and self._is_optimizer_step_complete(epoch_step):
            return model.no_sync()
        else:
            return nullcontext()

    def step(self, mode, model, step, input_dict, optimizers=None, schedulers=None, gradient_clipping=None, logger=None):

        operations = model.operations.keys()

        input_dict = self._input_to_device(input_dict)
        output_dict = {}

        if logger:
            logger.log_value(f'{mode}_step', step)

        for operation_name in operations:

            with self._forward_context(mode):
                # Forward Pass
                operation_output = model(
                    input_dict = shared_dict(input_dict, output_dict),
                    operation_name = operation_name,
                    mode = mode,
                    )
                
                update_output( 
                    operation_output,
                    output_dict,
                    operation_name,
                )

                # Loss computation            
                loss = model.compute_loss(
                    x = shared_dict(input_dict, output_dict),
                    operation_name = operation_name,
                    mode = mode,
                    logger = logger,
                    )
            
                # Backward Pass / Accumulate Gradients
                if loss:
                    loss = loss / self.grad_accum_steps
                    
                    if optimizers:
                        with self._backward_context(model, step):
                            loss.backward()

            # Check if one optimizer step is completed
            if optimizers and self._is_optimizer_step_complete(step):
                opt = optimizers[ operation_name ]                    

                # Gradient Clipping
                if gradient_clipping and operation_name in gradient_clipping:
                    gradient_clipping[ operation_name ]()

                # Update the parameters via optimizer using the gradients
                opt.step()

                if operation_name in schedulers:
                    schedulers[ operation_name ].step_done()

                # Zero all gradients
                opt.zero_grad()
            
            # Metrics
            metrics = model.compute_metrics(
                x = shared_dict(input_dict, output_dict),
                operation_name = operation_name,
                mode = mode,
                logger = logger,
            )

        if logger:
            logger.step_done()

        return output_dict, metrics


    def epoch(self, mode, model, epoch, dataset, optimizers=None, schedulers=None, gradient_clipping=None, logger=None):
        operations = model.operations.keys()

        # Zero all gradients
        if optimizers:
            for operation_name in operations:
                optimizers[ operation_name ].zero_grad()

        if logger:
            logger.log_value(f'{mode}_epoch', epoch)
        
        for epoch_step, input_dict in enumerate(dataset):           
            self.step(mode=mode, model=model, step=epoch_step, input_dict=input_dict, optimizers=optimizers, schedulers=schedulers, gradient_clipping=gradient_clipping, logger=logger)

        # End of epoch
        for operation_name in operations:
            if schedulers and operation_name in schedulers:
                schedulers[ operation_name ].epoch_done()

        if logger:
            logger.epoch_done()        
                



    def train(self, model: Model, train_dataset, loader_args, max_epochs, val_dataset=None, logger=None):
        # General Setup and Random Seed Initialization
        self._reset_random_seed()        
        torch.set_float32_matmul_precision(self.float32_matmul_precision)

        # Move model to device & compile if needed
        model.to(self.device)       
        if self.compile_model:            
            model = torch.compile(model)

        optimizers, schedulers, gradient_clipping = model.configure_optimizers()

        # Train Loop
        for epoch_idx in range(max_epochs):
            model.train()
            train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_args)
            self.epoch(mode='train', model=model, epoch=epoch_idx, dataset=train_loader, optimizers=optimizers, schedulers=schedulers, gradient_clipping=gradient_clipping, logger=logger)

            if val_dataset:
                model.eval()
                val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_args)
                self.epoch(mode='val', model=model, epoch=epoch_idx, dataset=val_loader, logger=logger)