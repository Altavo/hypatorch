import os
import torch
from contextlib import ExitStack, nullcontext

from .core import Model
from .utils import shared_dict, update_output

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

class Trainer:
    def __init__(self,
                 max_epochs,
                 device = None, 
                 logger=None,
                 seed=1234,
                 float32_matmul_precision='high',
                 compile_model=False,
                 distributed=False,
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
        if isinstance(self.autocast_dtype, str):
            self.autocast_dtype = getattr(torch, self.autocast_dtype)
        self.compile_model = compile_model
        self.distributed = distributed

        # Logging setup
        self.logger = logger

        # Optimizer setup
        self.grad_accum_steps = grad_accum_steps

        # Step
        self.global_step = 0
        self.train_step = 0
        self.val_step = 0
        self.epoch_idx = 0

        # Termination
        self.max_epochs = max_epochs

        # Training State
        self.optimizers = None
        self.schedulers = None
        self.gradient_clipping = None
        self.model = None
        self.rng_state_dict = None

    def _reset_steps(self):
        self.global_step = 0
        self.train_step = 0
        self.val_step = 0   
        self.epoch_idx = 0

    def _reset_random_seed(self):
        torch.manual_seed(self.seed)        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def get_rng_state_dict(self):
        rng_state_dict = {
            'torch_rng_state': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state_dict['torch_cuda_rng_state'] = torch.cuda.get_rng_state_all()
        return rng_state_dict
    
    def set_rng_state_dict(self, rng_state_dict):
        torch.set_rng_state(rng_state_dict['torch_rng_state'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state_dict['torch_cuda_rng_state'])

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

    def _next_step(self, mode):
        current_step = self.train_step if mode == 'train' else self.val_step
        current_global_step = self.global_step

        if mode == 'train':
            self.train_step += 1
        else:
            self.val_step += 1

        self.global_step += 1

        return current_step, current_global_step

    def save_checkpoint(self, name, model, optimizers=None, schedulers=None, chkpt_dir=None):
        from . import __version__

        checkpoint = {}
        checkpoint['hypatorch_version'] = __version__
        checkpoint['state_dict'] = model.state_dict()
        if optimizers:
            checkpoint['optimizers'] = {k: v.state_dict() for k, v in optimizers.items()}
        if schedulers:
            checkpoint['lr_schedulers'] = {k: v.state_dict() for k, v in schedulers.items()}            
        checkpoint['global_step'] = self.global_step
        checkpoint['train_step'] = self.train_step
        checkpoint['val_step'] = self.val_step
        checkpoint['epoch_idx'] = self.epoch_idx
        checkpoint['rng_state'] = self.get_rng_state_dict()

        # Save to output directory
        if chkpt_dir:
            name = os.path.join(chkpt_dir, name)
        torch.save(checkpoint, name)

    def load_checkpoint(self, name, model, optimizers=None, schedulers=None, chkpt_dir=None, strict=True, set_rng_state=False):
        if chkpt_dir:
            name = os.path.join(chkpt_dir, name)
        checkpoint = torch.load(name)

        model.load_checkpoint_state_dict(checkpoint['state_dict'], strict=strict)
        
        if optimizers:
            for k, v in optimizers.items():
                if k not in checkpoint['optimizers']:
                    raise ValueError(f"Optimizer {k} not found in checkpoint")
                
                v.load_state_dict(checkpoint['optimizers'][k])
        
        if schedulers:
            for k, v in schedulers.items():
                if k not in checkpoint['lr_schedulers']:
                    raise ValueError(f"LR Scheduler {k} not found in checkpoint")
                
                v.load_state_dict(checkpoint['lr_schedulers'][k])

        self.global_step = checkpoint['global_step']
        self.train_step = checkpoint['train_step']
        self.val_step = checkpoint['val_step']
        self.epoch_idx = checkpoint['epoch_idx']

        if set_rng_state and 'rng_state' in checkpoint:
            self.set_rng_state_dict(checkpoint['rng_state'])

    def step(self, mode, model, input_dict, optimizers=None, schedulers=None, gradient_clipping=None, logger=None):

        step, global_step = self._next_step(mode)

        if logger:
            logger.log_value(f'{mode}_step', step)

        # Move input_dict to device
        input_dict = self._input_to_device(input_dict)  
        output_dict = {}

        # Iterate over all operations
        for operation_name in model.operations.keys():

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

        return output_dict, metrics, step, global_step


    def epoch(self, mode, model, epoch, dataset, optimizers=None, schedulers=None, gradient_clipping=None, logger=None):
        operations = model.operations.keys()

        # Zero all gradients
        if optimizers:
            for operation_name in operations:
                optimizers[ operation_name ].zero_grad()

        if logger:
            logger.log_value(f'{mode}_epoch', epoch)
        
        for epoch_step, input_dict in enumerate(dataset):           
            output_dict, metrics, step, global_step = self.step(mode=mode, model=model, input_dict=input_dict, optimizers=optimizers, schedulers=schedulers, gradient_clipping=gradient_clipping, logger=logger)

            # On the first step in validation log the data
            if logger and epoch_step == 0 and mode == 'val':
                model.log_data(data_dict=shared_dict(input_dict, output_dict), step=global_step, logger=logger)

        # End of epoch
        for operation_name in operations:
            if schedulers and operation_name in schedulers:
                schedulers[ operation_name ].epoch_done()

        if logger:
            logger.epoch_done()        
                

    def _prepare_model_training(self, model: Model):
        # Move model to device & compile if needed
        model.to(self.device)       
        
        # Enable DDP before compiling
        if model.distributed:
            model = DDP(model, device_ids=[self.device])
        
        if self.compile_model:
            print("Compiling Model")      
            model = torch.compile(model)

        self.optimizers, self.schedulers, self.gradient_clipping = model.configure_optimizers()

        torch.set_float32_matmul_precision(self.float32_matmul_precision)

        self.model = model


    def _training_loop(self, train_dataset, loader_args, val_dataset=None, logger=None, checkpoint_path=None):
        
        # Train Loop from self.epoch_idx to self.max_epochs
        for epoch_idx in range(self.epoch_idx, self.max_epochs):            
            if val_dataset:
                self.model.eval()
                val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_args)
                self.epoch(mode='val', model=self.model, epoch=epoch_idx, dataset=val_loader, logger=logger)

            self.model.train()
            train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_args)
            self.epoch(mode='train', model=self.model, epoch=epoch_idx, dataset=train_loader, optimizers=self.optimizers, schedulers=self.schedulers, gradient_clipping=self.gradient_clipping, logger=logger)

            # Set to next epoch 
            self.epoch_idx = epoch_idx + 1

            # TODO: Check for checkpointing or early stopping

        # Save last.ckpt
        last_chkpt = 'last.ckpt'
        if checkpoint_path:
            last_chkpt = os.path.join(checkpoint_path, last_chkpt)

        self.save_checkpoint(last_chkpt, self.model, self.optimizers, self.schedulers)

        self.rng_state_dict = self.get_rng_state_dict()
    

    def train(self, model: Model, train_dataset, loader_args, val_dataset=None, logger=None, checkpoint_path=None):
        # Reset RNG and Steps
        self._reset_random_seed()        
        self._reset_steps()

        # Prepare Model for Training
        self._prepare_model_training(model)

        # Train Loop
        self._training_loop(train_dataset=train_dataset, loader_args=loader_args, val_dataset=val_dataset, logger=logger, checkpoint_path=checkpoint_path)

    def resume_training(self, model: Model, chkpt_name:str, train_dataset, loader_args, val_dataset=None, logger=None, checkpoint_path=None):
        # Prepare Model for Training
        self._prepare_model_training(model)

        # Load Checkpoint
        self.load_checkpoint(name=chkpt_name, model=model, optimizers=self.optimizers, schedulers=self.schedulers, chkpt_dir=checkpoint_path, set_rng_state=True)

        # Train Loop
        self._training_loop(train_dataset=train_dataset, loader_args=loader_args, val_dataset=val_dataset, logger=logger, checkpoint_path=checkpoint_path)


    def continue_training(self, train_dataset, loader_args, val_dataset=None, logger=None, checkpoint_path=None):
        # Check that the model is already prepared for training
        if not self.optimizers or not self.model or not self.rng_state_dict:
            raise ValueError("No training can be continued. Please call train() or resume_training() before contine_training()")

        self.set_rng_state_dict(self.rng_state_dict)

        # Train Loop
        self._training_loop(train_dataset=train_dataset, loader_args=loader_args, val_dataset=val_dataset, logger=logger, checkpoint_path=checkpoint_path)