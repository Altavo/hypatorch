import os
import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import ExitStack, nullcontext
import logging

from .core import Model
from .utils import shared_dict, update_output, get_rank, get_world_size

class Trainer:
    def __init__(self,
                 max_epochs,
                 device = None, 
                 log_every_n_steps = 25, 
                 logger=None,
                 seed=1234,
                 float32_matmul_precision='high',
                 compile_model=False,
                 autocast_dtype=None,
                 grad_accum_steps=1,
                 **kwargs):

        self.rank = get_rank()
        self.world_size = get_world_size()

        # Optimizer setup - parallelized gradient accumulation with world_size > 1
        if grad_accum_steps % self.world_size != 0:
            raise ValueError("world size %d must be a multiple of grad_accum_steps %d")

        self.grad_accum_steps = grad_accum_steps // self.world_size

        self._process_group_initialized = False

        # Device setup
        self.device = device       
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if (self.device == torch.device("cuda") or self.device == "cuda") and self.world_size > torch.cuda.device_count():
            raise ValueError(f"Trainer world_size is {self.world_size} but only {torch.cuda.device_count()} GPUs available.")

        logging.info(f"Using Device {self.device} with World Size {self.world_size}")

        self._init_process_group()

        # Computation setup
        self.float32_matmul_precision = float32_matmul_precision
        self.seed = seed
        self.autocast_dtype = autocast_dtype
        if isinstance(self.autocast_dtype, str):
            self.autocast_dtype = getattr(torch, self.autocast_dtype)
        self.compile_model = compile_model

        # Logging setup
        self.log_every_n_steps = log_every_n_steps
        self.logger = logger


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

    def __del__(self):
        self._cleanup_process_group()

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

    def _init_process_group(self):
        if self.world_size > 1:  
            torch.cuda.set_device(self.rank)
            torch.distributed.init_process_group("nccl", rank=self.rank, world_size=self.world_size)      
            self._process_group_initialized = True

    def _cleanup_process_group(self):
        if self._process_group_initialized:
            torch.distributed.destroy_process_group()

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
        # In DDP we only want to sync gradients when an optimizer step completes
        if hasattr(model, 'no_sync') and not self._is_optimizer_step_complete(epoch_step):
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

    def save_checkpoint(self, name, optimizers=None, schedulers=None, chkpt_dir=None):
        from . import __version__

        checkpoint = {}
        checkpoint['hypatorch_version'] = __version__
        checkpoint['state_dict'] = self.model.state_dict()
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

    def load_checkpoint(self, name, optimizers=None, schedulers=None, chkpt_dir=None, strict=True, set_rng_state=False):
        if chkpt_dir:
            name = os.path.join(chkpt_dir, name)
        checkpoint = torch.load(name, weights_only=True)

        self.model.load_checkpoint_state_dict(checkpoint['state_dict'], strict=strict)
        
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

    def step(self, mode, input_dict, optimizers=None, schedulers=None, gradient_clipping=None, logger=None):

        step, global_step = self._next_step(mode)

        if logger:
            logger.log_value(f'{mode}_step', step)

        # Move input_dict to device
        input_dict = self._input_to_device(input_dict)  
        output_dict = {}

        # Iterate over all operations
        for operation_name in self.model_operations.keys():

            with self._forward_context(mode):
                # Forward Pass
                operation_output = self.model(
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
                loss = self.compute_loss(
                    x = shared_dict(input_dict, output_dict),
                    operation_name = operation_name,
                    mode = mode,
                    logger = logger,
                    )
            
                # Backward Pass / Accumulate Gradients
                if loss:
                    loss = loss / self.grad_accum_steps
                    
                    if optimizers:
                        with self._backward_context(self.model, step):
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
            metrics = self.compute_metrics(
                x = shared_dict(input_dict, output_dict),
                operation_name = operation_name,
                mode = mode,
                logger = logger,
            )

        if logger:
            logger.step_done()

        return output_dict, metrics, step, global_step


    def epoch(self, mode, epoch, dataset, optimizers=None, schedulers=None, gradient_clipping=None, logger=None):

        # Zero all gradients
        if optimizers:
            for operation_name in self.model_operations:
                optimizers[ operation_name ].zero_grad()

        if logger:
            logger.log_value(f'{mode}_epoch', epoch)
        
        for epoch_step, input_dict in enumerate(dataset):           
            output_dict, metrics, step, global_step = self.step(mode=mode, input_dict=input_dict, optimizers=optimizers, schedulers=schedulers, gradient_clipping=gradient_clipping, logger=logger)

            # On the first step in validation log the data
            if logger and epoch_step == 0 and mode == 'val':
                self.log_data(data_dict=shared_dict(input_dict, output_dict), step=global_step, logger=logger)

        # End of epoch
        for operation_name in self.model_operations:
            if schedulers and operation_name in schedulers:
                schedulers[ operation_name ].epoch_done()

        if logger:
            logger.epoch_done()        
                

    def prepare_model_training(self, model: Model):
        # Move model to device & compile if needed
        model.to(self.device)       

        self.optimizers, self.schedulers, self.gradient_clipping = model.configure_optimizers()
        self.model_operations = model.operations
        self.log_data = model.log_data
        self.compute_loss = model.compute_loss
        self.compute_metrics = model.compute_metrics

        if self.world_size > 1:
            model = DDP(model, device_ids=[self.rank])

        if self.compile_model:
            print("Compiling Model")      
            model = torch.compile(model)


        torch.set_float32_matmul_precision(self.float32_matmul_precision)

        self.model = model


    def _training_loop(self, train_dataset, loader_args, val_dataset=None, logger=None, checkpoint_path=None):
        
        # Train Loop from self.epoch_idx to self.max_epochs
        for epoch_idx in range(self.epoch_idx, self.max_epochs):            
            if val_dataset:
                self.model.eval()
                val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_args)
                self.epoch(mode='val', epoch=epoch_idx, dataset=val_loader, logger=logger)

            self.model.train()
            train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_args)
            self.epoch(mode='train', epoch=epoch_idx, dataset=train_loader, optimizers=self.optimizers, schedulers=self.schedulers, gradient_clipping=self.gradient_clipping, logger=logger)

            # Set to next epoch 
            self.epoch_idx = epoch_idx + 1

            # TODO: Check for checkpointing or early stopping

        # Save last.ckpt
        last_chkpt = 'last.ckpt'
        if checkpoint_path:
            last_chkpt = os.path.join(checkpoint_path, last_chkpt)

        self.save_checkpoint(last_chkpt, self.optimizers, self.schedulers)

        self.rng_state_dict = self.get_rng_state_dict()
    

    def train(self, model: Model, train_dataset, loader_args, val_dataset=None, logger=None, checkpoint_path=None):
        # Reset RNG and Steps
        self._reset_random_seed()        
        self._reset_steps()

        # Prepare Model for Training
        self.prepare_model_training(model)

        # Train Loop
        self._training_loop(train_dataset=train_dataset, loader_args=loader_args, val_dataset=val_dataset, logger=logger, checkpoint_path=checkpoint_path)

    def resume_training(self, model: Model, chkpt_name:str, train_dataset, loader_args, val_dataset=None, logger=None, checkpoint_path=None):
        # Prepare Model for Training
        self.prepare_model_training(model)

        # Load Checkpoint
        self.load_checkpoint(name=chkpt_name, optimizers=self.optimizers, schedulers=self.schedulers, chkpt_dir=checkpoint_path, set_rng_state=True)

        # Train Loop
        self._training_loop(train_dataset=train_dataset, loader_args=loader_args, val_dataset=val_dataset, logger=logger, checkpoint_path=checkpoint_path)


    def continue_training(self, train_dataset, loader_args, val_dataset=None, logger=None, checkpoint_path=None):
        # Check that the model is already prepared for training
        if not self.optimizers or not self.model or not self.rng_state_dict:
            raise ValueError("No training can be continued. Please call train() or resume_training() before contine_training()")

        self.set_rng_state_dict(self.rng_state_dict)

        # Train Loop
        self._training_loop(train_dataset=train_dataset, loader_args=loader_args, val_dataset=val_dataset, logger=logger, checkpoint_path=checkpoint_path)