from abc import ABC, abstractmethod

class DataLogger(ABC):
    @abstractmethod
    def log_value(self, name:str, value:float):
        pass


class ConsoleLogger(DataLogger):
    def __init__(self, float_precision:int=4, log_step:bool=True, log_epoch:bool=True):
        super().__init__()

        self.log_step = log_step
        self.log_epoch = log_epoch

        self.step_log = {}
        self.epoch_log = {}

        self.epoch_no_log = []

        self.float_precision = float_precision

    def log_value(self, name:str, value:float|int):
        self.step_log[name] = value

    def _format(self, value) -> str:
        if isinstance(value, int):
            return str(value)

        if isinstance(value, float):
            return f"{value:.{self.float_precision}f}"
        
        if isinstance(value, list):
            mean_value = sum(value) / len(value)
            return f"{mean_value:.{self.float_precision}f}"
        

    def _update_epoch_log(self):
        for k, v in self.step_log.items():
            if k in self.epoch_no_log:
                continue
            
            if isinstance(v, int):
                if k in self.epoch_log and self.epoch_log[k] != v:
                    self.epoch_no_log.append(k)
                    self.epoch_log.pop(k)
                else:
                    self.epoch_log[k] = v
            else:
                if k not in self.epoch_log:
                    self.epoch_log[k] = []
                self.epoch_log[k].append(v)

    def step_done(self):
        if self.log_step:
            # Create a string with | separated key value pairs
            log_str = " | ".join([f"{k}={self._format(v)}" for k, v in self.step_log.items()])
            print("step: " + log_str)

        # Update the epoch log
        self._update_epoch_log()


    def epoch_done(self):
        if self.log_epoch:
            # Create a string with | separated key value pairs
            log_str = " | ".join([f"{k}={self._format(v)}" for k, v in self.epoch_log.items()])
            print("epoch: " + log_str)

        # Clear all logs
        self.step_log = {}
        self.epoch_log = {}
        self.epoch_no_log = []



