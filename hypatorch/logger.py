from abc import ABC, abstractmethod

class DataLogger(ABC):

    def __init__(self, log_every_n_steps:int=1):
        self.log_every_n_steps = log_every_n_steps

        self._reset()

    def _reset(self):
        self._step_log = {}
        self._epoch_log = {}
        self._epoch_no_log = []
        self._epoch_step = 0

    def log_value(self, name:str, value:float|int):
        self._step_log[name] = value

    @abstractmethod
    def report_step(self):
        pass

    @abstractmethod
    def report_epoch(self):
        pass

    def _update_epoch_log(self):
        for k, v in self._step_log.items():
            if k in self._epoch_no_log:
                continue
            
            if isinstance(v, int):
                if k in self._epoch_log and self._epoch_log[k] != v:
                    self._epoch_no_log.append(k)
                    self._epoch_log.pop(k)
                else:
                    self._epoch_log[k] = v
            else:
                if k not in self._epoch_log:
                    self._epoch_log[k] = []
                self._epoch_log[k].append(v)

    def step_done(self):
        if self._epoch_step % self.log_every_n_steps == 0:
            self.report_step()

        self._epoch_step += 1

        self._update_epoch_log()

    def epoch_done(self):
        self.report_epoch()

        self._reset()    
    

    def step_items(self):
        # Generator that yields key value pairs
        for k, v in self._step_log.items():
            if not k.endswith("_step") and not k.endswith("_epoch"):
                k = f"{k}_step"
            
            yield k, v

    def epoch_items(self):
        # Generator that yields key value pairs
        for k, v in self._epoch_log.items():
            if not k.endswith("_step") and not k.endswith("_epoch"):
                k = f"{k}_epoch"
            
            if isinstance(v, list):
                v = sum(v) / len(v)

            yield k, v


class ConsoleLogger(DataLogger):
    def __init__(self, log_every_n_steps:int=1, float_precision:int=4):
        super().__init__(log_every_n_steps=log_every_n_steps)
        self.float_precision = float_precision


    def _format(self, value) -> str:
        if isinstance(value, int):
            return str(value)

        if isinstance(value, float):
            return f"{value:.{self.float_precision}f}"
        
        if isinstance(value, list):
            mean_value = sum(value) / len(value)
            return f"{mean_value:.{self.float_precision}f}"

    def report_step(self):
        # Create a string with | separated key value pairs
        log_str = " | ".join([f"{k}={self._format(v)}" for k, v in self.step_items()])
        print("step: " + log_str)


    def report_epoch(self):
        # Create a string with | separated key value pairs
        log_str = " | ".join([f"{k}={self._format(v)}" for k, v in self.epoch_items()])
        print("epoch: " + log_str)




