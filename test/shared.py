import sys
import os

class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass

class add_envs():
    def __init__(self, **kwargs):
        self.envs = kwargs
        self.restore_envs = {}
    
    def __enter__(self):        
        for k,v in self.envs.items():
            if k in os.environ:
                self.restore_envs[k] = os.environ[k]
            
            os.environ[k] = str(v)

    def __exit__(self, exc_type, exc_value, traceback):
        for k in self.envs:
            if k in os.environ:
                if k in self.restore_envs:
                    os.environ[k] = self.restore_envs[k]
                else:
                    del os.environ[k]
                                                    