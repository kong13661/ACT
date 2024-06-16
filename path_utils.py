from uuid import uuid4
import os
import __main__


class classproperty(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


def create_path_decorator(func):
    def create_path(self):
        path = func(self)
        if self.create_path:
            if not os.path.exists(path):
                os.makedirs(path)
        return path
    return create_path


class PathAuxiliary:
    root = os.path.join(os.environ['Utils_path'], 'utils') if 'Utils_path' in os.environ else ''

    def __init__(self, dataset_path, ckpt_path, create_path=False, debug=False):
        self.project_name = PathAuxiliary.project_name_current
        self.table_name = PathAuxiliary.table_name_current
        self.config_path = ''
        self.dataset_path = dataset_path
        self.ckpt_path = ckpt_path

        if debug:
            self.env = 'debug'
        else:
            self.env = 'experiment'

        self.create_path = create_path

    def train(self, train=True):
        if train:
            self.env = 'experiment'
        else:
            self.env = 'debug'

    @classproperty
    def project_name_current(cls):
        return os.path.basename(os.path.dirname(os.path.realpath(__main__.__file__)))

    @classproperty
    def table_name_current(cls):
        return os.path.splitext(os.path.basename(__main__.__file__))[0]

    @property
    def dataset(self):
        return self.dataset_path

    @property
    def state_dict(self):
        return self.ckpt_path

    @property
    @create_path_decorator
    def log_tensorboard(self):
        return os.path.join(self.base_path, 'tensorboard')

    @property
    @create_path_decorator
    def log_wandb(self):
        return os.path.join(self.base_path, 'wandb')

    @property
    @create_path_decorator
    def checkpoint(self):
        return os.path.join(self.base_path, 'checkpoint')

    @property
    @create_path_decorator
    def tune(self):
        return os.path.join(self.base_path, 'tune')

    @property
    @create_path_decorator
    def sample(self):
        return os.path.join(self.base_path, 'sample')

    @property
    def base_path(self):
        return os.path.join(self.state_dict, self.project_name, self.env, self.table_name, self.config_path)

    def new_wandb_id(self):
        return uuid4().hex
