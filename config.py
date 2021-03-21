import os


class Config(object):
    def __init__(self, output_dir, epochs, batch_size, log_every_step, v1=False, **kwargs):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if v1:
            self.ckpt_dir_train = os.path.join(output_dir, 'ckpt_train')
            self.ckpt_dir_dev = os.path.join(output_dir, 'ckpt_dev')
            self.ckpt_train = os.path.join(self.ckpt_dir_train, 'ckpt')
            self.ckpt_dev = os.path.join(self.ckpt_dir_dev, 'ckpt')
        else:
            self.ckpt_dir_train = os.path.join(output_dir, 'ckpt/train')
            self.ckpt_dir_dev = os.path.join(output_dir, 'ckpt/dev')

        self.log_dir_train = os.path.join(output_dir, 'log/train')
        self.log_dir_dev = os.path.join(output_dir, 'log/dev')

        self.epochs = epochs
        self.batch_size = batch_size
        self.log_every_step = log_every_step

        for k, v in kwargs.items():
            self.__dict__[k] = v
