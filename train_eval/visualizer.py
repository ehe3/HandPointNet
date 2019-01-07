from visdom import Visdom
import os
import numpy as np

class Visualizer():

    def __init__(self, log_dir):
        DEFAULT_PORT = 8097
        DEFAULT_HOSTNAME = "http://localhost"
        self.log_dir  = log_dir
        self.viz      = Visdom(port=DEFAULT_PORT, server=DEFAULT_HOSTNAME, log_to_filename=os.path.join(self.log_dir, 'visdom.log'))
        self.window   = None
        self.legend   = ['train wld loss (mm)', 'test wld loss (mm)', 'lr']

    def plot(self, epoch, train_loss, test_loss, lr):
        x = np.array([[epoch] * 3])
        y = np.array([[train_loss,test_loss, lr]])
        
        if self.window == None:
            self.window = self.viz.line(X=x, 
                                Y=y,
                                name='Loss',
                                opts=dict(
                                    fillarea=False,
                                    legend=self.legend,
                                    width=500,
                                    height=500,
                                    xlabel='Epochs',
                                    ylabel='Loss',
                                    title='Loss Curves',
                                    marginleft=15,
                                    marginright=15,
                                    marginbottom=40,
                                    margintop=15,
                                ),
                        )
        else:
            self.viz.line(X=x, Y=y, win=self.window, update='append')