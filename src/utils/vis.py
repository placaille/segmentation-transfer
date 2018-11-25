import numpy as np
import visdom
import os


class Visualiser():
    def __init__(self, server, port, exp_name, reload, visdom_dir):
        if not server or not port:
            self.visualiser = None
            print('No server or no port defined. Visualisation are disabled.')
            return

        log_name = None
        if visdom_dir:
            log_name = os.path.join(visdom_dir, 'visdom.out')

        self.visualiser = visdom.Visdom(server, port=port, env=exp_name, use_incoming_socket=False, log_to_filename=log_name)
        if not reload:
            self.visualiser.delete_env(exp_name)

    def image(self, images, title=None, iteration=0, env=None):
        if not self.visualiser:
            return
        nrow = int(len(images)**(1/2))
        win = 'images%s%s' % (self.visualiser.env, iteration)
        self.visualiser.images(images, nrow=nrow, win=win, opts={'title': title}, env=env)

    def text(self, text, iteration=0, env=None):
        if not self.visualiser:
            return
        self.visualiser.text(text, win='text%s' % iteration, env=env)


    def plot(self, X, data, title=None, legend=None, iteration=0, update=None, env=None, name=None):
        if not self.visualiser:
            return
        # FIXME only work for 1 D data.
        if legend is None:
            legend = np.arange(len(np.array(data)))
        win = 'plot%s%s' % (self.visualiser.env, iteration)
        opts = {'title': title, 'legend': legend or range(len(data))}
        dataY = np.array([data[-1]]) if update == 'append' else data
        err = self.visualiser.line(X=X, Y=dataY, win=win, opts=opts, update=update, env=env, name=name)
        if win != err:
            self.visualiser.line(X=X, Y=data, win=win, opts=opts, env=env, name=name)


    def video(self, videofile, title=None, iteration=0):
        if not self.visualiser:
            return
        win = 'video%s%s' % (self.visualiser.env, iteration)
        opts = {'fps': 1, 'title': title}
        self.visualiser.video(videofile=videofile, win=win, opts=opts)


    def bar(self, bar, Y, title=None, iteration=0, env=None):
        if not self.visualiser:
            return
        win = 'bar%s%s' % (self.visualiser.env, iteration)
        opts = {'title': title}
        self.visualiser.bar(X=bar, Y=Y, win=win, opts=opts, env=env)


    def box(self, X, Y, title='', iteration=0, env=None):
        if not self.visualiser:
            return
        win = 'bar%s%s' % (self.visualiser.env, iteration)
        opts = {'title': title, 'legend': Y}
        self.visualiser.boxplot(X=X, win=win, opts=opts, env=env)
