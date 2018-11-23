import numpy as np
import visdom


def setup(server, port, exp_name, reload=False):
    visualiser = visdom.Visdom(server, port=port, env=exp_name, use_incoming_socket=False)
    if not reload:
        visualiser.delete_env(exp_name)
    return visualiser


def visualise_image(visualiser, images, title=None, iteration=0, env=None):
    nrow = int(len(images)**(1/2))
    win = 'images%s%s' % (visualiser.env, iteration)
    visualiser.images(images, nrow=nrow, win=win, opts={'title': title}, env=env)


def visualise_text(visualiser, text, iteration=0, env=None):
    visualiser.text(text, win='text%s' % iteration, env=env)


def visualise_plot(visualiser, X, data, title=None, legend=None, iteration=0, update=None, env=None, name=None):
    # FIXME only work for 1 D data.
    if legend is None:
        legend = np.arange(len(np.array(data)))
    win = 'plot%s%s' % (visualiser.env, iteration)
    opts = {'title': title, 'legend': legend or range(len(data))}
    dataY = np.array([data[-1]]) if update == 'append' else data
    err = visualiser.line(X=X, Y=dataY, win=win, opts=opts, update=update, env=env, name=name)
    if win != err:
        visualiser.line(X=X, Y=data, win=win, opts=opts, env=env, name=name)


def visualise_video(visualiser, videofile, title=None, iteration=0):
    win = 'video%s%s' % (visualiser.env, iteration)
    opts = {'fps': 1, 'title': title}
    visualiser.video(videofile=videofile, win=win, opts=opts)


def visualise_bar(visualiser, bar, Y, title=None, iteration=0, env=None):
    win = 'bar%s%s' % (visualiser.env, iteration)
    opts = {'title': title}
    visualiser.bar(X=bar, Y=Y, win=win, opts=opts, env=env)


def visualise_box(visualiser, X, Y, title='', iteration=0, env=None):
    win = 'bar%s%s' % (visualiser.env, iteration)
    opts = {'title': title, 'legend': Y}
    visualiser.boxplot(X=X, win=win, opts=opts, env=env)


def remove_point(visualiser, title, iteration, env=None):
    win = 'plot%s%s' % (visualiser.env, iteration)
    opts = {'title': title, 'legend': legend or range(len(data))}
    visualiser.line(X=None, Y=None, win=win, opts=opts, update='remove', env=env, name=iteration)
