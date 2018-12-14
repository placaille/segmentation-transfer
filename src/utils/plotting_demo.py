

def communicate(pipe):
    import numpy as np
    import matplotlib
    matplotlib.use('MacOSX')
    import matplotlib.pyplot as plt
    def action(img):
        print(img.shape)
        li.set_data(img)
        fig.canvas.draw()
        plt.pause(0.05)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    init = np.zeros((120, 160, 3))

    li = ax.imshow(init)
    # fig.canvas.draw()
    plt.show(block=False)

    while True:
        # a = np.random.uniform(0, 1, (120, 160, 3))
        # li.set_data(a)
        # fig.canvas.draw()
        plt.pause(0.05)
        print('Receiving')
        state = pipe.recv()
        print('Received')
        action(state)

if __name__ == '__main__':
    import multiprocessing as mp
    from multiprocessing import Process, Pipe
    import numpy as np
    import matplotlib.pyplot as plt
    mp.set_start_method('spawn')
    child = Process(target=communicate, args=(0,))
    child.start()
    child.join()
