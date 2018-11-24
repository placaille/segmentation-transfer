import numpy as np

def logit_to_img(logit):
    def func(pixel):
        if pixel == 1:
            out = [1,1,1]
        elif pixel == 2:
            out = [1,0,0]
        elif pixel == 3:
            out = [1,1,0]
        else:
            out = [0,0,0]
        return np.array(out)

    vfunc = np.vectorize(func, signature='()->(n)')
    img = vfunc(logit).astype('float')
    return img
