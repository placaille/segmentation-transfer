import numpy as np

def logit_to_img(logit):
    def func(pixel):
        if pixel == 1:
            return [1,1,1]
        elif pixel == 2:
            return [1,0,0]
        elif pixel == 3:
            return [1,1,0]
        else:
            return [0,0,0]

    vfunc = np.vectorize(func, signature='(n)->()')
    img = vfunc(logits).astype('float')
    return img
