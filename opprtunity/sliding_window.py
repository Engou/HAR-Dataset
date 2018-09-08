import numpy as np
from numpy.lib.stride_tricks import as_strided as ast

def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        pass

    raise TypeError('shape must be an int, or a tuple of ints')


def sliding_window(a, ws, ss=None, flatten=True):
    '''
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    '''

    if None is ss:
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    ls = [len(shape), len(ws), len(ss)]
    if 1 != len(set(ls)):
        raise ValueError( \
            'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    if np.any(ws > shape):
        raise ValueError( \
            'ws cannot be larger than a in any dimension.\
     a.shape was %s and ws was %s' % (str(a.shape), str(ws)))

    newshape = norm_shape(((shape - ws) // ss) + 1)
    newshape += norm_shape(ws)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a, shape=newshape, strides=newstrides)

    if not flatten:
        return strided

    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    dim = filter(lambda i: i != 1, dim)
    dim = list(dim)
    return np.reshape(strided,dim)


def opp_sliding_window(data_x, data_y, ws, ss):
    '''
    apply window function on data and label
    :param data_x: [total_time_length, channels]
    :param data_y: [total_time_length,] it is not 2 dims
    :param ws: int
    :param ss: int
    :return:
    '''
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = sliding_window(data_y, ws, ss)
    return data_x.astype(np.float32), data_y.astype(np.uint8)