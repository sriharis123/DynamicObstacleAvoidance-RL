from cmath import exp
import numpy as np

def get_arr_wts(a):
    # remove last value
    a = a[:-1]

    # get num lasers in each quadrant
    n_quad_1 = np.ceil(90 / (360.0 / a.size)).astype(np.int16)
    n_quad_2 = np.floor(90 / (360.0 / a.size)).astype(np.int16)

    # extract and form an array
    arr1 = np.flip(a[:n_quad_1])
    arr2 = np.flip(a[-n_quad_2:])
    arr = np.concatenate((arr1, arr2))
    norm_arr = (np.arange(-n_quad_1 + 1, n_quad_2 + 1))

    mu    = np.mean(norm_arr)
    sigma = np.std(norm_arr)

    wts = []
    for i in range(norm_arr.size):
        wts.append(np.round(0.4 * np.exp(-(0.5) * ((norm_arr[i] - mu) / (sigma)) ** 2), 2))

    return(arr, wts)


b = np.repeat(5, 18) * 0.01


arr, wts = get_arr_wts(b)
arr = arr / (18 * 5)

final_values = arr * wts
print(sum(final_values))