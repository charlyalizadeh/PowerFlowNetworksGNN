import copy
import numpy as np
from scipy.stats import pearsonr


def order_merge_sort(y_true, y_pred):
    sort_perm = np.argsort(y_pred)
    arr = y_true[sort_perm]
    n = len(arr)
    arr = copy.deepcopy(arr)
    temp_arr = [0] * n
    return _merge_sort(arr, temp_arr, 0, n - 1)


def _merge_sort(arr, temp_arr, left, right):
    inv_count = 0
    if left < right:
        mid = (left + right) // 2
        inv_count += _merge_sort(arr, temp_arr, left, mid)
        inv_count += _merge_sort(arr, temp_arr, mid + 1, right)
        inv_count += merge(arr, temp_arr, left, mid, right)
    return inv_count


def merge(arr, temp_arr, left, mid, right):
    i = left
    j = mid + 1
    k = left
    inv_count = 0
    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            temp_arr[k] = arr[i]
            k += 1
            i += 1
        else:
            temp_arr[k] = arr[j]
            inv_count += (mid-i + 1)
            k += 1
            j += 1
    while i <= mid:
        temp_arr[k] = arr[i]
        k += 1
        i += 1
    while j <= right:
        temp_arr[k] = arr[j]
        k += 1
        j += 1
    for loop_var in range(left, right + 1):
        arr[loop_var] = temp_arr[loop_var]
    return inv_count


def order_pearson_corr(y_true, y_pred):
    sort_perm_pred = np.argsort(y_pred)
    y_true = y_true[sort_perm_pred]
    sort_perm_true = np.argsort(y_true)
    x = list(range(len(sort_perm_true)))
    return pearsonr(x, sort_perm_true)[0]
