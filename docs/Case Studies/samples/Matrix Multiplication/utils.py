from accera.hat import HATFile, HATPackage
import numpy as np
import itertools
import re

def str_to_list(s):
    s = re.sub('["()]', "", s)
    return [int(i) for i in s.split(",")]

def permute(list, order):
    '''
        permute the items in the list according to the order,
        and return a new list
        for example: if list = [a, b, c] where a, b, and c are objects,
                     and order = [2, 1, 0], then output = [c, b, a]
    '''
    return [list[idx] for idx in order]

def get_order_permutations(initial_order, split=None):
    '''
        This function generates all the possible permutations of the elements in "initial_order"
        We use "split" to split the permutation space into two spaces,
        and the output would be the combination of the permutations from both spaces
    '''
    elements_list = np.arange(len(initial_order))
    elements_dict = {ele:i for i, ele in zip(elements_list, initial_order)}

    split_idx = 0
    if split:
        split_idx = elements_dict[split]
        elements_lists = [elements_list[0:split_idx], elements_list[split_idx:]]
    else:
        elements_lists = [elements_list]

    elements_order_lists = []
    for elements_list in elements_lists:
        elements_order_list = list(itertools.permutations(elements_list, len(elements_list)))
        elements_order_lists.append(elements_order_list)

    if split:
        elements_order_list = list(itertools.product(*elements_order_lists))
        elements_order_list = [[item for sublist in order for item in sublist] for order in elements_order_list]
    else:
        elements_order_list = elements_order_lists[0]
    return elements_order_list

def valid_order_choice(order_choice, initial_order, preserve_partial_order_list=None, fixed_indices=None, split=None):
    '''
        This function checks whether the 'order_choice' is valid or not according to the given "preserve_partial_order" and "fixed" constraints.
        - "order_choice": is a list of integer indices of the desired permutation.
                          For example: if we want to permute a list [i, j, k] to [k, i, j] the "order_choice" should be set to [2, 0, 1]
        - "initial_order": is a list of initial indices names as strings to make addressing them easier.
                           For example: if the original loop is [i, j, k], then ["i", "j", "k"]

        - "preserve_partial_order_list": is a list of tuples with indices whose order needs to be partially preserved,
                                         For example: if the original loop is [i, j, k, ii, jj],
                                         and we want to ensure that i always precedes ii and j always precedes jj,
                                         then "preserve_partial_order_list" needs to be set to [("i", "ii"), ("j", "jj")]
        - "fixed_indices": is the list of indices whose order should be fixed when compared to the original order,
                           For example: if the original loop is [i, j, k, ii, jj],
                           and we want to ensure that k is always the third loop, and jj is the last,
                           then "fixed_indices" needs to be set to ["k", "jj"]
    '''
    elements_list = np.arange(len(initial_order))
    elements_dict = {ele:i for i, ele in zip(elements_list, initial_order)}

    if preserve_partial_order_list:
        # Only keep the orders that satisfies the order of the preserve_partial_order_list
        for order in preserve_partial_order_list:
            if order_choice.index(elements_dict[order[0]]) > order_choice.index(elements_dict[order[1]]):
                return False

    if fixed_indices:
        # Only keep the orders where the fixed indices are not changed
        for fixed_index in fixed_indices:
            if order_choice.index(elements_dict[fixed_index]) != elements_dict[fixed_index]:
                return False

    return True

def valid_split_size(first_split_size, second_split_size):
    '''
        This function is one of the parameters grid filtering utils.
        It returns True if the second split size is smaller than the first split size.
    '''
    return second_split_size < first_split_size

def fits_in_l2(m_split_size, n_split_size, s_split_size, element_size, l2_cache_size):
    '''
        This function is one of the parameters grid filtering utils.
        It returns True if the total active tiles memory is smaller than the L2 cache size.
    '''
    tile_mem = element_size*(m_split_size*s_split_size + s_split_size*n_split_size + m_split_size*n_split_size) / 1024
    return tile_mem < l2_cache_size

def uses_enough_l2(m_split_size, n_split_size, s_split_size, element_size, l2_cache_size):
    '''
        This function is one of the parameters grid filtering utils.
        It returns True if the total active tiles memory is at least 50% of the L2 cache size.
    '''
    tile_mem = element_size*(m_split_size*s_split_size + s_split_size*n_split_size + m_split_size*n_split_size) / 1024
    return tile_mem >= 0.5 * l2_cache_size

def get_input_output_pairs(M, N, S):
    '''
        This function returns input/outpuput pairs for matrix multiplication of
        an "MxS" matrix A and an "SxN" Matrix B.
        It is used to check the value correctness of any generated matmul function.
    '''
    A = np.random.random((M,S)).astype(np.float32)
    B = np.random.random((S,N)).astype(np.float32)
    C = np.ones((M,N), dtype=np.float32)
    Out = C + A @ B
    return {"pre": [A, B, C], "post": [A, B, Out]}

def write_back_to_hat(hat_file_path, function_name, mean_time_secs):
    # Write back the runtime to the HAT file
    hat_file = HATFile.Deserialize(hat_file_path)
    hat_func = hat_file.function_map.get(function_name)
    hat_func.auxiliary["runtime"] = mean_time_secs

    hat_file.Serialize(hat_file_path)

    # Workaround to remove extra empty lines
    with open(hat_file_path, "r") as f:
        lines = f.readlines()
        lines = [lines[i] for i in range(len(lines)) if not(lines[i] == "\n" \
                                    and i < len(lines)-1 and lines[i+1] == "\n")]
    with open(hat_file_path, "w") as f:
        f.writelines(lines)

def get_auxiliary_data(directory):
    '''
        return a list of the functions auxilary data in a Accera package
    '''
    hat_package = HATPackage(directory)
    functions = [fn for fn in hat_package.get_functions()]

    data = []
    for function in functions:
        if "runtime" in function.auxiliary: # Only add if the function has benchmarking results
            data_point = function.auxiliary
            # Convert order numbers to nest indices
            data_point["order_indices"] = ",".join([["i", "j", "k", \
                                                     "jj", "kk", "kkk", "ii", "jjj", "jjjj"][idx]
                                          for idx in str_to_list(data_point["order"])])
            data_point["tile_volume"] = 4*(data_point["m_split_size"]*data_point["n_split_size"] + \
                                           data_point["m_split_size"]*data_point["s_split_size"] + \
                                           data_point["n_split_size"]*data_point["s_split_size"])/1024
            data.append(data_point)
    return data

def load_to_dataframe(data):
    import pandas as pd
    df = pd.DataFrame(data=data)
    df = df[df["runtime"] < 0.2]
    df = df.sort_values(by=['order_indices'])
    indices = np.arange(0, len(df))
    df["idx"] = indices
    return df
