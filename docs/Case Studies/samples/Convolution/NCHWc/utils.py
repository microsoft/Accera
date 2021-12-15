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
        we use "split" to split the permutation space into two spaces
    '''
    elements_list = np.arange(len(initial_order))
    elements_dict = {ele:i for i, ele in zip(elements_list, initial_order)} # {"out_f": 0, "out_c":1, ....}

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

def valid_order_choice(order_choice, initial_order, preserve_partial_order_list=None, fixed_indices=None):
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
    elements_dict = {ele:i for i, ele in zip(elements_list, initial_order)} # {"out_f": 0, "out_c":1, ....}

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

def naive_convolution(Input, Weights, Output, row_stride, column_stride):
    nchwc_input_channels, _, _, nchwc_input_channels_block = Input.shape
    nchwc_output_filters, output_rows, output_columns, nchwc_output_filters_block = Output.shape
    kernel_rows, kernel_columns, _, _ = Weights.shape
    for out_f in range(nchwc_output_filters):
        for in_ch in range(nchwc_input_channels):
            for out_r in range(output_rows):
                for out_c in range(output_columns):
                    for out_f_b in range(nchwc_output_filters_block):
                        for in_ch_b in range(nchwc_input_channels_block):
                            for k_r in range(kernel_rows):
                                for k_c in range(kernel_columns):
                                    in_r = out_r * row_stride + k_r
                                    in_c = out_c * column_stride + k_c
                                    k_ch = in_ch * nchwc_input_channels_block + in_ch_b
                                    k_f = out_f * nchwc_output_filters_block + out_f_b
                                    Output[out_f, out_r, out_c, out_f_b] += Input[in_ch, in_r, in_c, in_ch_b] \
                                                                            * Weights[k_r, k_c, k_ch, k_f]
    return Output

def get_input_output_pairs(nchwc_input_shape, nchwc_output_shape, weights_shape, row_stride, column_stride):
    '''
        This function returns input/outpuput pairs for NCHWc Convolution of an Input and Weights tensors,
        where the Input has an NCHWc memory layout (channel_blocks, rows, columns, channels),
        the weights shape is (kernel_rows, kernel_columns, total_input_channels, total_output_filters),
        and the resulting output has an NCHWc memory layout with dimensions (filters_blocks, rows, columns, filters)
        It is used to check the value correctness of any generated NCHWc convolution function.
    '''
    Input = np.random.random(nchwc_input_shape).astype(np.float32)
    Weights = np.random.random(weights_shape).astype(np.float32)
    Output_init = np.zeros(nchwc_output_shape, dtype=np.float32)
    Output = np.zeros(nchwc_output_shape, dtype=np.float32)

    naive_convolution(Input, Weights, Output, row_stride, column_stride)

    return {"pre": [Input, Weights, Output_init], "post": [Input, Weights, Output]}

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
        if function.auxiliary and "runtime" in function.auxiliary: # Only add if the function has benchmarking results
            data_point = function.auxiliary
            # Convert order numbers to nest indices
            data_point["order_indices"] = ",".join([["out_f", "in_ch", "out_r", "out_r2", "out_c", \
                                                     "k_r", "k_c", "in_ch_b", "out_f2", "out_c2", "out_f_b"][idx] \
                                          for idx in str_to_list(data_point["order"])])
            data.append(data_point)
    return data

def load_to_dataframe(data):
    import pandas as pd
    df = pd.DataFrame(data=data)
    df = df.sort_values(by=['order_indices'])
    indices = np.arange(0, len(df))
    df["idx"] = indices
    return df
