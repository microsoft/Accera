from accera.hat import HATFile, HATPackage
import numpy as np
import re

def str_to_list(s):
    s = re.sub('["()]', "", s)
    return [int(i) for i in s.split(",")]

def naive_convolution(Input, Weights, Output, row_stride, column_stride):
    input_rows, input_columns, input_channels = Input.shape
    output_rows, output_columns, out_filters = Output.shape
    kernel_rows, kernel_columns, _, _ = Weights.shape
    for out_f in range(out_filters):
        for out_r in range(output_rows):
            for out_c in range(output_columns):
                for in_ch in range(input_channels):
                    for k_r in range(kernel_rows):
                        for k_c in range(kernel_columns):
                            in_r = out_r * row_stride + k_r
                            in_c = out_c * column_stride + k_c
                            if in_r >= 0 and in_r < input_rows and in_c >= 0 and in_c < input_columns:
                                Output[out_r, out_c, out_f] += Input[in_r, in_c, in_ch] * Weights[k_r, k_c, in_ch, out_f]
    return Output

def valid_split_size(first_split_size, second_split_size):
    '''
        This function is one of the parameters grid filtering utils.
        It returns True if the second split size is smaller than the first split size.
    '''
    return second_split_size < first_split_size

def fits_in_l2(m_tile_size, n_tile_size, s_tile_size, element_size, l2_cache_size):
    '''
        This function is one of the parameters grid filtering utils.
        It returns True if the total active tiles memory is smaller than the L2 cache size.
    '''
    tile_mem = element_size*(m_tile_size*s_tile_size + s_tile_size*n_tile_size + m_tile_size*n_tile_size) / 1024
    return tile_mem < l2_cache_size

def uses_enough_l2(m_tile_size, n_tile_size, s_tile_size, element_size, l2_cache_size):
    '''
        This function is one of the parameters grid filtering utils.
        It returns True if the total active tiles memory is at least 50% of the L2 cache size.
    '''
    tile_mem = element_size*(m_tile_size*s_tile_size + s_tile_size*n_tile_size + m_tile_size*n_tile_size) / 1024
    return tile_mem >= 0.5 * l2_cache_size

def get_input_output_pairs(input_shape, kernel_shape, output_filters, row_stride, column_stride):
    '''
        This function returns input/outpuput pairs for 2D Convolution of an Input and Weights tensors,
        where the Input has dimensions (rows, columns, input_channels),
        the weights shape is (kernel_rows, kernel_columns, input_channels, output_filters),
        and the resulting output has dimensions (rows, columns, output_filters)
        It is used to check the value correctness of any generated NCHWc convolution function.
    '''
    input_rows, input_columns, input_channels = input_shape
    kernel_rows, kernel_columns = kernel_shape

    output_rows = int(((input_rows - kernel_rows) / row_stride) + 1)
    output_columns = int(((input_columns - kernel_columns) / column_stride) + 1)

    Input = np.random.random((input_rows, input_columns, input_channels)).astype(np.float32)
    Weights = np.random.random((kernel_rows, kernel_columns, input_channels, output_filters)).astype(np.float32)

    Output_init = np.zeros((output_rows, output_columns, output_filters), dtype=np.float32)

    Output = np.zeros((output_rows, output_columns, output_filters), dtype=np.float32)
    Output = naive_convolution(Input, Weights, Output, row_stride, column_stride)

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

def get_auxiliary_data(output_directory):
    '''
        return a list of the functions auxilary data in a Accera package
    '''
    hat_package = HATPackage(output_directory)
    functions = [fn for fn in hat_package.get_functions()]

    data = []
    for function in functions:
        if "runtime" in function.auxiliary: # Only add if the function has benchmarking results
            data_point = function.auxiliary
            # Convert order numbers to nest indices
            data_point["tile_volume"] = 4*(data_point["in_ch_split_size"]*data_point["outf_split1_size"] + \
                                           data_point["in_ch_split_size"]*data_point["outc_split_size"] + \
                                           data_point["outc_split_size"]*data_point["outf_split1_size"])/1024
            data.append(data_point)
    return data

def load_to_dataframe(data):
    import pandas as pd
    df = pd.DataFrame(data=data)
    indices = np.arange(0, len(df))
    df["idx"] = indices
    return df
