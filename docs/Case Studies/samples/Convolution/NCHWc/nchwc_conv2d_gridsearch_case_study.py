import accera as acc
from accera.tuning import AutoBenchmark

import os, itertools, csv, math, shutil
import random
import argparse

from utils import get_order_permutations, valid_order_choice
from utils import permute, get_input_output_pairs, write_back_to_hat, str_to_list
from utils import load_to_dataframe, get_auxiliary_data


def add_nchwc_conv2d_function(nchwc_input_shape, nchwc_output_shape, weights_shape, row_stride, column_stride, package, parameters_choice):
    '''
        This function is a accera implementation for nchwc 2D convolution between
        a 4D input Tensor of dimensions input_channels_blocks*input_rows*input_columns*block_input_channels
        and a 4D Weights Tensor of dimensions kernel_rows*kernel_columns*input_channels*output_filters
        resulting in a 4D output Tensor of dimensions output_filters_blocks*output_rows*output_columns*block_output_filters.
        The logic for the NCHWc 2D covolution can be expressed in python as
        ---------------------
        Python implemntation:
        ---------------------
        for out_f in range(output_filters_blocks):
            for in_ch in range(input_channels_blocks):
                for out_r in range(output_rows):
                    for out_c in range(output_columns):
                        for out_f_b in range(block_output_filters):
                            for in_ch_b in range(block_input_channels):
                                for k_r in range(kernel_rows):
                                    for k_c in range(kernel_columns):
                                        in_r = out_r * row_stride + k_r
                                        in_c = out_c * column_stride + k_c
                                        if in_r >= 0 and in_r < input_rows and in_c >= 0 and in_c < input_columns:
                                            Output[out_f, out_r, out_c, out_f_b] += Input[in_ch, in_r, in_c, in_ch_b] * \
                                                                                    Weights[k_r, k_c, in_ch * input_block_channels + in_ch_b,
                                                                                                out_f * output_block_filters + out_f_b]
    '''
    out_c_split_size, out_r_split_size, out_f_split_size, loop_order = parameters_choice

    input_channel_blocks, input_rows, input_columns, input_channels = nchwc_input_shape
    nchwc_output_filters, output_rows, output_columns, nchwc_output_filters_block = nchwc_output_shape
    kernel_rows, kernel_columns, total_input_channels, total_output_filters = weights_shape

    Input = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, \
                shape=(input_channel_blocks, input_rows, input_columns, input_channels))
    Weights = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, \
                shape=(kernel_rows, kernel_columns, total_input_channels, total_output_filters))
    Output = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, \
                shape=(nchwc_output_filters, output_rows, output_columns, nchwc_output_filters_block))

    nest = acc.Nest(shape=(nchwc_output_filters, input_channel_blocks, output_rows, kernel_rows, kernel_columns, \
                          input_channels, output_columns, nchwc_output_filters_block))

    out_f, in_ch, out_r, k_r, k_c, in_ch_b, out_c, out_f_b = nest.get_indices()

    @nest.iteration_logic
    def _():
        in_r = out_r * row_stride + k_r
        in_c = out_c * column_stride + k_c
        Output[out_f, out_r, out_c, out_f_b] += Input[in_ch, in_r, in_c, in_ch_b] * \
                                Weights[k_r, k_c, in_ch * input_channels + in_ch_b,
                                        out_f * nchwc_output_filters_block + out_f_b]

    # Create the schedule
    schedule = nest.create_schedule()

    out_c2 = schedule.split(out_c, out_c_split_size)
    out_r2 = schedule.split(out_r, out_r_split_size)
    out_f2 = schedule.split(out_f, out_f_split_size)

    schedule_items = [out_f, in_ch, out_r, out_r2, out_c, k_r, k_c, in_ch_b, out_f2, out_c2, out_f_b]
    schedule_items = permute(schedule_items, loop_order)
    schedule.reorder(*schedule_items) # apply re-ordering

    plan = schedule.create_action_plan()

    plan.cache(Input, index=in_ch_b)
    plan.cache(Output, index=out_f2)
    plan.cache(Weights, index=in_ch_b)

    plan.kernelize(unroll_indices=(schedule_items[-4:-1]), vectorize_indices=schedule_items[-1])

    auxiliary_data = {"out_c_split_size": out_c_split_size,
                      "out_r_split_size": out_r_split_size,
                      "out_f_split_size": out_f_split_size,
                      "order": ",".join([str(i) for i in loop_order])
                      }

    name = f"conv2d_{input_rows}_{input_columns}_{total_input_channels}_{total_output_filters}"
    function = package.add_function(plan, args=(Input, Weights, Output), base_name=name, auxiliary=auxiliary_data)

    return function

def create_parameters_grid(output_directory):
    '''
        While creating our function in s1_create_function.py and s2_create_function_with_paramaters.py,
        we chose some split sizes, unrolling factors as well as an order for our schedule loops.
        However, we are not sure that those chosen sizes would give the best performance,
        that is why we want to define a parameter grid, where our chosen parameters are:
            1. out_c_split_size
            2. out_r_split_size
            3. out_f_split_size
            4. loop_order

        and our grid will consist of a set of possible values for those parameters.

        For example, we might want to:
            1. define the `out_c_split_size` and `out_r_split_size`as any odd number between 1 and 7
               because those dimensions usually have smaller values in convolutional neural networks
            2. define the `out_f_split_size` as any power of 2 between 4 and 32.
            3. define the possible schedule loops order as any order where the large scale loops
               precede the high performance loops.
    '''
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # define the out_c_split_size and out_r_split_size as any odd number between 1 and 7
    out_c_split_size_choices = [1, 3, 5, 7]
    out_r_split_size_choices = [1, 3, 5, 7]

    # define out_f_split_size as any power of 2 between 4 and 32
    out_f_split_size_choices = [4, 8, 16, 32]

    # Split the loop dimensions into large scale loops that would optimize caching usuage,
    # and high performance loop that would optimize register re-use,
    # since we are using an NCHWc layout, that means that sequential elements in
    # the c (block channels) and W (columns) dimensions are closer together in memory
    # than sequential elements in the C (channel blocks) or H (rows) dimensions,
    # so we will keep the W and c dimensions for the input and the output in the high performance loop,
    # while the rest of the dimensions will be added to the large scale loop.
    # Then we define the order choices as all combinations of the possible large scale and high performance loops.
    # NOTE: split="in_ch_b" splits the loop dimensions spaces into two spaces starting the loop "in_ch_b",
    #       the first includes the large scale loops ("out_f", "in_ch", "out_r", "out_r2", "out_c", "k_r", "k_c"),
    #       while the second includes the high performance loops ("in_ch_b", "out_f2", "out_c2", "out_f_b")
    order_choices = get_order_permutations(initial_order=["out_f", "in_ch", "out_r", "out_r2", "out_c", "k_r", "k_c", \
                                                          "in_ch_b", "out_f2", "out_c2", "out_f_b"],
                                           split="in_ch_b"
                                          )

    parameters_choices = [out_c_split_size_choices, out_r_split_size_choices, out_f_split_size_choices, order_choices]

    # Write the options to a csv
    csv_file =  os.path.join(output_directory, "parameters_choices.csv")
    with open(csv_file, "w", newline='') as f:
        writer = csv.writer(f)
        for name, option in zip(parameters_names, parameters_choices):
            writer.writerow([name] + option)

    # Generate options combinations
    parameters_grid = list(itertools.product(*parameters_choices))
    return parameters_grid

def filter_parameters_grid(parameters_grid, output_directory):
    filtered_parameter_grid = []
    # Filter the choices that does not preserve the order constraints
    for parameters_choice in parameters_grid:
        _, _, _, loop_order = parameters_choice

        if valid_order_choice(order_choice=loop_order,
                              initial_order=["out_f", "in_ch", "out_r", "out_r2", "out_c", \
                                             "k_r", "k_c", "in_ch_b", "out_f2", "out_c2", "out_f_b"],
                              preserve_partial_order_list=[("out_f", "out_f2"),
                                                           ("out_f2", "out_f_b"),
                                                           ("in_ch", "in_ch_b"),
                                                           ("out_c", "out_c2"),
                                                           ("out_r", "out_r2")],
                              fixed_indices=["out_f", "k_r", "k_c"]):
            filtered_parameter_grid.append(parameters_choice)

    # Write the filtered parameters grid to csv
    csv_file =  os.path.join(output_directory, "filtered_parameters_grid.csv")
    with open(csv_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(parameters_names)
        writer.writerows(filtered_parameter_grid)

    return filtered_parameter_grid

def create_single_accera_package(output_directory, nchwc_input_shape, nchwc_output_shape, weights_shape,
                                    row_stride, column_stride, parameters_grid):
    # Define a accera package
    package = acc.Package()
    for parameters_choice in parameters_grid:
        add_nchwc_conv2d_function(nchwc_input_shape, nchwc_output_shape, weights_shape, row_stride, column_stride,
                                  package=package, parameters_choice=parameters_choice)

    # Build the accera package
    package.build("nchwc_conv", format=acc.Package.Format.HAT, output_dir=output_directory)

def create_multiple_accera_packages(base_output_directory, nchwc_input_shape, nchwc_output_shape, weights_shape,
                                        row_stride, column_stride, parameters_grid, functions_per_package=100):
    for i in range(0, len(parameters_grid), functions_per_package):
        start_idx = i
        end_idx = min(i+functions_per_package, len(parameters_grid))
        output_directory = os.path.join(base_output_directory, f"{start_idx}_{end_idx}")
        create_single_accera_package(output_directory, nchwc_input_shape, nchwc_output_shape, weights_shape,
                                        row_stride, column_stride, parameters_grid[start_idx:end_idx])

def worker(index, start, end, step, base_output_directory, nchwc_input_shape, nchwc_output_shape, weights_shape,
                            row_stride, column_stride, parameters_grid):
    for idx in range(start, end, step):
        start_idx = idx
        end_idx = min(idx+step, len(parameters_grid))
        print(f"Proc # {index} is creating a package with parameters {start_idx}-{end_idx}")
        output_directory = os.path.join(base_output_directory, f"{start_idx}_{end_idx}")
        create_single_accera_package(output_directory, nchwc_input_shape, nchwc_output_shape, weights_shape,
                                        row_stride, column_stride, parameters_grid[start_idx:end_idx])

# NOTE: We can't use multithreading here due to a conflict on resources during building the package
def create_multiple_accera_packages_simultaneous(base_output_directory, nchwc_input_shape, nchwc_output_shape, weights_shape,
                                                    row_stride, column_stride, parameters_grid, num_procs=3, functions_per_package=100):
    from multiprocessing import Process
    procs = []

    # Divide the parameter choices among the procs
    for i, block_idx in enumerate(range(0, len(parameters_grid), len(parameters_grid)//num_procs)):
        start = block_idx
        end = block_idx+len(parameters_grid)//num_procs
        step = min(functions_per_package, end-start)
        p = Process(target=worker, args=(i, start, end, step, base_output_directory, nchwc_input_shape, nchwc_output_shape, weights_shape,
                                            row_stride, column_stride, parameters_grid))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

def autobenchmark_package(package_directory, *args, write_back=False):
    benchmark = AutoBenchmark(package_directory)

    functions = benchmark.functions
    for function_name in functions:
        if "Initialize" in function_name: # Skip init functions
            continue
        correctness_check_values = get_input_output_pairs(*args)

        # NOTE: We place the benchmarking in try .. except block to prevent stopping
        #       the gridsearch in case of a function that failed the correctness check.
        #       However, this is redundunt in this case study since all the applied transformations are safe.
        try:
            mean_time_secs, _ = benchmark.run(function_name,
                                           min_timing_iterations=5,
                                           warmup_iterations=1,
                                           correctness_check_values=correctness_check_values)

            if write_back:
                # Write back the runtime to the HAT file
                hat_file_path = os.path.join(package_directory, "nchwc_conv" + ".hat")
                write_back_to_hat(hat_file_path, function_name, mean_time_secs)
        except:
            print(f"WARNING: function {function_name} failed correctness check.")

def plot_dataframe(df, output_directory):
    '''
        plot the runtime and annotate the points with iteration parameters
    '''
    import plotly.express as px
    df.sort_values(by=['order_indices'])
    fig = px.scatter(df, x="idx", y="runtime", color="order_indices", \
                        labels={"runtime": "Runtime (s)", "idx": "Run Index", "order_indices": "Loop Order"}, \
                        hover_data=["out_c_split_size", "out_r_split_size", "out_f_split_size", "order_indices"], \
                        title=f"NCHWc 2D Convolution Grid Search",
                        color_continuous_scale=px.colors.sequential.Bluered)

    fig.write_html(os.path.join(output_directory, f"package_and_benchmarking_visualization_color.html"))
    fig.write_image(os.path.join(output_directory, f"package_and_benchmarking_visualization_color.jpeg"))

def visualize_single_dir(output_directory):
    data = get_auxiliary_data(output_directory)
    if len(data) == 0:
        print(f"WARNING: No Benchmaring results found in {output_directory}")
        return
    dataframe = load_to_dataframe(data)
    plot_dataframe(dataframe, output_directory)

def visualize_multiple_dirs(output_directories, parent_directory):
    data = []
    for output_directory in output_directories:
        data.extend(get_auxiliary_data(output_directory))
    if len(data) == 0:
        print(f"WARNING: No Benchmaring results found in {output_directory}")
        return
    dataframe = load_to_dataframe(data)
    plot_dataframe(dataframe, parent_directory)

def get_optimal_point_single_dir(output_directory):
    data = get_auxiliary_data(output_directory)
    if len(data) == 0:
        print(f"WARNING: No Benchmaring results found in {output_directory}")
        return
    dataframe = load_to_dataframe(data)
    optimal_point_idx = dataframe['runtime'].idxmin()
    optimal_point = dataframe.iloc[optimal_point_idx]
    return optimal_point

def get_optimal_point_multiple_dirs(output_directories):
    import sys
    min_runtime = sys.float_info.max
    optimal_point = None
    for output_directory in output_directories:
        local_optimal_point = get_optimal_point_single_dir(output_directory)
        if local_optimal_point["runtime"] < min_runtime:
            optimal_point = local_optimal_point
            min_runtime = optimal_point["runtime"]
    return optimal_point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Accera NCHWc 2D Convolution case study')
    parser.add_argument('--input_shape', dest='input_shape', type=int, nargs='+',
                        help='Dimensions of the input 3D Matrix for Convolution.')
    parser.add_argument('--kernel_shape', dest='kernel_shape', type=int, nargs='+',
                        help='Dimensions of the 2D kernel for Convolution.')
    parser.add_argument('--output_filters', type=int,
                        help='Number of the required output filters from Convolution.')
    parser.add_argument('--stride', type=int, nargs='+',
                        help='Row and Column stride for Convolution.')
    parser.add_argument('--output_directory', type=str, default="nchwc_2d_conv",
                        help='Output directory.')
    parser.add_argument('--sample', type=int, default=None,
                        help='Optional parameter to choose a number of sample points of the parameter grid.')
    args = parser.parse_args()

    input_rows, input_columns, total_input_channels = args.input_shape
    kernel_rows, kernel_columns = args.kernel_shape
    row_stride, column_stride = args.stride

    total_output_filters = args.output_filters
    output_rows = int((input_rows - kernel_rows)/row_stride + 1)
    output_columns = int((input_columns - kernel_columns)/column_stride + 1)

    # For this case study we choose the NCHWc input channels block size,
    # as well as NCHWc output filters block size to be 8
    # because they optimize the usage of SIMD instructions in the target architecture (AVX2)
    # but for different architectures like (AVX512), 16 would be a better choice.
    output_filters_block_size = 8
    input_channels_block_size = 8

    output_filters_blocks = math.ceil(total_output_filters/output_filters_block_size)
    input_channels_blocks = math.ceil(total_input_channels/input_channels_block_size)

    nchwc_input_shape = [input_channels_blocks, input_rows, input_columns, input_channels_block_size]
    nchwc_output_shape = [output_filters_blocks, output_rows, output_columns, output_filters_block_size]
    weights_shape = [kernel_rows, kernel_columns, total_input_channels, total_output_filters]

    if os.path.isdir(args.output_directory):
        shutil.rmtree(args.output_directory)

    # Define the parameters
    parameters_names = ["out_c_split_size", "out_r_split_size", "out_f_split_size", "loop_order"]

    # Define the parameters grid
    print("Creating parameters grid ...")
    parameters_grid = create_parameters_grid(args.output_directory)
    print(f"Parameters grid created with {len(parameters_grid)} parameters choices.")

    # Filter parameters grid
    print("Filtering parameters grid ...")
    filtered_parameters_grid = filter_parameters_grid(parameters_grid, args.output_directory)
    print(f"Parameters grid filtered to have {len(filtered_parameters_grid)} parameters choices.")

    if args.sample:
        filtered_parameters_grid = random.sample(filtered_parameters_grid, args.sample)

    # Create a accera package using the filtered parameters grid
    if len(filtered_parameters_grid) < 200:
        create_single_accera_package(output_directory=args.output_directory,
                                       nchwc_input_shape=nchwc_input_shape,
                                       nchwc_output_shape=nchwc_output_shape,
                                       weights_shape=weights_shape,
                                       row_stride=row_stride,
                                       column_stride=column_stride,
                                       parameters_grid=filtered_parameters_grid)

        autobenchmark_package(args.output_directory,
                              nchwc_input_shape, nchwc_output_shape, weights_shape,
                              row_stride, column_stride, write_back=True)

        visualize_single_dir(args.output_directory)
        optimal_point = get_optimal_point_single_dir(args.output_directory)
    else:
        create_multiple_accera_packages_simultaneous(base_output_directory=args.output_directory,
                                                       nchwc_input_shape=nchwc_input_shape,
                                                       nchwc_output_shape=nchwc_output_shape,
                                                       weights_shape=weights_shape,
                                                       row_stride=row_stride,
                                                       column_stride=column_stride,
                                                       parameters_grid=filtered_parameters_grid)

        output_directories = [os.path.join(args.output_directory,item) for item in os.listdir(args.output_directory) \
                                if os.path.isdir(os.path.join(args.output_directory,item))]

        for output_directory in output_directories:
            print("Benchmarking directory", output_directory)
            autobenchmark_package(output_directory,
                                  nchwc_input_shape, nchwc_output_shape, weights_shape,
                                  row_stride, column_stride, write_back=True)

        visualize_multiple_dirs(output_directories, parent_directory=args.output_directory)
        optimal_point = get_optimal_point_multiple_dirs(output_directories)


    print("Optimal point by Grid Search:")
    print(optimal_point)

    # Create a new Accera package using the optimal parameters
    optimal_package = acc.Package()

    optimal_parameters = [int(optimal_point["out_c_split_size"]),
                          int(optimal_point["out_r_split_size"]),
                          int(optimal_point["out_f_split_size"]),
                          str_to_list(optimal_point["order"])]

    add_nchwc_conv2d_function(nchwc_input_shape=nchwc_input_shape,
                              nchwc_output_shape=nchwc_output_shape,
                              weights_shape=weights_shape,
                              row_stride=row_stride,
                              column_stride=column_stride,
                              package=optimal_package,
                              parameters_choice=optimal_parameters)

    # Build the accera package
    optimal_output_directory = args.output_directory + "_optimal"
    optimal_package.build("nchwc_conv", format=acc.Package.Format.HAT, output_dir=optimal_output_directory)
