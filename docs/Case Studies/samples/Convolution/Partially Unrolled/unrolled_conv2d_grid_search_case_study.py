import accera as acc
from accera.tuning import AutoBenchmark

import os, itertools, csv, shutil
import random
import argparse

from utils import uses_enough_l2, fits_in_l2, valid_split_size
from utils import get_input_output_pairs, write_back_to_hat
from utils import load_to_dataframe, get_auxiliary_data


def add_unrolled_conv2d_function(input_shape, kernel_shape, output_filters, row_stride, column_stride, package, parameters_choice):
    '''
    # The logic for the 2D covolution can be expressed in python as follows
    for out_f in range(output_filters):
        for out_r in range(output_rows):
            for out_c in range(output_columns):
                for in_ch in range(input_channels):
                    for k_r in range(kernel_rows):
                        for k_c in range(kernel_columns):
                            in_r = out_r * row_stride + k_r
                            in_c = out_c * column_stride + k_c
                            if in_r >= 0 and in_r < input_rows and in_c >= 0 and in_c < input_columns:
                                Output[out_r, out_c, out_f] += Input[in_r, in_c, in_ch] * Weights[k_r, k_c, in_ch, out_f]
    '''
    p_outf_split1_size, p_outf_split2_size, p_outf_split3_size, p_outc_split_size, p_in_ch_split_size = acc.create_parameters(5)

    input_rows, input_columns, input_channels = input_shape
    kernel_rows, kernel_columns= kernel_shape

    output_rows = int(((input_rows - kernel_rows) / row_stride) + 1)
    output_columns = int(((input_columns - kernel_columns) / column_stride) + 1)

    # Splitting an index and running the

    Input = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, \
                shape=(input_rows, input_columns, input_channels))
    Weights = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, \
                shape=(kernel_rows, kernel_columns, input_channels, output_filters))
    Output = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, \
                shape=(output_rows, output_columns, output_filters))

    nest = acc.Nest(shape=(output_filters, output_rows, output_columns, \
                          input_channels, kernel_rows, kernel_columns))

    out_f, out_r, out_c, in_ch, k_r, k_c = nest.get_indices()

    @nest.iteration_logic
    def _():
        in_r = out_r * row_stride + k_r
        in_c = out_c * column_stride + k_c
        Output[out_r, out_c, out_f] += Input[in_r, in_c, in_ch] * Weights[k_r, k_c, in_ch, out_f]

    # Create the schedule
    schedule = nest.create_schedule()

    out_f2 = schedule.split(out_f, p_outf_split1_size) # jj
    out_f3 = schedule.split(out_f2, p_outf_split2_size) # jjj
    out_f4 = schedule.split(out_f3, p_outf_split3_size) # jjjj

    out_c2 = schedule.split(out_c, p_outc_split_size) # ii
    in_ch2 = schedule.split(in_ch, p_in_ch_split_size) # kk

    schedule.reorder(out_f, # j
                     k_r, in_ch, # k
                     out_r, out_c, # i
                     out_f2, # jj
                     in_ch2, # kk
                     k_c, # kkk
                     out_c2, # ii
                     out_f3, # jjj
                     out_f4 # jjjj
                    )

    plan = schedule.create_action_plan()

    plan.cache(Input, index=in_ch2)
    plan.cache(Weights, index=out_f2)
    plan.cache(Output, index=out_f2)

    plan.unroll(out_c2)
    plan.unroll(out_f3)
    plan.vectorize(out_f4)

    outf_split1_size, outf_split2_size, outf_split3_size, outc_split_size, in_ch_split_size = parameters_choice

    auxiliary_data = {"outf_split1_size": outf_split1_size,
                      "outf_split2_size": outf_split2_size,
                      "outf_split3_size": outf_split3_size,
                      "outc_split_size": outc_split_size,
                      "in_ch_split_size": in_ch_split_size
                     }

    name = "unrolled_conv_using_caching"
    function = package.add_function(plan,
                                    args=(Input, Weights, Output),
                                    parameters={
                                        p_outf_split1_size: outf_split1_size,
                                        p_outf_split2_size: outf_split2_size,
                                        p_outf_split3_size: outf_split3_size,
                                        p_outc_split_size: outc_split_size,
                                        p_in_ch_split_size: in_ch_split_size,
                                    },
                                    auxiliary=auxiliary_data,
                                    base_name=name)

    return function

def create_parameters_grid(output_directory):
    '''
        While creating our parameterized function,
        we chose some split sizes as well as an order for our schedule loops.
        However, we are not sure that those chosen sizes would give the best performance,
        that is why we want to define a parameter grid, where our chosen parameters are:
            1. `outf_split1_size`
            2. `outf_split2_size`
            3. `outf_split3_size`
            4. `outc_split_size`
            5. `in_ch_split_size`

        and our grid will consist of a set of candidate values for those parameters.

        For example, we might want to:
            1. define the `outf_split1_size` and `in_ch_split_size` as any power of 2 between 32 and 256.
            2. define the `outf_split2_size` and `outf_split3_size` as any power of 2 between 4 and 32.
            3. define the `outc_split_size` as any even number betweem 4 and 8.
    '''
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # define the outf_split1_size and in_ch_split_size as any power of 2 between 32 and 256
    outf_split1_size_candidates = [32, 64, 128, 256]
    in_ch_split_size_candidates = [32, 64, 128, 256]

    # define the outf_split2_size and outf_split3_size as any power of 2 between 4 and 32
    outf_split2_size_candidates = [4, 8, 16, 32]
    outf_split3_size_candidates = [4, 8, 16, 32]

    # define the outc_split_size as any even number between 4 and 8
    outc_split_size_candidates= [4, 6, 8]

    parameters_choices = [outf_split1_size_candidates,
                          outf_split2_size_candidates,
                          outf_split3_size_candidates,
                          outc_split_size_candidates,
                          in_ch_split_size_candidates]

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
        outf_split1_size, outf_split2_size, outf_split3_size, outc_split_size, in_ch_split_size = parameters_choice

        if not valid_split_size(outf_split1_size, outf_split2_size) \
            or not valid_split_size(outf_split2_size, outf_split3_size) \
            or not fits_in_l2(outc_split_size, outf_split1_size, in_ch_split_size, 4, 256) \
            or not uses_enough_l2(outc_split_size, outf_split1_size, in_ch_split_size, 4, 256):
            continue
        else:
            filtered_parameter_grid.append(parameters_choice)

    # Write the filtered parameters grid to csv
    csv_file =  os.path.join(output_directory, "filtered_parameters_grid.csv")
    with open(csv_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(parameters_names)
        writer.writerows(filtered_parameter_grid)

    return filtered_parameter_grid

def create_single_accera_package(output_directory, input_shape, kernel_shape, output_filters,
                                    row_stride, column_stride, parameters_grid):
    # Define a accera package
    package = acc.Package()
    for parameters_choice in parameters_grid:
        add_unrolled_conv2d_function(input_shape, kernel_shape, output_filters, row_stride, column_stride,
                                     package=package, parameters_choice=parameters_choice)
    
    # Build the accera package
    package.build("unrolled_conv_using_caching", format=acc.Package.Format.MLIR, output_dir=output_directory)

def create_multiple_accera_packages(base_output_directory, input_shape, kernel_shape, output_filters,
                                      row_stride, column_stride, parameters_grid, functions_per_package=100):
    for i in range(0, len(parameters_grid), functions_per_package):
        start_idx = i
        end_idx = min(i+functions_per_package, len(parameters_grid))
        output_directory = os.path.join(base_output_directory, f"{start_idx}_{end_idx}")
        create_single_accera_package(output_directory, input_shape, kernel_shape, output_filters,
                                       row_stride, column_stride, parameters_grid[start_idx:end_idx])

def worker(index, start, end, step, base_output_directory, input_shape, kernel_shape, output_filters,
                            row_stride, column_stride, parameters_grid):
    for idx in range(start, end, step):
        start_idx = idx
        end_idx = min(idx+step, len(parameters_grid))
        print(f"Proc # {index} is creating a package with parameters {start_idx}-{end_idx}")
        output_directory = os.path.join(base_output_directory, f"{start_idx}_{end_idx}")
        create_single_accera_package(output_directory, input_shape, kernel_shape, output_filters,
                                       row_stride, column_stride, parameters_grid[start_idx:end_idx])

# NOTE: We can't use multithreading here due to a conflict on resources during building the package
def create_multiple_accera_packages_simultaneous(base_output_directory, input_shape, kernel_shape, output_filters,
                                row_stride, column_stride, parameters_grid, num_procs=3, functions_per_package=100):
    from multiprocessing import Process
    procs = []

    # Divide the parameter choices among the procs
    for i, block_idx in enumerate(range(0, len(parameters_grid), len(parameters_grid)//num_procs)):
        start = block_idx
        end = block_idx+len(parameters_grid)//num_procs
        step = min(functions_per_package, end-start)
        p = Process(target=worker, args=(i, start, end, step, base_output_directory, input_shape, kernel_shape, output_filters,
                                         row_stride, column_stride, parameters_grid))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

def autobenchmark_package(package_directory, *args, write_back=False):
    benchmark = AutoBenchmark(package_directory)

    functions = benchmark.functions
    for function_name in functions:
        if "Initialize" in function_name or "input" in function_name or "output" in function_name: # Skip init functions
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
                hat_file_path = os.path.join(package_directory, "unrolled_conv_using_caching" + ".hat")
                write_back_to_hat(hat_file_path, function_name, mean_time_secs)
        except:
            print(f"WARNING: function {function_name} failed correctness check.")

def plot_dataframe(df, output_directory):
    '''
        plot the runtime and annotate the points with iteration parameters
    '''
    import plotly.express as px
    fig = px.scatter(df, x="idx", y="runtime", \
                        labels={"runtime": "Runtime (s)", "idx": "Run Index"}, \
                        hover_data=["outf_split1_size", "outf_split2_size", "outf_split3_size", \
                                    "outc_split_size", "in_ch_split_size"], \
                        title=f"Unrolled 2D Convolution Grid Search",
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
    parser = argparse.ArgumentParser(description='Accera Unrolled 2D Convolution Grid Search Case Study')
    parser.add_argument('--input_shape', dest='input_shape', type=int, nargs='+',
                        help='Dimensions of the input 3D Matrix for Convolution.')
    parser.add_argument('--kernel_shape', dest='kernel_shape', type=int, nargs='+',
                        help='Dimensions of the 2D kernel for Convolution.')
    parser.add_argument('--output_filters', type=int,
                        help='Number of the required output filters from Convolution.')
    parser.add_argument('--stride', type=int, nargs='+',
                        help='Row and Column stride for Convolution.')
    parser.add_argument('--output_directory', type=str, default="unrolled_conv",
                        help='Output directory.')
    parser.add_argument('--sample', type=int, default=None,
                        help='Optional parameter to choose a number of sample points of the parameter grid.')
    args = parser.parse_args()

    input_shape = args.input_shape
    kernel_shape = args.kernel_shape
    output_filters = args.output_filters
    row_stride, column_stride = args.stride

    if os.path.isdir(args.output_directory):
        shutil.rmtree(args.output_directory)

    # Define the parameters
    parameters_names = ["outf_split1_size", "outf_split2_size", "outf_split3_size", \
                        "outc_split_size", "in_ch_split_size"]

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
        create_single_accera_package(args.output_directory, input_shape, kernel_shape, output_filters,
                                       row_stride, column_stride, filtered_parameters_grid)
        autobenchmark_package(args.output_directory, input_shape, kernel_shape, output_filters,
                              row_stride, column_stride, write_back=True)
        visualize_single_dir(args.output_directory)
        optimal_point = get_optimal_point_single_dir(args.output_directory)
    else:
        create_multiple_accera_packages_simultaneous(args.output_directory, input_shape, kernel_shape, output_filters,
                                                       row_stride, column_stride, filtered_parameters_grid)

        output_directories = [os.path.join(args.output_directory,item) for item in os.listdir(args.output_directory) \
                                if os.path.isdir(os.path.join(args.output_directory,item))]

        for output_directory in output_directories:
            print("Benchmarking directory", output_directory)
            autobenchmark_package(output_directory, input_shape, kernel_shape, output_filters,
                                  row_stride, column_stride, write_back=True)

        visualize_multiple_dirs(output_directories, parent_directory=args.output_directory)
        optimal_point = get_optimal_point_multiple_dirs(output_directories)

    print("Optimal point by Grid Search:")
    print(optimal_point)

    # Create a new Accera package using the optimal parameters
    optimal_package = acc.Package()

    optimal_parameters = [int(optimal_point["outf_split1_size"]),
                          int(optimal_point["outf_split2_size"]),
                          int(optimal_point["outf_split3_size"]),
                          int(optimal_point["outc_split_size"]),
                          int(optimal_point["in_ch_split_size"])]

    add_unrolled_conv2d_function(input_shape, kernel_shape, output_filters, row_stride, column_stride,
                                 package=optimal_package, parameters_choice=optimal_parameters)

    # Build the accera package
    optimal_output_directory = args.output_directory + "_optimal"
    optimal_package.build("unrolled_conv_using_caching", format=acc.Package.Format.HAT, output_dir=optimal_output_directory)
