import accera as acc
from accera.tuning import AutoBenchmark

import os, itertools, csv, shutil
import random
import argparse

from utils import get_order_permutations
from utils import permute, get_input_output_pairs, write_back_to_hat, str_to_list
from utils import fits_in_l2, uses_enough_l2, valid_split_size, valid_order_choice
from utils import load_to_dataframe, get_auxiliary_data

def add_matmul_functions_from_grid(M, N, S, package, parameters_grid):
    '''
        This function adds various functions with all the configurations in the parameter grid.
        Each function is an implementation for matrix multiplication
        of two matrices A and B, A has shape M*S and B has shape S*N.
    '''
    p_m_split_size, p_n_split_size, p_s_split_size, p_loop_order, \
            p_s_split_2_size, p_n_split_2_size, p_n_split_3_size = acc.create_parameters(7)

    A = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(M, S))
    B = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(S, N))
    C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N))

    # Define a simple affine loop nest and name its loops i, j, k
    nest = acc.Nest(shape=(M, N, S))
    i, j, k = nest.get_indices()

    # Define the logic of each iteration in the nest
    @nest.iteration_logic
    def _():
        C[i,j] += A[i,k] * B[k,j]

    schedule = nest.create_schedule()

    # Tile splits to place some blocks of the input and output matrices in the L2 cache
    ii, jj, kk = schedule.tile(indices=(i, j, k), shape=(p_m_split_size, p_n_split_size, p_s_split_size))

    # Kernel splits
    kkk = schedule.split(kk, p_s_split_2_size)
    jjj = schedule.split(jj, p_n_split_2_size)
    jjjj = schedule.split(jjj, p_n_split_3_size)

    # Apply re-ordering
    schedule.reorder(p_loop_order)

    plan = schedule.create_action_plan()

    # Cache input and output arrays
    plan.cache(B, index=jj)
    plan.cache(C, index=ii)

    # Unroll the non-vectorized kernel loops
    plan.unroll(p_loop_order[-2])
    plan.unroll(p_loop_order[-3])

    # Vectorize the innermost kernel loop
    plan.vectorize(p_loop_order[-1])

    auxiliary_data = {"m_split_size": p_m_split_size,
                      "n_split_size": p_n_split_size,
                      "s_split_size": p_s_split_size,
                      "n_split_2_size": p_n_split_2_size,
                      "n_split_3_size": p_n_split_3_size,
                      "s_split_2_size": p_s_split_2_size}

    function_name = f"matmul_{M}_{N}_{S}"
    function = acc.add_functions_from_grid(package,
                        plan,
                        args=(A, B, C),
                        parameter_grid={
                            p_m_split_size: parameters_grid["m_split_size"],
                            p_n_split_size: parameters_grid["n_split_size"],
                            p_s_split_size: parameters_grid["s_split_size"],
                            p_loop_order: [permute([i, j, k, jj, kk, kkk, ii, jjj, jjjj], order) \
                                                    for order in parameters_grid["loop_order"]],
                            p_n_split_2_size: parameters_grid["n_split_2_size"],
                            p_n_split_3_size: parameters_grid["n_split_3_size"],
                            p_s_split_2_size: parameters_grid["s_split_2_size"],
                        },
                        base_name=function_name,
                        auxiliary=auxiliary_data)

    return function

def add_matmul_function(M, N, S, package, parameters_choice):
    '''
        This function is a accera implementation for matrix multiplication
        of two matrices A and B, A has shape M*S and B has shape S*N.
        Python implemntation:
        ---------------------
        for i in range(M):
            for j in range(N):
                for k in range(S):
                    C[i,j] = A[i,k] * B[k,j]
    '''
    m_split_size, n_split_size, s_split_size, loop_order, \
            s_split_2_size, n_split_2_size, n_split_3_size = parameters_choice

    p_m_split_size, p_n_split_size, p_s_split_size, p_loop_order, \
            p_s_split_2_size, p_n_split_2_size, p_n_split_3_size = acc.create_parameters(7)

    A = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(M, S))
    B = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(S, N))
    C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N))

    # Define a simple affine loop nest and name its loops i, j, k
    nest = acc.Nest(shape=(M, N, S))
    i, j, k = nest.get_indices()

    # Define the logic of each iteration in the nest
    @nest.iteration_logic
    def _():
        C[i,j] += A[i,k] * B[k,j]

    schedule = nest.create_schedule()

    # Tile splits to place some blocks of the input and output matrices in the L2 cache
    ii, jj, kk = schedule.tile(indices=(i, j, k), shape=(p_m_split_size, p_n_split_size, p_s_split_size))

    # Kernel splits
    kkk = schedule.split(kk, p_s_split_2_size)
    jjj = schedule.split(jj, p_n_split_2_size)
    jjjj = schedule.split(jjj, p_n_split_3_size)

    schedule_items = [i, j, k, jj, kk, kkk, ii, jjj, jjjj]
    schedule_items = permute(schedule_items, loop_order)
    schedule.reorder(schedule_items) # apply re-ordering

    plan = schedule.create_action_plan()

    plan.cache(B, index=jj)
    plan.cache(C, index=ii)

    # Unroll the non-vectorized kernel loops
    plan.unroll(schedule_items[6])
    plan.unroll(schedule_items[7])

    # Vectorize the innermost kernel loop
    plan.vectorize(schedule_items[8])

    auxiliary_data = {"m_split_size": m_split_size,
                      "n_split_size": n_split_size,
                      "s_split_size": s_split_size,
                      "order": ",".join([str(i) for i in loop_order]),
                      "n_split_2_size": n_split_2_size,
                      "n_split_3_size": n_split_3_size,
                      "s_split_2_size": s_split_2_size
                     }

    function_name = f"matmul_{M}_{N}_{S}"
    function = package.add_function(plan,
                        args=(A, B, C),
                        parameters={
                            p_m_split_size: m_split_size,
                            p_n_split_size: n_split_size,
                            p_s_split_size: s_split_size,
                            p_s_split_2_size: s_split_2_size,
                            p_loop_order: schedule_items,
                            p_n_split_2_size: n_split_2_size,
                            p_n_split_3_size: n_split_3_size,
                        },
                        base_name=function_name,
                        auxiliary=auxiliary_data)

    return function

def create_parameters_grid(output_directory):
    '''
        While creating our parameterized function,
        we chose some split sizes as well as an order for our schedule loops.
        However, we are not sure that those chosen sizes would give the best performance,
        that is why we want to define a parameter grid, where our chosen parameters are:
            1. m_split_size
            2. n_split_size
            3. s_split_size
            4. loop_order
            5. s_split_2_size
            6. n_split_2_size
            7. n_split_3_size
        and our grid will consist of a set of possible values for those parameters.

        For example, we might want to:
            1. define the m_split_size, n_split_size, and s_split_size split sizes as
               the even numbers between 4 and 8, and the powers of 2 between 16 and 256
            2. define the n_split_2_size, n_split_3_size, and s_split_2_size as the powers of 2 between 4 and 16
            3. define the possible schedule loops order as any order where i, j, and k are the outer most loops in any order,
               and all the other loops follow in any order. The reason is that i, j, and k are the large scale loops,
               so having them as the outer most loops can improve caching utilization
    '''
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # define the m_split_size, n_split_size, and s_split_size split sizes as the even numbers between 4 and 8,
    # and the powers of 2 between 4 and 256
    m_split_size_choices = [4, 6, 8, 16, 32, 64, 128, 256]
    n_split_size_choices = [4, 6, 8, 16, 32, 64, 128, 256]
    s_split_size_choices = [4, 6, 8, 16, 32, 64, 128, 256]

    # define the n_split_2_size, n_split_3_size and s_split_2_size as the powers of 2 between 4 and 16
    n_split_2_size_choices = [4, 8, 16]
    n_split_3_size_choices = [4, 8, 16]
    s_split_2_size_choices = [4, 8, 16]

    # Define the possible schedule loops order as any order that preserves the accera
    # split="jj" splits the schedule loop space into two spaces, the first has ("i", "j", "k"),
    # while the second has ("jj", "kk", "kkk", "ii", "jjj", "jjjj").
    order_choices = get_order_permutations(initial_order=["i", "j", "k", "jj", "kk", "kkk", "ii", "jjj", "jjjj"],
                                           split="jj")

    parameters_choices = [m_split_size_choices, n_split_size_choices, s_split_size_choices, order_choices,
                s_split_2_size_choices, n_split_2_size_choices, n_split_3_size_choices]

    # Write the options to a csv
    csv_file =  os.path.join(output_directory, "parameters_choices.csv")
    with open(csv_file, "w", newline='') as f:
        writer = csv.writer(f)
        for name, option in zip(parameters_names, parameters_choices):
            writer.writerow([name] + option)

    # Generate options combinations
    parameters_grid = list(itertools.product(*parameters_choices))
    return parameters_grid

def filter_parameters_grid(parameters_grid, output_directory, l2_cache_size=256, element_size=4):
    filtered_parameter_grid = []
    for parameters_grid_point in parameters_grid:
        m_split_size, n_split_size, s_split_size, loop_order, \
            s_split_2_size, n_split_2_size, n_split_3_size = parameters_grid_point

        # Apply some filters to remove the redundant or non useful points from the paramters grid:
        # 1- valid_split_size: validates that the later splits are smaller than the former splits
        #    in case we did multiple splits along the same dimension.
        # 2- fits_in_l2: validates that the total outer tiles memory is smaller than the L2 cache size
        # 3- uses_enough_l2: validates that the total outer tiles memory uses at least 75% of the L2 cache size
        # 4- valid_order_choice: validates that the loop order meet some partial order and fixed order constraints,
        #    any order has to meet the Accera order constraint that "An inner dimension must not be ordered before its outer dimension"
        #    For example: The current kernel order in our function is
        #                 "i, j, k, jj, kk, kkk, ii, jjj, jjjj"
        #                 we want i to always precede ii and k to always precede kk and so on.
        #                 also we might want some indices to be fixed.
        valid = True
        if not valid_split_size(n_split_size, n_split_2_size) \
            or not valid_split_size(n_split_2_size, n_split_3_size) \
            or not valid_split_size(s_split_size, s_split_2_size) \
            or not fits_in_l2(m_split_size, n_split_size, s_split_size, element_size, l2_cache_size) \
            or not uses_enough_l2(m_split_size, n_split_size, s_split_size, element_size, l2_cache_size) \
            or not valid_order_choice(order_choice=loop_order,
                                      initial_order=["i", "j", "k", "jj", "kk", "kkk", "ii", "jjj", "jjjj"],
                                      preserve_partial_order_list=[("i", "ii"),
                                                                   ("j", "jj"),
                                                                   ("jj", "jjj"),
                                                                   ("jjj", "jjjj"),
                                                                   ("k", "kk"),
                                                                   ("kk", "kkk")],
                                      fixed_indices=["jj", "kk", "kkk"]):
            valid = False
        if valid:
            filtered_parameter_grid.append(parameters_grid_point)

    # Write the filtered parameters grid to csv
    csv_file =  os.path.join(output_directory, "filtered_parameters_grid.csv")
    with open(csv_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(parameters_names)
        writer.writerows(filtered_parameter_grid)

    return filtered_parameter_grid

def create_single_accera_package(output_directory, M, N, S, parameters_grid):
    # Define a accera package
    package = acc.Package()
    for parameters_choice in parameters_grid:
        add_matmul_function(M=M, N=N, S=S, package=package, parameters_choice=parameters_choice)

    # Build the accera package
    package.build("matmul", format=acc.Package.Format.HAT, output_dir=output_directory)

def create_multiple_accera_packages(base_output_directory, M, N, S, parameters_grid, functions_per_package=100):
    for i in range(0, len(parameters_grid), functions_per_package):
        start_idx = i
        end_idx = min(i+functions_per_package, len(parameters_grid))
        output_directory = os.path.join(base_output_directory, f"{start_idx}_{end_idx}")
        create_single_accera_package(output_directory, M, N, S, parameters_grid[start_idx:end_idx])

def worker(index, start, end, step, base_output_directory, M, N, S, parameters_grid):
    for idx in range(start, end, step):
        start_idx = idx
        end_idx = min(idx+step, len(parameters_grid))
        print(f"Proc # {index} is creating a package with parameters {start_idx}-{end_idx}")
        output_directory = os.path.join(base_output_directory, f"{start_idx}_{end_idx}")
        create_single_accera_package(output_directory, M, N, S, parameters_grid[start_idx:end_idx])

# NOTE: We can't use multithreading here due to a conflict on resources during building the package
def create_multiple_accera_packages_simultaneous(base_output_directory, M, N, S, parameters_grid, num_procs=3, functions_per_package=100):
    from multiprocessing import Process
    procs = []

    # Divide the parameter choices among the procs
    for i, block_idx in enumerate(range(0, len(parameters_grid), len(parameters_grid)//num_procs)):
        start = block_idx
        end = block_idx+len(parameters_grid)//num_procs
        step = min(functions_per_package, end-start)
        p = Process(target=worker, args=(i, start, end, step, base_output_directory, M, N, S, parameters_grid))
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
                hat_file_path = os.path.join(package_directory, "matmul" + ".hat")
                write_back_to_hat(hat_file_path, function_name, mean_time_secs)
        except:
            print(f"WARNING: function {function_name} failed correctness check.")

def plot_dataframe(df, output_directory):
    '''
        plot the runtime and annotate the points with iteration parameters
    '''
    import plotly.express as px

    fig = px.scatter(df, x="idx", y="runtime", color="order_indices", \
                        labels={"runtime": "Runtime (s)", "idx": "Run Index", "order_indices": "Loop Order", "tile_volume": "Active Tiles Volume (KB)"}, \
                        hover_data=["m_split_size", "n_split_size", "s_split_size", "s_split_2_size", "n_split_2_size",
                                    "n_split_3_size", 'order_indices', 'tile_volume', 'idx'], \
                        title=f"Matrix Multiplication Grid Search",
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
    parser = argparse.ArgumentParser(description='Accera MatMul tutorial')
    parser.add_argument('--matmul_dim', dest='matmul_dim', type=int, nargs='+',
                        help='Dimensions of the MatMul.')
    parser.add_argument('--output_directory', type=str, default="matmul_pkg",
                        help='Output directory.')
    parser.add_argument('--sample', type=int, default=None,
                        help='Optional parameter to choose a number of sample points of the parameter grid.')
    args = parser.parse_args()

    if os.path.isdir(args.output_directory):
        shutil.rmtree(args.output_directory)

    M, N, S = args.matmul_dim

    # Define the parameters
    parameters_names = ["m_split_size", "n_split_size", "s_split_size", "loop_order",
                        "s_split_2_size", "n_split_2_size", "n_split_3_size"]

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
        create_single_accera_package(args.output_directory, M, N, S, filtered_parameters_grid)

        autobenchmark_package(args.output_directory, M, N, S, write_back=True)
        visualize_single_dir(args.output_directory)
        optimal_point = get_optimal_point_single_dir(args.output_directory)
    else:
        create_multiple_accera_packages_simultaneous(args.output_directory, M, N, S, filtered_parameters_grid)

        output_directories = [os.path.join(args.output_directory,item) for item in os.listdir(args.output_directory) \
                              if os.path.isdir(os.path.join(args.output_directory,item))]

        for output_directory in output_directories: # Multiple packages
            autobenchmark_package(output_directory, M, N, S, write_back=True)
        visualize_multiple_dirs(output_directories, parent_directory=args.output_directory)
        optimal_point = get_optimal_point_multiple_dirs(output_directories)

    print("Optimal point by Grid Search:")
    print(optimal_point)

    # Create a new Accera package using the optimal parameters
    optimal_package = acc.Package()

    optimal_parameters = [int(optimal_point["m_split_size"]),
                          int(optimal_point["n_split_size"]),
                          int(optimal_point["s_split_size"]),
                          str_to_list(optimal_point["order"]),
                          int(optimal_point["s_split_2_size"]),
                          int(optimal_point["n_split_2_size"]),
                          int(optimal_point["n_split_3_size"])]

    add_matmul_function(M=M, N=N, S=S, package=optimal_package, parameters_choice=optimal_parameters)

    # Build the accera package
    optimal_output_directory = args.output_directory + "_optimal"
    optimal_package.build("matmul", format=acc.Package.Format.HAT, output_dir=optimal_output_directory)
