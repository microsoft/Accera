####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
# Authors: Mason Remy
# Requires: Python 3.7+
####################################################################################################

import os
import shutil
from enum import Enum, Flag, auto

from .utilities import *
from .parameters import *
from .accc_config import ACCCConfig
from .build_config import BuildConfig

build_config_types = ["Release", "Debug", "RelWithDebInfo"]
CPU_TARGET = "CPU"
GPU_TARGET = "GPU"
target_options = [CPU_TARGET, GPU_TARGET]
default_target = target_options[0]


class SystemTarget(Enum):
    HOST = "host"
    AVX512 = "avx512"
    RPI4 = "pi4"
    RPI3 = "pi3"
    RPI0 = "pi0"
    ARM_CORTEX_M4 = "cortex-m4"
    ARM_CORTEX_M4F = "cortex-m4f"


class Runtime(Enum):
    NONE = "none"
    CUDA = "cuda"
    ROCM = "rocm"
    VULKAN = "vulkan"
    OPENMP = "openmp"
    DEFAULT = "default"


system_target_options = [t.value for t in SystemTarget]

# Features are modeled by a kvp where the value is an optional lambda that does verification of a feature
# option (for example, ensuring that the number of threads is >= 0. If None is used instead, then the
# feature does not accept a value.
BASE_FEATURES = {
    'num_threads': lambda val: int(val) >= 0,
    'no_opt': None
}
# Options are modeled by creating a set of base options and the features they support, which in turn follows the model
# above

CPU_OPTS = {
    "arm": {},
    "intel": {
        'avx2': None,
        'avx512': None
    },
    "amd": {}
}
GPU_OPTS = {
    "none": {},
    "vulkan": {}
}
OS_OPTS = {
    "windows": {},
    "linux": {},
    "macos": {}
}


def DEFAULT_RC_MLIR_LOWERING_PASSES(
    dump=False,
    dump_intrapass_ir=False,
    system_target=SystemTarget.HOST.value,
    profile=False,
    runtime=Runtime.DEFAULT.value,
    gpu_only=False
):
    def bstr(val):
        return "true" if val else "false"

    acc_to_llvm_str = " ".join([
        f'dump-passes={bstr(dump)}',
        f'dump-intra-pass-ir={bstr(dump_intrapass_ir)}',
        f'runtime={str(runtime).lower()}',
        f'target={system_target}',
        f'enable-profiling={bstr(profile)}',
        f'gpu-only={bstr(gpu_only)}',
    ])

    return [f'--acc-to-llvm="{acc_to_llvm_str}"']


DEFAULT_RC_OPT_ARGS = ["--verify-each=false"]

DEFAULT_ACC_TRANSLATE_ARGS = []

DEFAULT_MLIR_TRANSLATE_ARGS = ["--mlir-print-op-on-diagnostic", "--acc-to-llvmir"]

DEFAULT_LOW_PRECISION_FLOAT_OPTS = ["-fp-contract=fast", "--enable-unsafe-fp-math"]
DEFAULT_HIGH_PRECISION_FLOAT_OPTS = ["-fp-contract=on"]

OPT_DISABLE_LOOP_UNROLLING_ARGS = ["--disable-loop-unrolling"]

LLVM_TOOLING_OPTS = {
    SystemTarget.HOST.value: ["-O3", "-mcpu=native"],
    SystemTarget.RPI4.value: [
        "-O3", "--march=arm", "-mcpu=cortex-a72", "--mtriple=armv7-linux-gnueabihf"
    ],
    SystemTarget.RPI3.value: [
        "-O3", "--march=arm", "-mcpu=cortex-a53", "--mtriple=armv7-linux-gnueabihf"
    ],
    SystemTarget.RPI0.value: [
        "-O3", "--march=arm", "-mcpu=arm1136jf-s", "--mtriple=armv6-linux-gnueabihf"
    ],
    SystemTarget.AVX512.value: ["-O3", "--march=x86-64", "-mcpu=skylake-avx512"],
    SystemTarget.ARM_CORTEX_M4.value: [
        "-Oz", "-mcpu=cortex-m4", "--mtriple=thumbv7em-arm-none-eabi",
    ],
    SystemTarget.ARM_CORTEX_M4F.value: [
        "-Oz", "-mcpu=cortex-m4", "--mtriple=thumbv7em-arm-none-eabi", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard"
    ],
}

DEFAULT_LLVM_TOOLING_OPTS = [
    '--enable-no-infs-fp-math',
    '--enable-no-nans-fp-math',
    '--enable-no-signed-zeros-fp-math',
    '--enable-no-trapping-fp-math'
]

DEFAULT_OPT_ARGS = DEFAULT_LLVM_TOOLING_OPTS + []

DEFAULT_LLC_ARGS = DEFAULT_LLVM_TOOLING_OPTS + ["-relocation-model=pic"]

class Options(Flag):
    NONE = auto() # (enable auto unroll | low precision float)
    DISABLE_AUTO_UNROLL = auto()
    HIGH_PRECISION_FLOATING_POINT_OPS = auto()

def _get_common_fp_options_args(options: Options):
    if options & Options.HIGH_PRECISION_FLOATING_POINT_OPS:
        return DEFAULT_HIGH_PRECISION_FLOAT_OPTS
    else:
        return DEFAULT_LOW_PRECISION_FLOAT_OPTS

def _get_options_opt_args(options: Options):
    args = []

    if options & Options.DISABLE_AUTO_UNROLL:
        args += OPT_DISABLE_LOOP_UNROLLING_ARGS

    args += _get_common_fp_options_args(options)

    return args

def _get_options_llc_args(options: Options):
    args = []

    args += _get_common_fp_options_args(options)

    return args


def get_default_deploy_shared_libraries(target=CPU_TARGET):
    if target == GPU_TARGET:
        if os.path.isfile(BuildConfig.vulkan_runtime_wrapper_shared_library):
            return [BuildConfig.vulkan_runtime_wrapper_shared_library]
        else:
            raise (ValueError("GPU support is not enabled"))
    else:
        return []


class BuiltAcceraEmittedLibrary:
    def __init__(self, library_path, dependencies=[]):
        self.library_path = os.path.abspath(library_path)
        self.library_dir = os.path.dirname(self.library_path)
        self.dependencies = dependencies


class BuiltAcceraProgram:
    def __init__(self, exe_path):
        self.exe_path = os.path.abspath(exe_path)
        self.exe_dir = os.path.dirname(self.exe_path)

    def run(
        self,
        parameters=ParameterCollection([]),
        working_dir=None,
        stdout=None,
        stderr=None,
        pretend=False,
        quiet=True
    ):
        if not isinstance(parameters, ParameterCollection):
            raise ValueError("Parameters must be a ParameterCollection")
        cmd = f"{self.exe_path} {parameters.to_cmd_argstring()}"
        if working_dir is None:
            working_dir = self.exe_dir
        run_command(cmd, working_dir, stdout=stdout, stderr=stderr, pretend=pretend, quiet=quiet)

    def run_on_high_performance_gpu(self, pretend=False, quiet=True):
        set_high_performance_gpu(self.exe_path, pretend=pretend, quiet=quiet)


class AcceraCMakeProject:
    class TargetType(Enum):
        Executable = 0
        StaticLibrary = 1
        SharedLibrary = 2

    def __init__(self, project_dir, target_name, target_type=TargetType.Executable, pretend=False):
        prefix = ""
        suffix = ""
        if target_type == self.TargetType.Executable:
            suffix = BuildConfig.exe_extension
        elif target_type == self.TargetType.StaticLibrary:
            prefix = BuildConfig.static_library_prefix
            suffix = BuildConfig.static_library_extension
        elif target_type == self.TargetType.SharedLibrary:
            prefix = BuildConfig.shared_library_prefix
            suffix = BuildConfig.shared_library_extension
        self.target_name = prefix + target_name + suffix
        self.project_dir = os.path.abspath(project_dir)
        self.target_type = target_type
        self.pretend = pretend

    def build(
        self,
        build_dir_name="build",
        build_config=build_config_types[0],
        stdout=None,
        stderr=None,
        pretend=False,
        quiet=True
    ):

        build_dir = os.path.join(self.project_dir, build_dir_name)

        if not quiet:
            print(
                f"\n### Building Accera cmake project for {self.target_name} in {build_dir} with config {build_config}...\n"
            )

        makedir(build_dir, pretend=pretend, quiet=quiet)

        run_command(
            get_cmake_initialization_cmd(build_config),
            build_dir,
            stdout=stdout,
            stderr=stderr,
            pretend=pretend,
            quiet=quiet
        )
        run_command(
            get_cmake_build_cmd(build_config), build_dir, stdout=stdout, stderr=stderr, pretend=pretend, quiet=quiet
        )

        built_target_path = get_built_target_path(build_dir, build_config, self.target_name)
        if self.target_type == self.TargetType.Executable:
            return BuiltAcceraProgram(built_target_path)
        else:
            return BuiltAcceraEmittedLibrary(built_target_path)


def create_simple_project_dir(
    project_root_dir, root_files=[], src_files=[], include_files=[], additional_dir_names_and_files={}
):
    # Create directory structure:
    # <project_root_dir>/
    #       root_files...
    #       include/
    #           include_files...
    #       src/
    #           src_files...
    #       additional_dir_0/
    #           deploy_files...
    #       additional_dir_1/
    #           additional_files...
    #       ...

    os.makedirs(project_root_dir, exist_ok=True)
    for root_file in root_files:
        shutil.copy(root_file, project_root_dir)

    project_src_dir = os.path.join(project_root_dir, "src")
    os.makedirs(project_src_dir, exist_ok=True)
    for src_file in src_files:
        shutil.copy(src_file, project_src_dir)

    project_include_dir = os.path.join(project_root_dir, "include")
    os.makedirs(project_include_dir, exist_ok=True)
    for include_file in include_files:
        shutil.copy(include_file, project_include_dir)

    for additional_dir_name in additional_dir_names_and_files:
        additional_dir = os.path.join(project_root_dir, additional_dir_name)
        os.makedirs(additional_dir, exist_ok=True)
        deploy_files = additional_dir_names_and_files[additional_dir_name]
        for filepath in deploy_files:
            shutil.copy(filepath, additional_dir)


def deploy_accera_generator_project(
    deploy_dir,
    generator_name,
    dsl_src_filepath,
    additional_text_replacements={},
    additional_filename_replacements={},
    additional_replace_file_exts=[],
    pretend=False
):
    # Configure file modifications for deployment
    default_text_replacements = {
        ACCCConfig.program_name_tag: generator_name,
        ACCCConfig.dsl_file_basename_tag: os.path.splitext(os.path.basename(dsl_src_filepath))[0]
    }

    text_replacements = {
        **default_text_replacements,
        **additional_text_replacements
    }

    default_filename_replacements = {
        "CMakeLists.txt.generator.in": "CMakeLists.txt"
    }
    filename_replacements = {
        **default_filename_replacements,
        **additional_filename_replacements
    }

    # File extensions to text-replace in
    default_replace_file_exts = [".cpp", ".h", ".txt"]
    replace_file_exts = default_replace_file_exts + additional_replace_file_exts

    # Produce directory structure:
    # <deploy_dir>\
    #       CMakeLists.txt
    #       src\
    #           <DSL src file>.cpp
    if not pretend:
        create_simple_project_dir(deploy_dir, root_files=[ACCCConfig.generator_cmakelist], src_files=[dsl_src_filepath])
        rename_files_in_dir(deploy_dir, filename_replacements)
        replace_file_text_in_dir(deploy_dir, text_replacements, replace_file_exts)

    return AcceraCMakeProject(
        deploy_dir, target_name=generator_name, target_type=AcceraCMakeProject.TargetType.Executable, pretend=pretend
    )


def deploy_accera_emitted_lib_project(
    deploy_dir,
    library_name,
    obj_files,
    additional_text_replacements={},
    additional_filename_replacements={},
    additional_replace_file_exts=[],
    pretend=False
):
    # Configure file modifications for deployment
    default_text_replacements = {
        ACCCConfig.library_name_tag: library_name
    }

    text_replacements = {
        **default_text_replacements,
        **additional_text_replacements
    }

    default_filename_replacements = {
        "CMakeLists.txt.emitted_library.in": "CMakeLists.txt"
    }
    filename_replacements = {
        **default_filename_replacements,
        **additional_filename_replacements
    }

    # File extensions to text-replace in
    default_replace_file_exts = [".txt"]
    replace_file_exts = default_replace_file_exts + additional_replace_file_exts

    # Produce directory structure:
    # <deploy_dir>\
    #       CMakeLists.txt
    #       src\
    #           <object_file_0>.obj
    #           <object_file_1>.obj
    if not pretend:
        create_simple_project_dir(deploy_dir, root_files=[ACCCConfig.emitted_lib_cmakelist], src_files=obj_files)
        rename_files_in_dir(deploy_dir, filename_replacements)
        replace_file_text_in_dir(deploy_dir, text_replacements, replace_file_exts)

    return AcceraCMakeProject(
        deploy_dir, target_name=library_name, target_type=AcceraCMakeProject.TargetType.StaticLibrary, pretend=pretend
    )


def deploy_accera_main_project(
    deploy_dir,
    main_name,
    library_name,
    main_src_filepath,
    emitted_header_path,
    emitted_library_path,
    target=default_target,
    deploy_shared_libraries=[],
    additional_text_replacements={},
    additional_filename_replacements={},
    additional_replace_file_exts=[],
    pretend=False
):
    default_text_replacements = {
        ACCCConfig.program_name_tag: main_name,
        ACCCConfig.library_name_tag: library_name,
        ACCCConfig.main_basename_tag: os.path.splitext(os.path.basename(main_src_filepath))[0],
        ACCCConfig.main_deploy_target_type_tag: target
    }
    text_replacements = {
        **default_text_replacements,
        **additional_text_replacements
    }

    default_filename_replacements = {
        "CMakeLists.txt.main.in": "CMakeLists.txt"
    }
    filename_replacements = {
        **default_filename_replacements,
        **additional_filename_replacements
    }

    # File extensions to text-replace in
    default_replace_file_exts = [".cpp", ".h", ".txt"]
    replace_file_exts = default_replace_file_exts + additional_replace_file_exts

    # Produce directory structure:
    # <deploy_dir>/
    #       CMakeLists.txt
    #       include/
    #           <library_name>.h
    #       src/
    #           <main src file>.cpp
    #           <library_name>.lib
    #       deploy/
    #           <shared_lib_0.dll>
    #           ...

    additional_deploy = {
        ACCCConfig.main_deploy_dir_name: deploy_shared_libraries
    }

    if not pretend:
        create_simple_project_dir(
            deploy_dir,
            root_files=[ACCCConfig.main_cmakelist],
            src_files=[main_src_filepath, emitted_library_path],
            include_files=[emitted_header_path],
            additional_dir_names_and_files=additional_deploy
        )

        rename_files_in_dir(deploy_dir, filename_replacements)
        replace_file_text_in_dir(deploy_dir, text_replacements, replace_file_exts)

    return AcceraCMakeProject(
        deploy_dir, target_name=main_name, target_type=AcceraCMakeProject.TargetType.Executable, pretend=pretend
    )


class ModuleOutputType(Enum):
    OBJECT = auto()
    CPP = auto()
    CUDA = auto()


class ModuleFileSet:
    def __init__(
        self,
        name,
        common_module_dir,
        output_type=ModuleOutputType.OBJECT,
        lowered_mlir_suffix="_llvm",
        mlir_ext=".mlir",
        ll_ext=".ll",
        opt_ext=".bc",
        cuda_ext=".cu",
        cpp_ext=".cpp"
    ):

        self.module_name = name
        self.common_module_dir = common_module_dir
        self.output_type = output_type
        self.module_dir = os.path.join(self.common_module_dir, self.module_name)
        self.generated_mlir_filepath = os.path.abspath(
            os.path.join(self.common_module_dir, self.module_name + mlir_ext)
        )
        self.lowered_mlir_filepath = os.path.abspath(
            os.path.join(self.module_dir, self.module_name + lowered_mlir_suffix + mlir_ext)
        )

        if self.output_type == ModuleOutputType.OBJECT:
            self.translated_ll_filepath = os.path.abspath(os.path.join(self.module_dir, self.module_name + ll_ext))
            self.optimized_ll_filepath = os.path.abspath(os.path.join(self.module_dir, self.module_name + opt_ext))
            self.object_filepath = os.path.abspath(
                os.path.join(self.module_dir, self.module_name + BuildConfig.obj_extension)
            )
            self.asm_filepath = os.path.abspath(
                os.path.join(self.module_dir, self.module_name + BuildConfig.asm_extension)
            )

        elif self.output_type in [ModuleOutputType.CUDA, ModuleOutputType.CPP]:
            ext = {
                ModuleOutputType.CUDA: cuda_ext,
                ModuleOutputType.CPP: cpp_ext
            }[self.output_type]
            self.translated_source_filepath = os.path.abspath(os.path.join(self.module_dir, self.module_name + ext))

    def __repr__(self):
        desc = [
            f"Name: {self.module_name}",
            f"Directory: {self.module_dir}",
            f"Output Type: {self.output_type.name.capitalize()}",
            f"Accera MLIR: {self.generated_mlir_filepath}",
            f"Lowered MLIR: {self.lowered_mlir_filepath}",
        ]

        if self.output_type == ModuleOutputType.OBJECT:
            desc += [
                f"LLVM IR: {self.translated_ll_filepath}",
                f"Optimized LLVM IR: {self.optimized_ll_filepath}",
                f"Object file: {self.object_filepath}",
                f"ASM Output: {self.asm_filepath}",
            ]
        else:
            desc += [f"Source Output: {self.translated_source_filepath}"]

        s = "\n".join(desc)

        return s


class AcceraProject:
    stdout_key = "stdout"
    stderr_key = "stderr"

    def __init__(
        self,
        output_dir,
        library_name,
        dsl_src_filepath=None,
        main_src_filepath=None,
        generator_dir_name="generator",
        library_dir_name="lib",
        main_dir_name="main",
        intermediate_working_dir_suffix="_intermediate",
        lowered_mlir_suffix="_llvm",
        mlir_ext=".mlir",
        ll_ext=".ll",
        opt_ext=".bc",
        cuda_ext=".cu",
        cpp_ext=".cpp",
        output_type=ModuleOutputType.OBJECT,
        print_subprocess_output=False,
        pretend=False,
        quiet=True
    ):

        self.library_name = library_name
        self.lowered_mlir_suffix = lowered_mlir_suffix
        self.mlir_ext = mlir_ext
        self.ll_ext = ll_ext
        self.opt_ext = opt_ext
        self.cuda_ext = cuda_ext
        self.cpp_ext = cpp_ext
        self.output_type = output_type
        self.output_dir = os.path.abspath(output_dir)
        self.print_subprocess_output = print_subprocess_output
        self.pretend = pretend
        self.quiet = quiet

        # Create the logs directory
        self.log_dir = os.path.join(self.output_dir, "logs")
        makedir(self.log_dir, pretend=pretend, quiet=self.quiet)

        # Create intermediate files working directory
        self.intermediate_working_dir = os.path.abspath(
            os.path.join(self.output_dir, self.library_name + intermediate_working_dir_suffix)
        )
        makedir(self.intermediate_working_dir, pretend=pretend, quiet=self.quiet)

        makedir(self.output_dir, pretend=pretend, quiet=self.quiet)

        # Create generator directory
        self.generator_dir = os.path.join(self.output_dir, generator_dir_name)
        makedir(self.generator_dir, pretend=pretend, quiet=self.quiet)
        if dsl_src_filepath:
            self.dsl_src_filepath = os.path.abspath(dsl_src_filepath)
            self.generator_name = self.library_name + "_generator"

            self.generator_project = deploy_accera_generator_project(
                deploy_dir=self.generator_dir,
                generator_name=self.generator_name,
                dsl_src_filepath=self.dsl_src_filepath,
                pretend=self.pretend
            )

        # Create lib directory
        self.library_dir = os.path.abspath(os.path.join(self.output_dir, library_dir_name))
        makedir(self.library_dir, pretend=pretend, quiet=self.quiet)
        # Can't deploy the emitted library project until we have the object files in hand produced by lowering the generated MLIR code

        # Create main runner program directory
        if main_src_filepath:
            self.main_dir = os.path.join(self.output_dir, main_dir_name)
            makedir(self.main_dir, pretend=pretend, quiet=self.quiet)
            self.main_src_filepath = os.path.abspath(main_src_filepath)
            self.main_name = self.library_name + "_main"

    def make_log_filepaths(self, tag):
        stdout_filename_template = "{}_stdout.txt"
        stderr_filename_template = "{}_stderr.txt"
        return {
            self.stdout_key: os.path.join(self.log_dir, stdout_filename_template.format(tag)),
            self.stderr_key: os.path.join(self.log_dir, stderr_filename_template.format(tag))
        }

    def build_generator(
        self,
        build_dir_name="build",
        build_config=build_config_types[0],
        stdout=None,
        stderr=None,
        pretend=False,
        quiet=None
    ):

        quiet = quiet if quiet is not None else self.quiet

        if self.print_subprocess_output:
            stdout = None
            stderr = None
        self.built_generator = self.generator_project.build(
            build_dir_name=build_dir_name,
            build_config=build_config,
            stdout=stdout,
            stderr=stderr,
            pretend=pretend,
            quiet=quiet
        )

    def build_main(
        self,
        deploy_shared_libraries=[],
        build_dir_name="build",
        build_config=build_config_types[0],
        target=default_target,
        stdout=None,
        stderr=None,
        pretend=False,
        system_target=SystemTarget.HOST.value,
        quiet=None
    ):

        quiet = quiet if quiet is not None else self.quiet

        if self.print_subprocess_output:
            stdout = None
            stderr = None
        self.main_project = deploy_accera_main_project(
            deploy_dir=self.main_dir,
            main_name=self.main_name,
            library_name=self.library_name,
            target=target,
            main_src_filepath=self.main_src_filepath,
            emitted_header_path=self.emitted_header_path,
            emitted_library_path=self.emitted_library.library_path,
            deploy_shared_libraries=deploy_shared_libraries,
            pretend=pretend
        )

        if system_target != SystemTarget.HOST.value:
            return

        self.built_main = self.main_project.build(
            build_dir_name=build_dir_name,
            build_config=build_config,
            stdout=stdout,
            stderr=stderr,
            pretend=pretend,
            quiet=quiet
        )
        return self.built_main

    def generate_mlir(
        self,
        generator_parameters,
        stdout=None,
        stderr=None,
        pretend=False,
        system_target=SystemTarget.HOST.value,
        quiet=None
    ):

        quiet = quiet if quiet is not None else self.quiet

        if not isinstance(generator_parameters, ParameterCollection):
            raise ValueError("Generator parameters must be a ParameterCollection")
        if not self.built_generator:
            self.build_generator(stdout=stdout, stderr=stderr, pretend=pretend)

        # Clear the working dir since we will interpet all .mlir files in it as products of the generator
        if os.path.exists(self.intermediate_working_dir):
            rmdir(self.intermediate_working_dir, pretend=pretend, quiet=quiet)
        makedir(self.intermediate_working_dir, pretend=pretend, quiet=quiet)

        if self.print_subprocess_output:
            stdout = None
            stderr = None
        self.built_generator.run(
            generator_parameters,
            working_dir=self.intermediate_working_dir,
            stdout=stdout,
            stderr=stderr,
            pretend=pretend,
            quiet=quiet
        )
        self.emitted_header_path = os.path.join(self.intermediate_working_dir, self.library_name + ".h")
        generated_mlir_filenames = [
            filename for filename in os.listdir(self.intermediate_working_dir) if filename.endswith(self.mlir_ext)
        ]
        generated_mlir_modulenames = [os.path.splitext(filename)[0] for filename in generated_mlir_filenames]
        self.module_file_sets = [
            ModuleFileSet(
                module_name,
                common_module_dir=self.intermediate_working_dir,
                lowered_mlir_suffix=self.lowered_mlir_suffix,
                mlir_ext=self.mlir_ext,
                ll_ext=self.ll_ext,
                opt_ext=self.opt_ext,
                cuda_ext=self.cuda_ext,
                cpp_ext=self.cpp_ext,
                output_type=self.output_type
            ) for module_name in generated_mlir_modulenames
        ]

    def lower_mlir(
        self,
        run_default_passes=True,
        dump_all_passes=False,
        dump_intrapass_ir=False,
        rc_opt_args=[],
        additional_passes=[],
        stderr=None,
        pretend=False,
        system_target=SystemTarget.HOST.value,
        runtime=Runtime.DEFAULT.value,
        profile=False,
        quiet=None,
        gpu_only=False
    ):

        quiet = quiet if quiet is not None else self.quiet

        default_passes = DEFAULT_RC_MLIR_LOWERING_PASSES(
            dump=dump_all_passes,
            dump_intrapass_ir=dump_intrapass_ir,
            system_target=system_target,
            runtime=runtime,
            profile=profile,
            gpu_only=gpu_only
        )

        if self.print_subprocess_output:
            stderr = None

        rc_opt_exe = os.path.abspath(ACCCConfig.rc_opt)
        rc_opt_base_args = rc_opt_args or DEFAULT_RC_OPT_ARGS

        for module_file_set in self.module_file_sets:
            current_output_path = module_file_set.module_dir

            makedir(current_output_path, pretend=pretend, quiet=quiet)
            all_passes = []    # Clear list every time the loop iterates
            all_passes += additional_passes
            if run_default_passes:
                all_passes += default_passes

            input_mlir_filepath = module_file_set.generated_mlir_filepath
            initial_filename = "0_Initial.mlir"
            initial_filepath = os.path.join(current_output_path, initial_filename)
            if not pretend:
                shutil.copyfile(input_mlir_filepath, initial_filepath)

            rc_opt_command = " ".join([f'"{rc_opt_exe}"'] + rc_opt_base_args + all_passes +
                                      [f'"{input_mlir_filepath}"'])
            with OpenFile(module_file_set.lowered_mlir_filepath, "w", pretend=pretend) as current_output_fd:
                run_command(
                    rc_opt_command,
                    working_directory=current_output_path,
                    stdout=current_output_fd,
                    stderr=stderr,
                    pretend=pretend,
                    quiet=quiet
                )

    def translate_mlir_with_acc_translate(
        self,
        acc_translate_args=None,
        stdout=None,
        stderr=None,
        pretend=False,
        system_target=SystemTarget.HOST.value,
        quiet=None
    ):

        quiet = quiet if quiet is not None else self.quiet

        if self.print_subprocess_output:
            stdout = None
            stderr = None
        for module_file_set in self.module_file_sets:

            output_type_args = {
                ModuleOutputType.CUDA: ["-print-cpp"],
                ModuleOutputType.CPP: ["-print-cpp"],
            }

            acc_translate_exe = os.path.abspath(ACCCConfig.acc_translate)
            full_acc_translate_args = output_type_args[self.output_type]
            full_acc_translate_args += acc_translate_args or DEFAULT_ACC_TRANSLATE_ARGS
            full_acc_translate_args += [f'-o="{module_file_set.translated_source_filepath}"']
            full_acc_translate_args += [f'"{module_file_set.lowered_mlir_filepath}"']
            acc_translate_command = " ".join([f'"{acc_translate_exe}"'] + full_acc_translate_args)
            run_command(
                acc_translate_command,
                working_directory=self.intermediate_working_dir,
                stdout=stdout,
                stderr=stderr,
                pretend=pretend,
                quiet=quiet
            )

    def translate_mlir_with_mlir_translate(
        self,
        mlir_translate_args=None,
        stdout=None,
        stderr=None,
        pretend=False,
        system_target=SystemTarget.HOST.value,
        quiet=None
    ):

        quiet = quiet if quiet is not None else self.quiet

        if self.print_subprocess_output:
            stdout = None
            stderr = None
        for module_file_set in self.module_file_sets:
            mlir_translate_exe = os.path.abspath(ACCCConfig.acc_translate)
            full_mlir_translate_args = []    # empty list every iteration
            full_mlir_translate_args += mlir_translate_args or DEFAULT_MLIR_TRANSLATE_ARGS
            full_mlir_translate_args += [f'-o="{module_file_set.translated_ll_filepath}"']
            full_mlir_translate_args += [f'"{module_file_set.lowered_mlir_filepath}"']
            mlir_translate_command = " ".join([f'"{mlir_translate_exe}"'] + full_mlir_translate_args)
            run_command(
                mlir_translate_command,
                working_directory=self.intermediate_working_dir,
                stdout=stdout,
                stderr=stderr,
                pretend=pretend,
                quiet=quiet
            )

    def optimize_llvm(
        self,
        llvm_opt_args=None,
        stdout=None,
        stderr=None,
        pretend=False,
        system_target=SystemTarget.HOST.value,
        quiet=None,
        _additional_llvm_opt_args=[] # Doesn't override defaults like llvm_opt_args does
    ):

        quiet = quiet if quiet is not None else self.quiet

        if self.print_subprocess_output:
            stdout = None
            stderr = None
        for module_file_set in self.module_file_sets:
            llvm_opt_exe = os.path.abspath(ACCCConfig.llvm_opt)
            full_llvm_opt_args = []    # empty list every iteration
            full_llvm_opt_args += llvm_opt_args or (LLVM_TOOLING_OPTS[system_target] + DEFAULT_OPT_ARGS)
            full_llvm_opt_args += [f'-o="{module_file_set.optimized_ll_filepath}"']
            full_llvm_opt_args += [f'"{module_file_set.translated_ll_filepath}"']
            full_llvm_opt_args += _additional_llvm_opt_args
            llvm_opt_command = " ".join([f'"{llvm_opt_exe}"'] + full_llvm_opt_args)
            run_command(
                llvm_opt_command,
                working_directory=self.intermediate_working_dir,
                stdout=stdout,
                stderr=stderr,
                pretend=pretend,
                quiet=quiet
            )

    def generate_object(
        self,
        llc_args=None,
        stdout=None,
        stderr=None,
        pretend=False,
        system_target=SystemTarget.HOST.value,
        quiet=None,
        _additional_llvm_llc_args=[] # Doesn't override defaults like llc_args does
    ):

        quiet = quiet if quiet is not None else self.quiet

        if self.print_subprocess_output:
            stdout = None
            stderr = None
        for module_file_set in self.module_file_sets:
            llc_exe = os.path.abspath(ACCCConfig.llc)
            full_llc_args = []    # empty list every iteration
            full_llc_args += llc_args or (LLVM_TOOLING_OPTS[system_target] + DEFAULT_LLC_ARGS)
            full_llc_args += ["-filetype=obj"]
            full_llc_args += _additional_llvm_llc_args
            full_llc_args += [f'-o="{module_file_set.object_filepath}"']
            full_llc_args += [f'"{module_file_set.optimized_ll_filepath}"']
            llc_command = " ".join([f'"{llc_exe}"'] + full_llc_args)
            run_command(
                llc_command,
                working_directory=self.intermediate_working_dir,
                stdout=stdout,
                stderr=stderr,
                pretend=pretend,
                quiet=quiet
            )

    def generate_asm(
        self,
        llc_args=None,
        stdout=None,
        stderr=None,
        pretend=False,
        system_target=SystemTarget.HOST.value,
        quiet=None
    ):

        quiet = quiet if quiet is not None else self.quiet

        if self.print_subprocess_output:
            stdout = None
            stderr = None
        for module_file_set in self.module_file_sets:
            llc_exe = os.path.abspath(ACCCConfig.llc)
            full_llc_args = []    # empty list every iteration
            full_llc_args += llc_args or (LLVM_TOOLING_OPTS[system_target] + DEFAULT_LLC_ARGS)
            full_llc_args += ["--filetype=asm"]
            full_llc_args += [f'-o="{module_file_set.asm_filepath}"']
            full_llc_args += [f'"{module_file_set.optimized_ll_filepath}"']
            llc_command = " ".join([f'"{llc_exe}"'] + full_llc_args)
            run_command(
                llc_command,
                working_directory=self.intermediate_working_dir,
                stdout=stdout,
                stderr=stderr,
                pretend=pretend,
                quiet=quiet
            )

    def build_static_lib(
        self,
        build_dir_name="build",
        build_config=build_config_types[0],
        stdout=None,
        stderr=None,
        pretend=False,
        system_target=SystemTarget.HOST.value,
        quiet=None
    ):

        quiet = quiet if quiet is not None else self.quiet

        if self.print_subprocess_output:
            stdout = None
            stderr = None
        obj_files = [module_file_set.object_filepath for module_file_set in self.module_file_sets]
        self.emitted_lib_project = deploy_accera_emitted_lib_project(
            deploy_dir=self.library_dir, library_name=self.library_name, obj_files=obj_files, pretend=pretend
        )
        self.emitted_library = self.emitted_lib_project.build(
            build_dir_name=build_dir_name,
            build_config=build_config,
            stdout=stdout,
            stderr=stderr,
            pretend=pretend,
            quiet=quiet
        )

    def generate_and_emit(
        self,
        generator_parameters=None,
        build_config=build_config_types[0],
        profile=False,
        dump_all_passes=False,
        dump_intrapass_ir=False,
        pretend=False,
        system_target=SystemTarget.HOST.value,
        runtime=Runtime.DEFAULT.value,
        quiet=None,
        gpu_only=False,
        _options: Options=Options.NONE
    ):
        # By default, save stdout and stderr for each phase to separate files

        quiet = quiet if quiet is not None else self.quiet

        emit_files = self.make_log_filepaths("emit")
        mlir_lowering_files = self.make_log_filepaths("mlir_lowering")
        translate_files = self.make_log_filepaths("translate_mlir")
        opt_files = self.make_log_filepaths("opt")
        llc_files = self.make_log_filepaths("llc")
        llc_asm_files = self.make_log_filepaths("llc_asm")
        emitted_lib_files = self.make_log_filepaths("emitted_library")

        if generator_parameters:
            with OpenFile(emit_files[self.stdout_key], "w", pretend=pretend) as stdout_file:
                with OpenFile(emit_files[self.stderr_key], "w", pretend=pretend) as stderr_file:
                    self.generate_mlir(
                        generator_parameters,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        pretend=pretend,
                        system_target=system_target,
                        quiet=quiet
                    )

        # Note: mlir-opt doesn't appear to support the -o option correctly, so all output goes to stdout
        #       therefore we can't capture and log stdout separately as we need it for the lowering pipeling
        with OpenFile(mlir_lowering_files[self.stderr_key], "w", pretend=pretend) as stderr_file:
            self.lower_mlir(
                dump_all_passes=dump_all_passes,
                dump_intrapass_ir=dump_intrapass_ir,
                stderr=stderr_file,
                pretend=pretend,
                system_target=system_target,
                runtime=runtime,
                profile=profile,
                quiet=quiet,
                gpu_only=gpu_only
            )

        if self.output_type == ModuleOutputType.OBJECT:

            with OpenFile(translate_files[self.stdout_key], "w", pretend=pretend) as stdout_file:
                with OpenFile(translate_files[self.stderr_key], "w", pretend=pretend) as stderr_file:
                    self.translate_mlir_with_mlir_translate(
                        stdout=stdout_file,
                        stderr=stderr_file,
                        pretend=pretend,
                        system_target=system_target,
                        quiet=quiet
                    )

            with OpenFile(opt_files[self.stdout_key], "w", pretend=pretend) as stdout_file:
                with OpenFile(opt_files[self.stderr_key], "w", pretend=pretend) as stderr_file:
                    self.optimize_llvm(
                        stdout=stdout_file,
                        stderr=stderr_file,
                        pretend=pretend,
                        system_target=system_target,
                        quiet=quiet,
                        _additional_llvm_opt_args=_get_options_opt_args(_options)
                    )

            with OpenFile(llc_files[self.stdout_key], "w", pretend=pretend) as stdout_file:
                with OpenFile(llc_files[self.stderr_key], "w", pretend=pretend) as stderr_file:
                    self.generate_object(
                        stdout=stdout_file,
                        stderr=stderr_file,
                        pretend=pretend,
                        system_target=system_target,
                        quiet=quiet,
                        _additional_llvm_llc_args=_get_options_llc_args(_options)
                    )

            with OpenFile(llc_asm_files[self.stdout_key], "w", pretend=pretend) as stdout_file:
                with OpenFile(llc_asm_files[self.stderr_key], "w", pretend=pretend) as stderr_file:
                    self.generate_asm(
                        stdout=stdout_file,
                        stderr=stderr_file,
                        pretend=pretend,
                        system_target=system_target,
                        quiet=quiet
                    )

        elif self.output_type in [ModuleOutputType.CPP, ModuleOutputType.CUDA]:

            with OpenFile(translate_files[self.stdout_key], "w", pretend=pretend) as stdout_file:
                with OpenFile(translate_files[self.stderr_key], "w", pretend=pretend) as stderr_file:
                    self.translate_mlir_with_acc_translate(
                        stdout=stdout_file,
                        stderr=stderr_file,
                        pretend=pretend,
                        system_target=system_target,
                        quiet=quiet
                    )


def accc(
    input_path,
    domain_list,
    library_name,
    output_dir,
    build_config=build_config_types[0],
    target=default_target,
    profile=False,
    generator_custom_args=ParameterCollection([]),
    main_cpp_path=None,
    main_custom_args=ParameterCollection([]),
    deploy_shared_libraries=[],
    run_main=False,
    dump_all_passes=False,
    dump_intrapass_ir=False,
    print_subprocess_output=False,
    pretend=False,
    system_target=SystemTarget.HOST.value,
    runtime=Runtime.DEFAULT.value
):
    if pretend:
        print()
        print("### Would run the following commands if 'pretend' was not set:")
        print()

    # Create Accera generator parameters
    library_name_param = LibraryNameParameter(library_name)
    target_device_param = BaseParameter("target", system_target)
    generator_parameters = ParameterCollection([domain_list, library_name_param, target_device_param])
    if generator_custom_args:
        generator_parameters.merge(generator_custom_args)

    project = AcceraProject(
        output_dir,
        library_name,
        input_path,
        main_cpp_path,
        print_subprocess_output=print_subprocess_output,
        pretend=pretend
    )

    generator_cmake_build_log_files = project.make_log_filepaths("generator")
    with OpenFile(generator_cmake_build_log_files[project.stdout_key], "w", pretend=pretend) as stdout_file:
        with OpenFile(generator_cmake_build_log_files[project.stderr_key], "w", pretend=pretend) as stderr_file:
            project.build_generator(build_config=build_config, stdout=stdout_file, stderr=stderr_file, pretend=pretend)

    project.generate_and_emit(
        generator_parameters,
        system_target=system_target,
        runtime=runtime,
        build_config=build_config,
        dump_all_passes=dump_all_passes,
        dump_intrapass_ir=dump_intrapass_ir,
        profile=profile,
        pretend=pretend
    )

    # If main c++ file is provided, create main program
    if main_cpp_path:
        main_cmake_build_log_files = project.make_log_filepaths("main")
        built_main = None
        with OpenFile(main_cmake_build_log_files[project.stdout_key], "w", pretend=pretend) as stdout_file:
            with OpenFile(main_cmake_build_log_files[project.stderr_key], "w", pretend=pretend) as stderr_file:
                built_main = project.build_main(
                    deploy_shared_libraries=deploy_shared_libraries,
                    build_config=build_config,
                    target=target,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    pretend=pretend,
                    system_target=system_target
                )
        if built_main:
            if target == GPU_TARGET:
                built_main.run_on_high_performance_gpu(pretend=pretend)
            if run_main:
                built_main.run(main_custom_args, pretend=pretend)


def get_available_features(**kwargs):
    cpu = kwargs.get('cpu')
    os = kwargs.get('os')

    avail_features = BASE_FEATURES
    avail_features.update(CPU_OPTS[cpu] if cpu else {})
    avail_features.update(OS_OPTS[os] if os else {})
    return avail_features
