	.text
	.def	 @feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.file	"LLVMDialectModule"
	.def	 hello_matmul_py_0f07b3ac_impl_16252232176815793891;
	.scl	2;
	.type	32;
	.endef
	.globl	hello_matmul_py_0f07b3ac_impl_16252232176815793891 # -- Begin function hello_matmul_py_0f07b3ac_impl_16252232176815793891
	.p2align	4, 0x90
hello_matmul_py_0f07b3ac_impl_16252232176815793891: # @hello_matmul_py_0f07b3ac_impl_16252232176815793891
.Lfunc_begin0:
	.file	1 "D:\\win\\repos\\accera-samples\\tutorials\\hello_matmul\\_tmp\\hello_matmul\\hello_matmul_llvm.mlir"
	.loc	1 6 0                           # hello_matmul\hello_matmul_llvm.mlir:6:0
# %bb.0:
	pushq	%rsi
	pushq	%rdi
	pushq	%rbx
	movq	152(%rsp), %rax
	movl	$3072, %r8d                     # imm = 0xC00
.Ltmp0:
	.loc	1 41 5 prologue_end             # hello_matmul\hello_matmul_llvm.mlir:41:5
	addq	96(%rsp), %r8
	addq	$12, %rdx
	xorl	%r9d, %r9d
	.p2align	4, 0x90
.LBB0_1:                                # %.preheader1
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_2 Depth 2
                                        #       Child Loop BB0_3 Depth 3
	.loc	1 0 5 is_stmt 0                 # hello_matmul\hello_matmul_llvm.mlir:0:5
	movq	%r9, %r10
	shlq	$8, %r10
	movq	%r8, %r11
	xorl	%ecx, %ecx
	.p2align	4, 0x90
.LBB0_2:                                # %.preheader
                                        #   Parent Loop BB0_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_3 Depth 3
	leaq	(%rcx,%r10), %rsi
	.loc	1 83 11 is_stmt 1               # hello_matmul\hello_matmul_llvm.mlir:83:11
	vmovss	(%rax,%rsi,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	movq	$-4, %rdi
	movq	%r11, %rbx
	.p2align	4, 0x90
.LBB0_3:                                #   Parent Loop BB0_1 Depth=1
                                        #     Parent Loop BB0_2 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	.loc	1 62 11                         # hello_matmul\hello_matmul_llvm.mlir:62:11
	vmovss	4(%rdx,%rdi,4), %xmm1           # xmm1 = mem[0],zero,zero,zero
	.loc	1 84 11                         # hello_matmul\hello_matmul_llvm.mlir:84:11
	vfmadd132ss	-3072(%rbx), %xmm0, %xmm1 # xmm1 = (xmm1 * mem) + xmm0
	.loc	1 114 5                         # hello_matmul\hello_matmul_llvm.mlir:114:5
	vmovss	%xmm1, (%rax,%rsi,4)
	.loc	1 125 12                        # hello_matmul\hello_matmul_llvm.mlir:125:12
	vmovss	8(%rdx,%rdi,4), %xmm0           # xmm0 = mem[0],zero,zero,zero
	.loc	1 147 12                        # hello_matmul\hello_matmul_llvm.mlir:147:12
	vfmadd132ss	-2048(%rbx), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
	.loc	1 177 5                         # hello_matmul\hello_matmul_llvm.mlir:177:5
	vmovss	%xmm0, (%rax,%rsi,4)
	.loc	1 188 12                        # hello_matmul\hello_matmul_llvm.mlir:188:12
	vmovss	12(%rdx,%rdi,4), %xmm1          # xmm1 = mem[0],zero,zero,zero
	.loc	1 210 12                        # hello_matmul\hello_matmul_llvm.mlir:210:12
	vfmadd132ss	-1024(%rbx), %xmm0, %xmm1 # xmm1 = (xmm1 * mem) + xmm0
	.loc	1 240 5                         # hello_matmul\hello_matmul_llvm.mlir:240:5
	vmovss	%xmm1, (%rax,%rsi,4)
	.loc	1 251 12                        # hello_matmul\hello_matmul_llvm.mlir:251:12
	vmovss	16(%rdx,%rdi,4), %xmm0          # xmm0 = mem[0],zero,zero,zero
	.loc	1 273 12                        # hello_matmul\hello_matmul_llvm.mlir:273:12
	vfmadd132ss	(%rbx), %xmm1, %xmm0    # xmm0 = (xmm0 * mem) + xmm1
	.loc	1 303 5                         # hello_matmul\hello_matmul_llvm.mlir:303:5
	vmovss	%xmm0, (%rax,%rsi,4)
	.loc	1 50 11                         # hello_matmul\hello_matmul_llvm.mlir:50:11
	addq	$4, %rdi
	addq	$4096, %rbx                     # imm = 0x1000
	cmpq	$252, %rdi
	.loc	1 51 5                          # hello_matmul\hello_matmul_llvm.mlir:51:5
	jb	.LBB0_3
# %bb.4:                                #   in Loop: Header=BB0_2 Depth=2
	.loc	1 307 12                        # hello_matmul\hello_matmul_llvm.mlir:307:12
	incq	%rcx
	.loc	1 46 5                          # hello_matmul\hello_matmul_llvm.mlir:46:5
	addq	$4, %r11
	.loc	1 45 11                         # hello_matmul\hello_matmul_llvm.mlir:45:11
	cmpq	$256, %rcx                      # imm = 0x100
	.loc	1 46 5                          # hello_matmul\hello_matmul_llvm.mlir:46:5
	jne	.LBB0_2
# %bb.5:                                #   in Loop: Header=BB0_1 Depth=1
	.loc	1 310 12                        # hello_matmul\hello_matmul_llvm.mlir:310:12
	incq	%r9
	.loc	1 41 5                          # hello_matmul\hello_matmul_llvm.mlir:41:5
	addq	$1024, %rdx                     # imm = 0x400
	.loc	1 40 11                         # hello_matmul\hello_matmul_llvm.mlir:40:11
	cmpq	$128, %r9
	.loc	1 41 5                          # hello_matmul\hello_matmul_llvm.mlir:41:5
	jne	.LBB0_1
# %bb.6:
	.loc	1 313 5                         # hello_matmul\hello_matmul_llvm.mlir:313:5
	popq	%rbx
	popq	%rdi
	popq	%rsi
	retq
.Ltmp1:
.Lfunc_end0:
                                        # -- End function
	.def	 hello_matmul_py_0f07b3ac;
	.scl	2;
	.type	32;
	.endef
	.globl	hello_matmul_py_0f07b3ac        # -- Begin function hello_matmul_py_0f07b3ac
	.p2align	4, 0x90
hello_matmul_py_0f07b3ac:               # @hello_matmul_py_0f07b3ac
.Lfunc_begin1:
	.loc	1 315 0                         # hello_matmul\hello_matmul_llvm.mlir:315:0
# %bb.0:
	pushq	%rsi
	pushq	%rdi
	pushq	%rbx
	.loc	1 41 5 prologue_end             # hello_matmul\hello_matmul_llvm.mlir:41:5
	addq	$3072, %rdx                     # imm = 0xC00
	addq	$12, %rcx
	xorl	%r9d, %r9d
	.p2align	4, 0x90
.LBB1_1:                                # %.preheader1.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_2 Depth 2
                                        #       Child Loop BB1_3 Depth 3
	.loc	1 0 5 is_stmt 0                 # hello_matmul\hello_matmul_llvm.mlir:0:5
	movq	%r9, %r10
	shlq	$8, %r10
	movq	%rdx, %r11
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB1_2:                                # %.preheader.i
                                        #   Parent Loop BB1_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB1_3 Depth 3
	leaq	(%rax,%r10), %rsi
	.loc	1 83 11 is_stmt 1               # hello_matmul\hello_matmul_llvm.mlir:83:11
	vmovss	(%r8,%rsi,4), %xmm0             # xmm0 = mem[0],zero,zero,zero
	movq	$-4, %rdi
	movq	%r11, %rbx
	.p2align	4, 0x90
.LBB1_3:                                #   Parent Loop BB1_1 Depth=1
                                        #     Parent Loop BB1_2 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	.loc	1 62 11                         # hello_matmul\hello_matmul_llvm.mlir:62:11
	vmovss	4(%rcx,%rdi,4), %xmm1           # xmm1 = mem[0],zero,zero,zero
	.loc	1 84 11                         # hello_matmul\hello_matmul_llvm.mlir:84:11
	vfmadd132ss	-3072(%rbx), %xmm0, %xmm1 # xmm1 = (xmm1 * mem) + xmm0
	.loc	1 114 5                         # hello_matmul\hello_matmul_llvm.mlir:114:5
	vmovss	%xmm1, (%r8,%rsi,4)
	.loc	1 125 12                        # hello_matmul\hello_matmul_llvm.mlir:125:12
	vmovss	8(%rcx,%rdi,4), %xmm0           # xmm0 = mem[0],zero,zero,zero
	.loc	1 147 12                        # hello_matmul\hello_matmul_llvm.mlir:147:12
	vfmadd132ss	-2048(%rbx), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
	.loc	1 177 5                         # hello_matmul\hello_matmul_llvm.mlir:177:5
	vmovss	%xmm0, (%r8,%rsi,4)
	.loc	1 188 12                        # hello_matmul\hello_matmul_llvm.mlir:188:12
	vmovss	12(%rcx,%rdi,4), %xmm1          # xmm1 = mem[0],zero,zero,zero
	.loc	1 210 12                        # hello_matmul\hello_matmul_llvm.mlir:210:12
	vfmadd132ss	-1024(%rbx), %xmm0, %xmm1 # xmm1 = (xmm1 * mem) + xmm0
	.loc	1 240 5                         # hello_matmul\hello_matmul_llvm.mlir:240:5
	vmovss	%xmm1, (%r8,%rsi,4)
	.loc	1 251 12                        # hello_matmul\hello_matmul_llvm.mlir:251:12
	vmovss	16(%rcx,%rdi,4), %xmm0          # xmm0 = mem[0],zero,zero,zero
	.loc	1 273 12                        # hello_matmul\hello_matmul_llvm.mlir:273:12
	vfmadd132ss	(%rbx), %xmm1, %xmm0    # xmm0 = (xmm0 * mem) + xmm1
	.loc	1 303 5                         # hello_matmul\hello_matmul_llvm.mlir:303:5
	vmovss	%xmm0, (%r8,%rsi,4)
	.loc	1 50 11                         # hello_matmul\hello_matmul_llvm.mlir:50:11
	addq	$4096, %rbx                     # imm = 0x1000
	addq	$4, %rdi
	cmpq	$252, %rdi
	.loc	1 51 5                          # hello_matmul\hello_matmul_llvm.mlir:51:5
	jb	.LBB1_3
# %bb.4:                                #   in Loop: Header=BB1_2 Depth=2
	.loc	1 307 12                        # hello_matmul\hello_matmul_llvm.mlir:307:12
	incq	%rax
	.loc	1 46 5                          # hello_matmul\hello_matmul_llvm.mlir:46:5
	addq	$4, %r11
	.loc	1 45 11                         # hello_matmul\hello_matmul_llvm.mlir:45:11
	cmpq	$256, %rax                      # imm = 0x100
	.loc	1 46 5                          # hello_matmul\hello_matmul_llvm.mlir:46:5
	jne	.LBB1_2
# %bb.5:                                #   in Loop: Header=BB1_1 Depth=1
	.loc	1 310 12                        # hello_matmul\hello_matmul_llvm.mlir:310:12
	incq	%r9
	.loc	1 41 5                          # hello_matmul\hello_matmul_llvm.mlir:41:5
	addq	$1024, %rcx                     # imm = 0x400
	.loc	1 40 11                         # hello_matmul\hello_matmul_llvm.mlir:40:11
	cmpq	$128, %r9
	.loc	1 41 5                          # hello_matmul\hello_matmul_llvm.mlir:41:5
	jne	.LBB1_1
.Ltmp2:
# %bb.6:                                # %hello_matmul_py_0f07b3ac_impl_16252232176815793891.exit
	.loc	1 377 5                         # hello_matmul\hello_matmul_llvm.mlir:377:5
	popq	%rbx
	popq	%rdi
	popq	%rsi
	retq
.Ltmp3:
.Lfunc_end1:
                                        # -- End function
	.section	.debug_abbrev,"dr"
.Lsection_abbrev:
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	14                              # DW_FORM_strp
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
	.ascii	"\264B"                         # DW_AT_GNU_pubnames
	.byte	25                              # DW_FORM_flag_present
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	110                             # DW_AT_linkage_name
	.byte	14                              # DW_FORM_strp
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	32                              # DW_AT_inline
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	14                              # DW_FORM_strp
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	5                               # DW_FORM_data2
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	0                               # DW_CHILDREN_no
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	88                              # DW_AT_call_file
	.byte	11                              # DW_FORM_data1
	.byte	89                              # DW_AT_call_line
	.byte	5                               # DW_FORM_data2
	.byte	87                              # DW_AT_call_column
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"dr"
.Lsection_info:
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.secrel32	.Lsection_abbrev        # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x6f DW_TAG_compile_unit
	.secrel32	.Linfo_string0          # DW_AT_producer
	.short	2                               # DW_AT_language
	.secrel32	.Linfo_string1          # DW_AT_name
	.secrel32	.Lline_table_start0     # DW_AT_stmt_list
	.secrel32	.Linfo_string2          # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin0       # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x2a:0x13 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	61                              # DW_AT_abstract_origin
	.byte	3                               # Abbrev [3] 0x3d:0xc DW_TAG_subprogram
	.secrel32	.Linfo_string3          # DW_AT_linkage_name
	.secrel32	.Linfo_string3          # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
                                        # DW_AT_external
	.byte	1                               # DW_AT_inline
	.byte	4                               # Abbrev [4] 0x49:0x30 DW_TAG_subprogram
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.secrel32	.Linfo_string4          # DW_AT_linkage_name
	.secrel32	.Linfo_string4          # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.short	315                             # DW_AT_decl_line
                                        # DW_AT_external
	.byte	5                               # Abbrev [5] 0x63:0x15 DW_TAG_inlined_subroutine
	.long	61                              # DW_AT_abstract_origin
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Ltmp2-.Lfunc_begin1            # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.short	376                             # DW_AT_call_line
	.byte	5                               # DW_AT_call_column
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"dr"
.Linfo_string:
.Linfo_string0:
	.asciz	"mlir"                          # string offset=0
.Linfo_string1:
	.asciz	"LLVMDialectModule"             # string offset=5
.Linfo_string2:
	.asciz	"/"                             # string offset=23
.Linfo_string3:
	.asciz	"hello_matmul_py_0f07b3ac_impl_16252232176815793891" # string offset=25
.Linfo_string4:
	.asciz	"hello_matmul_py_0f07b3ac"      # string offset=76
	.section	.debug_pubnames,"dr"
	.long	.LpubNames_end0-.LpubNames_begin0 # Length of Public Names Info
.LpubNames_begin0:
	.short	2                               # DWARF Version
	.secrel32	.Lcu_begin0             # Offset of Compilation Unit Info
	.long	122                             # Compilation Unit Length
	.long	73                              # DIE offset
	.asciz	"hello_matmul_py_0f07b3ac"      # External Name
	.long	61                              # DIE offset
	.asciz	"hello_matmul_py_0f07b3ac_impl_16252232176815793891" # External Name
	.long	0                               # End Mark
.LpubNames_end0:
	.section	.debug_pubtypes,"dr"
	.long	.LpubTypes_end0-.LpubTypes_begin0 # Length of Public Types Info
.LpubTypes_begin0:
	.short	2                               # DWARF Version
	.secrel32	.Lcu_begin0             # Offset of Compilation Unit Info
	.long	122                             # Compilation Unit Length
	.long	0                               # End Mark
.LpubTypes_end0:
	.globl	_fltused
	.section	.debug_line,"dr"
.Lsection_line:
.Lline_table_start0:
