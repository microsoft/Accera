	.text
	.def	 @feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.file	"LLVMDialectModule"
	.def	 optimized_matmul_py_4a6286d9_impl_17630232307017152746;
	.scl	2;
	.type	32;
	.endef
	.globl	optimized_matmul_py_4a6286d9_impl_17630232307017152746 # -- Begin function optimized_matmul_py_4a6286d9_impl_17630232307017152746
	.p2align	4, 0x90
optimized_matmul_py_4a6286d9_impl_17630232307017152746: # @optimized_matmul_py_4a6286d9_impl_17630232307017152746
.Lfunc_begin0:
	.file	1 "D:\\win\\repos\\accera-samples\\tutorials\\optimized_matmul\\_tmp\\optimized_matmul\\optimized_matmul_llvm.mlir"
	.loc	1 8 0                           # optimized_matmul\optimized_matmul_llvm.mlir:8:0
# %bb.0:
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rsi
	pushq	%rdi
	pushq	%rbp
	pushq	%rbx
	subq	$328, %rsp                      # imm = 0x148
	movl	$992, %ecx                      # imm = 0x3E0
.Ltmp0:
	.loc	1 163 5 prologue_end            # optimized_matmul\optimized_matmul_llvm.mlir:163:5
	addq	464(%rsp), %rcx
	addq	$12, %rdx
	movq	%rdx, 32(%rsp)                  # 8-byte Spill
	leaq	cache_17(%rip), %rdx
	leaq	cache_16(%rip), %r15
	xorl	%eax, %eax
	xorl	%ebp, %ebp
	.p2align	4, 0x90
.LBB0_1:                                # %.preheader40
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_2 Depth 2
                                        #     Child Loop BB0_4 Depth 2
                                        #       Child Loop BB0_5 Depth 3
                                        #         Child Loop BB0_6 Depth 4
	.loc	1 0 5 is_stmt 0                 # optimized_matmul\optimized_matmul_llvm.mlir:0:5
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	.loc	1 168 5 is_stmt 1               # optimized_matmul\optimized_matmul_llvm.mlir:168:5
	shlq	$8, %rax
	movq	%rax, 304(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$8, %rax
	movq	%rax, 296(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$16, %rax
	movq	%rax, 288(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$24, %rax
	movq	%rax, 280(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$32, %rax
	movq	%rax, 272(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$40, %rax
	movq	%rax, 264(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$48, %rax
	movq	%rax, 256(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$56, %rax
	movq	%rax, 248(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$64, %rax
	movq	%rax, 240(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$72, %rax
	movq	%rax, 232(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$80, %rax
	movq	%rax, 224(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$88, %rax
	movq	%rax, 216(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$96, %rax
	movq	%rax, 208(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$104, %rax
	movq	%rax, 200(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$112, %rax
	movq	%rax, 192(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$120, %rax
	movq	%rax, 184(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$128, %rax
	movq	%rax, 176(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$136, %rax
	movq	%rax, 168(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$144, %rax
	movq	%rax, 160(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$152, %rax
	movq	%rax, 152(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$160, %rax
	movq	%rax, 144(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$168, %rax
	movq	%rax, 136(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$176, %rax
	movq	%rax, 128(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$184, %rax
	movq	%rax, 120(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$192, %rax
	movq	%rax, 112(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$200, %rax
	movq	%rax, 104(%rsp)                 # 8-byte Spill
	movq	%rbp, %rax
	orq	$208, %rax
	movq	%rax, 96(%rsp)                  # 8-byte Spill
	movq	%rbp, %rax
	orq	$216, %rax
	movq	%rax, 88(%rsp)                  # 8-byte Spill
	movq	%rbp, %rax
	orq	$224, %rax
	movq	%rax, 80(%rsp)                  # 8-byte Spill
	movq	%rbp, %rax
	orq	$232, %rax
	movq	%rax, 72(%rsp)                  # 8-byte Spill
	movq	%rbp, %rax
	orq	$240, %rax
	movq	%rax, 64(%rsp)                  # 8-byte Spill
	movq	%rbp, %rax
	orq	$248, %rax
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	movq	$-8192, %rax                    # imm = 0xE000
	movq	%rcx, 48(%rsp)                  # 8-byte Spill
	.p2align	4, 0x90
.LBB0_2:                                # %.preheader38
                                        #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	.loc	1 188 12                        # optimized_matmul\optimized_matmul_llvm.mlir:188:12
	vmovups	-992(%rcx), %ymm0
	.loc	1 210 12                        # optimized_matmul\optimized_matmul_llvm.mlir:210:12
	vmovups	-960(%rcx), %ymm1
	.loc	1 232 12                        # optimized_matmul\optimized_matmul_llvm.mlir:232:12
	vmovups	-928(%rcx), %ymm2
	.loc	1 254 12                        # optimized_matmul\optimized_matmul_llvm.mlir:254:12
	vmovups	-896(%rcx), %ymm3
	.loc	1 276 12                        # optimized_matmul\optimized_matmul_llvm.mlir:276:12
	vmovups	-864(%rcx), %ymm4
	.loc	1 298 12                        # optimized_matmul\optimized_matmul_llvm.mlir:298:12
	vmovups	-832(%rcx), %ymm5
	.loc	1 320 12                        # optimized_matmul\optimized_matmul_llvm.mlir:320:12
	vmovups	-800(%rcx), %ymm16
	.loc	1 342 12                        # optimized_matmul\optimized_matmul_llvm.mlir:342:12
	vmovups	-768(%rcx), %ymm17
	.loc	1 364 12                        # optimized_matmul\optimized_matmul_llvm.mlir:364:12
	vmovups	-736(%rcx), %ymm18
	.loc	1 386 12                        # optimized_matmul\optimized_matmul_llvm.mlir:386:12
	vmovups	-704(%rcx), %ymm19
	.loc	1 408 12                        # optimized_matmul\optimized_matmul_llvm.mlir:408:12
	vmovups	-672(%rcx), %ymm20
	.loc	1 430 12                        # optimized_matmul\optimized_matmul_llvm.mlir:430:12
	vmovups	-640(%rcx), %ymm21
	.loc	1 452 12                        # optimized_matmul\optimized_matmul_llvm.mlir:452:12
	vmovups	-608(%rcx), %ymm22
	.loc	1 474 12                        # optimized_matmul\optimized_matmul_llvm.mlir:474:12
	vmovups	-576(%rcx), %ymm23
	.loc	1 496 12                        # optimized_matmul\optimized_matmul_llvm.mlir:496:12
	vmovups	-544(%rcx), %ymm24
	.loc	1 518 12                        # optimized_matmul\optimized_matmul_llvm.mlir:518:12
	vmovups	-512(%rcx), %ymm25
	.loc	1 579 5                         # optimized_matmul\optimized_matmul_llvm.mlir:579:5
	vmovaps	%ymm0, 8192(%rax,%rdx)
	.loc	1 628 5                         # optimized_matmul\optimized_matmul_llvm.mlir:628:5
	vmovaps	%ymm1, 8224(%rax,%rdx)
	.loc	1 661 5                         # optimized_matmul\optimized_matmul_llvm.mlir:661:5
	vmovaps	%ymm2, 16384(%rax,%rdx)
	.loc	1 707 5                         # optimized_matmul\optimized_matmul_llvm.mlir:707:5
	vmovaps	%ymm3, 16416(%rax,%rdx)
	.loc	1 740 5                         # optimized_matmul\optimized_matmul_llvm.mlir:740:5
	vmovaps	%ymm4, 24576(%rax,%rdx)
	.loc	1 786 5                         # optimized_matmul\optimized_matmul_llvm.mlir:786:5
	vmovaps	%ymm5, 24608(%rax,%rdx)
	.loc	1 819 5                         # optimized_matmul\optimized_matmul_llvm.mlir:819:5
	vmovaps	%ymm16, 32768(%rax,%rdx)
	.loc	1 865 5                         # optimized_matmul\optimized_matmul_llvm.mlir:865:5
	vmovaps	%ymm17, 32800(%rax,%rdx)
	.loc	1 898 5                         # optimized_matmul\optimized_matmul_llvm.mlir:898:5
	vmovaps	%ymm18, 40960(%rax,%rdx)
	.loc	1 944 5                         # optimized_matmul\optimized_matmul_llvm.mlir:944:5
	vmovaps	%ymm19, 40992(%rax,%rdx)
	.loc	1 977 5                         # optimized_matmul\optimized_matmul_llvm.mlir:977:5
	vmovaps	%ymm20, 49152(%rax,%rdx)
	.loc	1 1023 5                        # optimized_matmul\optimized_matmul_llvm.mlir:1023:5
	vmovaps	%ymm21, 49184(%rax,%rdx)
	.loc	1 1056 5                        # optimized_matmul\optimized_matmul_llvm.mlir:1056:5
	vmovaps	%ymm22, 57344(%rax,%rdx)
	.loc	1 1102 5                        # optimized_matmul\optimized_matmul_llvm.mlir:1102:5
	vmovaps	%ymm23, 57376(%rax,%rdx)
	.loc	1 1135 5                        # optimized_matmul\optimized_matmul_llvm.mlir:1135:5
	vmovaps	%ymm24, 65536(%rax,%rdx)
	.loc	1 1181 5                        # optimized_matmul\optimized_matmul_llvm.mlir:1181:5
	vmovaps	%ymm25, 65568(%rax,%rdx)
	.loc	1 188 12                        # optimized_matmul\optimized_matmul_llvm.mlir:188:12
	vmovups	-480(%rcx), %ymm0
	.loc	1 210 12                        # optimized_matmul\optimized_matmul_llvm.mlir:210:12
	vmovups	-448(%rcx), %ymm1
	.loc	1 232 12                        # optimized_matmul\optimized_matmul_llvm.mlir:232:12
	vmovups	-416(%rcx), %ymm2
	.loc	1 254 12                        # optimized_matmul\optimized_matmul_llvm.mlir:254:12
	vmovups	-384(%rcx), %ymm3
	.loc	1 276 12                        # optimized_matmul\optimized_matmul_llvm.mlir:276:12
	vmovups	-352(%rcx), %ymm4
	.loc	1 298 12                        # optimized_matmul\optimized_matmul_llvm.mlir:298:12
	vmovups	-320(%rcx), %ymm5
	.loc	1 320 12                        # optimized_matmul\optimized_matmul_llvm.mlir:320:12
	vmovups	-288(%rcx), %ymm16
	.loc	1 342 12                        # optimized_matmul\optimized_matmul_llvm.mlir:342:12
	vmovups	-256(%rcx), %ymm17
	.loc	1 364 12                        # optimized_matmul\optimized_matmul_llvm.mlir:364:12
	vmovups	-224(%rcx), %ymm18
	.loc	1 386 12                        # optimized_matmul\optimized_matmul_llvm.mlir:386:12
	vmovups	-192(%rcx), %ymm19
	.loc	1 408 12                        # optimized_matmul\optimized_matmul_llvm.mlir:408:12
	vmovups	-160(%rcx), %ymm20
	.loc	1 430 12                        # optimized_matmul\optimized_matmul_llvm.mlir:430:12
	vmovups	-128(%rcx), %ymm21
	.loc	1 452 12                        # optimized_matmul\optimized_matmul_llvm.mlir:452:12
	vmovups	-96(%rcx), %ymm22
	.loc	1 474 12                        # optimized_matmul\optimized_matmul_llvm.mlir:474:12
	vmovups	-64(%rcx), %ymm23
	.loc	1 496 12                        # optimized_matmul\optimized_matmul_llvm.mlir:496:12
	vmovups	-32(%rcx), %ymm24
	.loc	1 518 12                        # optimized_matmul\optimized_matmul_llvm.mlir:518:12
	vmovups	(%rcx), %ymm25
	.loc	1 579 5                         # optimized_matmul\optimized_matmul_llvm.mlir:579:5
	vmovaps	%ymm0, 73728(%rax,%rdx)
	.loc	1 628 5                         # optimized_matmul\optimized_matmul_llvm.mlir:628:5
	vmovaps	%ymm1, 73760(%rax,%rdx)
	.loc	1 661 5                         # optimized_matmul\optimized_matmul_llvm.mlir:661:5
	vmovaps	%ymm2, 81920(%rax,%rdx)
	.loc	1 707 5                         # optimized_matmul\optimized_matmul_llvm.mlir:707:5
	vmovaps	%ymm3, 81952(%rax,%rdx)
	.loc	1 740 5                         # optimized_matmul\optimized_matmul_llvm.mlir:740:5
	vmovaps	%ymm4, 90112(%rax,%rdx)
	.loc	1 786 5                         # optimized_matmul\optimized_matmul_llvm.mlir:786:5
	vmovaps	%ymm5, 90144(%rax,%rdx)
	.loc	1 819 5                         # optimized_matmul\optimized_matmul_llvm.mlir:819:5
	vmovaps	%ymm16, 98304(%rax,%rdx)
	.loc	1 865 5                         # optimized_matmul\optimized_matmul_llvm.mlir:865:5
	vmovaps	%ymm17, 98336(%rax,%rdx)
	.loc	1 898 5                         # optimized_matmul\optimized_matmul_llvm.mlir:898:5
	vmovaps	%ymm18, 106496(%rax,%rdx)
	.loc	1 944 5                         # optimized_matmul\optimized_matmul_llvm.mlir:944:5
	vmovaps	%ymm19, 106528(%rax,%rdx)
	.loc	1 977 5                         # optimized_matmul\optimized_matmul_llvm.mlir:977:5
	vmovaps	%ymm20, 114688(%rax,%rdx)
	.loc	1 1023 5                        # optimized_matmul\optimized_matmul_llvm.mlir:1023:5
	vmovaps	%ymm21, 114720(%rax,%rdx)
	.loc	1 1056 5                        # optimized_matmul\optimized_matmul_llvm.mlir:1056:5
	vmovaps	%ymm22, 122880(%rax,%rdx)
	.loc	1 1102 5                        # optimized_matmul\optimized_matmul_llvm.mlir:1102:5
	vmovaps	%ymm23, 122912(%rax,%rdx)
	.loc	1 1135 5                        # optimized_matmul\optimized_matmul_llvm.mlir:1135:5
	vmovaps	%ymm24, 131072(%rax,%rdx)
	.loc	1 1181 5                        # optimized_matmul\optimized_matmul_llvm.mlir:1181:5
	vmovaps	%ymm25, 131104(%rax,%rdx)
	.loc	1 167 12                        # optimized_matmul\optimized_matmul_llvm.mlir:167:12
	addq	$2048, %rcx                     # imm = 0x800
	addq	$64, %rax
	.loc	1 168 5                         # optimized_matmul\optimized_matmul_llvm.mlir:168:5
	jne	.LBB0_2
# %bb.3:                                # %.preheader37.preheader
                                        #   in Loop: Header=BB0_1 Depth=1
	.loc	1 0 5 is_stmt 0                 # optimized_matmul\optimized_matmul_llvm.mlir:0:5
	movq	32(%rsp), %r12                  # 8-byte Reload
	xorl	%r13d, %r13d
	movq	%rbp, 312(%rsp)                 # 8-byte Spill
	.p2align	4, 0x90
.LBB0_4:                                # %.preheader37
                                        #   Parent Loop BB0_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_5 Depth 3
                                        #         Child Loop BB0_6 Depth 4
	movq	%r13, 320(%rsp)                 # 8-byte Spill
	.loc	1 2469 5 is_stmt 1              # optimized_matmul\optimized_matmul_llvm.mlir:2469:5
	shlq	$9, %r13
	movq	304(%rsp), %rax                 # 8-byte Reload
	leaq	(%rax,%r13), %r14
	movl	$6144, %r8d                     # imm = 0x1800
	movq	%r15, %rcx
	xorl	%edx, %edx
	vzeroupper
	callq	memset
	movq	$-2, %r9
	leaq	cache_17+160(%rip), %rbp
	xorl	%ecx, %ecx
	.p2align	4, 0x90
.LBB0_5:                                # %.preheader33
                                        #   Parent Loop BB0_1 Depth=1
                                        #     Parent Loop BB0_4 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_6 Depth 4
	.loc	1 0 5 is_stmt 0                 # optimized_matmul\optimized_matmul_llvm.mlir:0:5
	movl	%ecx, %eax
	shrl	$4, %eax
	andl	$15, %eax
	shlq	$2, %rax
	leaq	(%rax,%rax,2), %rbx
	movq	%rbx, %rax
	shlq	$5, %rax
	leaq	(%r15,%rax), %r8
	xorl	%esi, %esi
	movq	$-1, %rdi
	negq	%rdi
	setl	%sil
	movl	$1, %edx
	cmovlq	%r9, %rdx
	movq	%rdx, %rdi
	shrq	$63, %rdi
	addq	%rdx, %rdi
	sarq	%rdi
	negq	%rsi
	xorq	%rdi, %rsi
	leaq	(%rsi,%rsi), %rdx
	movl	$1, %edi
	subq	%rdx, %rdi
	addq	%rbx, %rdi
	shlq	$5, %rdi
	vmovaps	(%rax,%r15), %ymm1
	leaq	(%r15,%rdi), %rbx
	vmovaps	(%rdi,%r15), %ymm0
	.loc	1 2487 5 is_stmt 1              # optimized_matmul\optimized_matmul_llvm.mlir:2487:5
	shlq	$6, %rsi
	negq	%rsi
	movq	$-4, %rdi
	movq	%rbp, %rax
	.p2align	4, 0x90
.LBB0_6:                                # %.preheader
                                        #   Parent Loop BB0_1 Depth=1
                                        #     Parent Loop BB0_4 Depth=2
                                        #       Parent Loop BB0_5 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	.loc	1 4305 13                       # optimized_matmul\optimized_matmul_llvm.mlir:4305:13
	vbroadcastss	4(%r12,%rdi,4), %ymm2
	.loc	1 4425 13                       # optimized_matmul\optimized_matmul_llvm.mlir:4425:13
	vfmadd231ps	-160(%rax), %ymm2, %ymm1 # ymm1 = (ymm2 * mem) + ymm1
	.loc	1 5203 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5203:13
	vfmadd231ps	-128(%rax,%rsi), %ymm2, %ymm0 # ymm0 = (ymm2 * mem) + ymm0
	.loc	1 4305 13                       # optimized_matmul\optimized_matmul_llvm.mlir:4305:13
	vbroadcastss	8(%r12,%rdi,4), %ymm2
	.loc	1 4425 13                       # optimized_matmul\optimized_matmul_llvm.mlir:4425:13
	vfmadd231ps	-96(%rax), %ymm2, %ymm1 # ymm1 = (ymm2 * mem) + ymm1
	.loc	1 5203 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5203:13
	vfmadd231ps	-64(%rax,%rsi), %ymm2, %ymm0 # ymm0 = (ymm2 * mem) + ymm0
	.loc	1 4305 13                       # optimized_matmul\optimized_matmul_llvm.mlir:4305:13
	vbroadcastss	12(%r12,%rdi,4), %ymm2
	.loc	1 4425 13                       # optimized_matmul\optimized_matmul_llvm.mlir:4425:13
	vfmadd231ps	-32(%rax), %ymm2, %ymm1 # ymm1 = (ymm2 * mem) + ymm1
	.loc	1 5203 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5203:13
	vfmadd231ps	(%rax,%rsi), %ymm2, %ymm0 # ymm0 = (ymm2 * mem) + ymm0
	.loc	1 4305 13                       # optimized_matmul\optimized_matmul_llvm.mlir:4305:13
	vbroadcastss	16(%r12,%rdi,4), %ymm2
	.loc	1 4425 13                       # optimized_matmul\optimized_matmul_llvm.mlir:4425:13
	vfmadd231ps	32(%rax), %ymm2, %ymm1  # ymm1 = (ymm2 * mem) + ymm1
	.loc	1 5203 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5203:13
	vfmadd231ps	64(%rax,%rsi), %ymm2, %ymm0 # ymm0 = (ymm2 * mem) + ymm0
	.loc	1 2486 13                       # optimized_matmul\optimized_matmul_llvm.mlir:2486:13
	addq	$4, %rdi
	addq	$256, %rax                      # imm = 0x100
	cmpq	$124, %rdi
	.loc	1 2487 5                        # optimized_matmul\optimized_matmul_llvm.mlir:2487:5
	jb	.LBB0_6
# %bb.7:                                #   in Loop: Header=BB0_5 Depth=3
	.loc	1 0 5 is_stmt 0                 # optimized_matmul\optimized_matmul_llvm.mlir:0:5
	vmovaps	%ymm1, (%r8)
	vmovaps	%ymm0, (%rbx)
	.loc	1 5649 13 is_stmt 1             # optimized_matmul\optimized_matmul_llvm.mlir:5649:13
	leaq	16(%rcx), %rax
	.loc	1 2482 5                        # optimized_matmul\optimized_matmul_llvm.mlir:2482:5
	addq	$8192, %rbp                     # imm = 0x2000
	.loc	1 2481 13                       # optimized_matmul\optimized_matmul_llvm.mlir:2481:13
	cmpq	$240, %rcx
	movq	%rax, %rcx
	.loc	1 2482 5                        # optimized_matmul\optimized_matmul_llvm.mlir:2482:5
	jb	.LBB0_5
# %bb.8:                                # %.preheader35
                                        #   in Loop: Header=BB0_4 Depth=2
	.loc	1 0 5 is_stmt 0                 # optimized_matmul\optimized_matmul_llvm.mlir:0:5
	movq	312(%rsp), %rbp                 # 8-byte Reload
	.loc	1 5667 13 is_stmt 1             # optimized_matmul\optimized_matmul_llvm.mlir:5667:13
	movq	%rbp, %rax
	addq	%r13, %rax
	movq	520(%rsp), %rcx
	.loc	1 5670 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5670:13
	vmovups	(%rcx,%rax,4), %ymm0
	.loc	1 5708 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5708:13
	vaddps	cache_16(%rip), %ymm0, %ymm0
	movq	296(%rsp), %rax                 # 8-byte Reload
	.loc	1 5727 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5727:13
	leaq	(%rax,%r13), %rax
	.loc	1 5730 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5730:13
	vmovups	(%rcx,%rax,4), %ymm1
	.loc	1 5770 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5770:13
	vaddps	cache_16+32(%rip), %ymm1, %ymm1
	movq	288(%rsp), %rax                 # 8-byte Reload
	.loc	1 5789 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5789:13
	leaq	(%rax,%r13), %rax
	.loc	1 5792 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5792:13
	vmovups	(%rcx,%rax,4), %ymm2
	.loc	1 5816 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5816:13
	vaddps	cache_16+384(%rip), %ymm2, %ymm2
	movq	280(%rsp), %rax                 # 8-byte Reload
	.loc	1 5835 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5835:13
	leaq	(%rax,%r13), %rax
	.loc	1 5838 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5838:13
	vmovups	(%rcx,%rax,4), %ymm3
	.loc	1 5875 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5875:13
	vaddps	cache_16+416(%rip), %ymm3, %ymm3
	movq	272(%rsp), %rax                 # 8-byte Reload
	.loc	1 5894 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5894:13
	leaq	(%rax,%r13), %rax
	.loc	1 5897 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5897:13
	vmovups	(%rcx,%rax,4), %ymm4
	.loc	1 5921 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5921:13
	vaddps	cache_16+768(%rip), %ymm4, %ymm4
	movq	264(%rsp), %rax                 # 8-byte Reload
	.loc	1 5940 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5940:13
	leaq	(%rax,%r13), %rax
	.loc	1 5943 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5943:13
	vmovups	(%rcx,%rax,4), %ymm5
	.loc	1 5980 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5980:13
	vaddps	cache_16+800(%rip), %ymm5, %ymm5
	movq	256(%rsp), %rax                 # 8-byte Reload
	.loc	1 5999 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5999:13
	leaq	(%rax,%r13), %rax
	.loc	1 6002 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6002:13
	vmovups	(%rcx,%rax,4), %ymm16
	.loc	1 6026 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6026:13
	vaddps	cache_16+1152(%rip), %ymm16, %ymm16
	movq	248(%rsp), %rax                 # 8-byte Reload
	.loc	1 6045 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6045:13
	leaq	(%rax,%r13), %rax
	.loc	1 6048 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6048:13
	vmovups	(%rcx,%rax,4), %ymm17
	.loc	1 6085 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6085:13
	vaddps	cache_16+1184(%rip), %ymm17, %ymm17
	movq	240(%rsp), %rax                 # 8-byte Reload
	.loc	1 6104 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6104:13
	leaq	(%rax,%r13), %rax
	.loc	1 6107 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6107:13
	vmovups	(%rcx,%rax,4), %ymm18
	.loc	1 6131 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6131:13
	vaddps	cache_16+1536(%rip), %ymm18, %ymm18
	movq	232(%rsp), %rax                 # 8-byte Reload
	.loc	1 6150 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6150:13
	leaq	(%rax,%r13), %rax
	.loc	1 6153 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6153:13
	vmovups	(%rcx,%rax,4), %ymm19
	.loc	1 6190 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6190:13
	vaddps	cache_16+1568(%rip), %ymm19, %ymm19
	movq	224(%rsp), %rax                 # 8-byte Reload
	.loc	1 6209 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6209:13
	leaq	(%rax,%r13), %rax
	.loc	1 6212 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6212:13
	vmovups	(%rcx,%rax,4), %ymm20
	.loc	1 6236 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6236:13
	vaddps	cache_16+1920(%rip), %ymm20, %ymm20
	movq	216(%rsp), %rax                 # 8-byte Reload
	.loc	1 6255 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6255:13
	leaq	(%rax,%r13), %rax
	.loc	1 6258 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6258:13
	vmovups	(%rcx,%rax,4), %ymm21
	.loc	1 6295 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6295:13
	vaddps	cache_16+1952(%rip), %ymm21, %ymm21
	movq	208(%rsp), %rax                 # 8-byte Reload
	.loc	1 6314 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6314:13
	leaq	(%rax,%r13), %rax
	.loc	1 6317 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6317:13
	vmovups	(%rcx,%rax,4), %ymm22
	.loc	1 6341 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6341:13
	vaddps	cache_16+2304(%rip), %ymm22, %ymm22
	movq	200(%rsp), %rax                 # 8-byte Reload
	.loc	1 6360 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6360:13
	leaq	(%rax,%r13), %rax
	.loc	1 6363 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6363:13
	vmovups	(%rcx,%rax,4), %ymm23
	.loc	1 6400 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6400:13
	vaddps	cache_16+2336(%rip), %ymm23, %ymm23
	movq	192(%rsp), %rax                 # 8-byte Reload
	.loc	1 6419 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6419:13
	leaq	(%rax,%r13), %rax
	.loc	1 6422 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6422:13
	vmovups	(%rcx,%rax,4), %ymm24
	.loc	1 6446 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6446:13
	vaddps	cache_16+2688(%rip), %ymm24, %ymm24
	movq	184(%rsp), %rax                 # 8-byte Reload
	.loc	1 6465 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6465:13
	leaq	(%rax,%r13), %rax
	.loc	1 6468 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6468:13
	vmovups	(%rcx,%rax,4), %ymm25
	.loc	1 6505 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6505:13
	vaddps	cache_16+2720(%rip), %ymm25, %ymm25
	.loc	1 6543 5                        # optimized_matmul\optimized_matmul_llvm.mlir:6543:5
	vmovups	%ymm0, (%rcx,%r14,4)
	vmovups	%ymm1, 32(%rcx,%r14,4)
	vmovups	%ymm2, 64(%rcx,%r14,4)
	vmovups	%ymm3, 96(%rcx,%r14,4)
	vmovups	%ymm4, 128(%rcx,%r14,4)
	vmovups	%ymm5, 160(%rcx,%r14,4)
	vmovups	%ymm16, 192(%rcx,%r14,4)
	vmovups	%ymm17, 224(%rcx,%r14,4)
	vmovups	%ymm18, 256(%rcx,%r14,4)
	vmovups	%ymm19, 288(%rcx,%r14,4)
	vmovups	%ymm20, 320(%rcx,%r14,4)
	vmovups	%ymm21, 352(%rcx,%r14,4)
	vmovups	%ymm22, 384(%rcx,%r14,4)
	vmovups	%ymm23, 416(%rcx,%r14,4)
	vmovups	%ymm24, 448(%rcx,%r14,4)
	vmovups	%ymm25, 480(%rcx,%r14,4)
	movq	176(%rsp), %rax                 # 8-byte Reload
	.loc	1 5667 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5667:13
	leaq	(%rax,%r13), %rax
	.loc	1 5670 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5670:13
	vmovups	(%rcx,%rax,4), %ymm0
	movq	168(%rsp), %rax                 # 8-byte Reload
	.loc	1 5727 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5727:13
	leaq	(%rax,%r13), %rax
	.loc	1 5730 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5730:13
	vmovups	(%rcx,%rax,4), %ymm1
	movq	160(%rsp), %rax                 # 8-byte Reload
	.loc	1 5789 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5789:13
	leaq	(%rax,%r13), %rax
	.loc	1 5792 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5792:13
	vmovups	(%rcx,%rax,4), %ymm2
	movq	152(%rsp), %rax                 # 8-byte Reload
	.loc	1 5835 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5835:13
	leaq	(%rax,%r13), %rax
	.loc	1 5838 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5838:13
	vmovups	(%rcx,%rax,4), %ymm3
	movq	144(%rsp), %rax                 # 8-byte Reload
	.loc	1 5894 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5894:13
	leaq	(%rax,%r13), %rax
	.loc	1 5897 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5897:13
	vmovups	(%rcx,%rax,4), %ymm4
	movq	136(%rsp), %rax                 # 8-byte Reload
	.loc	1 5940 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5940:13
	leaq	(%rax,%r13), %rax
	.loc	1 5943 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5943:13
	vmovups	(%rcx,%rax,4), %ymm5
	movq	128(%rsp), %rax                 # 8-byte Reload
	.loc	1 5999 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5999:13
	leaq	(%rax,%r13), %rax
	.loc	1 6002 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6002:13
	vmovups	(%rcx,%rax,4), %ymm16
	movq	120(%rsp), %rax                 # 8-byte Reload
	.loc	1 6045 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6045:13
	leaq	(%rax,%r13), %rax
	.loc	1 6048 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6048:13
	vmovups	(%rcx,%rax,4), %ymm17
	movq	112(%rsp), %rax                 # 8-byte Reload
	.loc	1 6104 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6104:13
	leaq	(%rax,%r13), %rax
	.loc	1 6107 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6107:13
	vmovups	(%rcx,%rax,4), %ymm18
	movq	104(%rsp), %rax                 # 8-byte Reload
	.loc	1 6150 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6150:13
	leaq	(%rax,%r13), %rax
	.loc	1 6153 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6153:13
	vmovups	(%rcx,%rax,4), %ymm19
	movq	96(%rsp), %rax                  # 8-byte Reload
	.loc	1 6209 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6209:13
	leaq	(%rax,%r13), %rax
	.loc	1 6212 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6212:13
	vmovups	(%rcx,%rax,4), %ymm20
	movq	88(%rsp), %rax                  # 8-byte Reload
	.loc	1 6255 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6255:13
	leaq	(%rax,%r13), %rax
	.loc	1 6258 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6258:13
	vmovups	(%rcx,%rax,4), %ymm21
	movq	80(%rsp), %rax                  # 8-byte Reload
	.loc	1 6314 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6314:13
	addq	%r13, %rax
	.loc	1 6317 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6317:13
	vmovups	(%rcx,%rax,4), %ymm22
	movq	72(%rsp), %rax                  # 8-byte Reload
	.loc	1 6360 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6360:13
	addq	%r13, %rax
	.loc	1 6363 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6363:13
	vmovups	(%rcx,%rax,4), %ymm23
	movq	64(%rsp), %rax                  # 8-byte Reload
	.loc	1 6419 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6419:13
	addq	%r13, %rax
	.loc	1 6422 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6422:13
	vmovups	(%rcx,%rax,4), %ymm24
	.loc	1 6465 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6465:13
	addq	56(%rsp), %r13                  # 8-byte Folded Reload
	.loc	1 6468 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6468:13
	vmovups	(%rcx,%r13,4), %ymm25
	.loc	1 5659 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5659:13
	leaq	(,%r14,4), %rax
	orq	$512, %rax                      # imm = 0x200
	.loc	1 5708 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5708:13
	vaddps	cache_16+3072(%rip), %ymm0, %ymm0
	.loc	1 5770 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5770:13
	vaddps	cache_16+3104(%rip), %ymm1, %ymm1
	.loc	1 5816 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5816:13
	vaddps	cache_16+3456(%rip), %ymm2, %ymm2
	.loc	1 5875 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5875:13
	vaddps	cache_16+3488(%rip), %ymm3, %ymm3
	.loc	1 5921 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5921:13
	vaddps	cache_16+3840(%rip), %ymm4, %ymm4
	.loc	1 5980 13                       # optimized_matmul\optimized_matmul_llvm.mlir:5980:13
	vaddps	cache_16+3872(%rip), %ymm5, %ymm5
	.loc	1 6026 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6026:13
	vaddps	cache_16+4224(%rip), %ymm16, %ymm16
	.loc	1 6085 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6085:13
	vaddps	cache_16+4256(%rip), %ymm17, %ymm17
	.loc	1 6131 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6131:13
	vaddps	cache_16+4608(%rip), %ymm18, %ymm18
	.loc	1 6190 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6190:13
	vaddps	cache_16+4640(%rip), %ymm19, %ymm19
	.loc	1 6236 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6236:13
	vaddps	cache_16+4992(%rip), %ymm20, %ymm20
	.loc	1 6295 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6295:13
	vaddps	cache_16+5024(%rip), %ymm21, %ymm21
	.loc	1 6341 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6341:13
	vaddps	cache_16+5376(%rip), %ymm22, %ymm22
	.loc	1 6400 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6400:13
	vaddps	cache_16+5408(%rip), %ymm23, %ymm23
	.loc	1 6446 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6446:13
	vaddps	cache_16+5760(%rip), %ymm24, %ymm24
	.loc	1 6505 13                       # optimized_matmul\optimized_matmul_llvm.mlir:6505:13
	vaddps	cache_16+5792(%rip), %ymm25, %ymm25
	.loc	1 6543 5                        # optimized_matmul\optimized_matmul_llvm.mlir:6543:5
	vmovups	%ymm0, (%rcx,%rax)
	vmovups	%ymm1, 32(%rcx,%rax)
	vmovups	%ymm2, 64(%rcx,%rax)
	vmovups	%ymm3, 96(%rcx,%rax)
	vmovups	%ymm4, 128(%rcx,%rax)
	vmovups	%ymm5, 160(%rcx,%rax)
	vmovups	%ymm16, 192(%rcx,%rax)
	vmovups	%ymm17, 224(%rcx,%rax)
	vmovups	%ymm18, 256(%rcx,%rax)
	vmovups	%ymm19, 288(%rcx,%rax)
	vmovups	%ymm20, 320(%rcx,%rax)
	vmovups	%ymm21, 352(%rcx,%rax)
	vmovups	%ymm22, 384(%rcx,%rax)
	vmovups	%ymm23, 416(%rcx,%rax)
	vmovups	%ymm24, 448(%rcx,%rax)
	vmovups	%ymm25, 480(%rcx,%rax)
	movq	320(%rsp), %r13                 # 8-byte Reload
	.loc	1 7694 13                       # optimized_matmul\optimized_matmul_llvm.mlir:7694:13
	incq	%r13
	.loc	1 2440 5                        # optimized_matmul\optimized_matmul_llvm.mlir:2440:5
	addq	$512, %r12                      # imm = 0x200
	.loc	1 2439 13                       # optimized_matmul\optimized_matmul_llvm.mlir:2439:13
	cmpq	$784, %r13                      # imm = 0x310
	.loc	1 2440 5                        # optimized_matmul\optimized_matmul_llvm.mlir:2440:5
	jne	.LBB0_4
# %bb.9:                                #   in Loop: Header=BB0_1 Depth=1
	.loc	1 7697 13                       # optimized_matmul\optimized_matmul_llvm.mlir:7697:13
	addq	$256, %rbp                      # imm = 0x100
	movq	40(%rsp), %rax                  # 8-byte Reload
	.loc	1 163 5                         # optimized_matmul\optimized_matmul_llvm.mlir:163:5
	incq	%rax
	movq	48(%rsp), %rcx                  # 8-byte Reload
	addq	$1024, %rcx                     # imm = 0x400
	.loc	1 162 12                        # optimized_matmul\optimized_matmul_llvm.mlir:162:12
	cmpq	$2, %rax
	leaq	cache_17(%rip), %rdx
	.loc	1 163 5                         # optimized_matmul\optimized_matmul_llvm.mlir:163:5
	jne	.LBB0_1
# %bb.10:
	.loc	1 7700 5                        # optimized_matmul\optimized_matmul_llvm.mlir:7700:5
	addq	$328, %rsp                      # imm = 0x148
	popq	%rbx
	popq	%rbp
	popq	%rdi
	popq	%rsi
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	vzeroupper
	retq
.Ltmp1:
.Lfunc_end0:
                                        # -- End function
	.def	 optimized_matmul_py_4a6286d9;
	.scl	2;
	.type	32;
	.endef
	.globl	__ymm@0000000000000200000000000000020000000000000003100000000000000000 # -- Begin function optimized_matmul_py_4a6286d9
	.section	.rdata,"dr",discard,__ymm@0000000000000200000000000000020000000000000003100000000000000000
	.p2align	5
__ymm@0000000000000200000000000000020000000000000003100000000000000000:
	.quad	0                               # 0x0
	.quad	784                             # 0x310
	.quad	512                             # 0x200
	.quad	512                             # 0x200
	.globl	__ymm@0000000000000200000000000000020000000000000000800000000000000000
	.section	.rdata,"dr",discard,__ymm@0000000000000200000000000000020000000000000000800000000000000000
	.p2align	5
__ymm@0000000000000200000000000000020000000000000000800000000000000000:
	.quad	0                               # 0x0
	.quad	128                             # 0x80
	.quad	512                             # 0x200
	.quad	512                             # 0x200
	.text
	.globl	optimized_matmul_py_4a6286d9
	.p2align	4, 0x90
optimized_matmul_py_4a6286d9:           # @optimized_matmul_py_4a6286d9
.Lfunc_begin1:
	.loc	1 7702 0                        # optimized_matmul\optimized_matmul_llvm.mlir:7702:0
# %bb.0:
	subq	$168, %rsp
.Ltmp2:
	.loc	1 7763 5 prologue_end           # optimized_matmul\optimized_matmul_llvm.mlir:7763:5
	vmovaps	__ymm@0000000000000200000000000000020000000000000003100000000000000000(%rip), %ymm0 # ymm0 = [0,784,512,512]
	vmovups	%ymm0, 128(%rsp)
	movq	%r8, 120(%rsp)
	vmovaps	__ymm@0000000000000200000000000000020000000000000000800000000000000000(%rip), %ymm0 # ymm0 = [0,128,512,512]
	movq	%r8, 112(%rsp)
	vmovups	%ymm0, 72(%rsp)
	movq	%rdx, 64(%rsp)
	movq	%rdx, 56(%rsp)
	movq	$1, 160(%rsp)
	movq	$1, 104(%rsp)
	movq	$1, 48(%rsp)
	movq	$128, 40(%rsp)
	movq	$128, 32(%rsp)
	movl	$784, %r9d                      # imm = 0x310
	movq	%rcx, %rdx
	xorl	%r8d, %r8d
	vzeroupper
	callq	optimized_matmul_py_4a6286d9_impl_17630232307017152746
	.loc	1 7764 5                        # optimized_matmul\optimized_matmul_llvm.mlir:7764:5
	addq	$168, %rsp
	retq
.Ltmp3:
.Lfunc_end1:
                                        # -- End function
	.lcomm	cache_17,131072,32              # @cache_17
	.lcomm	cache_16,6144,32                # @cache_16
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
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
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
	.byte	0                               # EOM(3)
	.section	.debug_info,"dr"
.Lsection_info:
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.secrel32	.Lsection_abbrev        # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x53 DW_TAG_compile_unit
	.secrel32	.Linfo_string0          # DW_AT_producer
	.short	2                               # DW_AT_language
	.secrel32	.Linfo_string1          # DW_AT_name
	.secrel32	.Lline_table_start0     # DW_AT_stmt_list
	.secrel32	.Linfo_string2          # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin0       # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x2a:0x19 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.secrel32	.Linfo_string3          # DW_AT_linkage_name
	.secrel32	.Linfo_string3          # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x43:0x1a DW_TAG_subprogram
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.secrel32	.Linfo_string4          # DW_AT_linkage_name
	.secrel32	.Linfo_string4          # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.short	7702                            # DW_AT_decl_line
                                        # DW_AT_external
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
	.asciz	"optimized_matmul_py_4a6286d9_impl_17630232307017152746" # string offset=25
.Linfo_string4:
	.asciz	"optimized_matmul_py_4a6286d9"  # string offset=80
	.section	.debug_pubnames,"dr"
	.long	.LpubNames_end0-.LpubNames_begin0 # Length of Public Names Info
.LpubNames_begin0:
	.short	2                               # DWARF Version
	.secrel32	.Lcu_begin0             # Offset of Compilation Unit Info
	.long	94                              # Compilation Unit Length
	.long	67                              # DIE offset
	.asciz	"optimized_matmul_py_4a6286d9"  # External Name
	.long	42                              # DIE offset
	.asciz	"optimized_matmul_py_4a6286d9_impl_17630232307017152746" # External Name
	.long	0                               # End Mark
.LpubNames_end0:
	.section	.debug_pubtypes,"dr"
	.long	.LpubTypes_end0-.LpubTypes_begin0 # Length of Public Types Info
.LpubTypes_begin0:
	.short	2                               # DWARF Version
	.secrel32	.Lcu_begin0             # Offset of Compilation Unit Info
	.long	94                              # Compilation Unit Length
	.long	0                               # End Mark
.LpubTypes_end0:
	.globl	_fltused
	.section	.debug_line,"dr"
.Lsection_line:
.Lline_table_start0:
