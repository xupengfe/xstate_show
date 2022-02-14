// SPDX-License-Identifier: GPL-2.0-only
/*
 * xsave_signal_handle.c - tests XSAVE with signal handling.
 *
 * XSAVE feature will save/restore xstate, each process and signal handling on
 * x86 CPUs has it's own xstates, and the xstates of user process and signal
 * handling are managed by kernel.
 * When user process raises a signal, kernel saves xstates of the process
 * by kernel with xsave/xsaves instruction, and after signal handling, the
 * xstate of process will be restored by kernel with xrstor/xrstors instruction.
 * So the xstate of the process should not change after signal handling.
 * It tests that:
 * 1. The xstates content of the process should not change after the entire
 *    signal handling.
 * 2. The xstates content of the child process should be the same as that of the
 *    parent process.
 * 3. The xstates content of the process should be the same across process
 *    switching.
 *
 * Updates:
 * - Because it tests FP SSE xstate, in order to prevent GCC from generating any
 *   FP code by mistake, "-mno-sse -mno-mmx -mno-sse2 -mno-avx -mno-pku"
 *   compiler parameter is added, it's referred to the parameters for compiling
 *   the x86 kernel.
 * - Remove the use of "kselftest.h", because kselftest.h includes <stdlib.h>,
 *   and "stdlib.h" uses sse instructions in it's libc, it conflicts with
 *   parameters for compiling "-mno-sse".
 * - Xstate like XMM, YMM are easily corrupted by libc, so write the key
 *   test function with assembly instructions only without any libc.
 * - Becasue Xstate like XMM, YMM are not preserved across function calls,
 *   so write the key test functions via inline functions.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <sched.h>
#include <stdbool.h>
#include <sys/wait.h>
#include <sys/syscall.h>

struct xsave_buffer *xstate_buf_init, *xstate_buf_compare, *xstate_buf_pollute;
static uint32_t xstate_size, err_num;
static uint64_t xsave_test_mask;

#define SUPPORT     1
#define NOT_SUPPORT 0
/* Define two different sets of initialized data and tainted data. */
#define XSTATE_INIT_DATA 0x1f2f3f4f
#define XSTATE_POLLUTE_DATA 0xf5f6f7f8
#define XSAVE_HDR_OFFSET	512
#define XSAVE_HDR_SIZE		64
/* The following definition is from arch/x86/include/asm/fpu/types.h */
#define XFEATURE_MASK_FP (1 << XFEATURE_FP)
#define XFEATURE_MASK_SSE (1 << XFEATURE_SSE)
#define XFEATURE_MASK_YMM (1 << XFEATURE_YMM)
#define XFEATURE_MASK_OPMASK (1 << XFEATURE_OPMASK)
#define XFEATURE_MASK_ZMM_Hi256 (1 << XFEATURE_ZMM_Hi256)
#define XFEATURE_MASK_Hi16_ZMM (1 << XFEATURE_Hi16_ZMM)
#define XFEATURE_MASK_PKRU (1 << XFEATURE_PKRU)

#define CPUID_LEAF1_ECX_XSAVE_MASK	(1 << 26)  /* XSAVE instructions */
#define CPUID_LEAF1_ECX_OSXSAVE_MASK	(1 << 27) /* OSXSAVE flag */

#define CPUID_LEAF7_EBX_AVX2_MASK	(1U << 5) /* AVX2 instructions */
#define CPUID_LEAF7_EBX_AVX512F_MASK	(1U << 16) /* AVX-512 Foundation */

#define CPUID_LEAF7_ECX_PKU_MASK   (1U << 3) /* Protection Keys for Userspace */
#define CPUID_LEAF7_ECX_OSPKE_MASK (1U << 4) /* OS Protection Keys Enable */

#define CPUID_LEAF_XSTATE		0xd
#define CPUID_SUBLEAF_XSTATE_USER	0x0

/*
 * It's from arch/x86/include/asm/fpu/types.h
 * List of XSAVE features Linux knows about:
 */
enum xfeature {
	XFEATURE_FP,
	XFEATURE_SSE,
	/*
	 * Values above here are "legacy states".
	 * Those below are "extended states".
	 */
	XFEATURE_YMM,
	XFEATURE_BNDREGS,
	XFEATURE_BNDCSR,
	XFEATURE_OPMASK,
	XFEATURE_ZMM_Hi256,
	XFEATURE_Hi16_ZMM,
	XFEATURE_PT_UNIMPLEMENTED_SO_FAR,
	XFEATURE_PKRU,
	XFEATURE_PASID,
	XFEATURE_RSRVD_COMP_11,
	XFEATURE_RSRVD_COMP_12,
	XFEATURE_RSRVD_COMP_13,
	XFEATURE_RSRVD_COMP_14,
	XFEATURE_LBR,
	XFEATURE_RSRVD_COMP_16,
	XFEATURE_XTILE_CFG,
	XFEATURE_XTILE_DATA,
	XFEATURE_MAX,
};

void free(void *ptr);
void *aligned_alloc(size_t alignment, size_t size);
void err(int eval, const char *fmt, ...);
/* err() exits and will not return */
#define fatal_error(msg, ...)	err(1, "[FAIL]\t" msg, ##__VA_ARGS__)

struct xsave_buffer {
	union {
		struct {
			char legacy[XSAVE_HDR_OFFSET];
			char header[XSAVE_HDR_SIZE];
			char extended[0];
		};
		char bytes[0];
	};
};

static struct {
	uint64_t xstate_mask;
	/* xstate_flag: 1 means support XFEATURE xstate, 0 means not support */
	uint32_t xstate_flag[XFEATURE_MAX];
	uint32_t xstate_size[XFEATURE_MAX];
	uint32_t xstate_offset[XFEATURE_MAX];
} xstate_data;

static inline void cpuid(uint32_t *eax, uint32_t *ebx, uint32_t *ecx,
	uint32_t *edx)
{
	asm volatile("cpuid;"
		     : "=a" (*eax), "=b" (*ebx), "=c" (*ecx), "=d" (*edx)
		     : "0" (*eax), "2" (*ecx));
}

static inline void check_cpuid_xsave(void)
{
	uint32_t eax, ebx, ecx, edx;

	/*
	 * CPUID.1:ECX.XSAVE[bit 26] enumerates general
	 * support for the XSAVE feature set, including
	 * XGETBV.
	 */
	eax = 1;
	ecx = 0;
	cpuid(&eax, &ebx, &ecx, &edx);
	if (!(ecx & CPUID_LEAF1_ECX_XSAVE_MASK))
		fatal_error("cpuid: no CPU xsave support");
	if (!(ecx & CPUID_LEAF1_ECX_OSXSAVE_MASK))
		fatal_error("cpuid: no OS xsave support");
}

static inline int cpu_has_avx2(void)
{
	unsigned int eax, ebx, ecx, edx;

	/* CPUID.7.0:EBX.AVX2[bit 5]: the support for AVX2 instructions */
	eax = 7;
	ecx = 0;
	cpuid(&eax, &ebx, &ecx, &edx);

	return !(!(ebx & CPUID_LEAF7_EBX_AVX2_MASK));
}

static inline int cpu_has_avx512f(void)
{
	unsigned int eax, ebx, ecx, edx;

	/* CPUID.7.0:EBX.AVX512F[bit 16]: the support for AVX512F instructions */
	eax = 7;
	ecx = 0;
	cpuid(&eax, &ebx, &ecx, &edx);

	return !(!(ebx & CPUID_LEAF7_EBX_AVX512F_MASK));
}

static inline int cpu_has_pkeys(void)
{
	unsigned int eax, ebx, ecx, edx;

	/* CPUID.7.0:ECX.PKU[bit 3]: the support for PKRU instructions */
	eax = 7;
	ecx = 0;
	cpuid(&eax, &ebx, &ecx, &edx);
	if (!(ecx & CPUID_LEAF7_ECX_PKU_MASK))
		return NOT_SUPPORT;
	/* CPUID.7.0:ECX.OSPKE[bit 4]: the support for OS set CR4.PKE */
	if (!(ecx & CPUID_LEAF7_ECX_OSPKE_MASK))
		return NOT_SUPPORT;

	return SUPPORT;
}

static uint32_t get_xstate_size(void)
{
	uint32_t eax, ebx, ecx, edx;

	eax = CPUID_LEAF_XSTATE;
	ecx = CPUID_SUBLEAF_XSTATE_USER;
	cpuid(&eax, &ebx, &ecx, &edx);
	/*
	 * EBX enumerates the size (in bytes) required by the XSAVE
	 * instruction for an XSAVE area containing all the user state
	 * components corresponding to bits currently set in XCR0.
	 *
	 * Stash that off so it can be used to allocate buffers later.
	 */

	return ebx;
}

struct xsave_buffer *alloc_xbuf(uint32_t buf_size)
{
	struct xsave_buffer *xbuf;

	/* XSAVE buffer should be 64B-aligned. */
	xbuf = aligned_alloc(64, buf_size);
	if (!xbuf)
		fatal_error("aligned_alloc()");

	return xbuf;
}

static inline void xsave(struct xsave_buffer *xbuf, uint64_t rfbm)
{
	uint32_t rfbm_lo = rfbm;
	uint32_t rfbm_hi = rfbm >> 32;

	asm volatile("xsave (%%rdi)"
		     : : "D" (xbuf), "a" (rfbm_lo), "d" (rfbm_hi)
		     : "memory");
}

static inline void xrstor(struct xsave_buffer *xbuf, uint64_t rfbm)
{
	uint32_t rfbm_lo = rfbm;
	uint32_t rfbm_hi = rfbm >> 32;

	asm volatile("xrstor (%%rdi)"
		     : : "D" (xbuf), "a" (rfbm_lo), "d" (rfbm_hi));
}

static void affinitize_cpu0(void)
{
	cpu_set_t cpuset;

	CPU_ZERO(&cpuset);
	CPU_SET(0, &cpuset);

	if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0)
		fatal_error("sched_setaffinity to CPU 0");
}

static void sethandler(int sig, void (*handler)(int, siginfo_t *, void *),
		       int flags)
{
	struct sigaction sa;

	memset(&sa, 0, sizeof(sa));
	sa.sa_sigaction = handler;
	sa.sa_flags = SA_SIGINFO | flags;
	sigemptyset(&sa.sa_mask);
	if (sigaction(sig, &sa, 0))
		fatal_error("sigaction");
}

static void clearhandler(int sig)
{
	struct sigaction sa;

	memset(&sa, 0, sizeof(sa));
	sa.sa_handler = SIG_DFL;
	sigemptyset(&sa.sa_mask);
	if (sigaction(sig, &sa, 0))
		fatal_error("sigaction");
}

static void fill_xstate_range(uint32_t xstate_id)
{
	uint32_t eax, ebx, ecx, edx;

	eax = (uint32_t)CPUID_LEAF_XSTATE;
	ecx = xstate_id;
	cpuid(&eax, &ebx, &ecx, &edx);
	xstate_data.xstate_flag[xstate_id] = SUPPORT;
	xstate_data.xstate_size[xstate_id] = eax;
	xstate_data.xstate_offset[xstate_id] = ebx;
	xstate_data.xstate_mask = xstate_data.xstate_mask | (1 << xstate_id);
}

static inline void set_xstatebv(struct xsave_buffer *buffer, uint64_t bv)
{
	/* XSTATE_BV is at the beginning of xstate header. */
	*(uint64_t *)(&buffer->header) = bv;
}

static uint64_t check_cpuid_xstate(void)
{
	/* CPU that support XSAVE could support FP and SSE by default. */
	xstate_data.xstate_mask = XFEATURE_MASK_FP | XFEATURE_MASK_SSE;

	if (!cpu_has_avx2())
		printf("[SKIP] No avx2 capability, skip avx2 part xstate.\n");
	else
		fill_xstate_range(XFEATURE_YMM);

	if (!cpu_has_avx512f())
		printf("[SKIP] No avx512f capability, skip avx512f part xstate.\n");
	else {
		fill_xstate_range(XFEATURE_OPMASK);
		fill_xstate_range(XFEATURE_ZMM_Hi256);
		fill_xstate_range(XFEATURE_Hi16_ZMM);
	}

	if (!cpu_has_pkeys())
		printf("[SKIP] No pkeys capability, skip pkru part xstate.\n");
	else
		fill_xstate_range(XFEATURE_PKRU);

	return xstate_data.xstate_mask;
}

static void fill_xstate_buf(char data, unsigned char *buf, int xstate_id)
{
	uint32_t i;

	if (xstate_data.xstate_flag[xstate_id] == SUPPORT) {
		for (i = 0; i < xstate_data.xstate_size[xstate_id]; i++)
			buf[xstate_data.xstate_offset[xstate_id] + i] = data;
	}
}

/* Write PKRU xstate with values by instruction. */
static inline void wrpkru(uint32_t pkey)
{
	uint32_t ecx = 0, edx = 0;

	asm volatile(".byte 0x0f, 0x01, 0xef\n\t"
	     : : "a" (pkey), "c" (ecx), "d" (edx));
}

static inline void prepare_fp_buf(uint32_t ui32_fp)
{
	uint64_t ui64_fp;

	/*
	 * Populate ui32_fp and ui64_fp and so on value onto FP registers stack
	 * and FP ST/MM xstates
	 */
	ui64_fp = (uint64_t)ui32_fp << 32;
	asm volatile("finit");
	ui64_fp = ui64_fp + ui32_fp;
	asm volatile("fldl %0" : : "m" (ui64_fp));
	asm volatile("flds %0" : : "m" (ui32_fp));
}

/* Fill FP/XMM/YMM/OPMASK and PKRU xstates into buffer. */
static void fill_xstates_buf(struct xsave_buffer *buf, uint32_t xsave_mask,
	uint32_t int_data)
{
	unsigned char *ptr = (unsigned char *)buf;
	uint32_t *int_ptr = (uint32_t *)buf;
	/* FP xstate(0-159 bytes) offset(0) and size(160bytes) are fixed. */
	uint32_t fp_offset = 0, fp_size = 160;
	/* XMM xstate(160-415 bytes) offset(160byte) and size(256bytes) are fixed */
	uint32_t xmm_offset = 160, xmm_size = 256, i, pkru_data;
	uint8_t byte_data;
	unsigned char fp_data[160] = {
		0x7f, 0x03, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00,
		0xf0, 0x19, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00,
		0x00, 0x78, 0xfa, 0x79, 0xf9, 0x78, 0xfa, 0xf9,
		0xf2, 0x3d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x78, 0xfa, 0x79, 0xf9, 0x78, 0xfa, 0xf9,
		0xf2, 0x3d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x78, 0xfa, 0x79, 0xf9, 0x78, 0xfa, 0xf9,
		0xf2, 0x3d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x78, 0xfa, 0x79, 0xf9, 0x78, 0xfa, 0xf9,
		0xf2, 0x3d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x78, 0xfa, 0x79, 0xf9, 0x78, 0xfa, 0xf9,
		0xf2, 0x3d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x78, 0xfa, 0x79, 0xf9, 0x78, 0xfa, 0xf9,
		0xf2, 0x3d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x78, 0xfa, 0x79, 0xf9, 0x78, 0xfa, 0xf9,
		0xf2, 0x3d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x78, 0xfa, 0x79, 0xf9, 0x78, 0xfa, 0xf9,
		0xf2, 0x3d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
	unsigned char fp_pollute[160] = {
		0x7f, 0x03, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00,
		0xf1, 0x19, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00,
		0x00, 0xc0, 0xbf, 0xb7, 0xaf, 0xc7, 0xbf, 0xb7,
		0x5f, 0xc3, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0xc0, 0xbf, 0xb7, 0xaf, 0xc7, 0xbf, 0xb7,
		0x5f, 0xc3, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0xc0, 0xbf, 0xb7, 0xaf, 0xc7, 0xbf, 0xb7,
		0x5f, 0xc3, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0xc0, 0xbf, 0xb7, 0xaf, 0xc7, 0xbf, 0xb7,
		0x5f, 0xc3, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0xc0, 0xbf, 0xb7, 0xaf, 0xc7, 0xbf, 0xb7,
		0x5f, 0xc3, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0xc0, 0xbf, 0xb7, 0xaf, 0xc7, 0xbf, 0xb7,
		0x5f, 0xc3, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0xc0, 0xbf, 0xb7, 0xaf, 0xc7, 0xbf, 0xb7,
		0x5f, 0xc3, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0xc0, 0xbf, 0xb7, 0xaf, 0xc7, 0xbf, 0xb7,
		0x5f, 0xc3, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

	pkru_data = int_data << 8;
	byte_data = int_data & 0xff;

	/* Fill specific bytes data value into FP xstate buffer(0-159 bytes). */
	if (int_data == XSTATE_INIT_DATA) {
		for (i = 0; i < fp_size; i++)
			ptr[fp_offset + i] = fp_data[i];
	} else {
	/* Fill pollute data into fp xstate buffer. */
		for (i = 0; i < fp_size; i++)
			ptr[fp_offset + i] = fp_pollute[i];
	}

	/* Fill specific byte data value into XMM xstate buffer(160-415 bytes). */
	for (i = 0; i < xmm_size; i++)
		ptr[xmm_offset + i] = byte_data;

	/* Fill xstate-component bitmap into xstate header buffer(416-512 bytes). */
	set_xstatebv(buf, xsave_mask);

	/* Fill specific byte data value into YMM xstate buffer(YMM offset/size). */
	fill_xstate_buf(byte_data, ptr, (int)XFEATURE_YMM);

	/*
	 * Fill specific byte data value into AVX512 OPMASK/ZMM xstates buffer
	 * (AVX512 OPMASK/ZMM offset/size).
	 */
	fill_xstate_buf(byte_data, ptr, (int)XFEATURE_OPMASK);
	fill_xstate_buf(byte_data, ptr, (int)XFEATURE_ZMM_Hi256);
	fill_xstate_buf(byte_data, ptr, (int)XFEATURE_Hi16_ZMM);

	if (xstate_data.xstate_flag[XFEATURE_PKRU] == SUPPORT) {
		/* Write the pkru data into pkru xstate. */
		wrpkru(pkru_data);
		/* Fill the pkru data into pkru xstate buffer. */
		int_ptr[(xstate_data.xstate_offset[XFEATURE_PKRU])/4] = pkru_data;
	}
}

static inline long long execute_syscall(int syscall64_num, long long rdi,
	long long rsi, long long rdx, long long r10, long long r8, long long r9)
{
	long long ret;

	register long long arg1 asm("rdi") = rdi;
	register long long arg2 asm("rsi") = rsi;
	register long long arg3 asm("rdx") = rdx;
	register long long arg4 asm("r10") = r10;
	register long long arg5 asm("r8")  = r8;
	register long long arg6 asm("r9")  = r9;
	long long nr = (unsigned int)syscall64_num;

	asm volatile("syscall"
		 : "=a" (ret)
		 : "a" (nr), "b" (nr),
		   "r" (arg1), "r" (arg2), "r" (arg3),
		   "r" (arg4), "r" (arg5), "r" (arg6)
		 : "rcx", "r11", "memory", "cc");

	return ret;
}

/*
 * Because xstate like XMM, YMM registers are not preserved across function
 * calls, so use inline function with assembly code only for fork syscall.
 */
static inline long long __fork(struct xsave_buffer *buf, uint64_t xsave_mask)
{
	long long ret;

	ret = execute_syscall((int)SYS_fork, 0, 0, 0, 0, 0, 0);

	return ret;
}

/*
 * Because xstate like XMM, YMM registers are not preserved across function
 * calls, so use inline function with assembly code only to raise signal.
 */
static inline long long __raise(long long SIG, long long pid_num)
{
	long long ret;

	ret = execute_syscall((int)SYS_kill, pid_num, SIG, 0, 0, 0, 0);

	return ret;
}

static inline bool __validate_xstate_regs(struct xsave_buffer *buf0)
{
	int ret;

	xrstor(buf0, xsave_test_mask);
	xsave(xstate_buf_compare, xsave_test_mask);
	ret = memcmp(&buf0->bytes[0], &xstate_buf_compare->bytes[0], xstate_size);
	/* Clear xstate_buf_compare for next test */
	memset(xstate_buf_compare, 0, xstate_size);

	if (ret == 0)
		return false;

	return true;
}

static inline void validate_xstate_regs_same(struct xsave_buffer *buf)
{
	int ret = __validate_xstate_regs(buf);

	if (ret != 0)
		fatal_error("Xstate registers changed");
}

static void usr1_pollute_xstate_handler(int signum, siginfo_t *info,
	void *__ctxp)
{
	if (signum == SIGUSR1) {
		fill_xstates_buf(xstate_buf_pollute, xsave_test_mask,
			XSTATE_POLLUTE_DATA);
		/* Pollute xstate of SIGUSR1, it should not affect xstate of process. */
		xrstor(xstate_buf_pollute, xsave_test_mask);
	}
}

static int test_xstate_sig_handle(void)
{
	pid_t process_pid;

	sethandler(SIGUSR1, usr1_pollute_xstate_handler, 0);
	printf("[RUN]\tCheck xstate around signal handling test.\n");
	/*
	 * The content of process xstate loaded by xrstor should be the same as
	 * the buffer xstate_buf_init.
	 */
	validate_xstate_regs_same(xstate_buf_init);

	process_pid = getpid();
	xrstor(xstate_buf_init, xsave_test_mask);
	__raise(SIGUSR1, process_pid);
	xsave(xstate_buf_compare, xsave_test_mask);

	if (memcmp(&xstate_buf_init->bytes[0], &xstate_buf_compare->bytes[0],
		xstate_size)) {
		printf("[FAIL]\tProcess xstate is not same after signal handling\n");
		err_num++;
	} else
		printf("[PASS]\tProcess xstate is same after signal handling.\n");

	clearhandler(SIGUSR1);

	memset(xstate_buf_compare, 0, xstate_size);
	memset(xstate_buf_pollute, 0, xstate_size);

	return 0;
}

static int test_xstate_fork(void)
{
	pid_t child;
	int status;

	/* Affinitize to the same CPU0 to force process switching. */
	affinitize_cpu0();
	printf("[RUN]\tParent pid:%d Check xstate around fork test.\n", getpid());
	memset(xstate_buf_compare, 0, xstate_size);

	/*
	 * Xrstor the xstate_buf_init and call syscall assembly instruction, then
	 * save the xstate to xstate_buf_compare in child process for comparison.
	 */
	xrstor(xstate_buf_init, xsave_test_mask);
	child = __fork(xstate_buf_compare, xsave_test_mask);
	if (child < 0)
		/* Fork syscall failed */
		fatal_error("fork failed");
	else if (child == 0) {
		/* Fork syscall succeeded, now in the child. */
		xsave(xstate_buf_compare, xsave_test_mask);
		if (memcmp(&xstate_buf_init->bytes[0], &xstate_buf_compare->bytes[0],
			xstate_size))
			printf("[FAIL]\tXstate of child process:%d is not same as xstate of parent\n",
				getpid());
		else
			printf("[PASS]\tXstate of child process:%d is same as xstate of parent\n",
				getpid());

		/* Child process pollute it's own xstate. */
		fill_xstates_buf(xstate_buf_pollute, xsave_test_mask,
			XSTATE_POLLUTE_DATA);
		xrstor(xstate_buf_pollute, xsave_test_mask);
	} else {
		/* Fork syscall succeeded, now in the parent. */
		xsave(xstate_buf_compare, xsave_test_mask);
		if (waitpid(child, &status, 0) != child || !WIFEXITED(status))
			fatal_error("Child exit with error status");

		if (memcmp(&xstate_buf_init->bytes[0], &xstate_buf_compare->bytes[0],
				xstate_size)) {
			printf("[FAIL]\tParent xstate changed after process switching.\n");
			err_num++;
		} else
			printf("[PASS]\tParent xstate is same after process switching.\n");
	}

	return 0;
}

static void init_buf(void)
{
	xstate_buf_init = alloc_xbuf(xstate_size);
	xstate_buf_compare = alloc_xbuf(xstate_size);
	xstate_buf_pollute = alloc_xbuf(xstate_size);
}

static void free_buf(void)
{
	free(xstate_buf_init);
	free(xstate_buf_compare);
	free(xstate_buf_pollute);
}

int main(void)
{
	/* Check hardware availability for xsave at first */
	check_cpuid_xsave();
	xstate_size = get_xstate_size();
	/* Check CPU capability by CPU id and set CPU capability flags */
	xsave_test_mask = check_cpuid_xstate();
	printf("[OK]\tWill test xstate mask:%lx\n", xsave_test_mask);

	init_buf();
	/* Fill the initialized data into the initialized xstate buffer */
	fill_xstates_buf(xstate_buf_init, xsave_test_mask, XSTATE_INIT_DATA);

	test_xstate_sig_handle();

	test_xstate_fork();

	free_buf();

	return err_num;
}
