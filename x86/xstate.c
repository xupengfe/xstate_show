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
 * - Because Xstate like XMM, YMM are not preserved across function calls,
 *   so write the key test functions via inline functions.
 */

#define _GNU_SOURCE
#include <err.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <sched.h>
#include <stdbool.h>
#include <sys/wait.h>
#include <sys/syscall.h>
#include <cpuid.h>
#include <malloc.h>

struct xsave_buffer *valid_xbuf, *compared_xbuf;
static uint32_t xstate_size;
static uint64_t xfeature_test_mask;

#define SUPPORT     1
#define NOT_SUPPORT 0
#define POPULATE_XSTATE_DATA 0x8f
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

#ifndef __cpuid_count
#define __cpuid_count(level, count, a, b, c, d)	\
	__asm__ __volatile__ ("cpuid\n\t"	\
			: "=a" (a), "=b" (b), "=c" (c), "=d" (d)	\
			: "0" (level), "2" (count))
#endif

/* It's from arch/x86/kernel/fpu/xstate.c. */
static const char * const xfeature_names[] = {
		"x87 floating point registers",
		"SSE registers",
		"AVX registers",
		"MPX bounds registers",
		"MPX CSR",
		"AVX-512 opmask",
		"AVX-512 Hi256",
		"AVX-512 ZMM_Hi256",
		"Processor Trace (unused)",
		"Protection Keys User registers",
		"PASID state",
		"unknown xstate feature",
		"unknown xstate feature",
		"unknown xstate feature",
		"unknown xstate feature",
		"unknown xstate feature",
		"unknown xstate feature",
		"AMX Tile config",
		"AMX Tile data",
		"unknown xstate feature",
};

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

void *aligned_alloc(size_t alignment, size_t size);

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
} xstate_info;

static inline void check_cpuid_xsave(void)
{
	uint32_t eax, ebx, ecx, edx;

	/*
	 * CPUID.1:ECX.XSAVE[bit 26] enumerates general
	 * support for the XSAVE feature set, including
	 * XGETBV.
	 */
	__cpuid_count(1, 0, eax, ebx, ecx, edx);
	if (!(ecx & CPUID_LEAF1_ECX_XSAVE_MASK))
		fatal_error("cpuid: no CPU xsave support");
	if (!(ecx & CPUID_LEAF1_ECX_OSXSAVE_MASK))
		fatal_error("cpuid: no OS xsave support");
}

static inline int cpu_has_avx2(void)
{
	unsigned int eax, ebx, ecx, edx;

	/* CPUID.7.0:EBX.AVX2[bit 5]: the support for AVX2 instructions */
	__cpuid_count(7, 0, eax, ebx, ecx, edx);

	return !(!(ebx & CPUID_LEAF7_EBX_AVX2_MASK));
}

static inline int cpu_has_avx512f(void)
{
	unsigned int eax, ebx, ecx, edx;

	/* CPUID.7.0:EBX.AVX512F[bit 16]: the support for AVX512F instructions */
	__cpuid_count(7, 0, eax, ebx, ecx, edx);

	return !(!(ebx & CPUID_LEAF7_EBX_AVX512F_MASK));
}

static inline int cpu_has_pkeys(void)
{
	unsigned int eax, ebx, ecx, edx;

	/* CPUID.7.0:ECX.PKU[bit 3]: the support for PKRU instructions */
	__cpuid_count(7, 0, eax, ebx, ecx, edx);
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

	__cpuid_count(CPUID_LEAF_XSTATE, CPUID_SUBLEAF_XSTATE_USER, eax, ebx,
		ecx, edx);
	/*
	 * EBX enumerates the size (in bytes) required by the XSAVE
	 * instruction for an XSAVE area containing all the user state
	 * components corresponding to bits currently set in XCR0.
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

/* Retrieve the bitmap mask, offset and size of a specific xstate. */
static void retrieve_xstate_info(uint32_t xfeature_num)
{
	uint32_t eax, ebx, ecx, edx;

	__cpuid_count(CPUID_LEAF_XSTATE, xfeature_num, eax, ebx, ecx, edx);
	xstate_info.xstate_flag[xfeature_num] = SUPPORT;
	/*
	 * CPUID.(EAX=0xd, ECX=xfeature_num), and output is as follow:
	 * eax: xfeature num state component size
	 * ebx: xfeature num state component offset in user buffer
	 */
	xstate_info.xstate_size[xfeature_num] = eax;
	xstate_info.xstate_offset[xfeature_num] = ebx;
	xstate_info.xstate_mask = xstate_info.xstate_mask | (1 << xfeature_num);
}

static inline void set_xstatebv(struct xsave_buffer *buffer, uint64_t bv)
{
	/* XSTATE_BV is at the beginning of xstate header. */
	*(uint64_t *)(&buffer->header) = bv;
}

static uint64_t check_cpuid_xstate(void)
{
	/* CPU that support XSAVE could support FP and SSE by default. */
	xstate_info.xstate_mask = XFEATURE_MASK_FP | XFEATURE_MASK_SSE;

	if (!cpu_has_avx2())
		printf("[SKIP]\tNo avx2 capability, skip avx2 part xstate.\n");
	else
		/* Retrieve the bitmap mask, offset and size of a specific xstate. */
		retrieve_xstate_info(XFEATURE_YMM);

	if (!cpu_has_avx512f())
		printf("[SKIP]\tNo avx512f capability, skip avx512f part xstate.\n");
	else {
		retrieve_xstate_info(XFEATURE_OPMASK);
		retrieve_xstate_info(XFEATURE_ZMM_Hi256);
		retrieve_xstate_info(XFEATURE_Hi16_ZMM);
	}

	if (!cpu_has_pkeys())
		printf("[SKIP]\tNo pkeys capability, skip pkru part xstate.\n");
	else
		retrieve_xstate_info(XFEATURE_PKRU);

	return xstate_info.xstate_mask;
}

static void fill_xstate_buf(uint8_t data, unsigned char *buf, int xfeature_num)
{
	uint32_t i;

	if (xstate_info.xstate_flag[xfeature_num] != SUPPORT)
		return;
	for (i = 0; i < xstate_info.xstate_size[xfeature_num]; i++)
		buf[xstate_info.xstate_offset[xfeature_num] + i] = data;
}

/* Write PKRU xstate with values by instruction. */
static inline void wrpkru(uint32_t pkey)
{
	uint32_t ecx = 0, edx = 0;

	asm volatile(".byte 0x0f, 0x01, 0xef\n\t"
	     : : "a" (pkey), "c" (ecx), "d" (edx));
}

/* Fill FP/XMM/YMM/OPMASK and PKRU xstates into buffer. */
static void fill_xstates_buf(struct xsave_buffer *buf, uint32_t xsave_mask)
{
	uint8_t byte_data = POPULATE_XSTATE_DATA;
	/* FP xstate(0-159 bytes) offset(0) and size(160bytes) are fixed. */
	uint32_t fp_size = 160;
	/* XMM xstate(160-415 bytes) offset(160byte) and size(256bytes) are fixed */
	uint32_t xmm_offset = 160, xmm_size = 256, i, pkru_data;
	/* Populate fp x87 state, MXCSR and MXCSR_MASK data as follow. */
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

	/* Fill fp x87 state, MXCSR and MXCSR_MASK data(0-159 bytes) into buffer. */
	memcpy(buf, fp_data, fp_size);

	/*
	 * Fill specific byte data value into XMM xstate buffer(160-415 bytes).
	 * 416-511 bytes are reserved as 0.
	 */
	for (i = 0; i < xmm_size; i++)
		*((unsigned char *)buf + xmm_offset + i) = byte_data;

	/*
	 * Fill xstate-component bitmap(512-519 bytes) into xstate header.
	 * xstate header range is 512-575 bytes.
	 */
	set_xstatebv(buf, xsave_mask);

	/* Fill specific byte data value into YMM xstate buffer(YMM offset/size). */
	fill_xstate_buf(byte_data, (unsigned char *)buf, (int)XFEATURE_YMM);

	/*
	 * Fill specific byte data value into AVX512 OPMASK/ZMM xstates buffer
	 * (AVX512 OPMASK/ZMM offset/size).
	 */
	fill_xstate_buf(byte_data, (unsigned char *)buf, (int)XFEATURE_OPMASK);
	fill_xstate_buf(byte_data, (unsigned char *)buf, (int)XFEATURE_ZMM_Hi256);
	fill_xstate_buf(byte_data, (unsigned char *)buf, (int)XFEATURE_Hi16_ZMM);

	if (xstate_info.xstate_flag[XFEATURE_PKRU] == SUPPORT) {
		/*
		 * PKRU bit 0-1 must be 0 for r/w access to linear address of
		 * process-self.
		 */
		pkru_data = (byte_data & 0xfc) | (byte_data << 8) | (byte_data << 16) |
			(byte_data << 24);
		/* Write the pkru data into pkru xstate. */
		wrpkru(pkru_data);
		/* Fill the pkru data into pkru xstate buffer. */
		memcpy((unsigned char *)buf + xstate_info.xstate_offset[XFEATURE_PKRU],
			&pkru_data, sizeof(pkru_data));
	}
}

/*
 * Because xstate like XMM, YMM registers are not preserved across function
 * calls, so use inline function with assembly code only for fork syscall.
 */
static inline long long __fork(void)
{
	long long ret, nr = SYS_fork;

	asm volatile("syscall"
		 : "=a" (ret)
		 : "a" (nr), "b" (nr)
		 : "rcx", "r11", "memory", "cc");

	return ret;
}

/*
 * Because xstate like XMM, YMM registers are not preserved across function
 * calls, so use inline function with assembly code only to raise signal.
 */
static inline long long __raise(long long pid_num, long long sig_num)
{
	long long ret, nr = SYS_kill;

	register long long arg1 asm("rdi") = pid_num;
	register long long arg2 asm("rsi") = sig_num;

	asm volatile("syscall"
		 : "=a" (ret)
		 : "a" (nr), "b" (nr), "r" (arg1), "r" (arg2)
		 : "rcx", "r11", "memory", "cc");

	return ret;
}

static void sigusr1_handler(int signum, siginfo_t *info, void *__ctxp)
{
	if (signum == SIGUSR1)
		printf("[NOTE]\tAccess SIGUSR1 handling.\n");
}

static int test_xstate_sig_handle(void)
{
	pid_t process_pid;

	sethandler(SIGUSR1, sigusr1_handler, 0);
	printf("[RUN]\tCheck xstate around signal handling test.\n");

	process_pid = getpid();
	xrstor(valid_xbuf, xfeature_test_mask);
	__raise(process_pid, SIGUSR1);
	xsave(compared_xbuf, xfeature_test_mask);

	if (memcmp(&valid_xbuf->bytes[0], &compared_xbuf->bytes[0],	xstate_size))
		printf("[FAIL]\tProcess xstate is not same after signal handling\n");
	else
		printf("[PASS]\tProcess xstate is same after signal handling.\n");

	clearhandler(SIGUSR1);

	return 0;
}

static int test_xstate_fork(void)
{
	pid_t child;
	int status;

	printf("[RUN]\tParent pid:%d Check xstate around fork test.\n", getpid());
	memset(compared_xbuf, 0, xstate_size);

	/*
	 * Xrstor the valid_xbuf and call syscall assembly instruction, then
	 * save the xstate to compared_xbuf in child process for comparison.
	 */
	xrstor(valid_xbuf, xfeature_test_mask);
	child = __fork();
	if (child < 0)
		/* Fork syscall failed */
		fatal_error("fork failed");
	else if (child == 0) {
		/* Fork syscall succeeded, now in the child. */
		xsave(compared_xbuf, xfeature_test_mask);
		if (memcmp(&valid_xbuf->bytes[0], &compared_xbuf->bytes[0],
			xstate_size))
			printf("[FAIL]\tXstate of child process:%d is not same as xstate of parent\n",
				getpid());
		else
			printf("[PASS]\tXstate of child process:%d is same as xstate of parent\n",
				getpid());
	} else {
		if (waitpid(child, &status, 0) != child || !WIFEXITED(status))
			fatal_error("Child exit with error status");
	}

	return 0;
}

static void free_buf(void)
{
	free(valid_xbuf);
	free(compared_xbuf);
}

static void prepare_buf(void)
{
	valid_xbuf = alloc_xbuf(xstate_size);
	compared_xbuf = alloc_xbuf(xstate_size);
	/* Populate the specified data into the validate xstate buffer. */
	fill_xstates_buf(valid_xbuf, xfeature_test_mask);
}

void xfeature_name(uint64_t xfeature_idx, const char **feature_name)
{
	*feature_name = xfeature_names[xfeature_idx];
}

static void print_xfeature_name(uint32_t xfeature_num)
{
	const char *feature_name;

	xfeature_name(xfeature_num, &feature_name);
	printf("[NOTE]\tXSAVE feature num %02d: '%s'\n", xfeature_num,
		feature_name);
}

static void show_xfeatures_name(uint64_t xfeature_mask)
{
	uint32_t i;

	for (i = 0; i < XFEATURE_MAX; i++) {
		if (!(xfeature_mask & (1 << i)))
			continue;
		print_xfeature_name(i);
	}
}

int main(void)
{
	/* Check hardware availability for xsave at first */
	check_cpuid_xsave();
	xstate_size = get_xstate_size();
	/* Check tested xstate by CPU id and retrieve CPU xstate info. */
	xfeature_test_mask = check_cpuid_xstate();
	printf("[NOTE]\tTest following mask:%lx xstates\n", xfeature_test_mask);
	show_xfeatures_name(xfeature_test_mask);

	prepare_buf();
	test_xstate_sig_handle();
	test_xstate_fork();
	free_buf();

	return 0;
}
