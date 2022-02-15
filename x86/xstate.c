// SPDX-License-Identifier: GPL-2.0-only
/*
 * xstate.c - tests XSAVE feature with fork and signal handling.
 *
 * The XSAVE feature set supports the saving and restoring of state components.
 * It tests "FP, SSE(XMM), AVX2(YMM), AVX512_OPMASK/AVX512_ZMM_Hi256/
 * AVX512_Hi16_ZMM and PKRU parts" xstates with following cases:
 * 1. The content of these xstates in the process should not change after the
 *    signal handling.
 * 2. The content of these xstates in the child process should be the same as
 *    those of the parent process after fork syscall.
 *
 * The regions and reserved bytes of the components tested for XSAVE are as
 * follows:
 * FP             (0 - 159 bytes)
 * SSE(XMM)       (160-415 bytes)
 * Reserved       (416-511 bytes)
 * Header_used    (512-527 bytes; XSTATE BV(bitmap vector) mask:512-519 bytes)
 * Headed_reserved(528-575 bytes must be 00)
 * YMM            (Offset:CPUID.(EAX=0D,ECX=2).EBX Size:CPUID(EAX=0D,ECX=2).EAX)
 * AVX512_OPMASK  (Offset:CPUID.(EAX=0D,ECX=5).EBX Size:CPUID(EAX=0D,ECX=5).EAX)
 * ZMM_Hi256      (Offset:CPUID.(EAX=0D,ECX=6).EBX Size:CPUID(EAX=0D,ECX=6).EAX)
 * Hi16_ZMM       (Offset:CPUID.(EAX=0D,ECX=7).EBX Size:CPUID(EAX=0D,ECX=7).EAX)
 * PKRU           (Offset:CPUID.(EAX=0D,ECX=9).EBX Size:CPUID(EAX=0D,ECX=9).EAX)
 *
 * Becasue xstate like XMM will not be preserved across function calls, it uses
 * assembly instruction to call a system call of fork or raise signal, and uses
 * the "inline" keyword in key test functions.
 * To prevent GCC from generating any FP/SSE(XMM)/AVX/PKRU code by mistake, add
 * "-mno-sse -mno-mmx -mno-sse2 -mno-avx -mno-pku" compiler arguments to avoid
 * spurious failure, so test code can't use <stdlib.h> because of "-mno-sse"
 * compile parameter. Not using <stdlib.h> makes it impossible to use
 * "kselftest.h", and it requires a declaration that defines *aligned_alloc().
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
static bool sigusr1_done;

/* Populate xstate with byte 0x8f. */
#define XSTATE_TESTBYTE 0x8f
/* Bits 0-1 in first byte of PKRU must be 0 for RW access to linear address. */
#define PKRU_TESTBYTE 0xfc
/* FP xstate(0-159 bytes) offset(0) and size(160 bytes) are fixed. */
#define FP_SIZE	160
/* XMM xstate(160-415 bytes) offset(160 byte) and size(256 bytes) are fixed. */
#define XMM_OFFSET	160
#define XMM_SIZE	256
/*
 * xstate 416-511 bytes are reserved, XSAVE header offset 512 bytes
 * and header size 64 bytes are fixed.
 */
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
#define __cpuid_count(level, count, a, b, c, d) ({	\
	__asm__ __volatile__ ("cpuid\n\t"	\
			: "=a" (a), "=b" (b), "=c" (c), "=d" (d)	\
			: "0" (level), "2" (count));	\
})
#endif

/* err() exits and will not return. */
#define fatal_error(msg, ...)	err(1, "[FAIL]\t" msg, ##__VA_ARGS__)

/*
 * XMM will be tested and compiled with -mno-sse to avoid XMM effects, which
 * makes stdlib.h unusable, so define declaration here to avoid warning.
 */
void *aligned_alloc(size_t alignment, size_t size);

enum support_type {
	NOT_SUPPORT,
	SUPPORT,
};

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
	uint64_t mask;
	uint32_t size[XFEATURE_MAX];
	uint32_t offset[XFEATURE_MAX];
} xstate_info;

static inline void check_cpuid_xsave_availability(void)
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

static inline bool xstate_tested(int xfeature_num)
{
	return !!(xstate_info.mask & (1 << xfeature_num));
}

static inline int cpu_has_avx2(void)
{
	unsigned int eax, ebx, ecx, edx;

	/* CPUID.7.0:EBX.AVX2[bit 5]: the support for AVX2 instructions */
	__cpuid_count(7, 0, eax, ebx, ecx, edx);

	return !!(ebx & CPUID_LEAF7_EBX_AVX2_MASK);
}

static inline int cpu_has_avx512f(void)
{
	unsigned int eax, ebx, ecx, edx;

	/* CPUID.7.0:EBX.AVX512F[bit 16]: the support for AVX512F instructions */
	__cpuid_count(7, 0, eax, ebx, ecx, edx);

	return !!(ebx & CPUID_LEAF7_EBX_AVX512F_MASK);
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

static struct xsave_buffer *alloc_xbuf(uint32_t buf_size)
{
	struct xsave_buffer *xbuf;

	/* XSAVE buffer should be 64B-aligned. */
	xbuf = aligned_alloc(64, buf_size);
	if (!xbuf)
		fatal_error("aligned_alloc()");

	return xbuf;
}

static inline void __xsave(struct xsave_buffer *xbuf, uint64_t rfbm)
{
	uint32_t rfbm_lo = rfbm;
	uint32_t rfbm_hi = rfbm >> 32;

	asm volatile("xsave (%%rdi)"
		     : : "D" (xbuf), "a" (rfbm_lo), "d" (rfbm_hi)
		     : "memory");
}

static inline void __xrstor(struct xsave_buffer *xbuf, uint64_t rfbm)
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
static void retrieve_xstate_size_and_offset(uint32_t xfeature_num)
{
	uint32_t eax, ebx, ecx, edx;

	__cpuid_count(CPUID_LEAF_XSTATE, xfeature_num, eax, ebx, ecx, edx);
	/*
	 * CPUID.(EAX=0xd, ECX=xfeature_num), and output is as follow:
	 * eax: xfeature num state component size
	 * ebx: xfeature num state component offset in user buffer
	 */
	xstate_info.size[xfeature_num] = eax;
	xstate_info.offset[xfeature_num] = ebx;
}

static inline void set_xstatebv(struct xsave_buffer *buffer, uint64_t bv)
{
	/* XSTATE_BV is at the beginning of xstate header. */
	*(uint64_t *)(&buffer->header) = bv;
}

static void check_cpuid_xstate_info(void)
{
	/* CPU that support XSAVE could support FP and SSE by default. */
	xstate_info.mask = XFEATURE_MASK_FP | XFEATURE_MASK_SSE;

	xstate_size = get_xstate_size();
	if (cpu_has_avx2()) {
		xstate_info.mask |= XFEATURE_MASK_YMM;
		retrieve_xstate_size_and_offset(XFEATURE_YMM);
	}

	if (cpu_has_avx512f()) {
		xstate_info.mask |= XFEATURE_MASK_OPMASK | XFEATURE_MASK_ZMM_Hi256 |
			XFEATURE_MASK_Hi16_ZMM;
		retrieve_xstate_size_and_offset(XFEATURE_OPMASK);
		retrieve_xstate_size_and_offset(XFEATURE_ZMM_Hi256);
		retrieve_xstate_size_and_offset(XFEATURE_Hi16_ZMM);
	}

	if (cpu_has_pkeys()) {
		xstate_info.mask |= XFEATURE_MASK_PKRU;
		retrieve_xstate_size_and_offset(XFEATURE_PKRU);
	}
}

static void fill_xstate_buf(uint8_t test_byte, unsigned char *buf,
	int xfeature_num)
{
	uint32_t i;

	for (i = 0; i < xstate_info.size[xfeature_num]; i++)
		buf[xstate_info.offset[xfeature_num] + i] = test_byte;
}

/* Fill FP/XMM/YMM/OPMASK and PKRU xstates into buffer. */
static void fill_xstates_buf(struct xsave_buffer *buf, uint32_t xsave_mask)
{
	uint32_t i;
	/* Populate fp x87 state for MXCSR and MXCSR_MASK data as follow. */
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

	/* Clean the buffer with all 0 first. */
	memset(buf, 0, xstate_size);

	/* Fill fp x87 state, MXCSR and MXCSR_MASK data(0-159 bytes) into buffer. */
	memcpy(buf, fp_data, FP_SIZE);

	/*
	 * Fill test byte value into XMM xstate buffer(160-415 bytes).
	 * xstate 416-511 bytes are reserved as 0.
	 */
	for (i = 0; i < XMM_SIZE; i++)
		*((unsigned char *)buf + XMM_OFFSET + i) = XSTATE_TESTBYTE;

	/*
	 * Fill xstate-component bitmap(512-519 bytes) into xstate header.
	 * xstate header range is 512-575 bytes.
	 */
	set_xstatebv(buf, xsave_mask);

	/* Fill test byte value into YMM xstate buffer(YMM offset/size). */
	if (xstate_tested(XFEATURE_YMM))
		fill_xstate_buf(XSTATE_TESTBYTE, (unsigned char *)buf, XFEATURE_YMM);

	/*
	 * Fill test byte value into AVX512 OPMASK/ZMM xstates buffer
	 * (AVX512_OPMASK/ZMM_Hi256/Hi16_ZMM offset/size).
	 */
	if (xstate_tested(XFEATURE_OPMASK))
		fill_xstate_buf(XSTATE_TESTBYTE, (unsigned char *)buf, XFEATURE_OPMASK);
	if (xstate_tested(XFEATURE_ZMM_Hi256)) {
		fill_xstate_buf(XSTATE_TESTBYTE, (unsigned char *)buf,
			XFEATURE_ZMM_Hi256);
	}
	if (xstate_tested(XFEATURE_Hi16_ZMM)) {
		fill_xstate_buf(XSTATE_TESTBYTE, (unsigned char *)buf,
			XFEATURE_Hi16_ZMM);
	}

	if (xstate_tested(XFEATURE_PKRU)) {
		/* Only 0-3 bytes of pkru xstates are allowed to be written. */
		memset((unsigned char *)buf + xstate_info.offset[XFEATURE_PKRU],
			PKRU_TESTBYTE, sizeof(uint32_t));
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
	sigusr1_done = true;
}

static void test_xstate_sig_handle(void)
{
	pid_t process_pid;

	sethandler(SIGUSR1, sigusr1_handler, 0);
	printf("[RUN]\tCheck xstate around signal handling test.\n");
	process_pid = getpid();

	/*
	 * Xrstor the valid_xbuf and call syscall assembly instruction, then
	 * save the xstate to compared_xbuf after signal handling for comparison.
	 */
	__xrstor(valid_xbuf, xstate_info.mask);
	__raise(process_pid, SIGUSR1);
	__xsave(compared_xbuf, xstate_info.mask);
	if (sigusr1_done == true)
		printf("[NOTE]\tSIGUSR1 handling is done.\n");
	else
		fatal_error("Didn't access SIGUSR1 handling after raised SIGUSR1");

	if (memcmp(&valid_xbuf->bytes[0], &compared_xbuf->bytes[0], xstate_size))
		printf("[FAIL]\tProcess xstate is not same after signal handling\n");
	else
		printf("[PASS]\tProcess xstate is same after signal handling.\n");

	clearhandler(SIGUSR1);
}

static void test_xstate_fork(void)
{
	pid_t child;
	int status;

	printf("[RUN]\tParent pid:%d check xstate around fork test.\n", getpid());
	memset(compared_xbuf, 0, xstate_size);

	/*
	 * Xrstor the valid_xbuf and call syscall assembly instruction, then
	 * save the xstate to compared_xbuf in child process for comparison.
	 */
	__xrstor(valid_xbuf, xstate_info.mask);
	child = __fork();
	if (child < 0) {
		/* Fork syscall failed */
		fatal_error("fork failed");
	} else if (child == 0) {
		/* Fork syscall succeeded, now in the child. */
		__xsave(compared_xbuf, xstate_info.mask);
		if (memcmp(&valid_xbuf->bytes[0], &compared_xbuf->bytes[0],
			xstate_size)) {
			printf("[FAIL]\tXstate of child process:%d is not same as xstate of parent\n",
				getpid());
		} else {
			printf("[PASS]\tXstate of child process:%d is same as xstate of parent.\n",
				getpid());
		}
	} else {
		if (waitpid(child, &status, 0) != child || !WIFEXITED(status))
			fatal_error("Child exit with error status");
	}
}

static void free_xbuf(void)
{
	free(valid_xbuf);
	free(compared_xbuf);
}

static void prepare_xbuf(void)
{
	valid_xbuf = alloc_xbuf(xstate_size);
	compared_xbuf = alloc_xbuf(xstate_size);
	/* Populate the specified data into the validate xstate buffer. */
	fill_xstates_buf(valid_xbuf, xstate_info.mask);
}

static void show_tested_xfeatures(void)
{
	uint32_t i;
	const char *feature_name;

	printf("[NOTE]\tTest xstates with following mask:%lx.\n", xstate_info.mask);
	for (i = 0; i < XFEATURE_MAX; i++) {
		if (!xstate_tested(i))
			continue;
		feature_name = xfeature_names[i];
		printf("[NOTE]\tXSAVE feature num %02d: '%s'\n", i, feature_name);
	}
}

int main(void)
{
	/* Check hardware availability for xsave at first. */
	check_cpuid_xsave_availability();
	/* Check tested xstate by CPU id and retrieve CPU xstate info. */
	check_cpuid_xstate_info();
	show_tested_xfeatures();
	prepare_xbuf();

	test_xstate_sig_handle();
	test_xstate_fork();
	free_xbuf();

	return 0;
}
