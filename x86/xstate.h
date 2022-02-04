/* SPDX-License-Identifier: GPL-2.0 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <cpuid.h>
#include <errno.h>
#include <sys/time.h>

#define RESULT_PASS 0
#define RESULT_FAIL 1
#define RESULT_ERROR 3
#define CHANGE    1
#define NO_CHANGE 0
#define SUPPORT     1
#define NOT_SUPPORT 0
/* The following definition is from arch/x86/include/asm/fpu/xstate.h */
#define XFEATURE_MASK_FP (1 << XFEATURE_FP)
#define XFEATURE_MASK_SSE (1 << XFEATURE_SSE)
#define XFEATURE_MASK_YMM (1 << XFEATURE_YMM)
#define XFEATURE_MASK_OPMASK (1 << XFEATURE_OPMASK)
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

#define XSAVE_HDR_OFFSET	512
#define XSAVE_HDR_SIZE		64
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
	/* xstate_flag 1 means support XFEATURE xstate, 0 means not support */
	uint32_t xstate_flag[XFEATURE_MAX];
	uint32_t xstate_size[XFEATURE_MAX];
	uint32_t xstate_offset[XFEATURE_MAX];
} xstate_data;

static inline void cpuid(uint32_t *eax, uint32_t *ebx, uint32_t *ecx, uint32_t *edx)
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

static inline int cpu_has_avx2(uint64_t xsave_mask)
{
	unsigned int eax, ebx, ecx, edx;

	/* CPUID.7.0:EBX.AVX2[bit 5]: the support for AVX2 instructions */
	eax = 7;
	ecx = 0;
	cpuid(&eax, &ebx, &ecx, &edx);
	xstate_data.xstate_flag[XFEATURE_YMM] = !(!(ebx & CPUID_LEAF7_EBX_AVX2_MASK));
	if (xstate_data.xstate_flag[XFEATURE_YMM])
		xsave_mask = xsave_mask | XFEATURE_MASK_YMM;

	return xstate_data.xstate_flag[XFEATURE_YMM];
}

static inline int cpu_has_avx512f(uint64_t xsave_mask)
{
	unsigned int eax, ebx, ecx, edx;

	/* CPUID.7.0:EBX.AVX512F[bit 16]: the support for AVX512F instructions */
	eax = 7;
	ecx = 0;
	cpuid(&eax, &ebx, &ecx, &edx);
	xstate_data.xstate_flag[XFEATURE_OPMASK] = !(!(ebx & CPUID_LEAF7_EBX_AVX512F_MASK));
	if (xstate_data.xstate_flag[XFEATURE_OPMASK])
		xsave_mask = xsave_mask | XFEATURE_MASK_OPMASK;

	return xstate_data.xstate_flag[XFEATURE_OPMASK];
}

static inline int cpu_has_pkeys(uint64_t xsave_mask)
{
	unsigned int eax, ebx, ecx, edx;

	/* CPUID.7.0:ECX.PKU[bit 3]: the support for PKRU instructions */
	eax = 7;
	ecx = 0;
	cpuid(&eax, &ebx, &ecx, &edx);
	if (!(ecx & CPUID_LEAF7_ECX_PKU_MASK)) {
		xstate_data.xstate_flag[XFEATURE_PKRU] = NOT_SUPPORT;
		return xstate_data.xstate_flag[XFEATURE_PKRU];
	}
	/* CPUID.7.0:ECX.OSPKE[bit 4]: the support for OS set CR4.PKE */
	if (!(ecx & CPUID_LEAF7_ECX_OSPKE_MASK)) {
		xstate_data.xstate_flag[XFEATURE_PKRU] = NOT_SUPPORT;
		return xstate_data.xstate_flag[XFEATURE_PKRU];
	}
	xstate_data.xstate_flag[XFEATURE_PKRU] = SUPPORT;
	xsave_mask = xsave_mask | XFEATURE_MASK_PKRU;

	return xstate_data.xstate_flag[XFEATURE_PKRU];
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

static uint32_t fill_xstate_range(int xstate_id, uint32_t xsave_mask)
{
	uint32_t eax, ebx, ecx, edx;

	eax = CPUID_LEAF_XSTATE;
	ecx = xstate_id;
	cpuid(&eax, &ebx, &ecx, &edx);
	xstate_data.xstate_size[xstate_id]=eax;
	xstate_data.xstate_offset[xstate_id]=ebx;
	xsave_mask = xsave_mask | (1 << xstate_id);

	return xsave_mask;
}

static inline void set_xstatebv(struct xsave_buffer *buffer, uint64_t bv)
{
	/* XSTATE_BV is at the beginning of xstate header. */
	*(uint64_t *)(&buffer->header) = bv;
}

static uint64_t check_cpuid_xstate(void)
{
	uint64_t xsave_mask = XFEATURE_MASK_FP | XFEATURE_MASK_SSE;

	if (!cpu_has_avx2(xsave_mask))
		printf("[SKIP] No avx2 capability, skip avx2 part xstate.\n");
	else
		xsave_mask = fill_xstate_range((int)XFEATURE_YMM, xsave_mask);

	if (!cpu_has_avx512f(xsave_mask))
		printf("[SKIP] No avx512f capability, skip avx512f part xstate.\n");
	else
		xsave_mask = fill_xstate_range((int)XFEATURE_OPMASK, xsave_mask);

	if (!cpu_has_pkeys(xsave_mask))
		printf("[SKIP] No pkeys capability, skip pkru part xstate.\n");
	else
		xsave_mask = fill_xstate_range((int)XFEATURE_PKRU, xsave_mask);

	return xsave_mask;
}

static void fill_xstate_buf(char data, unsigned char *buf, int xstate_id)
{
	uint32_t i;

	if (xstate_data.xstate_flag[xstate_id] == SUPPORT) {
		for (i = 0; i < xstate_data.xstate_size[xstate_id]; i++)
			buf[xstate_data.xstate_offset[xstate_id] + i] = data;
	}
}

/* Populate FP xstate with values by instruction. */
static inline void prepare_fp_buf(uint32_t ui32_fp)
{
	uint64_t ui64_fp;

	/* Populate ui32_fp and ui64_fp value onto FP registers stack. */
	ui64_fp = (uint64_t)ui32_fp << 32;
	asm volatile("finit");
	ui64_fp = ui64_fp + ui32_fp;
	asm volatile("fldl %0" : : "m" (ui64_fp));
	asm volatile("flds %0" : : "m" (ui32_fp));
}

/* Write PKRU xstate with values by instruction. */
static inline void wrpkru(uint32_t pkey)
{
	uint32_t ecx = 0, edx = 0;

	asm volatile(".byte 0x0f, 0x01, 0xef\n\t"
	     : : "a" (pkey), "c" (ecx), "d" (edx));
}

static void set_xstate_data(struct xsave_buffer *buf, uint32_t xsave_mask,
	uint32_t int_data)
{
	unsigned char *ptr = (unsigned char *)buf;
	/* XMM offset and size are fixed */
	uint32_t xmm_offset = 160, xmm_size = 256, i, pkru_data;
	uint8_t byte_data;

	pkru_data = int_data << 8;
	byte_data = int_data && 0xff;
	prepare_fp_buf(int_data);
	xsave(buf, XFEATURE_MASK_FP);
	xrstor(buf, XFEATURE_MASK_FP);
	/* MXCSR_MASK should set to 0x0000ffff for SSE component. */
	ptr[28]=0xff;
	ptr[29]=0xff;
	if (xstate_data.xstate_flag[XFEATURE_PKRU] == SUPPORT) {
		wrpkru(pkru_data);
		xsave(buf, XFEATURE_MASK_FP | XFEATURE_MASK_PKRU);
		xrstor(buf, XFEATURE_MASK_FP | XFEATURE_MASK_PKRU);
	}

	/* Fill XMM with specific byte data value */
	for (i = 0; i < xmm_size; i++)
		ptr[xmm_offset + i] = byte_data;

	set_xstatebv(buf, xsave_mask);
	fill_xstate_buf(byte_data, ptr, (int)XFEATURE_YMM);
	fill_xstate_buf(byte_data, ptr, (int)XFEATURE_OPMASK);
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
 * calls, so use inline function with assembly code only for fork test.
 */
static inline long long fork_test(struct xsave_buffer *buf, uint64_t xsave_mask)
{
	long long ret;

	ret = execute_syscall((int)SYS_fork, 0, 0, 0, 0, 0, 0);

	/* Save the xstates in buf */
	xsave((struct xsave_buffer *)buf, xsave_mask);

	return ret;
}

/*
 * Because xstate like XMM, YMM registers are not preserved across function
 * calls, so use inline function with assembly code only for singal test.
 */
static inline long long sig_test(struct xsave_buffer *buf, uint64_t xsave_mask,
	long long process_pid, long long SIG)
{
	long long ret;

	ret = execute_syscall((int)SYS_kill, process_pid, SIG, 0, 0, 0, 0);

	/* Save the xstates in buf2 */
	xsave((struct xsave_buffer *)buf, xsave_mask);

	return ret;
}
