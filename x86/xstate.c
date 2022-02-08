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
 * 1. In nested signal processing, the signal handling will use each signal's
 *    own xstates, and the xstates of the signal handling under test should
 *    not be changed after previous nested signal handling is completed;
 * 2. The xstates content of the process should not change after the entire
 *    nested signal handling.
 * 3. The xstates content of the child process should be the same as that of the
 *    parent process.
 * 4. The xstates content of the process should be the same across process
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

struct xsave_buffer *xstate_buf_init, *xstate_buf_compare, *xstate_buf_pollute,
		*xstate_buf_siginit;
static uint32_t xstate_size;
static uint64_t xsave_test_mask;
static pid_t process_pid;

#define CYCLE_MAX_NUM 10
#define CHANGE    1
#define NO_CHANGE 0
#define SUPPORT     1
#define NOT_SUPPORT 0
#define XSAVE_HDR_OFFSET	512
#define XSAVE_HDR_SIZE		64
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
	/* XMM xstate(160-415 bytes) offset(160) and size(256) are fixed */
	uint32_t xmm_offset = 160, xmm_size = 256, i, pkru_data;
	uint8_t byte_data;

	pkru_data = int_data << 8;
	byte_data = int_data & 0xff;
	prepare_fp_buf(int_data);
	/* Fill fp xstates values into xstate buffer (0-159 bytes) */
	xsave(buf, XFEATURE_MASK_FP);
	xrstor(buf, XFEATURE_MASK_FP);
	/* MXCSR_MASK(28-29 bytes) should set to 0xffff for SSE component. */
	ptr[28]=0xff;
	ptr[29]=0xff;
	/* Write PKRU xstate and then save into xstate buffer(PKRU offset/size). */
	if (xstate_data.xstate_flag[XFEATURE_PKRU] == SUPPORT) {
		wrpkru(pkru_data);
		xsave(buf, XFEATURE_MASK_FP | XFEATURE_MASK_PKRU);
		xrstor(buf, XFEATURE_MASK_FP | XFEATURE_MASK_PKRU);
	}

	/* Fill specific byte data value into XMM xstate buffer(160-415 bytes). */
	for (i = 0; i < xmm_size; i++)
		ptr[xmm_offset + i] = byte_data;

	/* Fill xstate-component bitmaps into xstate header buffer(416-512 bytes). */
	set_xstatebv(buf, xsave_mask);
	/* Fill specific byte data value into YMM xstate buffer(YMM offset/size). */
	fill_xstate_buf(byte_data, ptr, (int)XFEATURE_YMM);
	/* Fill specific byte data value into OPMASK xstate buffer(OPMASK offset/size). */
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
static inline long long __fork(struct xsave_buffer *buf, uint64_t xsave_mask)
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

static inline bool __validate_xstate_regs(struct xsave_buffer *buf0)
{
	int ret;

	xrstor(buf0, xsave_test_mask);
	xsave(xstate_buf_compare, xsave_test_mask);
	ret = memcmp(&buf0->bytes[0], &xstate_buf_compare->bytes[0], xstate_size);
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
	uint32_t usr1_pollute_data = 0x11223344;

	if (signum == SIGUSR1) {
		set_xstate_data(xstate_buf_siginit, xsave_test_mask, usr1_pollute_data);
		/* Xstate of SIGUSR1 handling is not affected by SIGUSR2 handling */
		xrstor(xstate_buf_siginit, xsave_test_mask);
		sig_test(xstate_buf_compare, xsave_test_mask, process_pid, SIGUSR2);

		if (memcmp(&xstate_buf_siginit->bytes[0], &xstate_buf_compare->bytes[0],
			xstate_size))
			printf("[FAIL]\tSIGUSR1 xstate changed after SIGUSR2 handling\n");
		else
			printf("[PASS]\tSIGUSR1 xstate is same after SIGUSR2 handling\n");
	}
}

static void usr2_nested_pollute_xstate_handler(int signum, siginfo_t *info,
	void *__ctxp)
{
	uint32_t usr2_pollute_data = 0x1020304;

	if (signum == SIGUSR2) {
		set_xstate_data(xstate_buf_pollute, xsave_test_mask, usr2_pollute_data);
		xrstor(xstate_buf_pollute, xstate_size);
	}
}

static int test_xstate_sig_handle(void)
{
	int cycle_num, fail_num = 0;
	uint32_t fill_data = 0x1f2f3f4f;

	sethandler(SIGUSR1, usr1_pollute_xstate_handler, 0);
	sethandler(SIGUSR2, usr2_nested_pollute_xstate_handler, 0);
	printf("[RUN]\tCheck xstate around signal handling test.\n");
	set_xstate_data(xstate_buf_init, xsave_test_mask, fill_data);
	validate_xstate_regs_same(xstate_buf_init);

	process_pid = getpid();
	for (cycle_num = 1; cycle_num <= CYCLE_MAX_NUM; cycle_num++) {
		xrstor(xstate_buf_init, xsave_test_mask);
		sig_test(xstate_buf_compare, xsave_test_mask, process_pid, SIGUSR1);
		if (memcmp(&xstate_buf_init->bytes[0], &xstate_buf_compare->bytes[0],
			xstate_size)) {
			fail_num++;
			printf("[FAIL]\tProcess xstate is not same after signal handling\n");
			break;
		}
	}
	if(fail_num == 0)
		printf("[PASS]\tProcess xstate is same after signal handling.\n");
	clearhandler(SIGUSR1);
	clearhandler(SIGUSR2);
	/* Keep initialized xstate_buf_init and clear other buffers for comparison. */
	memset(xstate_buf_compare, 0, xstate_size);
	memset(xstate_buf_pollute, 0, xstate_size);
	memset(xstate_buf_siginit, 0, xstate_size);

	return 0;
}

static int child_test_process_switch(void)
{
	pid_t grandchild;
	int status;
	uint32_t grand_change_data = 0x3f3f3f3f;

	xrstor(xstate_buf_init, xsave_test_mask);
	/* Child process performs process switching by forking grandchild process */
	grandchild = __fork(xstate_buf_compare, xsave_test_mask);
	if (grandchild < 0)
		fatal_error("grandchild fork failed\n");
	else if (grandchild == 0) {
		/* Fork syscall succeeded, now in the grandchild */
		printf("\tGrandchild pid:%d changed it's own xstates\n", getpid());
		set_xstate_data(xstate_buf_pollute, xsave_test_mask, grand_change_data);
		xrstor(xstate_buf_pollute, xsave_test_mask);
		_exit(0);
	} else {
		/* Fork syscall succeeded, still in the child process. */
		if (waitpid(grandchild, &status, 0) != grandchild ||
			!WIFEXITED(status))
			fatal_error("Grandchild exit with error status");
		else {
			printf("\tChild:%d check xstate with mask:0x%lx after process switching\n",
				getpid(), xsave_test_mask);
			if (memcmp(&xstate_buf_init->bytes[0], &xstate_buf_compare->bytes[0],
				xstate_size))
				return CHANGE;
		}
	}
	return NO_CHANGE;
}

static int test_xstate_fork(void)
{
	pid_t child;
	int status, cycle_num;

	/* Affinitize to the same CPU0 to force process switching. */
	affinitize_cpu0();
	printf("\tParent pid:%d\n", getpid());
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
		printf("[RUN]\tCheck xstate of child processs in process switching\n");
		if (memcmp(&xstate_buf_init->bytes[0], &xstate_buf_compare->bytes[0],
			xstate_size))
			printf("[FAIL]\tXstate of child process is not same as xstate of parent\n");
		else
			printf("[PASS]\tXstate of child process is same as xstate of parent\n");
		for (cycle_num = 1; cycle_num <= CYCLE_MAX_NUM; cycle_num++) {
			if(child_test_process_switch()) {
				printf("[FAIL]\tChild xstate changed after process switching.\n");
				_exit(0);
			}
		}
		printf("[PASS]\tChild xstate is same after process swiching.\n");
	} else {
		/* Fork syscall succeeded, now in the parent. */
		if (waitpid(child, &status, 0) != child || !WIFEXITED(status))
			fatal_error("Child exit with error status");
	}

	return 0;
}

static void init_xstate_buf(void)
{
	xstate_buf_init = alloc_xbuf(xstate_size);
	xstate_buf_compare = alloc_xbuf(xstate_size);
	xstate_buf_pollute = alloc_xbuf(xstate_size);
	xstate_buf_siginit = alloc_xbuf(xstate_size);
}

static void free_xstate_buf(void)
{
	free(xstate_buf_init);
	free(xstate_buf_compare);
	free(xstate_buf_pollute);
	free(xstate_buf_siginit);
}

int main(void)
{
	/* Check hardware availability for xsave at first */
	check_cpuid_xsave();
	xstate_size = get_xstate_size();
	/* Check CPU capability by CPU id and set CPU capability flags */
	xsave_test_mask = check_cpuid_xstate();

	init_xstate_buf();

	test_xstate_sig_handle();

	test_xstate_fork();

	free_xstate_buf();

	return 0;
}
