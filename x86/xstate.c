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
#include <sys/wait.h>
#include <sys/syscall.h>

#include "xstate.h"

#define CYCLE_MAX_NUM 10

struct xsave_buffer *xstate_buf0, *xstate_buf1, *xstate_buf2, *xstate_buf3;
static uint32_t xstate_size;
static uint64_t xsave_test_mask;
unsigned long long child_pid;

void free(void *ptr);
static void test_xstate_buffer()
{
	printf("[RUN]\tStashed xstate buffer should same as restored xstate.\n");
	set_xstate_data(xstate_buf0, xsave_test_mask);
	dump_buffer((unsigned char *)xstate_buf0, xstate_size);

	xrstor(xstate_buf0, xsave_test_mask);
	xsave(xstate_buf1, xsave_test_mask);
	if (compare_buf((unsigned char *)xstate_buf0,
			(unsigned char *)xstate_buf1, xstate_size))
		printf("[FAIL]\tStashed xstate is not same as restored xstate.\n");
	else
		printf("[PASS]\tStashed xstate is same as restored xstate.\n");
}

static void usr1_pollute_xstate_handler(int signum, siginfo_t *info,
	void *__ctxp)
{
	if (signum == SIGUSR1) {
		set_xstate_data(xstate_buf2, xsave_test_mask);
		/* Xstate of SIGUSR1 handling is not affected by SIGUSR2 handling */
		xsave_syscall_test((unsigned char *)xstate_buf2,
			(unsigned char *)xstate_buf3, xsave_test_mask, (int)SYS_kill,
			child_pid, SIGUSR2, 0, 0, 0, 0);

		if (compare_buf((unsigned char *)xstate_buf2,
			(unsigned char *)xstate_buf3, xstate_size))
			printf("[FAIL]\tSIGUSR1 xstate changed after SIGUSR2 handling\n");
		else
			printf("[PASS]\tSIGUSR1 xstate is same after SIGUSR2 handling\n");
	}
}

static void usr2_nested_pollute_xstate_handler(int signum, siginfo_t *info,
	void *__ctxp)
{
	if (signum == SIGUSR2) {
		set_xstate_data(xstate_buf3, xsave_test_mask);
		xrstor(xstate_buf3, xstate_size);
	}
}

static int test_xstate_sig_hanlde(void)
{
	pid_t child;
	int status, cycle_num;

	memset(xstate_buf1, 0, xstate_size);
	/* Use child process testing to avoid exceptions blocking the next test */
	child = fork();
	if (child < 0)
		fatal_error("Create child pid failed");
	else if	(child == 0) {
		printf("[RUN]\tCheck xstate around signal handling test.\n");
		child_pid = getpid();

		for (cycle_num = 1; cycle_num <= CYCLE_MAX_NUM; cycle_num++) {
			xsave_syscall_test((unsigned char *)xstate_buf0,
				(unsigned char *)xstate_buf1, xsave_test_mask, (int)SYS_kill,
				child_pid, SIGUSR1, 0, 0, 0, 0);
			if (compare_buf((unsigned char *)xstate_buf0,
					(unsigned char *)xstate_buf1, xstate_size)) {
				printf("[FAIL]\tProcess xstate is not same after signal handling\n");
				_exit(0);
			}
		}
		printf("[PASS]\tProcess xstate is same after signal handling.\n");
		_exit(0);
	}

	if (child) {
		if (waitpid(child, &status, 0) != child || !WIFEXITED(status))
			fatal_error("Child quit unexpectedly\n");
	}

	return 0;
}

static int test_process_switch(void)
{
	pid_t grandchild;
	int status;

	/* Child process performs process switching by forking grandchild process */
	grandchild = xsave_syscall_test((unsigned char *)xstate_buf0,
			(unsigned char *)xstate_buf3, xsave_test_mask,
			(int)SYS_fork, 0, 0, 0, 0, 0, 0);
	if (grandchild < 0)
		fatal_error("grandchild fork failed\n");
	if (grandchild == 0) {
		/* fork syscall succeeded, now in the grandchild */
		printf("\tGrandchild pid:%d changed it's own xstates\n", getpid());
		set_xstate_data(xstate_buf2, xsave_test_mask);
		xrstor(xstate_buf2, xsave_test_mask);
		_exit(0);
	}
	if (grandchild > 0) {
		/* fork syscall succeeded, still in the first child. */
		if (waitpid(grandchild, &status, 0) != grandchild ||
			!WIFEXITED(status))
			fatal_error("Grandchild exit with error status");
		else {
			//xsave(xstate_buf3, xsave_test_mask);
			printf("\tChild:%d check xstate with mask:0x%lx after process switching\n",
				getpid(), xsave_test_mask);
			if (compare_buf((unsigned char *)xstate_buf0,
					(unsigned char *)xstate_buf3, xstate_size)) {
				return 1;
			}
		}
	}
	return 0;
}

static int test_xstate_fork(void)
{
	pid_t child;
	int status, cycle_num;

	/* Affinitize to the same CPU0 to force process switching. */
	affinitize_cpu0();
	printf("\tParent pid:%d\n", getpid());
	memset(xstate_buf1, 0, xstate_size);
	memset(xstate_buf2, 0, xstate_size);
	memset(xstate_buf3, 0, xstate_size);
	/*
	 * Xrstor the xstate_buf0 and call syscall assembly instruction, then
	 * save the xstate to xstate_buf1 in child process for comparison.
	 */
	xrstor(xstate_buf0, xsave_test_mask);
	child = xsave_syscall_test((unsigned char *)xstate_buf0,
			(unsigned char *)xstate_buf1, xsave_test_mask,
			(int)SYS_fork, 0, 0, 0, 0, 0, 0);
	if (child < 0)
		/* fork syscall failed */
		fatal_error("fork failed");
	if (child == 0) {
		/* fork syscall succeeded, now in the child. */
		printf("[RUN]Check xstate of child processs in process switching\n");
		if (compare_buf((unsigned char *)xstate_buf0,
			(unsigned char *)xstate_buf1, xstate_size))
			printf("[FAIL]\tXstate of child process is not same as xstate of parent\n");
		else
			printf("[PASS]\tXstate of child process is same as xstate of parent\n");
		for (cycle_num = 1; cycle_num <= CYCLE_MAX_NUM; cycle_num++) {
			if(test_process_switch()) {
				printf("[FAIL]\tChild xstate changed after process swiching.\n");
				_exit(0);
			}
		}

		printf("[PASS]\tChild xstate is same after process swiching.\n");
	}

	if (child > 0) {
		/* fork syscall succeeded, now in the parent. */
		if (waitpid(child, &status, 0) != child || !WIFEXITED(status))
			fatal_error("Child exit with error status");
	}

	return 0;
}

static void free_xstate_buf(void)
{
	free(xstate_buf0);
	free(xstate_buf1);
	free(xstate_buf2);
	free(xstate_buf3);
}

int main(void)
{
	/* Check hardware availability for xsave at first */
	check_cpuid_xsave();
	xstate_size = get_xstate_size();
	/* Check CPU capability by CPU id and set CPU capability flags */
	xsave_test_mask = check_cpuid_xstate();
	xstate_buf0 = alloc_xbuf(xstate_size);
	xstate_buf1 = alloc_xbuf(xstate_size);
	xstate_buf2 = alloc_xbuf(xstate_size);
	xstate_buf3 = alloc_xbuf(xstate_size);
	test_xstate_buffer();

	sethandler(SIGUSR1, usr1_pollute_xstate_handler, 0);
	sethandler(SIGUSR2, usr2_nested_pollute_xstate_handler, 0);
	test_xstate_sig_hanlde();

	test_xstate_fork();

	clearhandler(SIGUSR1);
	clearhandler(SIGUSR2);
	free_xstate_buf();

	return 0;
}
