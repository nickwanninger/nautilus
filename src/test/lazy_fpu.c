/*
 * This file is part of the Nautilus AeroKernel developed
 * by the Hobbes and V3VEE Projects with funding from the
 * United States National  Science Foundation and the Department of Energy.
 *
 * The V3VEE Project is a joint project between Northwestern University
 * and the University of New Mexico.  The Hobbes Project is a collaboration
 * led by Sandia National Laboratories that includes several national
 * laboratories and universities. You can find out more at:
 * http://www.v3vee.org  and
 * http://xstack.sandia.gov/hobbes
 *
 * Copyright (c) 2017, The V3VEE Project  <http://www.v3vee.org>
 *                     The Hobbes Project <http://xstack.sandia.gov/hobbes>
 * All rights reserved.
 *
 * Author: Brian Richard Tauro <btauro@hawk.iit.edu>
 *         Nick Wanninger <nwanninger@hawk.iit.edu>
 *
 * This is free software.  You are permitted to use,
 * redistribute, and modify it as specified in the file "LICENSE.txt".
 */

/* This test only makes sense if the FPU_IRQ_SAVE is on */
#ifdef NAUT_CONFIG_FPU_IRQ_SAVE

#include <nautilus/fpu_irq.h>
#include <nautilus/nautilus.h>
#include <nautilus/shell.h>

#define DO_PRINT 0

#if DO_PRINT
#define PRINT(...) nk_vc_printf(__VA_ARGS__)
#else
#define PRINT(...)
#endif

#define ARRAY_SIZE 32
// Duration in nano seconds
#define NANOSECONDS 1000000000ULL

void init_array(float *i1, float *i2, float *res, int size) {
  for (int i = 0; i < size; i++) {
    i1[i] = i + 1;
    i2[i] = i + 2;
    res[i] = 0;
  }
}
void destroy_array(float *i1, float *i2, float *i3) {
  free(i1);
  free(i2);
  free(i3);
}

void print_array(char *array_name, float *array, int size) {
  nk_vc_printf("\n%s = { ", array_name);
  for (int i = 0; i < size; i++) {
    nk_vc_printf("%f ", array[i]);
  }
  nk_vc_printf("}\n");
}

/* Returns if it failed at any point, not how many failures. */
int check_array(char *tname, float *result, float *inp1, float *inp2,
		int size) {
  bool_t check = true;
  for (int i = 0; i < size; i++) {
    if (result[i] != (inp1[i] + inp2[i])) {
      check = false;
      break;
    }
  }
  if (!check) {
    nk_vc_printf("%s failed \n", tname);
    print_array("I1", inp1, size);
    print_array("I2", inp2, size);
    print_array("Result", result, size);
    return 1;
  }
  return 0;
}

void test_AVX(float *result, float *inp1, float *inp2, int number_of_elements) {
  long array_size = number_of_elements * sizeof(float);
  __asm__ __volatile__(
      "movq       $0, %%rcx                \n\t"  // Loop counter set to 0
      "movq       %3, %%rax                \n\t"  // Set Array size in rax
      "lfpu_loop_avx:                                  \n\t"
      "vmovaps    (%1,%%rcx), %%ymm0          \n\t"  // Load 8 elements from
						     // inp1
      "vaddps     (%2,%%rcx), %%ymm0, %%ymm0  \n\t"  // Add  8 elements from
						     // inp2
      "vmovaps    %%ymm0, (%0,%%rcx)          \n\t"  // Store result in result
      "addq       $0x20,  %%rcx               \n\t"  // 8 elements *
						     // sizeof(float) bytes = 32
						     // (0x20)
      "cmpq       %%rax, %%rcx                   \n\t"	// compare if we reached
							// end of array
      "jb         lfpu_loop_avx"			// Loop"
      :							// Outputs
      : "r"(result), "r"(inp1), "r"(inp2), "r"(array_size)  // Inputs
      : "%rcx", "%ymm0", "%rax",
	"memory"  // Modifies RAX, RCX, YMM0, and memory
  );
}
// WIll work only on SKYlake, ICE lake nodes or higher intel versions
void test_AVX2(float *result, float *inp1, float *inp2,
	       int number_of_elements) {
  long array_size = number_of_elements * sizeof(float);
  __asm__ __volatile__(
      "movq       $0, %%rcx                \n\t"  // Loop counter set to 0
      "movq       %3, %%rax                \n\t"  // Set Array size in rax
      "lfpu_loop_avx2:                                  \n\t"
      "vmovdqu32  (%1, %%rcx), %%zmm0 \n\t"
      "vmovdqu32  (%2, %%rcx), %%zmm1 \n\t"
      "vaddps     %%zmm0, %%zmm1, %%zmm0  \n\t"	 // Add  8 elements from inp2
      "vmovdqu32  %%zmm0, (%0,%%rcx) \n\t"
      "addq       $0x40,  %%rcx               \n\t"  // 8 elements *
						     // sizeof(float) bytes = 32
						     // (0x20)
      "cmpq       %%rax, %%rcx                   \n\t"	// compare if we reached
							// end of array
      "jb         lfpu_loop_avx2"			// Loop"
      :							// Outputs
      : "r"(result), "r"(inp1), "r"(inp2), "r"(array_size)  // Inputs
      : "%rcx", "%zmm0", "%rax",
	"memory"  // Modifies RAX, RCX, YMM0, and memory
  );
}
/*
  for (int i = 0; i < number_of_elements; i++) {
      result[i] = inp1[i] + inp2[i];
  }
*/
void test_SSE(float *result, float *inp1, float *inp2, int number_of_elements) {
#if 1
  long array_size = number_of_elements * sizeof(float);
  __asm__ __volatile__(
      "movq       $0, %%rcx                \n\t"  // Loop counter set to 0
      "movq       %3, %%rax                \n\t"  // Set Array size in rax
      "lfpu_loop_sse:                                  \n\t"
      "movaps    (%1,%%rcx), %%xmm1          \n\t"  // Load 4 elements from inp1
      "movaps    (%2,%%rcx), %%xmm2          \n\t"  // Load 4 elements from inp1
      "addps     %%xmm1, %%xmm2              \n\t"  // Add  4 elements from inp2
      "movaps    %%xmm2, (%0,%%rcx)          \n\t"  // Store result in result
      "addq       $0x10,  %%rcx               \n\t"  // 4 elements *
						     // sizeof(float) bytes = 16
						     // (0x10)
      "cmpq       %%rax, %%rcx                   \n\t"	// compare if we reached
							// end of array
      "jb         lfpu_loop_sse"			// Loop"
      :							// Outputs
      : "r"(result), "r"(inp1), "r"(inp2), "r"(array_size)  // Inputs
      : "%rcx", "%xmm1", "%xmm2", "%rax",
	"memory"  // Modifies RAX, RCX, YMM0, and memory
  );
#endif
}
void test_X87(float *result, float *inp1, float *inp2, int number_of_elements) {
#if 1
  long array_size = number_of_elements * sizeof(float);
  __asm__ __volatile__(
      "movq       $0, %%rcx                \n\t"  // Loop counter set to 0
      "movq       %3, %%rax                \n\t"  // Set Array size in rax
      "lfpu_loop_x87:                                  \n\t"
      "flds    (%1,%%rcx)          \n\t"	    // Load 1 elements from inp1
      "fadds   (%2,%%rcx)          \n\t"	    // Load 1 elements from inp1
      "fstps   (%0,%%rcx)          \n\t"	    // Load 1 elements from inp1
      "addq       $0x4,  %%rcx               \n\t"  // 1 elements *
						    // sizeof(float) bytes = 4
						    // (0x10)
      "cmpq       %%rax, %%rcx                   \n\t"	// compare if we reached
							// end of array
      "jb         lfpu_loop_x87"			// Loop"
      :							// Outputs
      : "r"(result), "r"(inp1), "r"(inp2), "r"(array_size)  // Inputs
      : "%rcx", "%xmm1", "%xmm2", "%rax",
	"memory"  // Modifies RAX, RCX, YMM0, and memory
  );
#endif
}

int benchmark(char *name, void (*test)(float *, float *, float *, int),
	      int array_length) {
  float *input1 = malloc(array_length * sizeof(float));
  float *input2 = malloc(array_length * sizeof(float));
  float *result = malloc(array_length * sizeof(float));
  // Initialization of float arrays
  init_array(input1, input2, result, array_length);

  test(result, input1, input2, array_length);

  int failed = check_array(name, result, input1, input2, array_length);

  destroy_array(input1, input2, result);

  return failed;
}

int parse_args(char *buf) {
  // DEFAULT
  int time_sec = 5;

  if (sscanf(buf, "lfpu %lu", &time_sec) == 1) {
    nk_vc_printf("Running FPU tests for %d seconds\n", time_sec);
    return time_sec;
  }

  nk_vc_printf("Default 5 seconds\n");
  return time_sec;
}

static int handle_lazy_fpu(char *buf, void *pvt) {
  nk_fpu_irq_begin_session();
  /* reset this metric */
  uint64_t duration = parse_args(buf) * NANOSECONDS;
  uint64_t end = nk_sched_get_realtime() + duration;
  uint64_t count = 0;
  uint64_t failed = 0;
  while (nk_sched_get_realtime() < end) {
    count++;
    failed += benchmark("X87", &test_X87, ARRAY_SIZE);

    count++;
    failed += benchmark("SSE", &test_SSE, ARRAY_SIZE);
    // benchmark("AVX",  &test_AVX,  ARRAY_SIZE);
    // benchmark("AVX2", &test_AVX2, ARRAY_SIZE);
  }

  int lazy = 0;
#ifdef NAUT_CONFIG_FPU_IRQ_SAVE_LAZY
  lazy = 1;
#endif

  nk_fpu_irq_session_t *session = nk_fpu_irq_end_session();

  nk_dump_fpu_irq_session(session);
  nk_free_fpu_irq_session(session);

  /*
nk_vc_printf("[%s] %llu failed of %llu. (%f%%) fpu states allocated: %llu\n",
	 lazy ? "lazy" : "eager", failed, count,
	 (float)failed / (float)count, count_fpu_state_alloc);
			   */

  return 0;
}

static struct shell_cmd_impl lazy_fpu_tests = {
    .cmd = "lfpu",
    .help_str = "lfpu (lazy fpu tests x87, SSE, AVX, AVX2)",
    .handler = handle_lazy_fpu,
};
nk_register_shell_cmd(lazy_fpu_tests);


static int handle_ipifpu(char *buf, void *pvt) {
  nk_fpu_irq_begin_session();
  /* reset this metric */
  uint64_t duration = parse_args(buf) * NANOSECONDS;
  uint64_t end = nk_sched_get_realtime() + duration;
  uint64_t count = 0;
  uint64_t failed = 0;
  while (nk_sched_get_realtime() < end) {
    count++;
    failed += benchmark("SSE", &test_SSE, ARRAY_SIZE);
  }

  nk_fpu_irq_session_t *session = nk_fpu_irq_end_session();
  nk_dump_fpu_irq_session(session);
  nk_free_fpu_irq_session(session);

  return 0;
}

static struct shell_cmd_impl ipi_fpu_tests = {
    .cmd = "ipifpu",
    .help_str = "ipifpu t",
    .handler = handle_ipifpu,
};
nk_register_shell_cmd(ipi_fpu_tests);

#endif
