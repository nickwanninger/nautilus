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
 * http://xtack.sandia.gov/hobbes
 *
 * Copyright (c) 2015, Kyle C. Hale <kh@u.northwestern.edu>
 * Copyright (c) 2015, The V3VEE Project  <http://www.v3vee.org>
 *                     The Hobbes Project <http://xstack.sandia.gov/hobbes>
 * All rights reserved.
 *
 * Author: Nick Wanninger <nwanninger@hawk.iit.edu>
 *         Brian Richard Tauro <btauro@hawk.iit.edu>
 *
 * This is free software.  You are permitted to use,
 * redistribute, and modify it as specified in the file "LICENSE.txt".
 */
#include <nautilus/fpu_irq.h>
#include <nautilus/idt.h>
#include <nautilus/mm.h>
#include <nautilus/scheduler.h>
#include <nautilus/spinlock.h>
#include <nautilus/vc.h>
#ifdef NAUT_CONFIG_PROVENANCE
#include <nautilus/provenance.h>
#endif

#ifdef NAUT_CONFIG_USE_CLANG
#define NOOPT __attribute__((optnone))
#else
#define NOOPT __attribute__((optimize("O0")))
#endif

static uint_t hash_long(addr_t val) { return val; }
static int nm_hash_eq_fn(addr_t key1, addr_t key2) { return key1 == key2; }

static spinlock_t session_lock = 0;
static nk_fpu_irq_session_t *current_session = NULL;

#define FPU_STATE_SIZE (4096)
#define FPU_BUFFERS_COUNT (32)
static spinlock_t fpu_buffers_lock = 0;
static struct {
  long uses;
  void *page;
} fpu_buffers[NAUT_CONFIG_FPU_IRQ_SAVE_BUFFER_COUNT];

void nk_fpu_irq_init(void) {
  for (int i = 0; i < FPU_BUFFERS_COUNT; i++) {
    fpu_buffers[i].uses = 0;
    fpu_buffers[i].page = malloc(FPU_STATE_SIZE);
  }
}

static void *NOOPT get_fpu_buffer(void) {
  uint8_t flags = spin_lock_irq_save(&fpu_buffers_lock);
  void *buf = NULL;

  for (int i = 0; i < FPU_BUFFERS_COUNT; i++) {
    if (fpu_buffers[i].page != NULL) {
      buf = fpu_buffers[i].page;
      fpu_buffers[i].uses++;
      fpu_buffers[i].page = NULL;
      break;
    }
  }

  if (buf == NULL) panic("No fpu buffer available");
  
  spin_unlock_irq_restore(&fpu_buffers_lock, flags);

  return buf;
}

static void NOOPT release_fpu_buffer(void *buf) {
  uint8_t flags = spin_lock_irq_save(&fpu_buffers_lock);
  int found_spot = 0;
  for (int i = 0; i < FPU_BUFFERS_COUNT; i++) {
    if (fpu_buffers[i].page == NULL) {
      fpu_buffers[i].page = buf;
      found_spot = 1;
      break;
    }
  }
  if (!found_spot) panic("No space for fpu buffer");
  spin_unlock_irq_restore(&fpu_buffers_lock, flags);
}

void nk_fpu_irq_record_usage(addr_t key) {
  /*
   * Take an IRQ lock. The critical section is pretty fast (and this is a debug
   * feature, so...).
   * TODO: in order to save time, we could do an atomic check on
   * `current_session` and avoid taking the lock just to check if there is a
   * session.
   */
  uint8_t flags = spin_lock_irq_save(&session_lock);

  if (current_session != NULL) {
    if (nk_htable_search(current_session->histogram, key) == 0) {
      nk_htable_insert(current_session->histogram, key, 1);
    } else {
      nk_htable_inc(current_session->histogram, key, 1);
    }
  }

  spin_unlock_irq_restore(&session_lock, flags);
}

static nk_fpu_irq_session_t *allocate_session(void) {
  nk_fpu_irq_session_t *session = malloc(sizeof(*session));
  session->histogram = nk_create_htable(1, hash_long, nm_hash_eq_fn);
  return session;
}

void nk_free_fpu_irq_session(nk_fpu_irq_session_t *session) {
  /* Don't free the keys or values, they are just ints */
  nk_free_htable(session->histogram, 0, 0);
  free(session);
}

void nk_dump_fpu_irq_session(nk_fpu_irq_session_t *session) {
  const char *type = "eager";
#ifdef NAUT_CONFIG_FPU_IRQ_SAVE_LAZY
  type = "lazy";
#endif

  nk_vc_printf("======= FPU IRQ Session Dump: ======= (%s saving)\n", type);
  if (nk_htable_count(session->histogram) > 0) {
    struct nk_hashtable_iter *iter = nk_create_htable_iter(session->histogram);
    do {
      addr_t ip = nk_htable_get_iter_key(iter);
      unsigned long count = nk_htable_get_iter_value(iter);
      const char *symbol_name = "??"; /* default, filled out by provenance */
#ifdef NAUT_CONFIG_PROVENANCE
      provenance_info *prov = nk_prov_get_info((uint64_t)ip);
      if (prov != NULL) {
	symbol_name = prov->symbol;
	/* AFAIK, it's okay to just free this, as it's a container that points
	 * to static strings
	 * TODO: make sure with Nanda. (maybe poke them to make a free api)
	 */
	free(prov);
      }
#endif
      nk_vc_printf("addr: %p, count: %3d in '%s'\n", ip, count, symbol_name);
    } while (nk_htable_iter_advance(iter) != 0);

    nk_destroy_htable_iter(iter);
  } else {
    nk_vc_printf("\nNo data to display...\n\n");
  }
  nk_vc_printf("=====================================\n");
}

int nk_fpu_irq_resume_session(nk_fpu_irq_session_t *session) {
  int res = 0;
  uint8_t flags = spin_lock_irq_save(&session_lock);
  if (current_session == NULL) {
    current_session = session;
  } else {
    /* A session is currently going. Don't allocate a new one. */
    res = -1;
  }
  spin_unlock_irq_restore(&session_lock, flags);

  return res;
}

int nk_fpu_irq_begin_session(void) {
  int res = 0;
  uint8_t flags = spin_lock_irq_save(&session_lock);
  if (current_session == NULL) {
    current_session = allocate_session();
  } else {
    /* A session is currently going. Don't allocate a new one. */
    res = -1;
  }
  spin_unlock_irq_restore(&session_lock, flags);

  return res;
}

nk_fpu_irq_session_t *nk_fpu_irq_end_session(void) {
  nk_fpu_irq_session_t *session = NULL;
  /* Take an IRQ lock. The critical section is pretty fast, and this is a debug
   * feature, so. */
  uint8_t flags = spin_lock_irq_save(&session_lock);
  /* TODO: since this is just a pointer, we could probaby do an atomic operation
   * to remove it. */
  session = current_session;
  current_session = NULL;
  spin_unlock_irq_restore(&session_lock, flags);

  return session;
}

void NOOPT nk_thread_push_irq_frame(struct thread_debug_fpu_frame *frame) {
  ASSERT(sizeof(struct thread_debug_fpu_frame) == 48);

  nk_thread_t *t = get_cur_thread();
  ulong_t cr0 = read_cr0();

  frame->prev = t->irq_fpu_stack;
  frame->old_cr0 = cr0;
  frame->state = NULL;

#ifndef NAUT_CONFIG_FPU_IRQ_SAVE_LAZY
  // allocate a buffer for the FPU state
  frame->state = get_fpu_buffer();
  ASSERT(frame->state != NULL);
  asm volatile("fxsave64 (%0);" ::"r"(frame->state));
  /* TODO: maybe re-init? Not sure if we need to worry about state leakage */
#endif

#if defined(NAUT_CONFIG_FPU_IRQ_SAVE_LAZY) || \
    defined(NAUT_CONFIG_FPU_IRQ_SAVE_RECORD)
  /* Disable floating point for now... Re-enabled upon
   * use or in the pop_irq_frame function
   */
  write_cr0(read_cr0() | CR0_TS);
#endif
  // "append" the fpu state onto the stack in the thread.
  t->irq_fpu_stack = frame;
}

void NOOPT nk_thread_pop_irq_frame(void) {
  nk_thread_t *t = get_cur_thread();

  struct thread_debug_fpu_frame *f = t->irq_fpu_stack;

  if (f != NULL) {
    /* Restore the CR0 from before */
    write_cr0(f->old_cr0);
    // Pop the entry off the list
    t->irq_fpu_stack = f->prev;
    if (f->state != NULL) {
      asm volatile("fxrstor64 (%0);" ::"r"(f->state));
      release_fpu_buffer(f->state);
      f->state = NULL;
    }
  }
}

int NOOPT nk_fpu_irq_nm_handler(excp_entry_t *excp, excp_vec_t vector,
				addr_t unused) {
  // reenable the FPU
  write_cr0(read_cr0() & ~CR0_TS);

  nk_thread_t *t = get_cur_thread();
  struct thread_debug_fpu_frame *frame = t->irq_fpu_stack;

#ifdef NAUT_CONFIG_FPU_IRQ_SAVE_RECORD
  nk_fpu_irq_record_usage((addr_t)excp->rip);
#endif

  if (frame != NULL) {
    /* save the FPU state into a buffer in the frame */
    if (frame->state == NULL)
      frame->state = get_fpu_buffer();
    /* Save into the buffer */
    asm volatile("fxsave64 (%0);" ::"r"(frame->state));
  }

  return 0;
}

int NOOPT nk_thread_fpu_irq_save_trampoline(
    excp_entry_t *excp, int irq, void *state,
    int (*handler)(excp_entry_t *, excp_vec_t, void *)) {
  struct thread_debug_fpu_frame f;

  nk_thread_push_irq_frame(&f);
  int res = handler(excp, irq, state);
  nk_thread_pop_irq_frame();
  return res;
}

struct nk_thread *NOOPT nk_thread_fpu_irq_need_resched(void) {
  struct thread_debug_fpu_frame f;
  nk_thread_push_irq_frame(&f);
  struct nk_thread *t = nk_sched_need_resched();
  nk_thread_pop_irq_frame();
  return t;
}
