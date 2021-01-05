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

#ifndef __FPU_IRQ_H__
#define __FPU_IRQ_H__

#include <nautilus/naut_types.h>
#include <nautilus/hashtable.h>


#ifdef NAUT_CONFIG_FPU_IRQ_SAVE


/*
 * An "FPU IRQ Session" encapsulates a range of time and allows you to
 * record a histogram of each address which uses floating point (when
 * lazy fpu saving is enabled).
 */
struct nk_fpu_irq_session {
	/* Mapping from instruction pointers to count */
	struct nk_hashtable *histogram;
};
typedef struct nk_fpu_irq_session nk_fpu_irq_session_t;

/* Print an irq session to the virtual console */
void nk_dump_fpu_irq_session(nk_fpu_irq_session_t *session);
void nk_free_fpu_irq_session(nk_fpu_irq_session_t *session);

/* Begin a session globally. Returns 0 on success and -1 if a session is already running */
int nk_fpu_irq_begin_session(void);
/* Return the current global session, if any. Stops recording */
nk_fpu_irq_session_t *nk_fpu_irq_end_session(void);
/* Resume a session that was previously ended */
int nk_fpu_irq_resume_session(nk_fpu_irq_session_t *session);
/* Tell the session that an instruction used floating point */
void nk_fpu_irq_record_usage(addr_t ip);


void nk_fpu_irq_init(void);


#endif

#endif
