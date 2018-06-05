/*

Header file for communication

*/

#ifndef __COMMLOOP_H__
#define __COMMLOOP_H__

int diagnostics_history_idx;
diagnostics_t * diagnostics_history;


int n_frames;

// different loops
void master_loop();
void slave_loop();


#endif

