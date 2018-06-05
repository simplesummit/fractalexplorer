/*

Header file for communication

*/

#ifndef __COMMLOOP_H__
#define __COMMLOOP_H__

int diagnostics_history_idx;
node_diagnostics_t * node_diagnostics_history;

// different loops
void master_loop();
void slave_loop();


#endif

