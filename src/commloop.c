/*

communication stuff between head and child nodes

*/


#include "fractalexplorer.h"
#include "commloop.h"


diagnostics_t * diagnostics_history = NULL;


void master_loop() {
    diagnostics_history = malloc(sizeof(diagnostics_t) * NUM_DIAGNOSTICS_SAVE);
    
    int i, j;
    for (i = 0; i < NUM_DIAGNOSTICS_SAVE; ++i) {
        diagnostics_history[i].node_information = (node_diagnostics_t *)malloc(sizeof(node_diagnostics_t) * world_size);
        for (j = 0; j < world_size; ++j) {
            diagnostics_history[i].node_information[j]
        }
    }

}


void slave_loop() {

}



