
#include "fractalexplorer.h"

#include <sys/time.h>

double get_time() {

    struct timeval cur;
    gettimeofday(&cur, NULL);

    double time_since_epoch = (double)cur.tv_sec + cur.tv_usec / 1000000.0;

    return time_since_epoch;
}


