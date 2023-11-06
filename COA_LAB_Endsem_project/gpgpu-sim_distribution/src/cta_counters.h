#ifndef CTA_COUNTERS_H
#define CTA_COUNTERS_H

// #define COUNTER_SIZE 1024

// extern int cta_count[COUNTER_SIZE];

extern int num_warps;
extern int* warp_insts_issued;
extern int* cta_status;
// bool is_first_cta_in_kernel=false;

#endif