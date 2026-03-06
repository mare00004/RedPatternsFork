#ifndef CLI_H
#define CLI_H
#include "sim_types.h"

#ifdef __cplusplus
extern "C" {
#endif

int parseArguments(int argc, char **argv, SimConfig *cfg);

#ifdef __cplusplus
}
#endif

#endif
