#ifndef CONFIG_H
#define CONFIG_H

#include "sim_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void setDefaults(SimConfig *c);
int printConfig(SimConfig *c);
int loadTOMLConfig(const char *path, SimConfig *c);
int applyCLIOverrides(int argc, char **argv, SimConfig *c);
int deriveAndValidateOrDie(SimConfig *c);
int writeResolvedConfig(const SimConfig *c);

#ifdef __cplusplus
}
#endif

#endif
