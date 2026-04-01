#ifndef CLI_H
#define CLI_H
#include "sim_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Parse command-line arguments into a simulation configuration.
 *
 * @param argc Number of CLI arguments.
 * @param argv CLI argument values.
 * @param cfg  Output `SimConfig` to populate.
 *
 * @return `0` on success, non-zero on parse/validation failure.
 */
int parseArguments(int argc, char **argv, SimConfig *cfg);

#ifdef __cplusplus
}
#endif

#endif
