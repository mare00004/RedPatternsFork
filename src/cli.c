#include "argtable3.h"
#include "sim_types.h"
#include <string.h>

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)
#define FIELD_WIDTH 30

/* TODO:
 * - Do some actual agrument validation
 */

typedef struct {
    struct arg_int *N;
    struct arg_dbl *T;
    struct arg_dbl *DT;
    struct arg_dbl *storeTime;
    struct arg_file *outDir;
    struct arg_file *phiFile;
    struct arg_str *gradient;
    struct arg_str *store;
} CommonCLIArguments;

void setCommonArguments(CommonCLIArguments *args, SimConfig *cfg) {
    if (args->N->count > 0) {
        cfg->run.N = args->N->ival[0];
    }
    if (args->T->count > 0) {
        cfg->run.T = args->T->dval[0];
    }
    if (args->DT->count > 0) {
        cfg->run.DT = args->DT->dval[0];
    }
    if (args->storeTime->count > 0) {
        cfg->run.storeTime = args->storeTime->dval[0];
    }
    if (args->outDir->count > 0) {
        strncpy(cfg->run.outDir, args->outDir->filename[0], 255);
    }
    if (args->phiFile->count > 0) {
        strncpy(cfg->model.initialPhiFile, args->phiFile->filename[0], sizeof(cfg->model.initialPhiFile) - 1);
        cfg->model.initialPhiFile[sizeof(cfg->model.initialPhiFile) - 1] = '\0';
    }
    if (args->gradient->count > 0) {
        if (strcmp(args->gradient->sval[0], "linear") == 0) {
            cfg->model.gradientType = LINEAR;
        } else if (strcmp(args->gradient->sval[0], "sigmoid") == 0) {
            cfg->model.gradientType = SIGMOID;
        } else if (strcmp(args->gradient->sval[0], "zero") == 0) {
            cfg->model.gradientType = ZERO;
        }
    }
    if (args->store->count > 0) {
        for (int i = 0; i < args->store->count; i++) {
            char const *str = args->store->sval[i];
            if (strcmp(str, "phi") == 0) {
                BITMAP_ADD(cfg->run.store, PHI);
            }
            if (strcmp(str, "psi") == 0) {
                BITMAP_ADD(cfg->run.store, PSI);
            }
            if (strcmp(str, "percoll") == 0) {
                printf("Percoll bitmap set\n");
                BITMAP_ADD(cfg->run.store, PERCOLL);
            }
        }
    }
}

int parseArguments(int argc, char **argv, SimConfig *cfg) {
    // COMMON
    struct arg_lit *cli_help =
        arg_litn(NULL, "help", 0, 1, "display this help and exit");
    struct arg_int *cli_N =
        arg_int0(NULL, "N", "<power of 2 int>", "number of grid points");
    struct arg_dbl *cli_T =
        arg_dbl0(NULL, "T", "<double>", "total simulation time in seconds");
    struct arg_dbl *cli_DT =
        arg_dbl0(NULL, "DT", "<double>", "time increment in seconds");
    struct arg_dbl *cli_storeTime =
        arg_dbl0(NULL, "storeTime", "<double>", "time between saves");
    struct arg_file *cli_outDir = arg_file0(
        "o",
        "out-dir",
        "<file>",
        "directory where simulation data is stored");
    struct arg_file *cli_phiFile =
        arg_file1(NULL, "phi-file", "<file>", "initial phi file");
    struct arg_str *cli_gradient = arg_str0(NULL, "gradient", "linear|sigmoid|zero", "Pressure gradient");
    struct arg_str *cli_store = arg_strn("s", "store", "phi|psi|percoll", 0, 3, "Arrays to store in HDF5 out file");

    CommonCLIArguments commonArgs = {
        .N = cli_N,
        .T = cli_T,
        .DT = cli_DT,
        .storeTime = cli_storeTime,
        .outDir = cli_outDir,
        .phiFile = cli_phiFile,
        .gradient = cli_gradient,
        .store = cli_store,
    };

    // TODO:
    // DEFAULT - ???
    struct arg_end *endDefault = arg_end(20);
    void *argtableDefault[] = { cli_help, cli_N, cli_T, cli_DT, cli_storeTime, cli_gradient, cli_outDir, cli_phiFile, endDefault };
    int nErrorsDefault;

    // CONVOLUTION - Options that are only valid for the convolution branch
    struct arg_lit *cli_conv =
        arg_lit1("c", "use-convolution", "use convolution integral");
    struct arg_file *cli_kernelFile =
        arg_file1(NULL, "kernel-file", "<file>", "external HDF5 convolution kernel file");
    struct arg_end *endConv = arg_end(20);
    void *argtableConv[] = { cli_help, cli_conv, cli_N, cli_T, cli_DT, cli_storeTime, cli_gradient, cli_store, cli_outDir, cli_phiFile, cli_kernelFile, endConv };
    int nErrorsConv;

    // TAYLOR - Options that are only valid for the taylor branch
    struct arg_lit *cli_tayl =
        arg_lit1("t", "use-taylor", "use taylor approximation");
    struct arg_dbl *cli_NU = arg_dbl0(NULL, "NU", "<double>", "interaction nu");
    struct arg_dbl *cli_MU = arg_dbl0(NULL, "MU", "<double>", "interaction mu");
    struct arg_end *endTayl = arg_end(20);
    void *argtableTayl[] = { cli_help, cli_tayl, cli_N, cli_T, cli_DT, cli_storeTime, cli_gradient, cli_store, cli_outDir, cli_phiFile, cli_NU, cli_MU, endTayl };
    int nErrorsTayl;

    // UNIQUE ARGUMENTS - for freeing argtables
    void *argtableUnique[] = {
        cli_help,
        cli_conv,
        cli_tayl,
        cli_N,
        cli_T,
        cli_DT,
        cli_storeTime,
        cli_gradient,
        cli_store,
        cli_outDir,
        cli_phiFile,
        cli_kernelFile,
        cli_NU,
        cli_MU,
        endDefault,
        endConv,
        endTayl
    };

    // Parsing
    int exitCode = 0;
    const char *progName = "red-patterns";

    if (arg_nullcheck(argtableDefault) != 0 || arg_nullcheck(argtableConv) != 0 ||
        arg_nullcheck(argtableTayl) != 0) {
        fprintf(stderr, "%s: insufficient memory for argument parsing\n", progName);
        exitCode = 1;
        goto exit;
    }

    nErrorsDefault = arg_parse(argc, argv, argtableDefault);
    nErrorsConv = arg_parse(argc, argv, argtableConv);
    nErrorsTayl = arg_parse(argc, argv, argtableTayl);

    if (cli_help->count > 0) {
        void *argtableCommon[] = { cli_N, cli_T, cli_DT, cli_storeTime, cli_gradient, cli_store, cli_outDir, cli_phiFile, endDefault };
        void *argsHelpConv[] = { cli_conv, cli_kernelFile, endConv };
        void *argsHelpTayl[] = { cli_tayl, cli_NU, cli_MU, endTayl };

        arg_dstr_t ds = arg_dstr_create();
        printf("Explanation TODO\n");

        printf("\n");

        printf("Usage:\n");

        printf("\t %s [COMMON...]\n", progName);

        arg_print_syntaxv_ds(ds, argsHelpConv, " [COMMON...]\n");
        printf("\t %s %s", progName, arg_dstr_cstr(ds));
        arg_dstr_reset(ds);

        arg_print_syntaxv_ds(ds, argsHelpTayl, " [COMMON...]\n");
        printf("\t %s %s", progName, arg_dstr_cstr(ds));
        arg_dstr_reset(ds);

        printf("\n");
        printf("COMMON:\n");
        arg_print_glossary(stdout, argtableCommon, "\t%-" STRINGIFY(FIELD_WIDTH) "s %s\n");

        printf("CONVOLUTION:\n");
        arg_print_glossary(stdout, argsHelpConv, "\t%-" STRINGIFY(FIELD_WIDTH) "s %s\n");

        printf("TAYLOR:\n");
        arg_print_glossary(stdout, argsHelpTayl, "\t%-" STRINGIFY(FIELD_WIDTH) "s %s\n");

        arg_dstr_destroy(ds);
        exitCode = 1;
        goto exit;
    }

    if (nErrorsDefault == 0 || nErrorsConv == 0) {
        setCommonArguments(&commonArgs, cfg);
        cfg->model.modelType = CONV;
        if (cli_kernelFile->count > 0) {
            strncpy(cfg->model.variant.Conv.kernelFile, cli_kernelFile->filename[0], sizeof(cfg->model.variant.Conv.kernelFile) - 1);
            cfg->model.variant.Conv.kernelFile[sizeof(cfg->model.variant.Conv.kernelFile) - 1] = '\0';
        }
    } else if (nErrorsTayl == 0) {
        setCommonArguments(&commonArgs, cfg);
        cfg->model.modelType = TAYL;
        if (cli_NU->count > 0) {
            cfg->model.variant.Tayl.NU = cli_NU->dval[0];
        }
        if (cli_MU->count > 0) {
            cfg->model.variant.Tayl.MU = cli_MU->dval[0];
        }
    } else {
        // No correct version found
        if (cli_conv->count > 0) {
            // User probably wants to use conv-version
            arg_print_errors(stderr, endConv, progName);
            printf("Usage: %s ", progName);
            arg_print_syntaxv(stdout, argtableConv, "\n");
            exitCode = 1;
            goto exit;
        } else if (cli_tayl->count > 0) {
            // User probably wants to use tayl-version
            arg_print_errors(stderr, endTayl, progName);
            printf("Usage: %s ", progName);
            arg_print_syntaxv(stdout, argtableTayl, "\n");
            exitCode = 1;
            goto exit;
        } else {
            printf("TODO: No correct version found\n");
            exitCode = 1;
            goto exit;
        }
    }

exit:
    arg_freetable(argtableUnique,
        sizeof(argtableUnique) / sizeof(argtableUnique)[0]);

    return exitCode;
}
