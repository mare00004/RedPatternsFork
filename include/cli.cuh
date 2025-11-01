#ifndef CLI_H
#define CLI_H
#include "argtable3.h"
#include "config.h"
#include "parameters.cuh"
#include <cmath>
#include <string>

typedef struct {
    struct arg_dbl *T;
    struct arg_dbl *DT;
    struct arg_int *NO;
    struct arg_dbl *U;
    struct arg_dbl *PSI;
    struct arg_dbl *gamma;
    struct arg_dbl *delta;
    struct arg_dbl *kappa;
    struct arg_file *outDir;
} CommonCLIArguments;

void setCommonArguments(CommonCLIArguments *args, SimConfig *cfg) {
    if (args->T->count > 0) {
        cfg->run.T = args->T->dval[0];
    }
    if (args->DT->count > 0) {
        cfg->run.DT = args->DT->dval[0];
    }
    if (args->NO->count > 0) {
        cfg->run.NO = args->NO->ival[0];
    }
    if (args->outDir->count > 0) {
        strncpy(cfg->run.outDir, args->outDir->filename[0], 256);
    }
    if (args->U->count > 0) {
        cfg->model.U = args->U->dval[0];
    }
    if (args->PSI->count > 0) {
        cfg->model.PSI = args->PSI->dval[0];
    }
    if (args->gamma->count > 0) {
        cfg->model.gamma = args->gamma->dval[0];
    }
    if (args->delta->count > 0) {
        cfg->model.delta = args->delta->dval[0];
    }
    if (args->kappa->count > 0) {
        cfg->model.kappa = args->kappa->dval[0];
    }
}

int parseArguments(int argc, char **argv, SimConfig *cfg) {
    // COMMON
    struct arg_lit *cli_help =
        arg_litn(NULL, "help", 0, 1, "display this help and exit");
    struct arg_dbl *cli_T =
        arg_dbl0(NULL, "T", "<double>", "total simulation time in seconds");
    struct arg_dbl *cli_DT =
        arg_dbl0(NULL, "DT", "<double>", "time increment in seconds");
    struct arg_int *cli_NO =
        arg_int0(NULL, "NO", "<int>", "time steps between saves");
    struct arg_dbl *cli_U = arg_dbl0(NULL, "U", "<double>", "RBC effective interaction energy in Joule");
    struct arg_dbl *cli_PSI =
        arg_dbl0(NULL, "PSI", "<double>", "RBC average volume fraction");
    struct arg_dbl *cli_gamma = arg_dbl0("g", "gamma", "<double>", NULL);
    struct arg_dbl *cli_delta = arg_dbl0("d", "delta", "<double>", NULL);
    struct arg_dbl *cli_kappa = arg_dbl0("k", "kappa", "<double>", NULL);
    struct arg_file *cli_outDir = arg_file0(
        "o",
        "out-dir",
        "<file>",
        "directory where simulation data is stored");

    CommonCLIArguments commonArgs = {
        .T = cli_T,
        .DT = cli_DT,
        .NO = cli_NO,
        .U = cli_U,
        .PSI = cli_PSI,
        .gamma = cli_gamma,
        .delta = cli_delta,
        .kappa = cli_kappa,
        .outDir = cli_outDir,
    };

    // DEFAULT
    struct arg_end *endDefault = arg_end(20);
    void *argtableDefault[] = { cli_help, cli_T, cli_DT, cli_NO, cli_U, cli_PSI, cli_gamma, cli_delta, cli_kappa, cli_outDir, endDefault };
    int nErrorsDefault;

    // CONVOLUTION
    struct arg_lit *cli_conv =
        arg_lit1("c", "use-convolution", "use convolution integral");
    struct arg_end *endConv = arg_end(20);
    void *argtableConv[] = { cli_help, cli_conv, cli_T, cli_DT, cli_NO, cli_U, cli_PSI, cli_gamma, cli_delta, cli_kappa, cli_outDir, endConv };
    int nErrorsConv;

    // TAYLOR
    struct arg_lit *cli_tayl =
        arg_lit1("t", "use-taylor", "use taylor approximation");
    struct arg_dbl *cli_NU = arg_dbl0(NULL, "NU", "<double>", "interaction nu");
    struct arg_dbl *cli_MU = arg_dbl0(NULL, "MU", "<double>", "interaction mu");
    struct arg_end *endTayl = arg_end(20);
    void *argtableTayl[] = { cli_help, cli_tayl, cli_T, cli_DT, cli_NO, cli_U, cli_PSI, cli_gamma, cli_delta, cli_kappa, cli_outDir, cli_NU, cli_MU, endTayl };
    int nErrorsTayl;

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
        void *argtableCommon[] = { cli_T, cli_DT, cli_NO, cli_U, cli_PSI, cli_gamma, cli_delta, cli_kappa, cli_outDir, endDefault };
        void *argsHelpConv[] = { cli_conv, endConv };
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
        arg_print_glossary(stdout, argtableCommon, "\t%-25s %s\n");

        printf("CONVOLUTION:\n");
        arg_print_glossary(stdout, argsHelpConv, "\t%-25s %s\n");

        printf("TAYLOR:\n");
        arg_print_glossary(stdout, argsHelpTayl, "\t%-25s %s\n");

        arg_dstr_destroy(ds);
        exitCode = 0;
        goto exit;
    }

    if (nErrorsDefault == 0 || nErrorsConv == 0) {
        setCommonArguments(&commonArgs, cfg);
        cfg->model.modelType = CONV;
    } else if (nErrorsTayl == 0) {
        setCommonArguments(&commonArgs, cfg);
        cfg->model.modelType = TAYL;
        if (cli_NU->count > 0) {
            cfg->model.u.Tayl.NU = cli_NU->dval[0];
        }
        if (cli_MU->count > 0) {
            cfg->model.u.Tayl.MU = cli_MU->dval[0];
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
        } else {
            printf("TODO\n");
        }
    }

exit:
    void *argtableUnique[] = {
        cli_help,
        cli_conv,
        cli_tayl,
        cli_T,
        cli_DT,
        cli_NO,
        cli_U,
        cli_PSI,
        cli_gamma,
        cli_delta,
        cli_kappa,
        cli_outDir,
        cli_NU,
        cli_MU,
        endDefault,
        endConv,
        endTayl
    };
    arg_freetable(argtableUnique,
        sizeof(argtableUnique) / sizeof(argtableUnique)[0]);

    return exitCode;
}

/* taking arguments */
void readParameters(int argc, char *argv[]) {
    int argIdx = 1;
    // U
    if (argc > argIdx)
        U = std::stod(argv[argIdx]);
    argIdx++;
    // PSI
    if (argc > argIdx)
        PSI = std::stod(argv[argIdx]);
    argIdx++;
    // IT
    if (argc > argIdx)
        IT = std::stod(argv[argIdx]);
    argIdx++;
    // T
    if (argc > argIdx)
        T = std::stod(argv[argIdx]);
    argIdx++;
    // NO
    if (argc > argIdx)
        NO = std::stod(argv[argIdx]);
    argIdx++;
    // gamma
    if (argc > argIdx)
        h_gamma = std::stod(argv[argIdx]);
    argIdx++;
    // delta
    if (argc > argIdx)
        h_delta = std::stod(argv[argIdx]);
    argIdx++;
    // kappa
    if (argc > argIdx)
        h_kappa = std::stod(argv[argIdx]);
    argIdx++;
    // re-evalutate parameters
    NT = ceil(T / IT);
}
#endif
