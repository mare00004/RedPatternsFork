#ifndef PARAMETERS_H
#define PARAMETERS_H

// density dimension
#define RC 1100.0 // central density
#define RL 30.0   // density range (RC +- RL/2)

// initial RBC density function
#define Rsigma 4.0f // [g/l] gaussian width
#define Rmu 1100.0f // [g/l] central RBC density

#define wingL 30  // [grid] length of gradient wings
#define P0 1100.0 // [g/l] central percoll density

// degenerate diffusion flux
#define mDeg 500
#define jDegDiffPhi0(i) (pow(1.0 - phi[i], mDeg))
#define jDegDiffPsi0(gi, i) (pow(1.0 - psi[i], mDeg) * abs(phi[gi]))
#define jDegDiffPsi1(gi, i) (-pow(psi[i], mDeg)) * abs(phi[gi])

/********
 * MISC *
 ********/

#define PI 3.141592653589793115997963468544185161590576171875

#endif
