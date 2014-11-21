#ifndef EXPERIMENTS_H
#define EXPERIMENTS_H

#include "Header.h"
#include "Support.h"

class Experiments
{
  private:
    int iterations;

  public:
    Experiments(int);

    void simulate(int);

    struct Parameters setParameters(int, int);

    void infer_components_exp1();
    void infer_components_exp2();

    void generateExperimentalMixtures(Mixture &, long double, string &, int, int);
    void inferExperimentalMixtures(Mixture &, long double, string &, struct Parameters &, int);
};

#endif

