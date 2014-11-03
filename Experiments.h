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

    void infer_components_exp1();

    void inferExperimentalMixtures(Mixture &, long double, string &, struct Parameters &);
};

#endif

