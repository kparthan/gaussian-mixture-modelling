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

    struct Parameters setParameters(int, int, int);

    void infer_components_exp1();
    void infer_components_exp1_compare();

    void infer_components_exp2();
    void infer_components_exp2_compare();

    void generateExperimentalMixtures(Mixture &, long double, string &, int, int);
    void inferExperimentalMixtures(Mixture &, long double, string &, struct Parameters &, int);

    void infer_components_increasing_sample_size_exp3();
    void infer_components_increasing_sample_size_exp4();
};

#endif

