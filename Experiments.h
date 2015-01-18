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
    void infer_components_exp1a();
    void infer_components_exp1_compare();
    void infer_components_exp1a_compare();

    void infer_components_exp2();
    void infer_components_exp2a();
    void infer_components_exp2b();
    void infer_components_exp2c();
    void infer_components_exp2_compare();
    void infer_components_exp2a_compare();
    void infer_components_exp2b_compare();
    void infer_components_exp2c_compare();

    void generateExperimentalMixtures(Mixture &, long double, string &, int, int);
    void inferExperimentalMixtures(Mixture &, long double, string &, struct Parameters &, int);

    void infer_components_increasing_sample_size_exp3();
    void infer_components_increasing_sample_size_exp4();
    void infer_components_increasing_sample_size_exp4a();

    void infer_components_exp_spiral();
    void infer_components_exp_spiral_compare();

    void plotMsglensDifferent();
};

#endif

