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

    void generateExperimentalMixtures(
      Mixture &, 
      long double, 
      string &, 
      int, 
      int
    );

    void inferExperimentalMixtures(
      Mixture &, 
      long double, 
      string &, 
      struct Parameters &, 
      int
    );

    Mixture mixture_exp1(long double);
    void exp1();
    void exp1_generate(int);
    void exp1_infer(int, int);
    void exp1_infer_compare();

    void infer_components_increasing_sample_size_exp3();
};

#endif

