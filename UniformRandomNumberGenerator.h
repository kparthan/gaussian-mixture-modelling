#ifndef UNIFORM_RANDOM_NUMBER_GENERATOR_H
#define UNIFORM_RANDOM_NUMBER_GENERATOR_H

#include "Header.h"

typedef boost::uniform_real<> UniformReal; 
typedef boost::mt19937 RandomNumberGenerator; 
typedef boost::variate_generator<RandomNumberGenerator&,UniformReal> Generator; 

class UniformRandomNumberGenerator
{
  private:
    Generator number_generator;

  public:
    UniformRandomNumberGenerator(Generator &num_gen) : number_generator(num_gen)
    {}

    long double operator()()
    {
      return number_generator();
    }
};

#endif

