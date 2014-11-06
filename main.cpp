#include "Support.h"
#include "UniformRandomNumberGenerator.h"

extern UniformRandomNumberGenerator *uniform_generator;
extern long double MIN_N;

int main(int argc, char **argv)
{
  MIN_N = 20;
  srand(time(NULL));
  UniformReal uniform_distribution(0,1);
  RandomNumberGenerator generator;
  Generator num_gen(generator,uniform_distribution); 
  generator.seed(time(NULL)); // seed with the current time 
  uniform_generator = new UniformRandomNumberGenerator(num_gen);

  struct Parameters parameters = parseCommandLineInput(argc,argv);

  if (parameters.test == SET) {
    TestFunctions();
  }

  if (parameters.experiments == SET) {
    RunExperiments(parameters.iterations);
  }

  if (parameters.read_profiles == SET && parameters.simulation == UNSET) {
    computeEstimators(parameters);
  } 

  if (parameters.simulation == SET) {
    simulateMixtureModel(parameters);
  }

  delete(uniform_generator);

  return 0;
}

