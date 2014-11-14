#include "Support.h"
#include "UniformRandomNumberGenerator.h"

extern UniformRandomNumberGenerator *uniform_generator;

int main(int argc, char **argv)
{

  //cout << "boost eps: " << BOOST_UBLAS_TYPE_CHECK_EPSILON << endl;
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

  if (parameters.read_profiles == SET && parameters.simulation == UNSET
       && parameters.comparison == UNSET) {
    computeEstimators(parameters);
  } 

  if (parameters.simulation == SET && parameters.comparison == UNSET) {
    simulateMixtureModel(parameters);
  }

  if (parameters.comparison == SET) {
    compareMixtures(parameters);
  }

  delete(uniform_generator);

  return 0;
}

