//#include "Support.h"
#include "SupportUnivariate.h"
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

  if (parameters.compute_responsibility_matrix == SET) {
    computeResponsibilityGivenMixture(parameters);
  }

  if (parameters.read_profiles == SET && parameters.simulation == UNSET
       && parameters.comparison == UNSET) {
    std::vector<Vector> coordinates;
    bool success = gatherData(parameters,coordinates);

    if (success) {
      if (parameters.D != 1) { // D > 1 (multivariate)
        computeEstimators(parameters,coordinates);
      } else if (parameters.D == 1) {
        Vector data(coordinates.size(),0);
        for (int i=0; i<coordinates.size(); i++) {
          data[i] = coordinates[i][0];
        }
        computeEstimatorsUnivariate(parameters,data);
      }
    } else {
      cout << "something wrong in reading data ...\n";
    }
  } 

  if (parameters.D != 1) {
    if (parameters.simulation == SET && parameters.comparison == UNSET) {
      simulateMixtureModel(parameters);
    }
    if (parameters.comparison == SET) {
      compareMixtures(parameters);
    }
  } else if (parameters.D == 1) {
    if (parameters.simulation == SET && parameters.comparison == UNSET) {
      simulateMixtureModelUnivariate(parameters);
    }
    if (parameters.comparison == SET) {
      compareMixturesUnivariate(parameters);
    }
  }

  delete(uniform_generator);

  return 0;
}

