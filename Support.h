#ifndef SUPPORT_H
#define SUPPORT_H

#include "Header.h"
#include "Mixture.h"
#include "vMF.h"

struct Parameters
{
  int test;                 // flag to test some modules
  int experiments;          // flag to run some experiments
  int iterations;           // number of iterations
  string profile_file;      // path to a single profile
  string profiles_dir;      // path to the directory containing the profiles
  int heat_map;             // flag to generate heat map images
  long double res;          // resolution used in heat map images
  int read_profiles;        // flag to read profile(s)
  int mixture_model;        // flag to model a mixture
  int fit_num_components;   // # of components in the mixture model
  int infer_num_components; // flag to infer # of components
  int min_components;       // min components to infer
  int max_components;       // max components to infer
  string infer_log;         // log file
  int continue_inference;   // flag to continue inference from some state
  int simulation;           // flag to run mixture model simulation
  int load_mixture;         // flag to read mixture from a file
  int simulated_components; // # of components to be simulated
  string mixture_file;      // file containing the mixture information
  int D;                    // dimensionality of data
  int sample_size;          // sample size to be generated from the simulated mixture
  int num_threads;          // flag to enable multithreading
  long double max_kappa;    // max value of kappa allowed
  int start_from;           // starting value of number of components
                            // during inference
  int estimate_all;         // estimate using all methods
  int compute_responsibility_matrix;  // flag
};

//! parameters associated with estimation
struct Estimates {
  Vector mean;
  long double R,Rbar,Neff;
  long double kappa_ml_approx,kappa_ml;
  long double kappa_tanabe,kappa_truncated_newton,kappa_song;
  long double kappa_mml_newton,kappa_mml_halley,kappa_mml_complete;
};

struct TestingConstants {
  long double a1,a2,b0,b1,b2,d1,d2;
  long double A2,B1,D1,D2;
  long double f11,f12;
  long double g11,g12,g21,g31;
};

// general functions
struct Parameters parseCommandLineInput (int, char **); 
void Usage (const char *, options_description &);
bool checkFile(string &);
void writeToFile(const char *, std::vector<Vector> &, int);
string extractName(string &);
void initializeMatrix(std::vector<Vector> &, int, int);
void print(ostream &, Vector &, int);
void print(ostream &, std::vector<int> &);
int sign(long double);
long double exponent(long double, long double);
long double uniform_random();
Matrix outer_prod(Vector &, Vector &);
Vector prod(Matrix &, Vector &);
Vector prod(Vector &, Matrix &);
long double prod_vMv(Vector &, Matrix &);
long double prod_xMy(Vector &, Matrix &, Vector &);
long double AIC(int, int, long double);
long double BIC(int, int, long double);
Vector sort(Vector &);
void quicksort(Vector &, std::vector<int> &, int, int);
int partition(Vector &, std::vector<int> &, int, int);
std::vector<Vector> flip(std::vector<Vector> &);
long double computeMedian(Vector &);
Vector computeMedians(std::vector<Vector> &);
long double computeMean(Vector &);
Vector computeMeans(std::vector<Vector> &);
long double computeVariance(Vector &);
int minimumIndex(Vector &);
int maximumIndex(Vector &);
void setEstimationMethod(int);
void TestFunctions();
void RunExperiments(int);

// bessel functions
long double computeLogModifiedBesselFirstKind_old(long double, long double);
long double computeLogModifiedBesselFirstKind(long double, long double);
long double computeRatioBessel(int, long double);
long double computeDerivativeOfRatioBessel(int, long double, long double);
long double computeSecondDerivativeOfRatioBessel(int, long double, long double);
long double computeThirdDerivativeOfRatioBessel(int, long double, long double, long double, long double);
long double computeDerivativeOf_Ader_A(int &, long double &, long double &, long double &); 
long double computeSecondDerivativeOf_Ader_A(int &, long double &, long double &, long double &, long double &);
long double computeDerivativeOf_A2der_Ader(int &, long double &, long double &, long double &, long double &); 
long double computeSecondDerivativeOf_A2der_Ader(int &, long double &, long double &, long double &, long double &, long double &);
struct TestingConstants computeTestingConstants(int, long double);

// geometry functions
std::vector<Vector> load_matrix(string &, int);
long double computeLogSurfaceAreaSphere(int);
long double normalize(Vector &, Vector &);
long double norm(Vector &);
void cartesian2spherical(Vector &, Vector &);
void spherical2cartesian(Vector &, Vector &);
long double computeDotProduct(Vector &, Vector &);
Vector crossProduct(Vector &, Vector &);
Vector computeVectorSum(std::vector<Vector> &);
Vector computeVectorSum(std::vector<Vector> &, Vector &, long double &);
Vector computeNormalizedVectorSum(std::vector<Vector> &);
Matrix computeDispersionMatrix(std::vector<Vector> &);
Matrix computeNormalizedDispersionMatrix(std::vector<Vector> &);
long double computeConstantTerm(int);
Matrix computeOrthogonalTransformation(Vector &);
Matrix computeOrthogonalTransformation(Vector &, Vector &);
std::vector<Vector> transform(std::vector<Vector> &, Matrix &);
bool invertMatrix(const Matrix &, Matrix &);
void eigenDecomposition(Matrix, Vector &, Matrix &);
void jacobiRotateMatrix(Matrix &, Matrix &, int, int);
Vector generateRandomUnitVector(int);
void generateRandomOrthogonalVectors(Vector &, Vector &, Vector &);  // 3D only
Matrix generateRandomBinghamMatrix(int);
Matrix generateRandomBinghamMatrix(Vector &, long double);
Vector generateRandomChisquare(int, int);
long double computeTestStatistic(Matrix &, struct TestingConstants &, int);
long double compute_pvalue(long double, chi_squared &);
long double compute_vc(long double);
long double compute_vs(long double);

// mixture functions
Vector generateFromSimplex(int);
std::vector<vMF> generateRandomComponents(int, int);
std::vector<Vector> generateRandomUnitMeans(int, int);
Vector generateRandomKappas(int);
std::vector<std::vector<int> > updateBins(std::vector<Vector> &, long double);
void outputBins(std::vector<std::vector<int> > &, long double);
void computeEstimators(struct Parameters &);
void computeResponsibilityGivenMixture(struct Parameters &);
bool gatherData(struct Parameters &, std::vector<Vector> &);
void modelOneComponent(struct Parameters &, std::vector<Vector> &);
void modelMixture(struct Parameters &, std::vector<Vector> &);
void simulateMixtureModel(struct Parameters &);
pair<std::vector<Mixture>,std::vector<Mixture> > 
estimateMixturesUsingAllMethods(int, std::vector<Vector> &, ostream &);
Mixture inferComponents(Mixture &, int, ostream &);
void updateInference(Mixture &, Mixture &, ostream &, int);
void inferStableMixtures(std::vector<Vector> &, int, int, string &);

#endif

