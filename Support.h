#ifndef SUPPORT_H
#define SUPPORT_H

#include "Header.h"
#include "Mixture.h"
#include "MultivariateNormal.h"

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
  int start_from;           // starting value of number of components during inference
};

//! parameters associated with estimation
struct Estimates {
  long double Neff;
  Vector mean;
  Matrix cov_ml,cov_mml;
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

// geometry functions
std::vector<Vector> load_matrix(string &, int);
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
Matrix computeDispersionMatrix(std::vector<Vector> &, Vector &);
void computeMeanAndCovariance(std::vector<Vector> &, Vector &, Vector &, Matrix &);
Matrix computeNormalizedDispersionMatrix(std::vector<Vector> &);
long double computeConstantTerm(int);
Matrix computeOrthogonalTransformation(Vector &, Vector &);
Matrix generateRandomCovarianceMatrix(int);
Matrix generateRandomCovarianceMatrix(int, long double);
std::vector<Vector> transform(std::vector<Vector> &, Matrix &);
int determinant_sign(const permutation_matrix<std::size_t> &);
bool invertMatrix(const Matrix &, Matrix &, long double &);
bool invertMatrix(const Matrix &, Matrix &);
void eigenDecomposition(Matrix, Vector &, Matrix &);
void jacobiRotateMatrix(Matrix &, Matrix &, int, int);
Vector generateRandomUnitVector(int);
long double computeEuclideanDistance(Vector &, Vector &);

// mixture functions
Vector generateFromSimplex(int);
std::vector<MultivariateNormal> generateRandomComponents(int, int);
std::vector<Vector> generateRandomGaussianMeans(int, int);
void computeEstimators(struct Parameters &);
bool gatherData(struct Parameters &, std::vector<Vector> &);
void modelOneComponent(struct Parameters &, std::vector<Vector> &);
void modelMixture(struct Parameters &, std::vector<Vector> &);
void simulateMixtureModel(struct Parameters &);
Mixture inferComponents(Mixture &, int, ostream &);
void updateInference(Mixture &, Mixture &, ostream &, int);

#endif

