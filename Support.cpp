#include "Support.h"
#include "Test.h"
#include "Normal.h"
#include "Structure.h"
#include "Experiments.h"
#include "UniformRandomNumberGenerator.h"

int MIXTURE_ID = 1;
int MIXTURE_SIMULATION;
int INFER_COMPONENTS;
int ENABLE_DATA_PARALLELISM;
int NUM_THREADS;
int ESTIMATION;
long double IMPROVEMENT_RATE;
int NUM_STABLE_COMPONENTS;
int MAX_ITERATIONS;
int TOTAL_ITERATIONS = 0;
UniformRandomNumberGenerator *uniform_generator;
int IGNORE_SPLIT;
long double MIN_N;
int STRATEGY;
int MSGLEN_FAIL;
int TRUE_MIX,COMPARE1,COMPARE2;

////////////////////// GENERAL PURPOSE FUNCTIONS \\\\\\\\\\\\\\\\\\\\\\\\\\\\

/*!
 *  \brief This function checks to see if valid arguments are given to the 
 *  command line output.
 *  \param argc an integer
 *  \param argv an std::vector of strings
 *  \return the parameters of the model
 */
struct Parameters parseCommandLineInput(int argc, char **argv)
{
  struct Parameters parameters;
  string parallelize,estimation_method;
  long double improvement_rate;
  int stop_after,strategy;

  bool noargs = 1;

  cout << "Checking command-line input ..." << endl;
  options_description desc("Allowed options");
  desc.add_options()
       ("help","produce help component")
       ("test","run some test cases")
       ("experiments","run some experiments")
       ("iter",value<int>(&parameters.iterations),"# of iterations while running experiments")
       ("profile",value<string>(&parameters.profile_file),"path to the profile")
       ("profiles",value<string>(&parameters.profiles_dir),"path to all profiles")
       ("mixture","flag to do mixture modelling")
       ("infer_components","flag to infer the number of components")
       ("strategy",value<int>(&STRATEGY),"strategy while inferring components")
       ("min_k",value<int>(&parameters.min_components),"min components to infer")
       ("max_k",value<int>(&parameters.max_components),"max components to infer")
       ("log",value<string>(&parameters.infer_log),"log file")
       ("continue","flag to continue inference from some state")
       ("begin",value<int>(&parameters.start_from),"# of components to begin inference from")
       ("k",value<int>(&parameters.fit_num_components),"number of components")
       ("simulate","to simulate a mixture model")
       ("load",value<string>(&parameters.mixture_file),"mixture file")
       ("components",value<int>(&parameters.simulated_components),"# of simulated components")
       ("d",value<int>(&parameters.D),"dimensionality of data")
       ("samples",value<int>(&parameters.sample_size),"sample size generated")
       ("heatmap","parameter to generate heat maps")
       ("res",value<long double>(&parameters.res),"resolution used in heat map images")
       ("mt",value<int>(&parameters.num_threads),"flag to enable multithreading")
       ("parallelize",value<string>(&parallelize),"section of the code to parallelize")
       ("improvement",value<long double>(&improvement_rate),"improvement rate")
       ("estimate",value<string>(&estimation_method),"ML/MML")
       ("compare","mixture comparison")
       ("true",value<string>(&parameters.true_mixture),"true mixture file")
       ("other1",value<string>(&parameters.other1_mixture),"other1 mixture file")
       ("other2",value<string>(&parameters.other2_mixture),"other1 mixture file")
       ("responsibility","flag to compute responsibility matrix")
  ;
  variables_map vm;
  store(command_line_parser(argc,argv).options(desc).run(),vm);
  notify(vm);

  if (vm.count("help")) {
    Usage(argv[0],desc);
  }

  if (vm.count("test")) {
    parameters.test = SET;
  } else {
    parameters.test = UNSET;
  }

  if (vm.count("responsibility")) {
    parameters.compute_responsibility_matrix = SET;
  } else {
    parameters.compute_responsibility_matrix = UNSET;
  }

  if (vm.count("experiments")) {
    parameters.experiments = SET;
    if (!vm.count("iter")) {
      parameters.iterations = 1;
    }
  } else {
    parameters.experiments = UNSET;
  }

  if (vm.count("heatmap")) {
    parameters.heat_map = SET;
    if (!vm.count("res")) {
      parameters.res = DEFAULT_RESOLUTION;
    }
  } else {
    parameters.heat_map = UNSET;
  }

  if (vm.count("profiles") || vm.count("profile")) {
    parameters.read_profiles = SET;
  } else {
    parameters.read_profiles = UNSET;
  }

  if (vm.count("mixture")) {
    parameters.mixture_model = SET;
    if (!vm.count("k")) {
      parameters.fit_num_components = DEFAULT_FIT_COMPONENTS;
    }
    if (vm.count("infer_components")) {
      parameters.infer_num_components = SET;
      INFER_COMPONENTS = SET;
      if (!vm.count("max_k")) {
        parameters.max_components = -1;
        if (vm.count("continue")) {
          parameters.continue_inference = SET;
        } else {
          parameters.continue_inference = UNSET;
        }
        if (!vm.count("begin")) {
          parameters.start_from = 1;
        }
      }
      if (!vm.count("strategy")) {
        STRATEGY = ACCEPT_DEFINITE;
      }
    } else {
      parameters.infer_num_components = UNSET;
      INFER_COMPONENTS = UNSET;
    }
  } else {
    parameters.mixture_model = UNSET;
  }

  if (vm.count("simulate")) {
    parameters.simulation = SET;
    MIXTURE_SIMULATION = SET;
    if (!vm.count("samples")) {
      parameters.sample_size = DEFAULT_SAMPLE_SIZE;
    }
    if (vm.count("load")) {
      parameters.load_mixture = SET;
    } else {
      parameters.load_mixture = UNSET;
      if (!vm.count("components")) {
        parameters.simulated_components = DEFAULT_SIMULATE_COMPONENTS;
      }
    }
    if (!vm.count("d")) {
      cout << "Dimensionality (D) of data not supplied ...\n";
      Usage(argv[0],desc);
    }
  } else {
    parameters.simulation = UNSET;
    MIXTURE_SIMULATION = UNSET;
  }

  if (vm.count("compare")) {
    parameters.comparison = SET;
    if (!vm.count("profile")) { // should generate random data first
      if (!vm.count("samples")) {
        parameters.sample_size = DEFAULT_SAMPLE_SIZE;
      }
    }
    parameters.comparison_type = BOUNDS;  // default
    if(vm.count("mc")) {
      parameters.comparison_type = MC;
    } else if (vm.count("bounds")) {
      parameters.comparison_type = BOUNDS;
    }
    TRUE_MIX = UNSET; COMPARE1 = UNSET; COMPARE2 = UNSET;
    if (vm.count("true")) {
      TRUE_MIX = SET;
    }
    if (vm.count("other1") && vm.count("other2")) {
      COMPARE2 = SET;
    } else if (vm.count("other1")) {
      COMPARE1 = SET;
    }
  } else {
    parameters.comparison = UNSET;
  }

  if (vm.count("mt")) {
    NUM_THREADS = parameters.num_threads;
    ENABLE_DATA_PARALLELISM = SET;
  } else {
    ENABLE_DATA_PARALLELISM = UNSET;
    NUM_THREADS = 1;
  }

  if (vm.count("improvement")) {
    IMPROVEMENT_RATE = improvement_rate;
  } else {
    IMPROVEMENT_RATE = 0.00001; // 0.01 % default
  }

  ESTIMATION = MML;
  if (vm.count("estimate")) {
    if (estimation_method.compare("ml") == 0) {
      ESTIMATION = ML;
    }   
  }

  return parameters;
}

/*!
 *  \brief This module prints the acceptable input format to the program
 *  \param exe a reference to a const char
 *  \param desc a reference to a options_description object
 */
void Usage(const char *exe, options_description &desc)
{
  cout << "Usage: " << exe << " [options]" << endl;
  cout << desc << endl;
  exit(1);
}

/*!
 *  \brief This module checks whether the input file exists or not.
 *  \param file_name a reference to a string
 *  \return true or false depending on whether the file exists or not.
 */
bool checkFile(string &file_name)
{
  ifstream file(file_name.c_str());
  return file;
}

/*!
 *  \brief This module prints the elements of a std::vector<std::vector<> > to a file
 *  \param v a reference to std::vector<Vector>
 *  \param file_name a pointer to a const char
 */
void writeToFile(const char *file_name, std::vector<Vector> &v, int precision)
{
  ofstream file(file_name);
  for (int i=0; i<v.size(); i++) {
    for (int j=0; j<v[i].size(); j++) {
      file << fixed << setw(10) << setprecision(precision) << v[i][j];
    }
    file << endl;
  }
  file.close(); 
}

void writeToFile(const char *file_name, std::vector<Vector> &v)
{
  ofstream file(file_name);
  for (int i=0; i<v.size(); i++) {
    for (int j=0; j<v[i].size(); j++) {
      file << setw(15) << scientific << v[i][j];
    }
    file << endl;
  }
  file.close(); 
}

void writeToFile(string &file_name, std::vector<Vector> &v)
{
  ofstream file(file_name.c_str());
  for (int i=0; i<v.size(); i++) {
    for (int j=0; j<v[i].size(); j++) {
      file << setw(15) << scientific << v[i][j];
    }
    file << endl;
  }
  file.close(); 
}

/*!
 *  \brief This module extracts the file name from the path
 *  \param file a reference to a string
 *  \return the extracted portion of the file name
 */
string extractName(string &file)
{
  unsigned pos1 = file.find_last_of("/");
  unsigned pos2 = file.find(".");
  int length = pos2 - pos1 - 1;
  string sub = file.substr(pos1+1,length);
  return sub;
}

/*!
 *  \brief This function initializes a matrix.
 *  \param matrix a reference to a std::vector<Vector>
 *  \param rows an integers
 *  \param cols an integer
 */
void initializeMatrix(std::vector<Vector> &matrix, int rows, int cols)
{
  Vector tmp(cols,0);
  for (int i=0; i<rows; i++) {
    matrix.push_back(tmp);
  }
}

/*!
 *  \brief This function prints the elements of an std::vector.
 *  \param os a reference to a ostream
 *  \param v a reference to a Vector
 */
void print(ostream &os, Vector &v, int precision)
{
  if (precision == 0) { // scientific notation
    if (v.size() == 1) {
      os << scientific << setprecision(6) << "(" << v[0] << ")";
    } else if (v.size() > 1) {
      os << scientific <<  setprecision(6) <<"(" << v[0] << ", ";
      for (int i=1; i<v.size()-1; i++) {
        os << scientific << setprecision(6) << v[i] << ", ";
      }
      os << scientific << setprecision(6) << v[v.size()-1] << ")\t";
    } else {
      os << "No elements in v ...";
    }
  } else if (precision != 0) { 
    if (v.size() == 1) {
      os << fixed << setprecision(3) << "(" << v[0] << ")";
    } else if (v.size() > 1) {
      os << fixed << setprecision(3) << "(" << v[0] << ", ";
      for (int i=1; i<v.size()-1; i++) {
        os << fixed << setprecision(3) << v[i] << ", ";
      }
      os << fixed << setprecision(3) << v[v.size()-1] << ")\t";
    } else {
      os << "No elements in v ...";
    }
  }
}

/*!
 *  \brief This function prints the elements of an std::vector.
 *  \param os a reference to a ostream
 *  \param v a reference to a std::vector<int>
 */
void print(ostream &os, std::vector<int> &v)
{
  if (v.size() == 1) {
    os << "(" << v[0] << ")";
  } else if (v.size() > 1) {
    os << "(" << v[0] << ", ";
    for (int i=1; i<v.size()-1; i++) {
      os << v[i] << ", ";
    }
    os << v[v.size()-1] << ")\t";
  } else {
    os << "No elements in v ...";
  }
}

/*!
 *  \brief This module returns the sign of a number.
 *  \param number a long double
 *  \return the sign
 */
int sign(long double number)
{
  if (fabs(number) <= ZERO) {
    return 0;
  } else if (number > 0) {
    return 1;
  } else {
    return -1;
  }
}

/*!
 *  \brief This function computes the exponent a^x
 *  \param a a long double
 *  \param x a long double
 *  \return the exponent value
 */
long double exponent(long double a, long double x)
{
  assert(a > 0);
  long double tmp = x * log(a);
  return exp(tmp);
}

long double uniform_random()
{
  return (*uniform_generator)();
  //return rand()/(long double)RAND_MAX;
}

/*!
 *  v1 and v2 are considered to be a column std::vectors
 *  output: v1 * v2' (the outer product matrix)
 */
Matrix outer_prod(Vector &v1, Vector &v2)
{
  assert(v1.size() == v2.size());
  int m = v1.size();
  Matrix ans(m,m);
  for (int i=0; i<m; i++) {
    for (int j=0; j<m; j++) {
      ans(i,j) = v1[i] * v2[j];
    }
  }
  return ans;
}

/*!
 *  v is considered to be a column std::vector
 *  output: m * v (a row std::vector)
 */
Vector prod(Matrix &m, Vector &v)
{
  assert(m.size2() == v.size());
  Vector ans(m.size1(),0);
  for (int i=0; i<m.size1(); i++) {
    for (int j=0; j<m.size2(); j++) {
      ans[i] += m(i,j) * v[j];
    }
  }
  return ans;
}

/*!
 *  v is considered to be a column std::vector
 *  output: v' * m (a row std::vector)
 */
Vector prod(Vector &v, Matrix &m)
{
  assert(m.size1() == v.size());
  Vector ans(m.size2(),0);
  for (int i=0; i<m.size2(); i++) {
    for (int j=0; j<m.size1(); j++) {
      ans[i] += v[j] * m(j,i);
    }
  }
  return ans;
}

/*!
 *  v is considered to be a column std::vector
 *  output: v' M v
 */
long double prod_vMv(Vector &v, Matrix &M)
{
  Vector vM = prod(v,M);
  return computeDotProduct(vM,v);
}

/*!
 *  x,y are considered to be a column std::vectors
 *  output: x' M y
 */
long double prod_xMy(Vector &x, Matrix &M, Vector &y)
{
  Vector xM = prod(x,M);
  return computeDotProduct(xM,y);
}

/*!
 *  \brief This function computes the Akaike information criteria (AIC)
 *  http://en.wikipedia.org/wiki/Akaike_information_criterion
 *  \param k an integer
 *  \param n an integer
 *  \param neg_log_likelihood a double
 *  \return the AIC value (bits)
 */
long double AIC(int k, int n, long double neg_log_likelihood)
{
  long double ans = 2 * k * n / (long double) (n - k -1);
  ans += 2 * neg_log_likelihood;
  return ans / (2*log(2));
}

/*!
 *  \brief This function computes the Bayesian information criteria (BIC)
 *  http://en.wikipedia.org/wiki/Bayesian_information_criterion
 *  \param k an integer
 *  \param n an integer
 *  \param neg_log_likelihood a double
 *  \return the BIC value (bits)
 */
long double BIC(int k, int n, long double neg_log_likelihood)
{
  long double ans = 2 * neg_log_likelihood;
  ans += k * (log(n) + log(2*PI));
  return ans / (2*log(2));
}

Vector sort(Vector &list)
{
  int num_samples = list.size();
	Vector sortedList(list);
  std::vector<int> index(num_samples,0);
	for(int i=0; i<num_samples; i++) {
			index[i] = i;
  }
	quicksort(sortedList,index,0,num_samples-1);
  return sortedList;
}

/*!
 *  This is an implementation of the classic quicksort() algorithm to sort a
 *  list of data values. The module uses the overloading operator(<) to 
 *  compare two Point<T> objects. 
 *  Pivot is chosen as the right most element in the list(default)
 *  This function is called recursively.
 *  \param list a reference to a Vector
 *	\param index a reference to a std::vector<int>
 *  \param left an integer
 *  \param right an integer
 */
void quicksort(Vector &list, std::vector<int> &index, int left, int right)
{
	if(left < right)
	{
		int pivotNewIndex = partition(list,index,left,right);
		quicksort(list,index,left,pivotNewIndex-1);
		quicksort(list,index,pivotNewIndex+1,right);
	}
}

/*!
 *  This function is called from the quicksort() routine to compute the new
 *  pivot index.
 *  \param list a reference to a Vector
 *	\param index a reference to a std::vector<int>
 *  \param left an integer
 *  \param right an integer
 *  \return the new pivot index
 */
int partition(Vector &list, std::vector<int> &index, int left, int right)
{
	long double temp,pivotPoint = list[right];
	int storeIndex = left,temp_i;
	for(int i=left; i<right; i++) {
		if(list[i] < pivotPoint) {
			temp = list[i];
			list[i] = list[storeIndex];
			list[storeIndex] = temp;
			temp_i = index[i];
			index[i] = index[storeIndex];
			index[storeIndex] = temp_i;
			storeIndex += 1;	
		}
	}
	temp = list[storeIndex];
	list[storeIndex] = list[right];
	list[right] = temp;
	temp_i = index[storeIndex];
	index[storeIndex] = index[right];
	index[right] = temp_i;
	return storeIndex;
}

std::vector<Vector> flip(std::vector<Vector> &table)
{
  int num_rows = table.size();
  Vector empty_vector(num_rows,0);
  int num_cols = table[0].size();
  std::vector<Vector> inverted_table(num_cols,empty_vector);
  for (int i=0; i<num_cols; i++) {
    for (int j=0; j<num_rows; j++) {
      inverted_table[i][j] = table[j][i];
    }
  }
  return inverted_table;
}

/*!
 *  \brief This module computes the median of a sorted set of samples
 *  \param list a reference to a std::vector<double>
 *  \return the median value
 */
long double computeMedian(Vector &list)
{
  Vector sorted_list = sort(list);
  int n = sorted_list.size();
  if (n % 2 == 1) {
    return sorted_list[n/2];
  } else {
    return (sorted_list[n/2-1]+sorted_list[n/2])/2;
  }
}

Vector computeMedians(std::vector<Vector> &table)
{
  std::vector<Vector> inverted_table = flip(table);
  int num_cols = table[0].size();
  Vector medians(num_cols,0);
  for (int i=0; i<num_cols; i++) {
    medians[i] = computeMedian(inverted_table[i]);
  }
  return medians;
}

/*!
 *  \brief This module computes the mean of a set of samples
 *  \param list a reference to a std::vector<double>
 *  \return the mean value
 */
long double computeMean(Vector &list)
{
  long double sum = 0;
  for (int i=0; i<list.size(); i++) {
    sum += list[i];
  }
  return sum / (long double)list.size();
}

Vector computeMeans(std::vector<Vector> &table)
{
  std::vector<Vector> inverted_table = flip(table);
  int num_cols = table[0].size();
  Vector means(num_cols,0);
  for (int i=0; i<num_cols; i++) {
    means[i] = computeMean(inverted_table[i]);
  }
  return means;
}

/*!
 *  \brief Computes the variance
 */
long double computeVariance(Vector &list)
{
  long double mean = computeMean(list);
  long double sum = 0;
  for (int i=0; i<list.size(); i++) {
    sum += (list[i]-mean) * (list[i]-mean);
  }
  return sum / (long double) (list.size()-1);
}

int minimumIndex(Vector &values)
{
  int min_index = 0;
  long double min_val = values[0];
  for (int i=1; i<values.size(); i++) { 
    if (values[i] <= min_val) {
      min_index = i;
      min_val = values[i];
    }
  }
  return min_index;
}

int maximumIndex(Vector &values)
{
  int max_index = 0;
  long double max_val = values[0];
  for (int i=1; i<values.size(); i++) { 
    if (values[i] > max_val) {
      max_index = i;
      max_val = values[i];
    }
  }
  return max_index;
}

long double absolute_maximum(std::vector<Vector> &data)
{
  long double max = fabs(data[0][0]);
  for (int i=0; i<data.size(); i++) {
    for (int j=0; j<data[0].size(); j++) {
      if (fabs(data[i][j]) > max) max = fabs(data[i][j]);
    }
  }
  return max;
}

/*!
 *  \brief This function sets the estimation method.
 *  \param estimation an integer
 */
void setEstimationMethod(int estimation)
{
  ESTIMATION = estimation;
}

void TestFunctions()
{
  Test test;

  //test.fisher();

  //test.determinant();

  //test.random_data_generation();

  //test.all_estimates_univariate();

  //test.all_estimates();

    test.factor_analysis_spiral_data();
}

void RunExperiments(int iterations)
{
  Experiments experiments(iterations);

  //experiments.simulate(3);
  //experiments.simulate(5);

  //experiments.infer_components_exp1();
  //experiments.infer_components_exp1a();
  //experiments.infer_components_exp2();
  //experiments.infer_components_exp2a();
  //experiments.infer_components_exp2b();
  //experiments.infer_components_exp2c();

  //experiments.infer_components_exp1_compare();
  //experiments.infer_components_exp1a_compare();
  //experiments.infer_components_exp2_compare();
  //experiments.infer_components_exp2a_compare();
  //experiments.infer_components_exp2b_compare();
  //experiments.infer_components_exp2c_compare();

  //experiments.infer_components_increasing_sample_size_exp3();
  //experiments.infer_components_increasing_sample_size_exp4();
  //experiments.infer_components_increasing_sample_size_exp4a();

  //experiments.infer_components_exp_spiral();
  //experiments.infer_components_exp_spiral_compare();

  experiments.plotMsglensDifferent();
}

void computeResponsibilityGivenMixture(struct Parameters &parameters)
{
  std::vector<Vector> coordinates;
  bool success = gatherData(parameters,coordinates);
  if (success) {
    Mixture mixture;
    mixture.load(parameters.mixture_file,parameters.D);
    mixture.computeResponsibilityMatrix(coordinates,parameters.infer_log);
  } else {
    cout << "Something wrong in reading data ...\n";
    exit(1);
  }
}

std::vector<std::vector<TwoPairs> > generatePairs(int D)
{
  array<int,2> x;
  std::vector<array<int,2> > pairs;

  for (int i=0; i<D; i++) {
    x[0] = i;
    for (int j=i; j<D; j++) {
      x[1] = j;
      pairs.push_back(x);
    }
  }
  //print(pairs);

  int dim = 0.5 * D * (D+1);  // == pairs.size()
  assert(dim == pairs.size());
  std::vector<TwoPairs> row(dim);
  std::vector<std::vector<TwoPairs> > table;
  TwoPairs instance;

  for (int i=0; i<dim; i++) {
    instance.p1 = pairs[i];
    for (int j=0; j<dim; j++) {
      instance.p2 = pairs[j];
      row[j] = instance;
    }
    table.push_back(row);
  }
  //print(table);

  return table;
}

void deal_with_improper_covariances(Matrix &A, Matrix &inv, long double &det)
{
  cout << "before fixing ...\n";
  cout << "cov: " << A << endl;
  cout << "inverse: " << inv << endl;
  cout << "det_cov: " << det << endl;

  int D = A.size1();
  Vector eigen_values(D,0);
  Matrix V = IdentityMatrix(D,D);
  eigenDecomposition(A,eigen_values,V);

  cout << "eigen_values: "; print(cout,eigen_values,0); cout << endl;
  cout << "V: " << V << endl;

  Matrix diag = ZeroMatrix(D,D);
  for (int i=0; i<D; i++) {
    if (eigen_values[i] <= 0) {
      cout << "Improper covariance matrix ...\n";
      diag(i,i) = TOLERANCE;
    } else {
      diag(i,i) = eigen_values[i];
    }
  }
  Matrix Vt = trans(V);

  Matrix tmp1 = prod(V,diag);
  A = prod(tmp1,Vt);
  invertMatrix(A,inv,det);

  cout << "after fixing ...\n";
  cout << "cov: " << A << endl;
  cout << "inverse: " << inv << endl;
  cout << "det_cov: " << det << endl;
}

// D  = 3
std::vector<Vector> generate_spiral_data(int N)
{
  Vector emptyvec(3,0);
  std::vector<Vector> data(N,emptyvec);
  long double ti;
  Vector ni;
  Normal normal(0,1);
  for (int i=0; i<N; i++) {
    ti = uniform_random() * 4 * PI;
    ni = normal.generate(3);
    data[i][0] = (13 - 0.5 * ti) * cos(ti) + ni[0];
    data[i][1] = (0.5 * ti - 13) * sin(ti) + ni[1];
    data[i][2] = ti  + ni[2];
  }
  writeToFile("random_sample.dat",data);
  return data;
}

// cov = LL' +  Psi
bool factor_analysis_3d(Matrix &cov, Vector &L, Matrix &Psi)
{
  L = Vector(3,0);
  Psi = ZeroMatrix(3,3);
  long double product = cov(0,1) * cov(0,2) * cov(1,2);
  if (product < 0) {
    cout << "product: " << product << endl;
    cout << "cov: " << cov << endl;
    return 0;
  }
  long double a1a2a3 = sqrt(product);
  long double a1 = a1a2a3 / cov(1,2);
  long double a2 = a1a2a3 / cov(0,2);
  long double a3 = a1a2a3 / cov(0,1);

  if (!(cov(0,1) >= 0 && cov(0,2) >= 0 && cov(1,2) >= 0)) {
    if (cov(0,1) < 0 && cov(0,2) < 0) {
      a1 *= -1;
    } else if (cov(0,1) < 0 && cov(1,2) < 0) {
      a2 *= -1;
    } else if (cov(0,2) < 0 && cov(1,2) < 0) {
      a3 *= -1;
    }
  }

  L[0] = a1; L[1] = a2; L[2] = a3;
  for (int i=0; i<3; i++) {
    Psi(i,i) = cov(i,i) - (L[i]*L[i]);
  }
  return 1;
}

////////////////////// GEOMETRY FUNCTIONS \\\\\\\\\\\\\\\\\\\\\\\\\\\\

// not unit vectors ...
std::vector<Vector> load_data_table(string &file_name, int D)
{
  std::vector<Vector> sample;
  ifstream file(file_name.c_str());
  string line;
  Vector numbers(D,0);
  int i;
  while (getline(file,line)) {
    boost::char_separator<char> sep(" \t");
    boost::tokenizer<boost::char_separator<char> > tokens(line,sep);
    i = 0;
    BOOST_FOREACH(const string &t, tokens) {
      istringstream iss(t);
      long double x;
      iss >> x;
      numbers[i++] = x;
    }
    sample.push_back(numbers);
  }
  file.close();
  return sample;
}

long double normalize(Vector &x, Vector &unit)
{
  long double normsq = 0;
  for (int i=0; i<x.size(); i++) {
    normsq += x[i] * x[i];
  }
  long double norm = sqrt(normsq);
  for (int i=0; i<x.size(); i++) {
    unit[i] = x[i] / norm;
  }
  return norm;
}

long double norm(Vector &v)
{
  long double normsq = 0;
  for (int i=0; i<v.size(); i++) {
    normsq += v[i] * v[i];
  }
  return sqrt(normsq);
}

void cartesian2spherical(Vector &cartesian, Vector &spherical)
{
  Vector unit(3,0);
  long double r = normalize(cartesian,unit);

  long double x = unit[0];
  long double y = unit[1];
  long double z = unit[2];

  // theta \in [0,PI]: angle with Z-axis
  long double theta = acos(z);

  // phi \in[0,2 PI]: angle with positive X-axis
  long double ratio = x/sin(theta);
  if (ratio > 1) {
    ratio = 1;
  } else if (ratio < -1) {
    ratio = -1;
  }
  long double angle = acos(ratio);
  long double phi = 0;
  if (x == 0 && y == 0) {
    phi = 0;
  } else if (x == 0) {
    if (y > 0) {
      phi = angle;
    } else {
      phi = 2 * PI - angle;
    }
  } else if (y >= 0) {
    phi = angle;
  } else if (y < 0) {
    phi = 2 * PI - angle;
  }

  spherical[0] = r;
  spherical[1] = theta;
  spherical[2] = phi;
}

/*!
 *  \brief This function converts the spherical coordinates into cartesian.
 *  \param spherical a reference to a Vector 
 *  \param cartesian a reference to a Vector 
 */
void spherical2cartesian(Vector &spherical, Vector &cartesian)
{
  cartesian[0] = spherical[0] * sin(spherical[1]) * cos(spherical[2]);
  cartesian[1] = spherical[0] * sin(spherical[1]) * sin(spherical[2]);
  cartesian[2] = spherical[0] * cos(spherical[1]);
}

/*!
 *  \brief This funciton computes the dot product between two std::vectors.
 *  \param v1 a reference to a Vector
 *  \param v2 a reference to a Vector
 *  \return the dot product
 */
long double computeDotProduct(Vector &v1, Vector &v2) 
{
  assert(v1.size() == v2.size());
  long double dot_product = 0;
  for (int i=0; i<v1.size(); i++) {
    dot_product += v1[i] * v2[i];
  }
  return dot_product;
}

// 3D only
Vector crossProduct(Vector &v1, Vector &v2) 
{
  Vector ans(3,0);
  ans[0] = v1[1] * v2[2] - v1[2] * v2[1];
  ans[1] = v1[2] * v2[0] - v1[0] * v2[2];
  ans[2] = v1[0] * v2[1] - v1[1] * v2[0];
  return ans;
}

long double computeSum(Vector &data)
{
  long double sum = 0;
  for (int i=0; i<data.size(); i++) {
    sum += data[i];
  }
  return sum;
}

long double computeSum(Vector &data, Vector &weights, long double &Neff)
{
  long double sum = 0;
  Neff = 0;
  for (int i=0; i<data.size(); i++) {
    sum += (data[i] * weights[i]);
    Neff += weights[i];
  }
  return sum;
}

Vector computeVectorSum(std::vector<Vector> &sample) 
{
  int d = sample[0].size();
  Vector resultant(d,0);  // resultant direction

  std::vector<Vector> _resultants;
  for (int i=0; i<NUM_THREADS; i++) {
    _resultants.push_back(resultant);
  }
  int tid;
  #pragma omp parallel if(ENABLE_DATA_PARALLELISM) num_threads(NUM_THREADS) private(tid) 
  {
    tid = omp_get_thread_num();
    #pragma omp for
    for (int i=0; i<sample.size(); i++) {
      for (int j=0; j<d; j++) {
        _resultants[tid][j] += sample[i][j];
      }
    } // i loop ends ...
  }

  for (int i=0; i<NUM_THREADS; i++) {
    for (int j=0; j<d; j++) {
      resultant[j] += _resultants[i][j];
    }
  }
  return resultant;
}

Vector computeVectorSum(std::vector<Vector> &sample, Vector &weights, long double &Neff) 
{
  int d = sample[0].size();
  Vector resultant(d,0);  // resultant direction

  std::vector<Vector> _resultants;
  for (int i=0; i<NUM_THREADS; i++) {
    _resultants.push_back(resultant);
  }
  int tid;
  long double sum_neff = 0;
  #pragma omp parallel if(ENABLE_DATA_PARALLELISM) num_threads(NUM_THREADS) private(tid) reduction(+:sum_neff) 
  {
    tid = omp_get_thread_num();
    #pragma omp for
    for (int i=0; i<sample.size(); i++) {
      for (int j=0; j<d; j++) {
        _resultants[tid][j] += sample[i][j] * weights[i];
      }
      sum_neff += weights[i];
    } // i loop ends ...
  }
  Neff = sum_neff;

  for (int i=0; i<NUM_THREADS; i++) {
    for (int j=0; j<d; j++) {
      resultant[j] += _resultants[i][j];
    }
  }
  return resultant;
}

/*!
 *  Computes \sum x / N (x is a vector)
 */
Vector computeNormalizedVectorSum(std::vector<Vector> &sample) 
{
  Vector sum = computeVectorSum(sample);
  for (int j=0; j<sum.size(); j++) {
    sum[j] /= sample.size();
  }
  return sum;
}

/*!
 *  Computes \sum x * x' (x is a vector)
 */
Matrix computeDispersionMatrix(std::vector<Vector> &sample)
{
  int d = sample[0].size();
  Matrix dispersion = ZeroMatrix(d,d);
  for (int i=0; i<sample.size(); i++) {
    dispersion += outer_prod(sample[i],sample[i]);
  }
  return dispersion;
}

Matrix computeDispersionMatrix(std::vector<Vector> &sample, Vector &weights)
{
  int d = sample[0].size();
  Matrix dispersion = ZeroMatrix(d,d);
  Matrix tmp;
  for (int i=0; i<sample.size(); i++) {
    tmp = outer_prod(sample[i],sample[i]);
    dispersion += (weights[i] * tmp);
  }
  return dispersion;
}

/*!
 *  Computes \sum x * x' / N (x is a vector)
 */
Matrix computeNormalizedDispersionMatrix(std::vector<Vector> &sample)
{
  Matrix dispersion = computeDispersionMatrix(sample);
  return dispersion/sample.size();
}

void computeMeanAndCovariance(
  std::vector<Vector> &data, 
  Vector &weights,
  Vector &mean, 
  Matrix &cov
) {
  int N = data.size();
  int D = data[0].size();

  long double Neff;
  mean = computeVectorSum(data,weights,Neff);
  for (int i=0; i<D; i++) {
    mean[i] /= Neff;
  }

  std::vector<Vector> x_mu(N);
  Vector diff(D,0);
  for (int i=0; i<N; i++) {
    for (int j=0; j<D; j++) {
      diff[j] = data[i][j] - mean[j];
    }
    x_mu[i] = diff;
  }
  Matrix S = computeDispersionMatrix(x_mu,weights);
  if (Neff > 1) {
    cov = S / (Neff - 1);
  } else {
    cov = S / Neff;
  }
}

Matrix computeCovariance(
  std::vector<Vector> &data, 
  Vector &weights,
  Vector &mean
) {
  int N = data.size();
  int D = data[0].size();

  long double Neff = 0;
  for (int i=0; i<N; i++) {
    Neff += weights[i];
  }

  std::vector<Vector> x_mu(N);
  Vector diff(D,0);
  for (int i=0; i<N; i++) {
    for (int j=0; j<D; j++) {
      diff[j] = data[i][j] - mean[j];
    }
    x_mu[i] = diff;
  }
  Matrix S = computeDispersionMatrix(x_mu,weights);
  Matrix cov;
  if (Neff > 1) {
    cov = S / (Neff - 1);
  } else {
    cov = S / Neff;
  }
  return cov;
}

/*!
 *  \brief This function computes the approximation of the constant term for
 *  the constant term in the message length expression (pg. 257 Wallace)
 *  \param d an integer
 *  \return the constant term
 */
long double computeConstantTerm(int d)
{
  long double ad = 0;
  ad -= 0.5 * d * log(2 * PI);
  ad += 0.5 * log(d * PI);
  return ad;
}

long double logLatticeConstant(int d)
{
  long double cd = computeConstantTerm(d);
  long double ans = -1;
  ans += (2.0 * cd / d);
  return ans;
}

// n-dimensional rotation matrix x -> y (H x = y)
// x and y need not be unit vectors
Matrix computeOrthogonalTransformation(Vector &x, Vector &y)
{
  int D = y.size();
  Vector u(D,0),v(D,0),tmp(D,0);
  normalize(x,u);

  long double dp = computeDotProduct(u,y);
  for (int i=0; i<D; i++) {
    tmp[i] = y[i] - dp * u[i];
  }
  normalize(tmp,v);

  Matrix uut,vvt;
  uut = outer_prod(u,u);
  vvt = outer_prod(v,v);
  Matrix P = uut + vvt;
  Matrix I = IdentityMatrix(D,D);
  Matrix Q = I - P;

  Matrix uv(D,2);
  for (int i=0; i<D; i++) {
    uv(i,0) = u[i];
    uv(i,1) = v[i];
  }

  dp = computeDotProduct(x,y);
  long double norm_x = norm(x);
  long double norm_y = norm(y);
  long double cost = dp / (norm_x * norm_y);
  long double sint = sqrt(1-cost*cost);
  Matrix R(2,2);
  R(0,0) = cost;
  R(0,1) = -sint;
  R(1,0) = sint;
  R(1,1) = cost;
  Matrix uvr = prod(uv,R);
  Matrix uvt = trans(uv);
  Matrix uvruvt = prod(uvr,uvt);
  Matrix H = Q + uvruvt;

  return H;
}

Matrix generateRandomCovarianceMatrix(int D)
{
  Matrix A = ZeroMatrix(D,D);
  for (int i=0; i<D; i++) {
    for (int j=0; j<D; j++) {
      A(i,j) = uniform_random() * R2;
    }
  }
  Matrix At = trans(A);
  Matrix cov = prod(At,A);
  return cov;
}

Matrix generateRandomCovarianceMatrix(int D, long double sigma)
{
  Matrix cov = IdentityMatrix(D,D);
  long double sigmasq = sigma * sigma;
  for (int i=0; i<D; i++) {
    cov(i,i) = sigmasq;
  }
  return cov;
}

/*
 *  \brief Transformation of x using T
 *  \param x a reference to a vector<vector<long double> >
 *  \param T a reference to a Matrix<long double>
 *  \return the transformed vector list
 */
std::vector<Vector> transform(std::vector<Vector> &x, Matrix &T)
{
  std::vector<Vector> y(x.size());
  for (int i=0; i<x.size(); i++) {
    y[i] = prod(T,x[i]);
  }
  return y;
}

// liblcb inverse()
/*bool invertMatrix(const Matrix &input, Matrix &inverse, long double &det)
{
  Matrix A(input);

  // convert 'A' to lcb::Matrix
  lcb::Matrix<long double> lcb_matrix = convert_to_lcb_matrix(A);
  det = lcb_matrix.determinant();

  lcb::Matrix<long double> lcb_inv = lcb_matrix.inverse();

  inverse = convert_to_ublas_matrix(lcb_inv);

  return true;
}

// liblcb inverse()
bool invertMatrix(const Matrix &input, Matrix &inverse)
{
  Matrix A(input);

  // convert 'A' to lcb::Matrix
  lcb::Matrix<long double> lcb_matrix = convert_to_lcb_matrix(A);
  lcb::Matrix<long double> lcb_inv = lcb_matrix.inverse();

  inverse = convert_to_ublas_matrix(lcb_inv);

  return true;
}

lcb::Matrix<long double> convert_to_lcb_matrix(Matrix &ublas_matrix)
{
  int rows = ublas_matrix.size1();
  int cols = ublas_matrix.size2();

  lcb::Matrix<long double> lcb_matrix(rows,cols);
  for (int i=0; i<rows; i++) {
    for (int j=0; j<cols; j++) {
      lcb_matrix[i][j] = ublas_matrix(i,j);
    }
  }

  return lcb_matrix;
}

Matrix convert_to_ublas_matrix(lcb::Matrix<long double> &lcb_matrix)
{
  int rows = lcb_matrix.rows();
  int cols = lcb_matrix.columns();

  Matrix ublas_matrix = ZeroMatrix(rows,cols);
  for (int i=0; i<rows; i++) {
    for (int j=0; j<cols; j++) {
      ublas_matrix(i,j) = lcb_matrix[i][j];
    }
  }

  return ublas_matrix;
}*/

// computes the determinant as well
int determinant_sign(const permutation_matrix<std::size_t> &pm)
{
  int pm_sign = 1;
  std::size_t size = pm.size();
  for (std::size_t i=0; i<size; i++) {
    if (i != pm(i)) {
      pm_sign *= -1;
    }
  }
  return pm_sign;
}

bool invertMatrix(const Matrix &input, Matrix &inverse, long double &det)
{
  typedef permutation_matrix<std::size_t> pmatrix;

  // create a working copy of the input
  Matrix A(input);

  // create a permutation matrix for the LU-factorization
  pmatrix pm(A.size1());

  // perform LU-factorization
  int res = lu_factorize(A, pm);
  if (res != 0) {
    det = 0;
    return false;
  }

  //cout << "A: " << A << endl; 
  det = 1;
  for (int i=0; i<A.size1(); i++) {
    det *= A(i,i);
  }
  det *= determinant_sign(pm);

  // create identity matrix of "inverse"
  inverse.assign(IdentityMatrix (A.size1()));

  // backsubstitute to get the inverse
  lu_substitute(A, pm, inverse);

  return true;
}

/*!
 *  Matrix inverse C++ Boost::ublas
 */
bool invertMatrix(const Matrix &input, Matrix &inverse)
{
  typedef permutation_matrix<std::size_t> pmatrix;

  // create a working copy of the input
  Matrix A(input);

  // create a permutation matrix for the LU-factorization
  pmatrix pm(A.size1());

  // perform LU-factorization
  int res = lu_factorize(A, pm);
  if (res != 0)
    return false;

  // create identity matrix of "inverse"
  inverse.assign(IdentityMatrix (A.size1()));

  // backsubstitute to get the inverse
  lu_substitute(A, pm, inverse);

  return true;
}

void eigenDecomposition(
  Matrix m, 
  Vector &eigen_values,
  Matrix &eigen_vectors
) {
  // check if m is symmetric
  int num_rows = m.size1();
  int num_cols = m.size2();
  if (num_rows != num_cols) {
    cout << "Error: rows: " << num_rows << " != cols: " << num_cols << endl;
    exit(1);
  }
  for (int i=0; i<num_rows; i++) {
    for (int j=0; j<num_cols; j++) {
      if (fabs(m(i,j)-m(j,i)) >= TOLERANCE) {
        cout << "Error: Matrix is not symmetric ...\n";
        cout << "m: " << m << endl;
        cout << "m(" << i << "," << j << ") != m(" << j << "," << i << ")\n";
        exit(1);
      }
    }
  }

  // matrix is now validated ...
  int MAX_ITERATIONS = 100;
  for (int i=0; i < MAX_ITERATIONS; i++) {
    //find the largest off-diagonal 
    int max_row = 0, max_col = 1;
    int cur_row, cur_col;
    long double max_val = m(max_row,max_col);
    for (cur_row = 0; cur_row < num_rows-1; ++cur_row) {
      for (cur_col = cur_row + 1; cur_col < num_cols; ++cur_col) {
        if (fabs(m(cur_row,cur_col)) > max_val) {
          max_row = cur_row;
          max_col = cur_col;
          max_val = fabs(m(cur_row,cur_col));
        }
      }
    }

    if (max_val <= ZERO) {
      break; //finished
    }

    jacobiRotateMatrix(m,eigen_vectors,max_row,max_col);
  }

  for (int i = 0; i < num_cols; i++) {
    eigen_values[i] = m(i,i);
  }

  //cout << "eigen_values: "; print(cout,eigen_values,0); cout << endl;
  //cout << "eigen_vectors: " << eigen_vectors << endl;
}

void jacobiRotateMatrix(
  Matrix &m,
  Matrix &eigen_vectors, 
  int max_row, 
  int max_col
) {
  long double diff = m(max_col,max_col) - m(max_row,max_row);
  long double phi, t, c, s, tau, temp;
  int i;
  
  phi = diff / (2.0 * m(max_row,max_col));
  t = 1.0 / (std::fabs(phi) + std::sqrt((phi*phi) + 1.0));
  if(phi < 0){ t = -t; }

  c = 1.0 / std::sqrt(t*t + 1.0);
  s = t*c;
  tau = s/(1.0 + c);

  temp = m(max_row,max_col);
  m(max_row,max_col) = 0;
  m(max_row,max_row) = m(max_row,max_row) - (t*temp);
  m(max_col,max_col) = m(max_col,max_col) + (t*temp);
  for(i = 0; i < max_row; i++){ // Case i < max_row
    temp = m(i,max_row);
    m(i,max_row) = temp - (s*(m(i,max_col) + (tau*temp)));
    m(i,max_col) = m(i,max_col) + (s*(temp - (tau*m(i,max_col))));
  }
  for(i = max_row + 1; i < max_col; i++){ // Case max_row < i < max_col
    temp = m(max_row,i);
    m(max_row,i) = temp - (s*(m(i,max_col) + (tau*m(max_row,i))));
    m(i,max_col) = m(i,max_col) + (s*(temp - (tau*m(i,max_col))));
  }
  for(i = max_col + 1; i < m.size2(); i++){ // Case i > max_col
    temp = m(max_row,i);
    m(max_row,i) = temp - (s*(m(max_col,i) + (tau*temp)));
    m(max_col,i) = m(max_col,i) + (s*(temp - (tau*m(max_col,i))));
  }

  for (i = 0; i < eigen_vectors.size1(); i++) { // update the transformation matrix
    temp = eigen_vectors(i,max_row);
    eigen_vectors(i,max_row) = temp
      - (s*(eigen_vectors(i,max_col) + (tau*temp)));
    eigen_vectors(i,max_col) = eigen_vectors(i,max_col)
      + (s*(temp - (tau*eigen_vectors(i,max_col))));
  }
  return;
}

Vector generateRandomUnitVector(int D)
{
  Vector unit_vector(D,0);
  if (D == 3) {
    Vector spherical(3,1);
    spherical[1] = uniform_random() * PI;
    spherical[2] = uniform_random() * 2 * PI;
    spherical2cartesian(spherical,unit_vector);
  } else {
    Normal normal(0,1);
    Vector random_vector = normal.generate(D);
    normalize(random_vector,unit_vector);
  }
  return unit_vector;
}

long double computeEuclideanDistance(Vector &p1, Vector &p2)
{
  int D = p1.size();
  long double distsq = 0;
  for (int i=0; i<D; i++) {
    distsq += (p1[i] - p2[i]) * (p1[i] - p2[i]);
  }
  //return sqrt(distsq);
  return distsq;
}

long double computeLogMultivariateGamma(int p, long double a)
{
  long double ans = 0.25 * p * (p-1) * log(PI);
  long double x;
  for (int i=1; i<=p ;i++) {
    x = a + 0.5 * (1 - i);
    ans += boost::math::lgamma<long double>(x);
  }
  return ans;
}

bool verify(Matrix &m)
{
  int d = m.size1();
  for (int i=0; i<d; i++) {
    for (int j=i+1; j<d; j++) {
      if (fabs(m(i,j)-m(j,i)) >= TOLERANCE) {
        cout << "Error: Matrix is not symmetric ...\n";
        cout << "m: " << m << endl;
        cout << "m(" << i << "," << j << ") != m(" << j << "," << i << ")\n";
        return 0;
      }
    }
  }
  return 1;
}

////////////////////// MIXTURE FUNCTIONS \\\\\\\\\\\\\\\\\\\\\\\\\\\\

/*!
 *  \brief This function is used to generate random weights such that
 *  0 < w_i < 1 and \sum_i w_i = 1
 *  \param D an integer
 *  \return the list of weights
 */
Vector generateFromSimplex(int D)
{
  Vector values(D,0);
  long double random,sum = 0;
  for (int i=0; i<D; i++) {
    // generate a random value in (0,1)
    random = uniform_random(); 
    assert(random > 0 && random < 1);
    // sampling from an exponential distribution with \lambda = 1
    values[i] = -log(1-random);
    sum += values[i];
  }
  for (int i=0; i<D; i++) {
    values[i] /= sum;
  }
  return values;
}

/*!
 *  \brief This function is used to generate random components.
 *  \param num_components an integer
 *  \param D an integer
 *  \return the list of components
 */
std::vector<MultivariateNormal> generateRandomComponents(int num_components, int D)
{
  // generate random means
  std::vector<Vector> means = generateRandomGaussianMeans(num_components,D);

  std::vector<MultivariateNormal> components;
  Matrix A,cov;
  for (int i=0; i<num_components; i++) {
    cov = generateRandomCovarianceMatrix(D);
    MultivariateNormal mvnorm(means[i],cov);
    components.push_back(mvnorm);
  }
  return components;
}

/*!
 *  \brief This function generates random unit means
 *  \param num_components an integer
 *  \param D an integer
 *  \return the list of random unit means 
 */
std::vector<Vector> generateRandomGaussianMeans(int num_components, int D)
{
  std::vector<Vector> random_means;
  Vector mean(D,0);
  long double random;
  for (int i=0; i<num_components; i++) {
    for (int j=0; j<D; j++) {
      random = uniform_random() * R1;
      mean[j] = random - 0.5 * R1;
    }
    random_means.push_back(mean);
  }
  return random_means;
}

/*!
 *  \brief This function is used to read the angular profiles and use this data
 *  to estimate parameters of a Von Mises distribution.
 *  \param parameters a reference to a struct Parameters
 */
void computeEstimators(struct Parameters &parameters, std::vector<Vector> &coordinates)
{
  if (parameters.mixture_model == UNSET) {  // no mixture modelling
    modelOneComponent(parameters,coordinates);
  } else if (parameters.mixture_model == SET) { // mixture modelling
    modelMixture(parameters,coordinates);
  }
}

/*!
 *  \brief This function reads through the profiles from a given directory
 *  and collects the data to do mixture modelling.
 *  \param parameters a reference to a struct Parameters
 *  \param coordinates a reference to a std::vector<Vector>
 */
bool gatherData(struct Parameters &parameters, std::vector<Vector> &coordinates)
{
  if (parameters.profile_file.compare("") == 0) {
    path p(parameters.profiles_dir);
    cout << "path: " << p.string() << endl;
    if (exists(p)) { 
      if (is_directory(p)) { 
        std::vector<path> files; // store paths,
        copy(directory_iterator(p), directory_iterator(), back_inserter(files));
        cout << "# of profiles: " << files.size() << endl;
        int tid;
        std::vector<std::vector<Vector> > _coordinates(NUM_THREADS);
        #pragma omp parallel num_threads(NUM_THREADS) private(tid)
        {
          tid = omp_get_thread_num();
          if (tid == 0) {
            cout << "# of threads: " << omp_get_num_threads() << endl;
          }
          #pragma omp for 
          for (int i=0; i<files.size(); i++) {
            Structure structure;
            structure.load(files[i]);
            std::vector<Vector> coords = structure.getCoordinates();
            for (int j=0; j<coords.size(); j++) {
              _coordinates[tid].push_back(coords[j]);
            }
          }
        }
        for (int i=0; i<NUM_THREADS; i++) {
          for (int j=0; j<_coordinates[i].size(); j++) {
            coordinates.push_back(_coordinates[i][j]);
          }
        }
        cout << "# of profiles read: " << files.size() << endl;
        parameters.D = coordinates[0].size();
        return 1;
      } else {
        cout << p << " exists, but is neither a regular file nor a directory\n";
      }
    } else {
      cout << p << " does not exist\n";
    }
    return 0;
  } else if (parameters.profiles_dir.compare("") == 0) {
    if (checkFile(parameters.profile_file)) {
      // read a single profile
      Structure structure;
      structure.load(parameters.profile_file);
      coordinates = structure.getCoordinates();
      parameters.D = coordinates[0].size();
      return 1;
    } else {
      cout << "Profile " << parameters.profile_file << " does not exist ...\n";
      return 0;
    }
  }
}

/*!
 *  \brief This function models a single component.
 *  \param parameters a reference to a struct Parameters
 *  \param data a reference to a std::vector<Vector>
 */
void modelOneComponent(struct Parameters &parameters, std::vector<Vector> &data)
{
  cout << "Sample size: " << data.size() << endl;
  MultivariateNormal mvnorm;
  Vector weights(data.size(),1);
  struct Estimates estimates;
  mvnorm.computeAllEstimators(data,estimates,1);
}

/*!
 *  \brief This function models a mixture of several components.
 *  \param parameters a reference to a struct Parameters
 *  \param data a reference to a std::vector<std::vector<long double,3> >
 */
void modelMixture(struct Parameters &parameters, std::vector<Vector> &data)
{
  Vector data_weights(data.size(),1);
  // if the optimal number of components need to be determined
  if (parameters.infer_num_components == SET) {
    if (parameters.max_components == -1) {
      Mixture mixture;
      if (parameters.continue_inference == UNSET) {
        Mixture m(parameters.start_from,data,data_weights);
        mixture = m;
        mixture.estimateParameters();
      } else if (parameters.continue_inference == SET) {
        mixture.load(parameters.mixture_file,parameters.D,data,data_weights);
      } // continue_inference
      strategic_inference(parameters,mixture,data);
    }   
  } else if (parameters.infer_num_components == UNSET) {
    // for a given value of number of components
    // do the mixture modelling
    Mixture mixture(parameters.fit_num_components,data,data_weights);
    mixture.estimateParameters();
    cout << "First part: " << mixture.first_part() << endl;
    cout << "Second part: " << mixture.second_part() << endl;
    cout << "First part: " << mixture.getMinimumMessageLength() << endl;
  }
}

/*!
 *  \brief This function is used to simulate the mixture model.
 *  \param parameters a reference to a struct Parameters
 */
void simulateMixtureModel(struct Parameters &parameters)
{
  std::vector<Vector> data;
  if (parameters.load_mixture == SET) {
    Mixture original;
    original.load(parameters.mixture_file,parameters.D);
    bool save = 1;
    if (parameters.read_profiles == SET) {
      bool success = gatherData(parameters,data);
      if (!success) {
        cout << "Error in reading data...\n";
        exit(1);
      }
    } else if (parameters.read_profiles == UNSET) {
      data = original.generate(parameters.sample_size,save);
    }
    if (parameters.heat_map == SET && (parameters.D == 2 || parameters.D == 3)) {
      original.generateHeatmapData(parameters.sample_size,parameters.res,parameters.D);
    }
  } else if (parameters.load_mixture == UNSET) {
    int k = parameters.simulated_components;
    //srand(time(NULL));
    Vector weights = generateFromSimplex(k);
    std::vector<MultivariateNormal> components = generateRandomComponents(k,parameters.D);
    Mixture original(k,components,weights);
    bool save = 1;
    data = original.generate(parameters.sample_size,save);
    // save the simulated mixture
    ofstream file("./simulation/simulated_mixture");
    for (int i=0; i<k; i++) {
      file << fixed << setw(10) << setprecision(5) << weights[i];
      file << "\t";
      components[i].printParameters(file);
    }
    file.close();
  }
  // model a mixture using the original data
  if (parameters.mixture_model == UNSET) {
    modelOneComponent(parameters,data);
  } else if (parameters.mixture_model == SET) {
    modelMixture(parameters,data);
  }
}

std::pair<Vector,Vector> compareMixtures(struct Parameters &parameters)
{
  Mixture original,other1,other2;
  std::vector<Vector> data,Data;
  Vector dw;
  long double upper,lower,kldiv,msg,msg_approx;
  std::pair<Vector,Vector> results;
  Vector msglens,kldivs;
  int N = 100000;

  if (parameters.read_profiles == SET) {
    bool success = gatherData(parameters,data);
    if (!success) {
      cout << "Error in reading data...\n";
      exit(1);
    }
  }

  if (COMPARE1 == SET) {  // true mix is given
    cout << "KL-DIVERGENCE:\n";
    original.load(parameters.true_mixture,parameters.D);
    Data = original.generate(N,0);
    cout << "*** OTHER1 MIX ***\n";
    other1.load(parameters.other1_mixture,parameters.D);
    kldiv = original.computeKLDivergence(other1,Data);
    cout << "kldiv (data): " << kldiv << endl;
    kldiv = original.computeKLDivergenceAverageBound(other1);
    cout << "kldiv (bound): " << kldiv << endl;

    if (parameters.read_profiles == UNSET) {
      data = Data; 
    }
    dw = Vector(data.size(),1);

    cout << "\nMESSAGE LENGTHS:\n";
    cout << "*** TRUE MIX ***\n";
    original.load(parameters.true_mixture,parameters.D,data,dw);
    msg = original.computeMinimumMessageLength(0);
    cout << "msg (true): " << msg << endl;

    cout << "\n*** OTHER1 MIX ***\n";
    other1.load(parameters.other1_mixture,parameters.D,data,dw);
    msg = other1.computeMinimumMessageLength(0);
    cout << "msg (other1): " << msg << endl;
    msg_approx = other1.computeApproximatedMessageLength();
    cout << "msg_approx (other1): " << msg_approx << endl;
  }

  if (COMPARE2 == SET) {
    if (TRUE_MIX == SET) {
      cout << "KL-DIVERGENCE:\n";
      original.load(parameters.true_mixture,parameters.D);
      Data = original.generate(N,0);

      cout << "*** OTHER1 MIX ***\n";
      other1.load(parameters.other1_mixture,parameters.D);
      kldiv = original.computeKLDivergence(other1,Data);
      cout << "kldiv (data): " << kldiv << endl;
      kldivs.push_back(kldiv);
      kldiv = original.computeKLDivergenceUpperBound(other1); upper = kldiv;
      cout << "kldiv (upper bound): " << kldiv << endl;
      kldivs.push_back(kldiv);
      kldiv = original.computeKLDivergenceLowerBound(other1); lower = kldiv;
      cout << "kldiv (lower bound): " << kldiv << endl;
      kldivs.push_back(kldiv);
      kldiv = 0.5 * (upper+lower);
      kldivs.push_back(kldiv);

      cout << "*** OTHER2 MIX ***\n";
      other2.load(parameters.other2_mixture,parameters.D);
      kldiv = original.computeKLDivergence(other2,Data);
      cout << "kldiv (data): " << kldiv << endl;
      kldivs.push_back(kldiv);
      kldiv = original.computeKLDivergenceUpperBound(other2); upper = kldiv;
      cout << "kldiv (upper bound): " << kldiv << endl;
      kldivs.push_back(kldiv);
      kldiv = original.computeKLDivergenceLowerBound(other2); lower = kldiv;
      cout << "kldiv (lower bound): " << kldiv << endl;
      kldivs.push_back(kldiv);
      kldiv = 0.5 * (upper+lower);
      kldivs.push_back(kldiv);
    }

    if (parameters.read_profiles == UNSET) {
      data = Data; 
    }
    dw = Vector(data.size(),1);

    cout << "\nMESSAGE LENGTHS:\n";
    cout << "*** OTHER1 MIX ***\n";
    other1.load(parameters.other1_mixture,parameters.D,data,dw);
    msg = other1.computeMinimumMessageLength(0);
    cout << "msg (other1): " << msg << endl;
    msglens.push_back(msg);
    msg_approx = other1.computeApproximatedMessageLength();
    cout << "msg_approx (other1): " << msg_approx << endl;
    msglens.push_back(msg_approx);
    cout << "\n*** OTHER2 MIX ***\n";
    other2.load(parameters.other2_mixture,parameters.D,data,dw);
    msg = other2.computeMinimumMessageLength(0);
    cout << "msg (other2): " << msg << endl;
    msglens.push_back(msg);
    msg_approx = other2.computeApproximatedMessageLength();
    cout << "msg_approx (other2): " << msg_approx << endl;
    msglens.push_back(msg_approx);
  } // if (compare2 == SET)

  results.first = msglens;
  results.second = kldivs;
  return results;
}

void strategic_inference(
  struct Parameters &parameters, 
  Mixture &mixture, 
  std::vector<Vector> &data
) {
  switch(STRATEGY) {
    case ACCEPT_DEFINITE:
    {
      ofstream log(parameters.infer_log.c_str());
      Mixture stable = inferComponents(mixture,data.size(),data[0].size(),log);
      cout << "# of components: " << stable.getNumberOfComponents() << endl;
      log.close();
      string ans = "./simulation/inferred_mixture_1";
      ofstream out(ans.c_str());
      Vector weights = stable.getWeights();
      std::vector<MultivariateNormal> components = stable.getComponents();
      for (int k=0; k<components.size(); k++) {
        out << "\t" << fixed << setw(10) << setprecision(5) << weights[k];
        out << "\t";
        //components[k].printParameters(out);
        components[k].printParameters(out,1);
      }
      out.close();
      break;
    }

    case ACCEPT_PROBABILISTIC:
    {
      ofstream log(parameters.infer_log.c_str());
      Mixture stable = inferComponentsProbabilistic(mixture,data.size(),data[0].size(),log);
      cout << "# of components: " << stable.getNumberOfComponents() << endl;
      log.close();
      break;
    }

    case ACCEPT_BEST_ITER:
    {
      int NUM_ATTEMPTS = 3;
      string log_file;
      ofstream summary("inference_summary");
      std::vector<Mixture> stable_mixtures;
      Mixture stable,starting;
      long double minimum_msglen,current_msglen;
      int min_index;
      for (int i=0; i<NUM_ATTEMPTS; i++) {
        log_file = parameters.infer_log + "_attempt_" + boost::lexical_cast<string>(i+1);
        ofstream log(log_file.c_str());
        if (i == 0) {
          starting = mixture;
        } else if (stable.getNumberOfComponents() > starting.getNumberOfComponents()) {
          starting = stable;
        }
        stable = inferComponents(starting,data.size(),data[0].size(),log);
        if (i == 0) {
          minimum_msglen = stable.getMinimumMessageLength();
          min_index = 0;
        } else {
          current_msglen = stable.getMinimumMessageLength(); 
          if (current_msglen < minimum_msglen) {
            minimum_msglen = current_msglen;
            min_index = i;
          }
        }
        log.close();
        NUM_STABLE_COMPONENTS = stable.getNumberOfComponents();
        stable_mixtures.push_back(stable);
        summary << "stable mixture [" << i << "] : " << NUM_STABLE_COMPONENTS << endl;
        stable.printParameters(summary,2);
      }
      summary << "\n\nBest mixture: \n";
      stable_mixtures[min_index].printParameters(summary,2);
      summary.close();
      break;
    }

    case ACCEPT_LAST_SPLIT:
    {
      break;
    }
  } // switch()
}

/*!
 *  \brief This function is used to infer optimum number of mixture components.
 *  \param mixture a reference to a Mixture
 *  \param N an integer
 *  \param log a reference to a ostream 
 */
Mixture inferComponents(Mixture &mixture, int N, int D, ostream &log)
{
  int K,iter = 0;
  std::vector<MultivariateNormal> components;
  Mixture modified,improved,parent;
  Vector sample_size;

  MIN_N = 0.25 * D * (D + 3);

  improved = mixture;
  TOTAL_ITERATIONS = 0;

  while (1) {
    parent = improved;
    iter++;
    log << "Iteration #" << iter << endl;
    log << "Parent:\n";
    parent.printParameters(log,1);
    parent.printIndividualMsgLengths(log);
    components = parent.getComponents();
    sample_size = parent.getSampleSize();
    K = components.size();
    /*for (int i=0; i<K; i++) { // split() ...
      if (sample_size[i] > MIN_N) {
        IGNORE_SPLIT = 0;
        modified = parent.split(i,log);
        if (IGNORE_SPLIT == 0) {
          updateInference(modified,improved,N,log,SPLIT);
        }
      }
    }*/
    if (K >= 2) {  // kill() ...
      for (int i=0; i<K; i++) {
        modified = parent.kill(i,log);
        updateInference(modified,improved,N,log,KILL);
      } // killing each component
    } // if (K > 2) loop
    if (K > 1) {  // join() ...
      for (int i=0; i<K; i++) {
        int j = parent.getNearestComponent(i); // closest component
        modified = parent.join(i,j,log);
        updateInference(modified,improved,N,log,JOIN);
      } // join() ing nearest components
    } // if (K > 1) loop
    if (improved == parent) {
      for (int i=0; i<K; i++) { // split() ...
        if (sample_size[i] > MIN_N) {
          IGNORE_SPLIT = 0;
          modified = parent.split(i,log);
          if (IGNORE_SPLIT == 0) {
            updateInference(modified,improved,N,log,SPLIT);
          } else {
            log << "\t\tIGNORING SPLIT ... \n\n";
          }
        }
      } // for()
    }
    if (improved == parent) goto finish;
  } // if (improved == parent || iter%2 == 0) loop

  finish:
  cout << "TOTAL_ITERATIONS: " << TOTAL_ITERATIONS << endl;
  return parent;
}

/*!
 *  \brief Updates the inference
 *  \param modified a reference to a Mixture
 *  \param current a reference to a Mixture
 *  \param log a reference to a ostream
 *  \param operation an integer
 */
void updateInference(Mixture &modified, Mixture &current, int N, ostream &log, int operation)
{
  long double modified_msglen = modified.getMinimumMessageLength();
  long double current_msglen = current.getMinimumMessageLength();

  long double dI = current_msglen - modified_msglen;
  long double dI_n = dI / N;
  long double improvement_rate = (current_msglen - modified_msglen) / current_msglen;

  if (operation == KILL || operation == JOIN || operation == SPLIT) {
    if (improvement_rate >= 0) {
      log << "\t ... IMPROVEMENT ... (+ " << fixed << setprecision(3) 
          << 100 * improvement_rate << " %) ";
      log << "\t\t[ACCEPT]\n\n";
      current = modified;
    } else {
      log << "\t ... NO IMPROVEMENT\t\t\t[REJECT]\n\n";
    }
  } /*else if (operation == SPLIT) {
    if (improvement_rate > IMPROVEMENT_RATE) {
      log << "\t ... IMPROVEMENT ... (+ " << fixed << setprecision(3) 
          << 100 * improvement_rate << " %) ";
      log << "\t\t[ACCEPT]\n\n";
      current = modified;
    } else if (improvement_rate > 0 && improvement_rate <= IMPROVEMENT_RATE) {
      log << "\t ... IMPROVEMENT (" << 100 * improvement_rate << " %) < " << fixed << setprecision(3) 
          << 100 * IMPROVEMENT_RATE << " %\t\t\t[REJECT]\n\n";
      log << "\t\tdI: " << dI << " bits.\n";
      log << "\t\tdI/N: " << dI_n << " bits.\n\n";
    } else {
      log << "\t ... NO IMPROVEMENT\t\t\t[REJECT]\n\n";
    }
  }*/
}

Mixture inferComponentsProbabilistic(Mixture &mixture, int N, int D, ostream &log)
{
  int K,iter = 0;
  std::vector<MultivariateNormal> components;
  Mixture modified,improved,parent;
  Vector sample_size;

  if (D >= 10) {
    MIN_N = 2 * (D + 3);
  } else {
    MIN_N = D + 3;
  }

  improved = mixture;
  TOTAL_ITERATIONS = 0;

  if (D <= 5) {
    IMPROVEMENT_RATE = 0.001;
  } else {
    IMPROVEMENT_RATE = 0.005;
  }
  while (1) {
    parent = improved;
    iter++;
    log << "Iteration #" << iter << endl;
    log << "Parent:\n";
    parent.printParameters(log,1);
    components = parent.getComponents();
    sample_size = parent.getSampleSize();
    K = components.size();
    if (K >= 2) {  // kill() ...
      for (int i=0; i<K; i++) {
        modified = parent.kill(i,log);
        updateInferenceProbabilistic(modified,improved,N,log,KILL);
      } // killing each component
    } // if (K > 2) loop
    if (K > 1) {  // join() ...
      for (int i=0; i<K; i++) {
        int j = parent.getNearestComponent(i); // closest component
        modified = parent.join(i,j,log);
        updateInferenceProbabilistic(modified,improved,N,log,JOIN);
      } // join() ing nearest components
    } // if (K > 1) loop
    if (improved == parent) {
      for (int i=0; i<K; i++) { // split() ...
        if (sample_size[i] > MIN_N) {
          IGNORE_SPLIT = 0;
          modified = parent.split(i,log);
          if (IGNORE_SPLIT == 0) {
            updateInferenceProbabilistic(modified,improved,N,log,SPLIT);
          }
        }
      } // for()
    }
    if (improved == parent) goto finish;
  } // if (improved == parent || iter%2 == 0) loop

  finish:
  return parent;
}

void updateInferenceProbabilistic(
  Mixture &modified, 
  Mixture &current, 
  int N, 
  ostream &log, 
  int operation
) {
  long double modified_msglen = modified.getMinimumMessageLength();
  long double current_msglen = current.getMinimumMessageLength();

  long double improvement_rate = (current_msglen - modified_msglen) / current_msglen;

  if (operation == KILL || operation == JOIN) {
    if (improvement_rate >= 0) {
      log << "\t ... IMPROVEMENT ... (+ " << fixed << setprecision(3) 
          << 100 * improvement_rate << " %) ";
      log << "\t\t[ACCEPT]\n\n";
      current = modified;
    } else {
      log << "\t ... NO IMPROVEMENT\t\t\t[REJECT]\n\n";
    }
  } else if (operation == SPLIT) {
    if (improvement_rate > IMPROVEMENT_RATE) {
      log << "\t ... IMPROVEMENT ... (+ " << fixed << setprecision(3) 
          << 100 * improvement_rate << " %) ";
      log << "\t\t[ACCEPT]\n\n";
      current = modified;
    } else if (improvement_rate <= IMPROVEMENT_RATE) {  // slight improvement ...
      log << "\tslight improvment.. accept/reject probabilistically ...\n";
      long double dI = (current_msglen - modified_msglen) / N;
      long double ratio = exponent(2,dI);
      long double reject_prob = 1.0 / (1 + ratio);
      long double random = uniform_random();
      if (random > reject_prob) {
        log << "\t ... IMPROVEMENT < " << fixed << setprecision(3) 
            << 100 * IMPROVEMENT_RATE << " %\t\t\t[ACCEPT PROBABILISTIC]\n\n";
        current = modified;
      } else {
        log << "\t ... IMPROVEMENT < " << fixed << setprecision(3) 
            << 100 * IMPROVEMENT_RATE << " %\t\t\t[REJECT]\n\n";
      } // slight impr (accept or reject?)
    } else {  // no improvement
      log << "\t ... NO IMPROVEMENT\t\t\t[REJECT]\n\n";
    }
  } // operation = split
}

void inferStableMixtures_MML(
  std::vector<Vector> &data, 
  int true_number,
  int min_k, 
  int max_k, 
  string &log_file
) {
  long double mml,current_msglen,part1,part2,prev_msglen,prev_part2,prev_part1;
  int num_repeats;
  Mixture best_mix;
  Vector data_weights(data.size(),1);
  ESTIMATION = MML;

  ofstream log(log_file.c_str());
  string msglens_file = log_file + ".msglens.dat";
  ofstream msglens(msglens_file.c_str());
  string parts_file = log_file + ".parts.dat";
  ofstream parts(parts_file.c_str());
  for (int k=min_k; k<=max_k; k++) {
    num_repeats = 0;
    repeat:
    Mixture mixture(k,data,data_weights);
    mixture.estimateParameters();
    current_msglen = mixture.getMinimumMessageLength();
    part1 = mixture.first_part();
    part2 = mixture.second_part();

    /*if (k == true_number) { 
      if (current_msglen >= prev_msglen) goto repeat;
      mml = current_msglen;
    }*/
    if (k > true_number) { // check if current is better
      if (current_msglen < prev_msglen && num_repeats <= 5) {
        num_repeats++;
        goto repeat;
      }
      if (num_repeats > 5) {
        if (current_msglen <= mml) goto repeat;
      }
    }
    mixture.printParameters(log,1);
    prev_msglen = current_msglen;
    prev_part1 = part1;
    prev_part2 = part2;

    msglens << fixed << setw(10) << k;
    msglens << fixed << setw(15) << setprecision(3) << current_msglen << endl;

    parts << fixed << setw(10) << k;
    parts << fixed << setw(15) << setprecision(3) << part1;
    parts << fixed << setw(15) << setprecision(3) << part2;
    parts << fixed << setw(15) << setprecision(3) << part1+part2 << endl;
  } // for loop ends ...
}
