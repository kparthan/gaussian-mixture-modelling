#include "Support.h"
#include "Test.h"
#include "Normal.h"
#include "Structure.h"
#include "Experiments.h"
#include "UniformRandomNumberGenerator.h"

string CURRENT_DIRECTORY;
int CONSTRAIN_KAPPA;
int MIXTURE_ID = 1;
int MIXTURE_SIMULATION;
int INFER_COMPONENTS;
int ENABLE_DATA_PARALLELISM;
int NUM_THREADS;
int ESTIMATION;
long double MAX_KAPPA;
long double IMPROVEMENT_RATE;
int NUM_STABLE_COMPONENTS;
int MAX_ITERATIONS;
UniformRandomNumberGenerator *uniform_generator;
int ML_NEWTON_FAIL,ML_HALLEY_FAIL,MML_NEWTON_FAIL,MML_HALLEY_FAIL;

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
  string constrain,parallelize,estimation_method,joining;
  long double improvement_rate;
  int stop_after;

  bool noargs = 1;

  cout << "Checking command-line input ..." << endl;
  options_description desc("Allowed options");
  desc.add_options()
       ("help","produce help component")
       ("test","run some test cases")
       ("experiments","run some experiments")
       ("iter",value<int>(&parameters.iterations),"# of iterations while running experiments")
       ("truncate",value<int>(&stop_after),"# of iterations in the root finding approximation")
       ("profile",value<string>(&parameters.profile_file),"path to the profile")
       ("profiles",value<string>(&parameters.profiles_dir),"path to all profiles")
       ("constrain",value<string>(&constrain),"to constrain kappa")
       ("max_kappa",value<long double>(&parameters.max_kappa),"maximum value of kappa allowed")
       ("mixture","flag to do mixture modelling")
       ("infer_components","flag to infer the number of components")
       ("min_k",value<int>(&parameters.min_components),"min components to infer")
       ("max_k",value<int>(&parameters.max_components),"max components to infer")
       ("log",value<string>(&parameters.infer_log),"log file")
       ("continue","flag to continue inference from some state")
       ("begin",value<int>(&parameters.start_from),"# of components to begin inference from")
       ("join",value<string>(&joining),"criterion to join two components while inferring")
       ("k",value<int>(&parameters.fit_num_components),"number of components")
       ("simulate","to simulate a mixture model")
       ("load",value<string>(&parameters.mixture_file),"mixture file")
       ("components",value<int>(&parameters.simulated_components),"# of simulated components")
       ("d",value<int>(&parameters.D),"dimensionality of data")
       ("samples",value<int>(&parameters.sample_size),"sample size generated")
       ("bins","parameter to generate heat maps")
       ("res",value<long double>(&parameters.res),"resolution used in heat map images")
       ("mt",value<int>(&parameters.num_threads),"flag to enable multithreading")
       ("parallelize",value<string>(&parallelize),"section of the code to parallelize")
       ("estimation",value<string>(&estimation_method),"type of estimation")
       ("improvement",value<long double>(&improvement_rate),"improvement rate")
       ("estimate_all","flag to estimate using all methods")
       ("responsibility","flag to compute responsibility matrix")
  ;
  variables_map vm;
  //store(parse_command_line(argc,argv,desc),vm);
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

  if (vm.count("experiments")) {
    parameters.experiments = SET;
    if (!vm.count("iter")) {
      parameters.iterations = 1;
    }
  } else {
    parameters.experiments = UNSET;
  }

  if (vm.count("truncate")) {
    MAX_ITERATIONS = stop_after;
  } else {
    MAX_ITERATIONS = 2; // default
  }

  if (vm.count("bins")) {
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
    } else {
      parameters.infer_num_components = UNSET;
      INFER_COMPONENTS = UNSET;
    }
  } else {
    parameters.mixture_model = UNSET;
  }

  if (vm.count("estimate_all")) {
    parameters.estimate_all = SET;
  } else {
    parameters.estimate_all = UNSET;
  }

  if (vm.count("responsibility")) {
    parameters.compute_responsibility_matrix = SET;
  } else {
    parameters.compute_responsibility_matrix = UNSET;
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

  if (constrain.compare("kappa") == 0) {
    CONSTRAIN_KAPPA = SET;
  } else {
    CONSTRAIN_KAPPA = UNSET;
  }

  if (!vm.count("max_kappa")) {
    MAX_KAPPA = DEFAULT_MAX_KAPPA;
  } else {
    MAX_KAPPA = parameters.max_kappa;
  }

  if (vm.count("mt")) {
    NUM_THREADS = parameters.num_threads;
    ENABLE_DATA_PARALLELISM = SET;
  } else {
    ENABLE_DATA_PARALLELISM = UNSET;
    NUM_THREADS = 1;
  }

  if (vm.count("estimation")) {
    if (estimation_method.compare("ml_approx") == 0) {
      ESTIMATION = ML_APPROX;
    } else if (estimation_method.compare("ml") == 0) {
      ESTIMATION = ML;
    } else if (estimation_method.compare("tanabe") == 0) {
      ESTIMATION = TANABE;
    } else if (estimation_method.compare("trunc_newton") == 0) {
      ESTIMATION = TRUNCATED_NEWTON;
    } else if (estimation_method.compare("song") == 0) {
      ESTIMATION = SONG;
    } else if (estimation_method.compare("mml_newton") == 0) {
      ESTIMATION = MML_NEWTON;
    } else if (estimation_method.compare("mml_halley") == 0) {
      ESTIMATION = MML_HALLEY;
    } else if (estimation_method.compare("mml_complete") == 0) {
      ESTIMATION = MML_COMPLETE;
    } else {
      cout << "Invalid estimation method ...\n";
      Usage(argv[0],desc);
    }
  } else {  // default is MML estimation ...
    ESTIMATION = MML_HALLEY;
  }

  if (vm.count("improvement")) {
    IMPROVEMENT_RATE = improvement_rate;
  } else {
    IMPROVEMENT_RATE = 0.0001; // 0.01 % default
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
  if (precision == 0) {
    if (v.size() == 1) {
      os << scientific << "(" << v[0] << ")";
    } else if (v.size() > 1) {
      os << scientific << "(" << v[0] << ", ";
      for (int i=1; i<v.size()-1; i++) {
        os << scientific << v[i] << ", ";
      }
      os << scientific << v[v.size()-1] << ")\t";
    } else {
      os << "No elements in v ...";
    }
  } else if (precision != 0) { // scientific notation
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

  //test.orthogonal_transformation();

  //test.bessel();

  //test.random_sample_generation();

  //test.generate_random_chisq();

  test.all_estimates();

  //test.generate_random_bingham();

  //test.chi_square_goodness_of_fit();

  //test.ks_test();

  //test.hypothesis_testing_fb_1();

  //test.hypothesis_testing_fb_2();

  //test.estimate_mixtures_all();
}

void RunExperiments(int iterations)
{
  Experiments experiments(iterations);

  //experiments.simulate(3,100);
  //experiments.simulate(50,100);
  //experiments.simulate(2,10);
  experiments.simulate(100,10);
}

////////////////////// BESSEL RELATED FUNCTIONS \\\\\\\\\\\\\\\\\\\\\\\\\\\\

long double computeLogModifiedBesselFirstKind_old(long double alpha, long double x)
{
  if (!(alpha >= 0 && fabs(x) >= 0)) {
    cout << "Error logModifiedBesselFirstKind: (alpha,x) = (" << alpha << "," << x << ")\n";
    exit(1);
  }
  long double t;
  if (alpha == 0) {
    t = 1;
  } else if (fabs(x) <= TOLERANCE) {
    t = 0;
    return 0;
  } 
  long double x2_4 = x * x * 0.25;
  long double R = 1.0,             // 0-th term
         I = 1.0,             // m=0 sum
         m = 1.0;             // next, m=1
  // m! = m*(m-1)!, and
  // Gamma(m+alpha+1) = (m+alpha)*Gamma(m+alpha), 
  // because Gamma(x+1) = x*Gamma(x), for all x > 0.
  do { 
    //long double tmp = log(x2_4) - log(m) - log(alpha+m);
    long double tmp = x2_4 / (m*(alpha+m));  // the m-th term
    R *= tmp;  // the m-th term
    I += R;                     // total
    if (R >= INFINITY || I >= INFINITY) {
      return INFINITY;
    }
    m += 1.0;
    //cout << "m: " << m << "; tmp: " << tmp << "; R: " << R << "; I: " << I << endl;
  } while( R >= I * TOLERANCE);
  long double log_mod_bessel = log(I) + (alpha * log(x/2.0)) - boost::math::lgamma<long double>(alpha+1);
  /*if (log_mod_bessel >= INFINITY) {
    log_mod_bessel = approximate_bessel(alpha,x);
  }*/
  return log_mod_bessel;
}

long double computeLogModifiedBesselFirstKind(long double alpha, long double x)
{
  if (!(alpha >= 0 && fabs(x) >= 0)) {
    cout << "Error logModifiedBesselFirstKind: (alpha,x) = (" << alpha << "," << x << ")\n";
    exit(1);
  }
  if (fabs(x) <= TOLERANCE) {
    return -LARGE_NUMBER;
  } 

  // constant term log(x^2/4)
  long double log_x2_4 = 2.0 * log(x/2.0);
  long double four_x2 = 4.0 / (x * x);

  long double m = 1;
  long double log_sm_prev = -boost::math::lgamma<long double>(alpha+1); // log(t0)
  long double log_sm_current;
  long double log_tm_prev = log_sm_prev; // log(t0)
  long double log_tm_current;
  long double cm_prev = 0,cm_current; 
  long double ratio = (alpha+1) * four_x2;  // t0/t1
  while(ratio < 1) {
    cm_current = (cm_prev + 1) * ratio;
    log_tm_current = log_tm_prev - log(ratio); 
    log_sm_prev = log_tm_current + log(cm_current + 1);
    m++;
    ratio = m * (m+alpha) * four_x2;
    log_tm_prev = log_tm_current;
    cm_prev = cm_current;
  } // while() ends ...
  long double k = m;
  log_tm_current = log_tm_prev - log(ratio);  // log(tk)
  long double c = log_tm_current - log_sm_prev;
  long double tk_sk_1 = exp(c);
  long double y = log_sm_prev;
  long double zm = 1;
  long double log_rm_prev = 0,log_rm_current,rm;
  while(1) {
    log_sm_current = y + log(1 + tk_sk_1 * zm);
    m++;
    log_rm_current = (log_x2_4 - log(m) - log(m+alpha)) + log_rm_prev;
    rm = exp(log_rm_current);
    zm += rm;
    if (rm/zm < 1e-10)  break;
    log_sm_prev = log_sm_current;
    log_rm_prev = log_rm_current;
  } // while() ends ...
  log_sm_current = y + log(1 + tk_sk_1 * zm);
  return (log_sm_current + (alpha * log(x/2.0)));
}

/*!
 *  \brief This function computes the ratio of modified Bessel functions
 *  of order d. 
 *              A_d = I_{d/2} / I_{d/2-1}   ... Mardia (2000) book pg. 350
 *  \param d a reference to an integer
 *  \param kappa a reference to a long double
 *  \return the ratio of Bessel functions
 */
long double computeRatioBessel(int d, long double kappa)
{
  long double d_over_2 = d / 2.0;
  long double log_I1 = computeLogModifiedBesselFirstKind(d_over_2,kappa);
  long double log_I2 = computeLogModifiedBesselFirstKind(d_over_2-1,kappa);
  long double log_ratio = log_I1 - log_I2;
  return exp(log_ratio);
}

/*!
 *  \brief This function computes the first derivative of the ratio of modified 
 *  Bessel functions of order d. 
 *          A'_d = 1 - A_d^2 - ((d-1)/kappa) A_d  ... Mardia (2000) book pg. 350
 *  \param d a reference to an integer
 *  \param kappa a reference to a long double
 *  \param Ad a reference to a long double
 *  \return the first derivative of the ratio of Bessel functions
 */
long double computeDerivativeOfRatioBessel(int d, long double kappa, long double A)
{
  long double ans = 1 - (A * A);
  long double tmp = (A * (d-1)) / kappa;
  ans -= tmp;
  return ans;
}

/*!
 *  A''(d)
 */
long double computeSecondDerivativeOfRatioBessel(int d, long double kappa, long double A)
{
  long double ans = 2 * A * A * A;
  long double tmp = 3 * (d-1) * A * A / kappa;
  ans += tmp;
  tmp =  (d * d - d - 2 * kappa * kappa) * A / (kappa * kappa);
  ans += tmp; 
  tmp = (d-1)/kappa;
  ans -= tmp;
  return ans;
}

/*!
 *  A'''(d)
 */
long double computeThirdDerivativeOfRatioBessel(
  int d, long double kappa, long double A, long double Ader, long double A2der
) {
  long double A2der_Ader = A2der / Ader;
  long double A_Ader = A / Ader;
  long double ans,tmp;
  tmp = (d - 1) / kappa;

  ans = (-2 * A) * A2der_Ader;
  ans -= (2 * Ader);
  ans -= (tmp * A2der_Ader);
  ans -= ((2 * tmp * A_Ader) / (kappa * kappa));
  ans += ((2 * tmp) / kappa);
  ans *= Ader;

  return ans;
}

/*!
 *  \brief This function computes the first derivative of A' over A.
 *    d/dk (A'/A) = -A'/A^2 - A' + (d-1)/k^2
 *  \param d a reference to an integer
 *  \param kappa a reference to a long double
 *  \param Ad a reference to a long double
 *  \param Ader a reference to a long double
 *  \return the first derivative of A' over A
 */
long double computeDerivativeOf_Ader_A(
  int &d, 
  long double &kappa, 
  long double &A, 
  long double &Ader
) {
  long double ans = -Ader/(A * A);
  ans -= Ader;
  ans += (d-1)/(kappa*kappa);
  return ans;
}

/*!
 *  d^2/dk^2 (A'/A)
 */
long double computeSecondDerivativeOf_Ader_A(
  int &d, long double &kappa, long double &A, long double &Ader, long double &A2der
) {
  long double ans,tmp;
  tmp = (2 * Ader * Ader) - (A * A2der);
  ans = tmp / (A * A * A);
  ans -= A2der;
  tmp = (2 * (d - 1)) / (kappa * kappa * kappa);
  ans -= tmp;
  return ans;
}

/*!
 *  \brief This function computes the first derivative of A'' over A'.
 *    d/dk (A''/A') = -2 A' + 2(d-1)/k^2 - (d-1)/k^3  (A/A') (k A''/A' + 2)
 *  \param d a reference to an integer
 *  \param kappa a reference to a long double
 *  \param Ader a reference to a long double
 *  \param Ader_A a reference to a long double
 *  \param A2der_Ader a reference to a long double
 *  \return the first derivative of A'' over A'
 */
long double computeDerivativeOf_A2der_Ader(
  int &d, 
  long double &kappa, 
  long double &Ader, 
  long double &Ader_A, 
  long double &A2der_Ader
) {
  long double tmp = (d-1)/(kappa*kappa);
  long double ans = -2 * Ader;
  ans += 2 * tmp;
  long double tmp2 = (tmp/(kappa * Ader_A)) * (kappa*A2der_Ader + 2);
  ans -= tmp2;
  return ans;
}

/*!
 *  d^2/dk^2 (A''/A')
 */
long double computeSecondDerivativeOf_A2der_Ader(
  int &d, 
  long double &kappa, 
  long double &A, 
  long double &Ader, 
  long double &A2der, 
  long double &A3der
) {
  long double ans,tmp,tmp2,tmp3;
  ans = -2 * A2der;
  tmp = (4 * (d - 1)) / (kappa * kappa * kappa);
  ans -= tmp;

  // d(A A'' / K^2 A'^2)
  tmp = kappa * A * Ader * A3der;
  tmp += kappa * Ader * Ader * A2der;
  tmp -= 2 * kappa * A * A2der * A2der;
  tmp -= 2 * A * Ader * A2der;
  tmp2 = kappa * Ader;
  ans -= ((d - 1) * tmp) / (tmp2 * tmp2 * tmp2);

  // d(A / K^3 A')
  tmp2 = kappa * kappa * kappa;
  tmp3 = A / (tmp2 * kappa * Ader * Ader);
  tmp = kappa * A2der  + 3 * Ader;
  tmp *= tmp3;
  tmp *= -1;
  tmp += (1 / tmp2);
  ans -= (2 * (d-1) * tmp);

  return ans;
}

struct TestingConstants computeTestingConstants(int d, long double k)
{
  long double diff,tmp;
  long double Ivp1_Iv,Ivp2_Iv,Ivp3_Iv;

  long double v = (d/2.0) - 1;

  long double log_Iv = computeLogModifiedBesselFirstKind(v,k);
  long double log_Ivp1 = computeLogModifiedBesselFirstKind(v+1,k);
  diff = log_Ivp1 - log_Iv;
  Ivp1_Iv = exp(diff);

  long double log_Ivp2 = computeLogModifiedBesselFirstKind(v+2,k);
  diff = log_Ivp2 - log_Iv;
  Ivp2_Iv = exp(diff);

  long double log_Ivp3 = computeLogModifiedBesselFirstKind(v+3,k);
  diff = log_Ivp3 - log_Iv;
  Ivp3_Iv = exp(diff);

  struct TestingConstants constants;
  constants.a1 = Ivp1_Iv;
  constants.a2 = ((d-1) * Ivp2_Iv + 1) / d;
  constants.b0 = Ivp1_Iv / k;
  constants.b1 = Ivp2_Iv / k;
  constants.b2 = ((d+1) * Ivp3_Iv + Ivp1_Iv) / ((d+2) * k);
  constants.d1 = Ivp2_Iv / (k*k);
  constants.d2 = 3 * constants.d1;

  constants.A2 = constants.a2 - (constants.a1 * constants.a1);
  constants.D2 = constants.d2 - (constants.b0 * constants.b0);
  constants.D1 = constants.d1 - (constants.b0 * constants.b0);
  constants.B1 = constants.b1 - (constants.a1 * constants.b0);

  constants.f11 = constants.D2 - ((constants.B1 * constants.B1) / constants.A2);
  constants.f12 = constants.D1 - ((constants.B1 * constants.B1) / constants.A2);

  long double num,denom;
  num = constants.f11 + ((d-3) * constants.f12);
  denom = 2 * constants.d1 * (constants.f11 + ((d-2) * constants.f12));
  constants.g11 = num / denom;
  constants.g12 = -constants.f12 / denom;
  constants.g21 = 1 / (constants.b2 - (constants.b1 * constants.b1 / constants.b0));
  constants.g31 = 1 / constants.d1;

  /*cout << "g11: " << constants.g11 << endl;
  cout << "g21: " << constants.g21 << endl;
  cout << "g12: " << constants.g12 << endl;
  cout << "g31: " << constants.g31 << endl;
  cout << "b0: " << constants.b0 << endl;*/

  return constants;
}

////////////////////// GEOMETRY FUNCTIONS \\\\\\\\\\\\\\\\\\\\\\\\\\\\

std::vector<Vector> load_matrix(string &file_name, int D)
{
  std::vector<Vector> sample;
  ifstream file(file_name.c_str());
  string line;
  Vector numbers(D,0),unit_vector(D,0);
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
    normalize(numbers,unit_vector);
    sample.push_back(unit_vector);
  }
  file.close();
  return sample;
}

long double computeLogSurfaceAreaSphere(int d)
{
  long double log_num = log(d) + ((d/2.0) * log(PI));
  long double log_denom = boost::math::lgamma<long double>(d/2.0+1);
  return (log_num - log_denom);
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

/*!
 *  Computes \sum x * x' / N (x is a vector)
 */
Matrix computeNormalizedDispersionMatrix(std::vector<Vector> &sample)
{
  Matrix dispersion = computeDispersionMatrix(sample);
  return dispersion/sample.size();
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

/*!
 *  \brief This function computes the orthogonal transformation matrix
 *  to align a std::vector x = (0,...,0,1)^T with another std::vector y
 *  Source: http://math.stackexchange.com/questions/598750/finding-the-rotation-matrix-in-n-dimensions
 *  \param y a reference to a Vector
 *  \return the transformation matrix
 */
Matrix computeOrthogonalTransformation(Vector &y)
{
  int D = y.size();

  Vector x(D,0);
  x[D-1] = 1;
  Vector u = x;
  long double uy = y[D-1];
  Vector vn(D,0),v(D,0);
  for (int i=0; i<D; i++) {
    vn[i] = y[i] - uy * x[i];
  }
  normalize(vn,v);
  Matrix I = IdentityMatrix(D,D);
  Matrix UUt = outer_prod(u,u);
  Matrix VVt = outer_prod(v,v);
  Matrix tmp1 = UUt + VVt;
  Matrix tmp2 = I - tmp1;
  Matrix UV(D,2);
  for (int i=0; i<D; i++) {
    UV(i,0) = u[i];
    UV(i,1) = v[i];
  }
  Matrix UVt = trans(UV);
  long double cost = 0;
  for (int i=0; i<D; i++) {
    cost += x[i] * y[i];
  }
  long double sint = sqrt(1-cost*cost);
  Matrix R(2,2);
  R(0,0) = cost;
  R(0,1) = -sint;
  R(1,0) = sint;
  R(1,1) = cost;
  Matrix tmp3 = prod(UV,R);
  Matrix tmp4 = prod(tmp3,UVt);
  Matrix Q = tmp2 + tmp4;
  return Q;
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

  // check
  /*Vector Hx = prod(H,x);  // == y ?
  Matrix Ht = trans(H); // H transpose = H inverse
  Vector Hty = prod(Ht,y);  // == x ?
  cout << "x: "; print(cout,x,3); cout << endl;
  cout << "y: "; print(cout,y,3); cout << endl;
  cout << "H: " << H << endl;
  cout << "Hx: "; print(cout,Hx,3); cout << endl;
  cout << "Hty: "; print(cout,Hty,3); cout << endl;*/

  return H;
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

// 3D only
void generateRandomOrthogonalVectors(
  Vector &mean,
  Vector &major_axis,
  Vector &minor_axis
) {
  long double phi = (2 * PI) * uniform_random();
  Vector spherical(3,1),major1(3,0);
  spherical[1] = PI/2;
  spherical[2] = phi;
  spherical2cartesian(spherical,major1); // major axis
  Vector zaxis(3,0); zaxis[2] = 1; // z-axis
  Vector mu1 = zaxis;

  long double theta = PI * uniform_random();
  phi = (2 * PI) * uniform_random();
  //phi = rand()*2*PI/(long double)RAND_MAX;
  spherical[1] = theta;
  spherical[2] = phi;
  mean = Vector(3,0);
  spherical2cartesian(spherical,mean); 

  Matrix r = computeOrthogonalTransformation(zaxis,mean);
  major_axis = prod(r,major1);
  //minor_axis = prod(r,minor1);
  minor_axis = crossProduct(mean,major_axis);
}

// Resultant A should be a symmetric matrix
// Trace(A) = 0
Matrix generateRandomBinghamMatrix(int D)
{
  long double MAX = 10;
  Matrix A = ZeroMatrix(D,D);
  long double trace = 0,random;
  // fill the diagonal entries
  for (int i=0; i<D-1; i++) {
    random = (uniform_random() * 2) - 1;
    A(i,i) = random * MAX;
    trace += A(i,i);
  }
  A(D-1,D-1) = -trace;
  // fill the off-diagonal entries
  for (int i=0; i<D; i++) {
    for (int j=0; j<D; j++) {
      if (i < j) {
        random = (uniform_random() * 2) - 1;
        A(i,j) = random * MAX;
      } else if (i > j) {
        A(i,j) = A(j,i);
      }
    }
  }
  cout << "A: " << A << endl;
  return -A;
}

// equivalent to Kent (n-dimensional)
Matrix generateRandomBinghamMatrix(Vector &mu, long double kappa)
{
  int D = mu.size();
  Vector betas(D-1,0);
  int attempts = 0;
  do {
    long double sum_betas=0;
    for (int i=0; i<D-2; i++) {
      betas[i] = (uniform_random() - 0.5) * kappa;
      sum_betas += betas[i];
    }
    betas[D-2] = -sum_betas;
    attempts++;
  } while (fabs(betas[D-2]) >= 0.5 * kappa);
  cout << "attempts: " << attempts << endl;
  //assert(fabs(betas[D-2]) < 0.5 * kappa);
  cout << "kappa: " << kappa << "; betas: "; print(cout,betas,3); cout << endl;

  Matrix A1 = ZeroMatrix(D,D);
  for (int i=0; i<D-1; i++) {
    A1(i,i) = betas[i];
  }
  Vector zaxis(D,0); zaxis[D-1] = 1;
  Matrix H = computeOrthogonalTransformation(zaxis,mu);
  Matrix Ht = trans(H);
  Matrix tmp = prod(H,A1);
  Matrix A = prod(tmp,Ht);
  cout << "A: " << A << endl;
  return -A;
}

Vector generateRandomChisquare(int df, int N)
{
  Normal normal(0,1);
  Vector random_chi(N,0);
  Vector random_normal;
  for (int i=0; i<N; i++) {
    random_normal = normal.generate(df);
    for (int j=0; j<df; j++) {
      random_chi[i] += (random_normal[j] * random_normal[j]);
    }
  }
  return random_chi;
}

long double computeTestStatistic(Matrix &T, struct TestingConstants &constants, int N)
{
  int D = T.size1();

  //cout << "T: " << T << endl;

  long double tmp1,tmp2;
  long double term1 = 0;
  for (int j=1; j<D; j++) {
    tmp1 = T(j,j) - constants.b0;
    term1 += (tmp1 * tmp1);
  }
  //cout << "term1: " << term1;
  term1 *= constants.g11;

  long double term2 = 0;
  for (int j=1; j<D; j++) {
    tmp1 = T(j,j) - constants.b0;
    for (int k=j+1; k<D; k++) {
      tmp2 = T(k,k) - constants.b0;
      term2 += (tmp1 * tmp2);
    }
  }
  term2 *= (2 * constants.g12);

  long double term3 = 0;
  for (int j=1; j<D; j++) {
    tmp1 = T(0,j);
    term3 += (tmp1 * tmp1);
  }
  //cout << "; term3: " << term3 << endl;
  term3 *= constants.g21;

  long double term4 = 0;
  for (int j=1; j<D; j++) {
    for (int k=j+1; k<D; k++) {
      tmp1 = T(j,k);
      term4 += (tmp1 * tmp1);
    }
  }
  term4 *= constants.g31;

  //cout << "term2: " << term2 << "; term4: " << term4 << endl;

  long double t = N * (term1 + term2 + term3 + term4);
  return t;
}

// return the p-value of the test
long double compute_pvalue(long double t, chi_squared &chisq)
{
  //long double critical_value = quantile(chisq,1-alpha);
  long double pvalue = 1 - cdf(chisq,t);
  return pvalue;
}

long double compute_vc(long double k)
{
  long double log_I0 = computeLogModifiedBesselFirstKind(0,k);
  long double log_I1 = computeLogModifiedBesselFirstKind(1,k);
  long double log_I2 = computeLogModifiedBesselFirstKind(2,k);
  long double log_I3 = computeLogModifiedBesselFirstKind(3,k);
  long double log_I4 = computeLogModifiedBesselFirstKind(4,k);
  
  long double diff = log_I1 - log_I0;
  long double I1_I0 = exp(diff);
  diff = log_I2 - log_I0;
  long double I2_I0 = exp(diff);
  diff = log_I3 - log_I0;
  long double I3_I0 = exp(diff);
  diff = log_I4 - log_I0;
  long double I4_I0 = exp(diff);

  long double tmp;
  tmp = (1 + I4_I0) * 0.5;
  long double vc = tmp - (I2_I0 * I2_I0);
  long double num,denom;
  tmp = I3_I0 + I1_I0 - (2 * I1_I0 * I2_I0);
  num = tmp * tmp;
  tmp = 1 + I2_I0 - (2 * I1_I0 * I1_I0);
  denom = 2 * tmp;
  tmp = num / denom;
  vc -= tmp;
  return vc;
}

long double compute_vs(long double k)
{
  long double log_I0 = computeLogModifiedBesselFirstKind(0,k);
  long double log_I1 = computeLogModifiedBesselFirstKind(1,k);
  long double log_I2 = computeLogModifiedBesselFirstKind(2,k);
  long double log_I3 = computeLogModifiedBesselFirstKind(3,k);
  long double log_I4 = computeLogModifiedBesselFirstKind(4,k);
  
  long double diff = log_I1 - log_I0;
  long double I1_I0 = exp(diff);
  diff = log_I2 - log_I0;
  long double I2_I0 = exp(diff);
  diff = log_I3 - log_I0;
  long double I3_I0 = exp(diff);
  diff = log_I4 - log_I0;
  long double I4_I0 = exp(diff);

  long double tmp,num,denom,vs;
  tmp = (1 - I4_I0) * (1 - I2_I0);
  num = tmp;
  tmp = (I1_I0 - I3_I0);
  num -= (tmp * tmp);
  denom = 2 * (1 - I2_I0);
  vs = num / denom;
  return vs;
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
std::vector<vMF> generateRandomComponents(int num_components, int D)
{
  // generate random unit means
  std::vector<Vector> unit_means = generateRandomUnitMeans(num_components,D);

  // generate random kappas
  Vector kappas = generateRandomKappas(num_components);

  std::vector<vMF> components;
  Vector mean;
  long double kappa;
  for (int i=0; i<num_components; i++) {
    // initialize component parameters
    mean = unit_means[i];
    kappa = kappas[i];
    vMF vmf(mean,kappa);
    components.push_back(vmf);
  }
  return components;
}

/*!
 *  \brief This function generates random unit means
 *  \param num_components an integer
 *  \param D an integer
 *  \return the list of random unit means 
 */
std::vector<Vector> generateRandomUnitMeans(int num_components, int D)
{
  std::vector<Vector> random_unit_means;
  Normal normal(0,1);
  Vector mean(D,0),unit_mean(D,0);
  for (int i=0; i<num_components; i++) {
    mean = normal.generate(D);
    normalize(mean,unit_mean);
    random_unit_means.push_back(unit_mean);
  }
  return random_unit_means;
}

/*!
 *  \brief This function generates random kappas 
 *  \param num_components an integer
 *  \return the list of random kappas 
 */
Vector generateRandomKappas(int num_components)
{
  Vector random_kappas;
  for (int i=0; i<num_components; i++) {
    long double kappa = uniform_random() * MAX_KAPPA;
    random_kappas.push_back(kappa);
  }
  return random_kappas;
}

/*!
 *  \brief This function bins the sample data 
 *  \param res a long double
 *  \param unit_coordinates a reference to a vector<vector<long double> > 
 */
std::vector<std::vector<int> > updateBins(
  std::vector<Vector> &unit_coordinates, long double res
) {
  std::vector<std::vector<int> > bins;
  int num_rows = 180 / res;
  int num_cols = 360 / res;
  std::vector<int> tmp(num_cols,0);
  for (int i=0; i<num_rows; i++) {
    bins.push_back(tmp);
  }

  long double theta,phi;
  int row,col;
  Vector spherical(3,0);
  for (int i=0; i<unit_coordinates.size(); i++) {
    //cout << "i: " << i << endl; 
    cartesian2spherical(unit_coordinates[i],spherical);
    theta = spherical[1] * 180 / PI;
    if (fabs(theta) <= ZERO) {
      row = 0;
    } else {
      row = (int)(ceil(theta/res) - 1);
    }
    phi = spherical[2] * 180 / PI;
    if (fabs(phi) <= ZERO) {
      col = 0;
    } else {
      col = (int)(ceil(phi/res) - 1);
    }
    if (row >= bins.size() || col >= bins[0].size()) {
      cout << "outside bounds: " << row << " " << col << "\n";
      cout << "theta: " << theta << " phi: " << phi << endl;
      cout << "spherical_1: " << spherical[1] << " spherical_2: " << spherical[2] << endl;
      cout << "unit_coordinates[i]_1: " << unit_coordinates[i][1] << " unit_coordinates[i]_2: " << unit_coordinates[i][2] << endl;
      fflush(stdout);
    }
    bins[row][col]++;
    //cout << "row,col: " << row << "," << col << endl;
  }
  return bins;
}

/*!
 *  \brief This function outputs the bin data.
 *  \param bins a reference to a std::vector<std::vector<int> >
 */
void outputBins(std::vector<std::vector<int> > &bins, long double res)
{
  long double theta=0,phi;
  string fbins2D_file,fbins3D_file;
  fbins2D_file = "./visualize/bins2D.dat";
  fbins3D_file = "./visualize/bins3D.dat";
  ofstream fbins2D(fbins2D_file.c_str());
  ofstream fbins3D(fbins3D_file.c_str());
  Vector cartesian(3,0);
  Vector spherical(3,1);
  for (int i=0; i<bins.size(); i++) {
    phi = 0;
    spherical[1] = theta * PI / 180;
    for (int j=0; j<bins[i].size(); j++) {
      fbins2D << fixed << setw(10) << bins[i][j];
      phi += res;
      spherical[2] = phi * PI / 180;
      spherical2cartesian(spherical,cartesian);
      for (int k=0; k<3; k++) {
        fbins3D << fixed << setw(10) << setprecision(4) << cartesian[k];
      }
      fbins3D << fixed << setw(10) << bins[i][j] << endl;
    }
    theta += res;
    fbins2D << endl;
  }
  fbins2D.close();
  fbins3D.close();
}

/*!
 *  \brief This function is used to read the angular profiles and use this data
 *  to estimate parameters of a Von Mises distribution.
 *  \param parameters a reference to a struct Parameters
 */
void computeEstimators(struct Parameters &parameters)
{
  std::vector<Vector> unit_coordinates;
  bool success = gatherData(parameters,unit_coordinates);
  if (parameters.heat_map == SET && unit_coordinates[0].size() == 3) {
    std::vector<std::vector<int> > bins = updateBins(unit_coordinates,parameters.res);
    outputBins(bins,parameters.res);
  }
  if (success && parameters.mixture_model == UNSET) {  // no mixture modelling
    modelOneComponent(parameters,unit_coordinates);
  } else if (success && parameters.mixture_model == SET) { // mixture modelling
    modelMixture(parameters,unit_coordinates);
  }
}

/*!
 *
 */
void computeResponsibilityGivenMixture(struct Parameters &parameters)
{
  std::vector<Vector> unit_coordinates;
  bool success = gatherData(parameters,unit_coordinates);
  if (success) {
    Mixture mixture;
    mixture.load(parameters.mixture_file,parameters.D);
    mixture.computeResponsibilityMatrix(unit_coordinates,parameters.infer_log);
  } else {
    cout << "Something wrong in reading data ...\n";
    exit(1);
  }
}

/*!
 *  \brief This function reads through the profiles from a given directory
 *  and collects the data to do mixture modelling.
 *  \param parameters a reference to a struct Parameters
 *  \param unit_coordinates a reference to a std::vector<Vector>
 */
bool gatherData(struct Parameters &parameters, std::vector<Vector> &unit_coordinates)
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
        std::vector<std::vector<Vector> > _unit_coordinates(NUM_THREADS);
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
            std::vector<Vector> coords = structure.getUnitCoordinates();
            for (int j=0; j<coords.size(); j++) {
              _unit_coordinates[tid].push_back(coords[j]);
            }
          }
        }
        for (int i=0; i<NUM_THREADS; i++) {
          for (int j=0; j<_unit_coordinates[i].size(); j++) {
            unit_coordinates.push_back(_unit_coordinates[i][j]);
          }
        }
        cout << "# of profiles read: " << files.size() << endl;
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
      unit_coordinates = structure.getUnitCoordinates();
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
  vMF vmf;
  Vector weights(data.size(),1);
  struct Estimates estimates;
  vmf.computeAllEstimators(data,estimates,1);
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
      ofstream log(parameters.infer_log.c_str());
      Mixture stable = inferComponents(mixture,data.size(),log);
      NUM_STABLE_COMPONENTS = stable.getNumberOfComponents();
      log.close();
    } else {  // parameters.max_components == -1
      inferStableMixtures(data,parameters.min_components,parameters.max_components,
                          parameters.infer_log);
    }
  } else if (parameters.infer_num_components == UNSET) {
    // for a given value of number of components
    // do the mixture modelling
    if (parameters.estimate_all == UNSET) { // estimate using only one method (MML)
      Mixture mixture(parameters.fit_num_components,data,data_weights);
      mixture.estimateParameters();
    } else if (parameters.estimate_all == SET) { // estimate using all methods
      ofstream log(parameters.infer_log.c_str());
      estimateMixturesUsingAllMethods(parameters.fit_num_components,data,log);
      log.close();
    }
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
    if (parameters.heat_map == SET && parameters.D == 3) {
      original.generateHeatmapData(parameters.res);
      std::vector<std::vector<int> > bins = updateBins(data,parameters.res);
      outputBins(bins,parameters.res);
    }
  } else if (parameters.load_mixture == UNSET) {
    int k = parameters.simulated_components;
    //srand(time(NULL));
    Vector weights = generateFromSimplex(k);
    std::vector<vMF> components = generateRandomComponents(k,parameters.D);
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

/*!
 *  \brief Estimates mixtures using all estimation methods
 */
pair<std::vector<Mixture>,std::vector<Mixture> > estimateMixturesUsingAllMethods(
  int num_components, 
  std::vector<Vector> &data,
  ostream &log
) {
  Vector data_weights(data.size(),1);
  int NUM_ITERATIONS = 5;
  std::vector<Mixture> best_mixtures_mml,best_mixtures_all;
  int best_mml_iter;
  long double best_mml_msglen;
  std::vector<int> wins(NUM_METHODS,0);

  log << "\nEstimating mixtures using all methods ...\n";
  log << "No. of components: " << num_components << endl << endl;
  for (int iter=1; iter<=NUM_ITERATIONS; iter++) {
    std::vector<Mixture> current_mixtures;
    Vector current_msglens(NUM_METHODS,0);
    log << "Iteration # " << iter << " ...\n";
    for (int i=0; i<NUM_METHODS; i++) {
      setEstimationMethod(i);
      Mixture mixture(num_components,data,data_weights);
      mixture.estimateParameters();
      current_msglens[i] = mixture.getMinimumMessageLength();
      if (i == ML_APPROX) {
        log << "ML_approx mixture\n";
      } else if (i == TANABE) {
        log << "Tanabe mixture\n";
      } else if (i == TRUNCATED_NEWTON) {
        log << "Truncated Newton mixture\n";
      } else if (i == SONG) {
        log << "Song mixture\n";
      } else if (i == MML_NEWTON) {
        log << "MML_Newton mixture\n";
      } else if (i == MML_HALLEY) {
        log << "MML_Halley mixture\n";
      }
      mixture.printParameters(log,1);
      current_mixtures.push_back(mixture);
    } // i loop ends ...
    int winner = minimumIndex(current_msglens);
    wins[winner]++;
    if (winner != MML_HALLEY) {
      log << "Iteration # " << iter << " unsuccessful ...\n\n";
      print(log,current_msglens,3); 
      log << endl;
    }
    if (iter != 1) {
      // update best mml mixtures
      if (current_msglens[MML_HALLEY] < best_mml_msglen) {
        best_mml_iter = iter;
        best_mml_msglen = current_msglens[MML_HALLEY];
        best_mixtures_mml = current_mixtures;
      }
      // update best all mixtures
      for (int i=0; i<NUM_METHODS; i++) {
        if (current_msglens[i] < best_mixtures_all[i].getMinimumMessageLength()) {
          best_mixtures_all[i] = current_mixtures[i];
        }
      } // i loop ends ...
    } else if (iter == 1) {
      best_mml_iter = iter;
      best_mml_msglen = current_msglens[MML_HALLEY];
      best_mixtures_mml = current_mixtures;
      best_mixtures_all = current_mixtures;
    }
  } // iter loop ends ...

  log << "Wins (ML_APPROX:Tanabe:TruncNewton:Song:MML_NEWTON:MML_Halley): (" 
      << wins[ML_APPROX] << ":" << wins[TANABE] << ":" << wins[TRUNCATED_NEWTON] << ":"
      << wins[SONG] << ":" << wins[MML_NEWTON] << ":" << wins[MML_HALLEY] << ")\n";
      
  log << "Best MML mixture @ iteration: " << best_mml_iter << endl;
  log << "BEST MIXTURES ALL:\n";
  for (int i=0; i<best_mixtures_all.size(); i++) {
    best_mixtures_all[i].printParameters(log,1);
  }
  return pair<std::vector<Mixture>,std::vector<Mixture> >(best_mixtures_mml,best_mixtures_all);
}

/*!
 *  \brief This function is used to infer optimum number of mixture components.
 *  \param mixture a reference to a Mixture
 *  \param N an integer
 *  \param log a reference to a ostream 
 */
Mixture inferComponents(Mixture &mixture, int N, ostream &log)
{
  int K,iter = 0;
  std::vector<vMF> components;
  Mixture modified,improved,parent;
  Vector sample_size;
  //long double min_n = 0.01 * N;
  long double min_n = 1;
  long double null_msglen = mixture.computeNullModelMessageLength();
  log << "Null model encoding: " << null_msglen << " bits."
      << "\t(" << null_msglen/N << " bits/point)\n\n";

  improved = mixture;

  while (1) {
    parent = improved;
    iter++;
    log << "Iteration #" << iter << endl;
    log << "Parent:\n";
    parent.printParameters(log,1);
    components = parent.getComponents();
    sample_size = parent.getSampleSize();
    K = components.size();
    for (int i=0; i<K; i++) { // split() ...
      if (sample_size[i] > min_n) {
        modified = parent.split(i,log);
        updateInference(modified,improved,log,SPLIT);
      }
    }
    if (K >= 2) {  // kill() ...
      for (int i=0; i<K; i++) {
        modified = parent.kill(i,log);
        updateInference(modified,improved,log,KILL);
      } // killing each component
    } // if (K > 2) loop
    if (K > 1) {  // join() ...
      for (int i=0; i<K; i++) {
        int j = parent.getNearestComponent(i); // closest component
        modified = parent.join(i,j,log);
        updateInference(modified,improved,log,JOIN);
      } // join() ing nearest components
    } // if (K > 1) loop
    if (improved == parent) goto finish;
  } // if (improved == parent || iter%2 == 0) loop

  finish:
  return parent;
}

/*!
 *  \brief Updates the inference
 *  \param modified a reference to a Mixture
 *  \param current a reference to a Mixture
 *  \param log a reference to a ostream
 *  \param operation an integer
 */
void updateInference(Mixture &modified, Mixture &current, ostream &log, int operation)
{
  long double modified_msglen = modified.getMinimumMessageLength();
  long double current_msglen = current.getMinimumMessageLength();

  if (modified_msglen < current_msglen) {   // ... improvement
    long double improvement_rate = (current_msglen - modified_msglen) / current_msglen;
    if (operation == JOIN || 
        improvement_rate > IMPROVEMENT_RATE) {  // there is > 0.001 % improvement
      current = modified;
      log << "\t ... IMPROVEMENT ... (+ " << fixed << setprecision(3) 
          << 100 * improvement_rate << " %) ";
      if (operation == JOIN && improvement_rate < IMPROVEMENT_RATE) {
        log << "\t\t[ACCEPT] with negligible improvement (while joining)!\n\n";
      } else {
        log << "\t\t[ACCEPT]\n\n";
      }
    } else {  // ... no substantial improvement
      log << "\t ... IMPROVEMENT < " << fixed << setprecision(3) 
          << 100 * IMPROVEMENT_RATE << " %\t\t\t[REJECT]\n\n";
    }
  } else {    // ... no improvement
    log << "\t ... NO IMPROVEMENT\t\t\t[REJECT]\n\n";
  }
}

/*!
 *  \brief Infers stable mixtures from K=min_k to K=max_k and plots them
 */
void inferStableMixtures(
  std::vector<Vector> &data, 
  int min_k, 
  int max_k, 
  string &log_file
) {
  ofstream log(log_file.c_str());

  string msglens_file = log_file + ".msglens.dat";
  string aic_file = log_file + ".aic.dat";
  string bic_file = log_file + ".bic.dat";
  string parts_file = log_file + ".parts.dat";
  ofstream msglens(msglens_file.c_str());
  ofstream aic(aic_file.c_str());
  ofstream bic(bic_file.c_str());
  ofstream parts(parts_file.c_str());

  Vector current_msglens(NUM_METHODS,0),current_aic(NUM_METHODS,0),current_bic(NUM_METHODS,0);
  Vector individual_msglens(2,0);

  for (int k=min_k; k<=max_k; k++) {
    pair<std::vector<Mixture>,std::vector<Mixture> > 
      mixtures = estimateMixturesUsingAllMethods(k,data,log);
    std::vector<Mixture> best_mixtures_mml = mixtures.first;
    std::vector<Mixture> best_mixtures_all = mixtures.second;
    for (int i=0; i<NUM_METHODS; i++) {
      current_msglens[i] = best_mixtures_all[i].getMinimumMessageLength();
      current_aic[i] = best_mixtures_all[i].computeAIC();
      current_bic[i] = best_mixtures_all[i].computeBIC();
      if (i == MML_HALLEY) {
        individual_msglens[0] = best_mixtures_all[i].first_part();
        individual_msglens[1] = best_mixtures_all[i].second_part();
      } 
    } // i loop ends ...
    msglens << fixed << setw(10) << k;
    aic << fixed << setw(10) << k;
    bic << fixed << setw(10) << k;
    parts << fixed << setw(10) << k;

    for (int i=0; i<NUM_METHODS; i++) {
      msglens << fixed << setw(15) << setprecision(3) << current_msglens[i];
      aic << fixed << setw(15) << setprecision(3) << current_aic[i];
      bic << fixed << setw(15) << setprecision(3) << current_bic[i];
    }
    msglens << endl; aic << endl; bic << endl;
    parts << fixed << setw(15) << setprecision(3) << individual_msglens[0];
    parts << fixed << setw(15) << setprecision(3) << individual_msglens[1];
    parts << fixed << setw(15) << setprecision(3) 
          << individual_msglens[0]+individual_msglens[1] << endl;

  } // k loop ends ...
  string script_file,data_file,plot_file,ylabel,title;
  for (int i=0; i<3; i++) {
    if (i == 0) {
      script_file = log_file + ".msglens.p";
      data_file = msglens_file;
      plot_file = log_file + ".msglens.eps";
      title = "COMPARISON OF MESSAGE LEGTHS";
      ylabel = "Message length (in bits)";
    } else if (i == 1) {
      script_file = log_file + ".aic.p";
      data_file = aic_file;
      plot_file = log_file + ".aic.eps";
      title = "COMPARISON OF AIC VALUES";
      ylabel = "AIC (measured in bits)";
    } else if (i == 2) {
      script_file = log_file + ".bic.p";
      data_file = bic_file;
      plot_file = log_file + ".bic.eps";
      title = "COMPARISON OF BIC VALUES";
      ylabel = "BIC (measured in bits)";
    }
    ofstream script(script_file.c_str());
    script << "set terminal post eps" << endl ;
    script << "set autoscale\t" ;
    script << "# scale axes automatically" << endl ;
    script << "set xtic auto\t" ;
    script << "# set xtics automatically" << endl ;
    script << "set ytic auto\t" ;
    script << "# set ytics automatically" << endl ;
    script << "set style data lines" << endl;
    script << "set title \"" << title << "\"" << endl ;
    script << "set xlabel \"Number of components (K)\"" << endl ;
    script << "set ylabel \"" << ylabel << "\"" << endl ;
    script << "set xrange [" << min_k << ":]" << endl;
    script << "set output \"" << plot_file << "\"" << endl ;
    script << "plot \"" << data_file << "\" using 1:2 title \"Banerjee\" ";
    script << "lt 1 lc rgb \"red\", \\" << endl;
    script << "\"" << data_file << "\" using 1:3 title \"MLE\" ";
    script << "lt 1 lc rgb \"blue\", \\" << endl;
    script << "\"" << data_file << "\" using 1:4 title \"Tanabe\" ";
    script << "lt 1 lc rgb \"green\", \\" << endl ;
    script << "\"" << data_file << "\" using 1:5 title \"Truncated Newton\" ";
    script << "lt 1 lc rgb \"gold\", \\" << endl ;
    script << "\"" << data_file << "\" using 1:6 title \"Song\" ";
    script << "lt 1 lc rgb \"orange\", \\" << endl ;
    script << "\"" << data_file << "\" using 1:7 title \"MML_Newton\" ";
    script << "lt 1 lc rgb \"purple\", \\" << endl ;
    script << "\"" << data_file << "\" using 1:8 title \"MML_Halley\" ";
    script << "lt 1 lc rgb \"black\", \\" << endl ;
    script << "\"" << data_file << "\" using 1:9 title \"MML_Complete\" ";
    script << "lt 1 lc rgb \"brown\"" << endl ;
    script.close();
    string cmd = "gnuplot -persist " + script_file; 
    if(system(cmd.c_str()));
  } // i loop ends ...
}

