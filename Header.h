#ifndef HEADER_H
#define HEADER_H

#include <iostream>
#include <memory>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <ctime>
#include <cassert>
#include <omp.h>

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include <boost/filesystem.hpp>
#include <boost/random.hpp>
#include <boost/multiprecision/mpfr.hpp>

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/special_functions/detail/erf_inv.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/distributions.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace std;
using namespace boost::program_options;
using namespace boost::filesystem;
using namespace boost::math;
using namespace boost::numeric::ublas;
using namespace boost::multiprecision;
using namespace boost::math::tools;

typedef std::vector<long double> Vector;
typedef boost::numeric::ublas::matrix<long double> Matrix;
typedef boost::numeric::ublas::identity_matrix<long double> IdentityMatrix;
typedef boost::numeric::ublas::zero_matrix<long double> ZeroMatrix;
typedef number<mpfr_float_backend<1000> >  my_float; 

// numeric constants
#define AOM 0.001
#define LARGE_NUMBER 1000000000
#define PI boost::math::constants::pi<double>()
#define LOG_PI log(PI)
#define ZERO std::numeric_limits<long double>::epsilon()
#define TOLERANCE 1e-6

#define SET 1 
#define UNSET 0

#define ML_APPROX 0 
#define TANABE 1
#define TRUNCATED_NEWTON 2
#define SONG 3
#define MML_NEWTON 4
#define MML_HALLEY 5
#define NUM_METHODS 6

#define ML 6 
#define MML_COMPLETE 7

#define DEFAULT_RESOLUTION 1
#define MAX_COMPONENTS 100
#define DEFAULT_FIT_COMPONENTS 2
#define DEFAULT_SIMULATE_COMPONENTS 2
#define DEFAULT_SAMPLE_SIZE 2000
#define DEFAULT_MAX_KAPPA 10000 
#define MIN_KAPPA 0.001

#define SPLIT 0
#define KILL 1
#define JOIN 2

#endif

