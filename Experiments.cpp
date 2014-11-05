#include "Experiments.h"

extern int MIXTURE_SIMULATION;
extern int INFER_COMPONENTS;
extern int ENABLE_DATA_PARALLELISM;
extern int NUM_THREADS;
extern int ESTIMATION;
extern long double IMPROVEMENT_RATE;
extern int NUM_STABLE_COMPONENTS;
extern int TOTAL_ITERATIONS;

Experiments::Experiments(int iterations) : iterations(iterations)
{}

void Experiments::simulate(int D)
{
  int N = 10000;

  Vector mean(D,0);
  Matrix cov = IdentityMatrix(D,D);
  MultivariateNormal mvnorm(mean,cov),mvnorm_est;

  string D_str = boost::lexical_cast<string>(D);
  string size_str = boost::lexical_cast<string>(N);
  string folder = "./experiments/D_" + D_str + "/";
  string negloglkhd_file = folder + "n_" + size_str + "_negloglikelihood";
  string kldvg_file = folder + "n_" + size_str + "_kldiv";
  string msglens_file = folder + "n_" + size_str + "_msglens";
  ofstream logneg(negloglkhd_file.c_str(),ios::app);
  ofstream logkldiv(kldvg_file.c_str(),ios::app);
  ofstream logmsg(msglens_file.c_str(),ios::app);

  Vector emptyvec(2,0);
  std::vector<Vector> negloglkhd(iterations,emptyvec),kldiv(iterations,emptyvec),msglens(iterations,emptyvec);
  long double actual_negloglkhd,actual_msglen;
  for (int iter=0; iter<iterations; iter++) {
    std::vector<Vector> data = mvnorm.generate(N);
    struct Estimates estimates;
    mvnorm_est = MultivariateNormal(mean,cov);
    mvnorm_est.computeAllEstimators(data,estimates);

    actual_negloglkhd = mvnorm.computeNegativeLogLikelihood(data) / log(2);
    actual_msglen = mvnorm.computeMessageLength(data);
    logneg << scientific << actual_negloglkhd << "\t";
    logmsg << scientific << actual_msglen << "\t";

    // ML
    MultivariateNormal fit(estimates.mean,estimates.cov_ml);
    negloglkhd[iter][0] = fit.computeNegativeLogLikelihood(data) / log(2);
    msglens[iter][0] = fit.computeMessageLength(data);
    kldiv[iter][0] = mvnorm.computeKLDivergence(fit);
    logneg << scientific << negloglkhd[iter][0] << "\t";
    logmsg << scientific << msglens[iter][0] << "\t";
    logkldiv << scientific << kldiv[iter][0] << "\t";

    // MML
    fit = MultivariateNormal(estimates.mean,estimates.cov_mml);
    negloglkhd[iter][1] = fit.computeNegativeLogLikelihood(data) / log(2);
    msglens[iter][1] = fit.computeMessageLength(data);
    kldiv[iter][1] = mvnorm.computeKLDivergence(fit);
    logneg << scientific << negloglkhd[iter][1] << "\t";
    logmsg << scientific << msglens[iter][1] << "\t";
    logkldiv << scientific << kldiv[iter][1] << "\t";
    
    logneg << endl; logmsg << endl; logkldiv << endl; 
  }
  logneg.close(); logmsg.close(); logkldiv.close();
}

// ./experiments/infer_components/exp1/
// bivariate data, covariance = I
void Experiments::infer_components_exp1()
{
  int N = 800;
  long double delta = 1.10;

  int D = 10;
  Vector mu1(D,0);
  Vector mu2(D,0); mu2[0] = delta;
  Matrix C1 = IdentityMatrix(D,D);
  Matrix C2 = IdentityMatrix(D,D);

  MultivariateNormal mvnorm1(mu1,C1);
  MultivariateNormal mvnorm2(mu2,C2);

  Vector weights(2,0.5);
  std::vector<MultivariateNormal> components;
  components.push_back(mvnorm1);
  components.push_back(mvnorm2);
  Mixture original(2,components,weights);

  // iterations = 50 (in paper)
  struct Parameters parameters;
  parameters.simulation = SET;
  parameters.load_mixture = UNSET;
  parameters.D = D;
  parameters.read_profiles = UNSET;
  parameters.mixture_model = SET;
  parameters.infer_num_components = SET;
  parameters.max_components = -1;
  parameters.continue_inference = UNSET;
  parameters.start_from = 1;
  parameters.heat_map = UNSET;
  parameters.sample_size = N;
  INFER_COMPONENTS = SET;
  MIXTURE_SIMULATION = SET;
  NUM_THREADS = 1;
  ENABLE_DATA_PARALLELISM = UNSET;
  ESTIMATION = MML;
  IMPROVEMENT_RATE = 0.002;
    
  string results_folder = "./experiments/infer_components/exp2/";
  //for (delta=2.4; delta<=2.6; delta+=0.1) {
    inferExperimentalMixtures(original,delta,results_folder,parameters);
  //}
}

void Experiments::inferExperimentalMixtures(
  Mixture &original, 
  long double delta,
  string &results_folder,
  struct Parameters &parameters
) {
  std::ostringstream ss;
  ss << fixed << setprecision(2);
  ss << delta;
  string delta_str = "delta_" + ss.str();
  //cout << "delta_str: " << delta_str << endl;

  string infer_log,summary_log;
  summary_log = results_folder + "summary/" + delta_str;
  ofstream summary(summary_log.c_str());
  std::vector<Vector> data;
  for (int iter=1; iter<=iterations; iter++) {
    string iter_str = boost::lexical_cast<string>(iter);
    infer_log = results_folder + "logs/" + delta_str + "_iter_" + iter_str;
    parameters.infer_log = infer_log;

    //data = original.generate(parameters.sample_size,0);
    string data_file = "./support/mixturecode2/exp2/data/" + delta_str + "/"
                       + "mvnorm_iter_" + iter_str + ".dat";
    data = load_matrix(data_file,parameters.D);
    modelMixture(parameters,data);

    // update summary file
    summary << "\t\t" << iter << "\t\t" << NUM_STABLE_COMPONENTS << "\t\t" 
            << TOTAL_ITERATIONS << endl;
  }
  summary.close();
}

