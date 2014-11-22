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

struct Parameters Experiments::setParameters(int N, int D)
{
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
  return parameters;
}

// ./experiments/infer_components/exp1/
// bivariate data, covariance = I
void Experiments::infer_components_exp1()
{
  int N = 800;
  int D = 2;
  int precision = 1;
  long double delta = 2.0;

  // iterations = 50 (in paper)
  struct Parameters parameters = setParameters(N,D);
    
  string results_folder = "./experiments/infer_components/exp1/";
  //for (long double delta=1.8; delta<=2.6; delta+=0.1) {
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
    //generateExperimentalMixtures(original,delta,results_folder,N,precision);
    inferExperimentalMixtures(original,delta,results_folder,parameters,precision);
  //}
}

void Experiments::infer_components_exp2()
{
  int N = 800;
  int D = 10;
  int precision = 2;
  long double delta = 1.25;

  // iterations = 50 (in paper)
  struct Parameters parameters = setParameters(N,D);
    
  string results_folder = "./experiments/infer_components/exp2/";
  //for (long double delta=1.15; delta<=1.60; delta+=0.05) {
    Vector mu1(D,0);
    Vector mu2(D,delta);
    Matrix C1 = IdentityMatrix(D,D);
    Matrix C2 = IdentityMatrix(D,D);

    MultivariateNormal mvnorm1(mu1,C1);
    MultivariateNormal mvnorm2(mu2,C2);

    Vector weights(2,0.5);
    std::vector<MultivariateNormal> components;
    components.push_back(mvnorm1);
    components.push_back(mvnorm2);
    Mixture original(2,components,weights);
    //generateExperimentalMixtures(original,delta,results_folder,N,precision);
    inferExperimentalMixtures(original,delta,results_folder,parameters,precision);
  //}
}

void Experiments::generateExperimentalMixtures(
  Mixture &original, 
  long double delta,
  string &results_folder,
  int sample_size,
  int precision
) {
  std::ostringstream ss;
  ss << fixed << setprecision(precision);
  ss << delta;
  string delta_str = "delta_" + ss.str();
  
  string iter_str,data_file;
  std::vector<Vector> data;
  for (int iter=1; iter<=iterations; iter++) {
    iter_str = boost::lexical_cast<string>(iter);
    data_file = results_folder + "data/" + delta_str + "/mvnorm_iter_" + iter_str + ".dat";
    data = original.generate(sample_size,0);
    writeToFile(data_file,data);
  }
}

void Experiments::inferExperimentalMixtures(
  Mixture &original, 
  long double delta,
  string &results_folder,
  struct Parameters &parameters,
  int precision
) {
  std::ostringstream ss;
  ss << fixed << setprecision(precision);
  ss << delta;
  string delta_str = "delta_" + ss.str();

  string iter_str,infer_log,data_file,summary_log;
  summary_log = results_folder + "summary/" + delta_str;
  ofstream summary(summary_log.c_str());
  std::vector<Vector> data;
  int num_success = 0;
  for (int iter=1; iter<=iterations; iter++) {
    iter_str = boost::lexical_cast<string>(iter);
    infer_log = results_folder + "logs/" + delta_str + "/mvnorm_iter_" + iter_str + ".log";
    parameters.infer_log = infer_log;

    data_file = results_folder + "data/" + delta_str + "/mvnorm_iter_" + iter_str + ".dat";
    //data_file = "./support/mixturecode2/exp2/data/" + delta_str + "/"
    //            + "mvnorm_iter_" + iter_str + ".dat";
    cout << "data_file: " << data_file << endl;
    data = load_matrix(data_file,parameters.D);

    Vector data_weights(data.size(),1.0);
    Mixture m(parameters.start_from,data,data_weights);
    Mixture mixture = m;
    mixture.estimateParameters();
    ofstream log(parameters.infer_log.c_str());
    Mixture stable = inferComponents(mixture,data.size(),data[0].size(),log);
    log.close();
    string ans = results_folder + "mixtures/" + delta_str + "/mvnorm_iter_" + iter_str;
    ofstream out(ans.c_str());
    Vector weights = stable.getWeights();
    std::vector<MultivariateNormal> components = stable.getComponents();
    for (int k=0; k<components.size(); k++) {
      out << "\t" << fixed << setw(10) << setprecision(5) << weights[k];
      out << "\t";
      components[k].printParameters(out,1);
    }
    out.close();

    // update summary file
    if (stable.getNumberOfComponents() == original.getNumberOfComponents()) num_success++;
    summary << "\t\t" << iter << "\t\t" << stable.getNumberOfComponents() << "\t\t" 
            << TOTAL_ITERATIONS << endl;
  }
  summary << "\nsuccess rate: " << setprecision(2) 
          << num_success * 100 / (long double)(iterations) << " %\n";
  summary.close();
}

