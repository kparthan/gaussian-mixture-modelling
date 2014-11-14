#ifndef MIXTURE_H
#define MIXTURE_H

#include "MultivariateNormal.h"

class Mixture
{
  private:
    //! ID
    int id;

    //! Sample size
    int N;

    //! Dimensionality of data
    int D;

    //! Number of components
    int K;

    //! List of components
    std::vector<MultivariateNormal> components;
    
    //! Sample (x_i) -- Cartesian coordinates
    std::vector<Vector> data;

    //! Data weights
    Vector data_weights;

    //! Responsibility matrix (K X N)
    std::vector<Vector> responsibility;

    //! Effective sample size for each component (n_k)
    Vector sample_size;

    //! Weights of the components (a_k)
    Vector weights;

    //! List of message lengths over several iterations
    Vector msglens;

    //! Null model message length
    long double null_msglen;

    //! Optimal encoding length
    long double Ik,Iw,sum_It,Il,kd_term,part1,part2,minimum_msglen;
    Vector It;

  public:
    //! Null constructor
    Mixture();

    //! Constructor
    Mixture(int, std::vector<MultivariateNormal> &, Vector &);

    //! Constructor
    Mixture(int, std::vector<Vector> &, Vector &);

    //! Constructor
    Mixture(int, std::vector<MultivariateNormal> &, Vector &, Vector &, 
            std::vector<Vector> &, std::vector<Vector> &, Vector &);

    //! Overloading = operator
    Mixture operator=(const Mixture &);

    //! Overloading == operator
    bool operator==(const Mixture &);

    //! Prepare log file
    string getLogFile();

    //! Gets the list of weights
    Vector getWeights();

    //! Gets the list of components
    std::vector<MultivariateNormal> getComponents();

    //! Returns number of components
    int getNumberOfComponents();

    //! Gets the responsibility matrix
    std::vector<Vector> getResponsibilityMatrix();

    //! Gets the sample size
    Vector getSampleSize();

    //! Initialize parameters
    void initialize();
    void initialize2();
    void initialize3();
    void initialize4();

    //! Updates the effective sample size
    void updateEffectiveSampleSize();
    void updateEffectiveSampleSize(int);

    //! Update the component weights
    void updateWeights();
    void updateWeights(int);

    void updateWeights_ML();
    void updateWeights_ML(int);

    //! Update components
    int updateComponents();
    void updateComponents(int);

    //! Update the responsibility matrix
    int updateResponsibilityMatrix();
    void updateResponsibilityMatrix(int);

    //! Computes the responsibility matrix
    void computeResponsibilityMatrix(std::vector<Vector> &, string &);
                                          
    //! Probability of a datum
    long double log_probability(Vector &);

    //! Computes the negative log likelihood
    long double negativeLogLikelihood(std::vector<Vector> &);

    //! Computes the minimum message length
    long double computeMinimumMessageLength();

    void printIndividualMsgLengths(ostream &);

    //! Gets the minimum message length
    long double getMinimumMessageLength();

    //! Gets the first part
    long double first_part();

    //! Gets the second part
    long double second_part();

    //! Estimate mixture parameters
    long double estimateParameters();

    //! EM loop
    void EM();

    void CEM();

    //! Prints the model parameters
    void printParameters(ostream &, int, long double);

    //! Prints the model parameters
    void printParameters(ostream &, int);

    //! Prints the model parameters
    void printParameters(ostream &);

    //! Loads the mixture file
    void load(string &, int);

    //! Loads the mixture file with the corresponding data
    void load(string &, int, std::vector<Vector> &, Vector &);

    //! Randomly choose a component
    int randomComponent();

    //! Saves the data generated from a component
    void saveComponentData(int, std::vector<Vector> &);

    //! Generate random data from the distribution using mixture proportions
    std::vector<Vector> generate(int, bool);

    //! Splits a component
    Mixture split(int, ostream &);

    //! Deltes a component
    Mixture kill(int, ostream &);

    //! Joins two  components
    Mixture join(int, int, ostream &);

    //! Generate heat maps (for d=3)
    void generateHeatmapData(int, long double, int);

    //! Get the nearest component
    int getNearestComponent(int);

    //! Computes the approx KL divergence between two mixtures
    long double computeKLDivergence(Mixture &);

    long double computeKLDivergence(Mixture &, std::vector<Vector> &);

    long double computeKLDivergenceUpperBound(Mixture &);

    long double computeKLDivergenceLowerBound(Mixture &);

    long double computeKLDivergenceAverageBound(Mixture &);
};

#endif

