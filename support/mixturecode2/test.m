data = load('./data/mvnorm_iter_1.dat')
y = data'
clear covar mu mu1 mu2 mu3
[bestk,bestpp,bestmu,bestcov,dl,countf] = mixtures4(y,1,25,0,1e-4,0)
