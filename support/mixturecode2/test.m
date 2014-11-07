clear;
clc;
%data = load('./data/mvnorm_iter_1.dat')
%data = load('../../visualize/test_4.dat')
data = load('../../random_sample.dat')
y = data'
[bestk,bestpp,bestmu,bestcov,dl,countf] = mixtures4(y,1,25,0,1e-4,0)
