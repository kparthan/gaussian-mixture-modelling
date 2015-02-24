clear;
clc;
%data = load('../../random_sample.dat')
data = load('../acidity.dat')
%data = load('../iris.dat')
y = data'
%delta = 2.5
%pp = [0.5];
%mu1 = [0 0];
%mu2 = [delta 0];
%mu = [mu1' mu2'];
%covar(:,:,1) = [1 0; 0 1];
%covar(:,:,2) = [1 0; 0 1];
%y = genmix(100,mu,covar,pp);
[bestk,bestpp,bestmu,bestcov,dl,countf] = mixtures4(y,1,25,0,1e-5,0)

% print the inferred mixture
file = fopen('../../simulation/inferred_mixture_2','w');
D = size(data,2);
for k=1:bestk
  w = bestpp(k);
  mu = bestmu(:,k);
  C = bestcov(:,:,k);
  fprintf(file,'\t%.5f\t\t',w);
  fprintf(file,'[mu]: (');
  for i=1:D-1
    fprintf(file,'%.6e, ',mu(i,1));
  end
  fprintf(file,'%.6e)',mu(D,1));
  fprintf(file,'\t\t[cov]: (');
  for i=1:D
    fprintf(file,'(');
    for j=1:D-1
      fprintf(file,'%.6e, ',C(i,j));
    end
    fprintf(file,'%.6e)',C(i,D));
  end
  fprintf(file,')\n');
end
fclose(file);
