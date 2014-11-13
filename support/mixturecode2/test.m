clear;
clc;
%data = load('./data/mvnorm_iter_1.dat')
%data = load('../../visualize/test_4.dat')
data = load('../../random_sample.dat')
y = data'
[bestk,bestpp,bestmu,bestcov,dl,countf] = mixtures4(y,1,25,0,1e-4,0)

% print the inferred mixture
file = fopen('../../inferred_mixture_2','w');
D = size(data,2);
for k=1:bestk
  w = bestpp(k);
  mu = bestmu(:,k);
  C = bestcov(:,:,k);
  fprintf(file,'\t%.5f\t\t',w);
  fprintf(file,'[mu]: (');
  for i=1:D-1
    fprintf(file,'%.3f, ',mu(i,1));
  end
  fprintf(file,'%.3f)',mu(D,1));
  fprintf(file,'\t\t[cov]: (');
  for i=1:D
    fprintf(file,'(');
    for j=1:D-1
      fprintf(file,'%.3f, ',C(i,j));
    end
    fprintf(file,'%.3f)',C(i,D));
  end
  fprintf(file,')\n');
end
fclose(file);
