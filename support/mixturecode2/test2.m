pp = [0.5];
mu1 = [0 0];
mu2 = [10 0];
mu = [mu1' mu2'];
covar(:,:,1) = [1 0; 0 1];
covar(:,:,2) = [1 0; 0 1];

common_file_prefix = './data/mvnorm_iter_';
dat = '.dat';
summary = fopen('summary','w');
parameters = fopen('parameters','w');
for iter = 1:50
  y = genmix(800,mu,covar,pp);
  %clear covar mu mu1 mu2 mu3
  [bestk,bestpp,bestmu,bestcov,dl,countf] = mixtures4(y,1,25,0,1e-4,0)
  sample = y';
  iter_str = int2str(iter);
  file_name = strcat(common_file_prefix,iter_str,dat);
  save(file_name,'sample','-ascii');
  fprintf(summary,'%6d %6d %6d\n',iter,bestk,countf);
  fprintf(parameters,'\nIter: %d\n',iter);
  fprintf(parameters,'bestpp:\n');
  fprintf(parameters,'%f ',bestpp); fprintf(parameters,'\n')
  fprintf(parameters,'bestmu:\n');
  fprintf(parameters,'%f ',bestmu); fprintf(parameters,'\n')
  fprintf(parameters,'bestcov:\n');
  fprintf(parameters,'%f ',bestcov); fprintf(parameters,'\n')
end
fclose(summary);
fclose(parameters);
