clear
iterations = 50;
formatspec = '%.2f';
for delta=1.10:0.05:1.61
  %delta = 1.30
  pp = [0.5];
  mu1 = zeros(1,10);
  mu2 = delta * ones(1,10);
  mu = [mu1' mu2'];
  clear covar;
  covar(:,:,1) = eye(10);
  covar(:,:,2) = eye(10);
  
  delta_str = num2str(delta,formatspec);
  %common_file_prefix = strcat('./exp2_45/data/delta_',delta_str,'/mvnorm_iter_');
  summary_file = strcat('./exp2_45/summary/delta_',delta_str);
  summary = fopen(summary_file,'w');
  params_file = strcat('./exp2_45/summary/delta_',delta_str,'_parameters');
  parameters = fopen(params_file,'w');
  data_folder = strcat('./exp2/data/delta_',delta_str);
  success_rate = 0;
  inferred = []
  for iter = 1:iterations
    iter_str = int2str(iter);
    data_file = strcat(data_folder,'/mvnorm_iter_',iter_str,'.dat')
    sample = load(data_file);
    y = sample';
%   y = genmix(800,mu,covar,pp);
    [bestk,bestpp,bestmu,bestcov,dl,countf] = mixtures4(y,1,45,0,1e-5,0)
%    sample = y';
    if (bestk == 2)
      success_rate = success_rate + 1;
    end
%    file_name = strcat(common_file_prefix,iter_str,'.dat');
%    save(file_name,'sample','-ascii');
    fprintf(summary,'%6d %6d %6d\n',iter,bestk,countf);
    fprintf(parameters,'\nIter: %d\n',iter);
    fprintf(parameters,'bestpp:\n');
    fprintf(parameters,'%f ',bestpp); fprintf(parameters,'\n')
    fprintf(parameters,'bestmu:\n');
    fprintf(parameters,'%f ',bestmu); fprintf(parameters,'\n')
    fprintf(parameters,'bestcov:\n');
    fprintf(parameters,'%f ',bestcov); fprintf(parameters,'\n')
    mix_file = strcat('./exp2_45/mixtures/delta_',delta_str,'/mvnorm_iter_',iter_str);
    mixout = fopen(mix_file,'w');
    inferred = [inferred bestk];
    for k=1:bestk
      w = bestpp(k);
      mu_est = bestmu(:,k);
      cov_est = bestcov(:,:,k);
      fprintf(mixout,'\t%.5f\t\t',w);
      fprintf(mixout,'[mu]: (');
      D = 10;
      for i=1:D-1
        fprintf(mixout,'%.6e, ',mu_est(i,1));
      end
      fprintf(mixout,'%.6e)',mu_est(D,1));
      fprintf(mixout,'\t\t[cov]: (');
      for i=1:D
        fprintf(mixout,'(');
        for j=1:D-1
          fprintf(mixout,'%.6e, ',cov_est(i,j));
        end
        fprintf(mixout,'%.6e)',cov_est(i,D));
      end
      fprintf(mixout,')\n');
    end
    fclose(mixout);
  end
  avg_number = mean(inferred);
  variance = var(inferred);
  fprintf(summary,'\nsuccess rate: %.2f %%\n',success_rate*100/iterations);
  fprintf(summary,'Avg:%f\n',avg_number);
  fprintf(summary,'Variance:%f\n',variance);
  fclose(summary);
  fclose(parameters);
end
