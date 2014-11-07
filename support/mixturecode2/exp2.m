for delta=1.1:0.05:1.5
    %delta = 1.3
    pp = [0.5];
    mu1 = zeros(1,10);
    mu2 = delta * ones(1,10);
    mu = [mu1' mu2'];
    clear covar;
    covar(:,:,1) = eye(10);
    covar(:,:,2) = eye(10);
    
    delta_str = num2str(delta);
    common_file_prefix = strcat('./exp2/data/delta_',delta_str,'/mvnorm_iter_');
    summary_file = strcat('./exp2/summary/',delta_str);
    summary = fopen(summary_file,'w');
    params_file = strcat('./exp2/summary/',delta_str,'_parameters')
    parameters = fopen(params_file,'w');
    for iter = 1:50
      y = genmix(800,mu,covar,pp);
      [bestk,bestpp,bestmu,bestcov,dl,countf] = mixtures4(y,1,25,0,1e-4,0)
      sample = y';
      iter_str = int2str(iter);
      file_name = strcat(common_file_prefix,iter_str,'.dat');
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
end
