for delta=1.8:0.1:2.6
    pp = [0.5];
    mu1 = [0 0];
    mu2 = [delta 0];
    mu = [mu1' mu2'];
    covar(:,:,1) = [1 0; 0 1];
    covar(:,:,2) = [1 0; 0 1];
    
    delta_str = num2str(delta);
    common_file_prefix = strcat('./exp1/data/delta_',delta_str,'/mvnorm_iter_');
    summary_file = strcat('./exp1/summary/',delta_str);
    summary = fopen(summary_file,'w');
    params_file = strcat('./exp1/summary/',delta_str,'_parameters')
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
      mix_file = strcat('./exp1/mixtures/delta_',delta_str,'/mvnorm_iter_',iter_str);
      mixout = fopen(mix_file,'w');
      for k=1:bestk
        w = bestpp(k);
        mu = bestmu(:,k);
        C = bestcov(:,:,k);
        fprintf(mixout,'\t%.5f\t\t',w);
        fprintf(mixout,'[mu]: (');
        D = 2;
        for i=1:D-1
          fprintf(mixout,'%.6e, ',mu(i,1));
        end
        fprintf(mixout,'%.6e)',mu(D,1));
        fprintf(mixout,'\t\t[cov]: (');
        for i=1:D
          fprintf(mixout,'(');
          for j=1:D-1
            fprintf(mixout,'%.6e, ',C(i,j));
          end
          fprintf(mixout,'%.6e)',C(i,D));
        end
        fprintf(mixout,')\n');
      end
      fclose(mixout);
    end
    fclose(summary);
    fclose(parameters);
end
