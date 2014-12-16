clear
iterations = 50;

summary_file = './exp_spiral/summary';
summary = fopen(summary_file,'w');
inferred = []
D = 3;
for iter = 1:iterations
  iter_str = int2str(iter);
  data_file = strcat('../../experiments/infer_components/exp_spiral/data/spiral_iter_',iter_str,'.dat')
  sample = load(data_file);
  y = sample';
  [bestk,bestpp,bestmu,bestcov,dl,countf] = mixtures4(y,1,30,0,1e-5,0)
  fprintf(summary,'%6d %6d %6d\n',iter,bestk,countf);
  mix_file = strcat('./exp_spiral/mixtures/mixture_iter_',iter_str);
  mixout = fopen(mix_file,'w');
  inferred = [inferred bestk];
  for k=1:bestk
    w = bestpp(k);
    mu_est = bestmu(:,k);
    cov_est = bestcov(:,:,k);
    fprintf(mixout,'\t%.5f\t\t',w);
    fprintf(mixout,'[mu]: (');
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
fprintf(summary,'\nAvg:%f\n',avg_number);
fprintf(summary,'Variance:%f\n',variance);
fclose(summary);
