clear
iterations = 50;
formatspec = '%.2f';
delta = 2.0
pp = [0.5];
mu1 = [0 0];
mu2 = [delta 0];
mu = [mu1' mu2'];
clear covar;
covar(:,:,1) = eye(2);
covar(:,:,2) = eye(2);

folder = '../../experiments/infer_components/exp3/';
fout = fopen('./exp3/avg_inference','w');
for N=600:50:4001
  inferred = []
  N_str = strcat('N_',num2str(N));
  data_folder = strcat(folder,N_str);
  summary_file = strcat('./exp3/',N_str)
  summary = fopen(summary_file,'w');
  for iter = 1:iterations
    iter_str = int2str(iter);
    data_file = strcat(data_folder,'/mvnorm_iter_',iter_str,'.dat')
    sample = load(data_file);
    y = sample';
    [bestk,bestpp,bestmu,bestcov,dl,countf] = mixtures4(y,1,25,0,1e-5,0)
    fprintf(summary,'%d\n',bestk);
    inferred = [inferred bestk];
  end
  fclose(summary);
  avg_inferred = mean(inferred);
  variance = var(inferred);
  fprintf(fout,'%d\t\t%f\t\t%f\n',N,avg_inferred,variance);
end
fclose(fout);
