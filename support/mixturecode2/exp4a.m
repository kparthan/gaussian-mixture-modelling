clear
iterations = 50;
formatspec = '%.2f';
delta = 10
pp = [0.5];
mu1 = zeros(1,10);
mu2 = delta * ones(1,10);
mu = [mu1' mu2'];
clear covar;
covar(:,:,1) = eye(10);
covar(:,:,2) = eye(10);

folder = '../../experiments/infer_components/exp4a/delta_10/';
fout = fopen('./exp4a/delta_10/avg_inference','w');
for N=50:50:201
  inferred = []
  N_str = strcat('N_',num2str(N));
  data_folder = strcat(folder,N_str);
  summary_file = strcat('./exp4a/delta_10/',N_str)
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
