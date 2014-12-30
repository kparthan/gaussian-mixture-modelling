iterations = 50;

folder1 = '../../experiments/infer_components/exp_spiral/';
folder2 = './exp_spiral/';

summary1 = strcat(folder1,'summary');
res1 = load(summary1);
summary2 = strcat(folder2,'summary');
res2 = load(summary2);

for iter=1:iterations
  iter_str = int2str(iter);
  
  K = res1(iter,2);
  mix = strcat(folder1,'mixtures/mixture_iter_',iter_str);
  [mus,covs] = parse_mixture_file(K,3,mix);
  output_file = strcat(folder1,'connectors/connectors_iter_',iter_str);
  fw = fopen(output_file,'w');
  D = zeros(3);
  for i=1:K
    mu = mus(:,i);
    covar = covs(:,:,i);
    [lambda,psi,T] = factoran(covar,1,'xtype','cov');
    for j=1:3
      D(j,j) = sqrt(covar(j,j));
      Lambda = D * lambda;
    end
    sp = mu - 2 * Lambda;
    ep = mu + 2 * Lambda;
    for j=1:3
      fprintf(fw,'%.6e\t',sp(j,1));
    end
    for j=1:3
      fprintf(fw,'%.6e\t',ep(j,1));
    end
    fprintf(fw,'\n');
  end
  fclose(fw);

  K = res2(iter,2);
  mix = strcat(folder2,'mixtures/mixture_iter_',iter_str);
  [mus,covs] = parse_mixture_file(K,3,mix);
  output_file = strcat(folder2,'connectors/connectors_iter_',iter_str);
  fw = fopen(output_file,'w');
  D = zeros(3);
  for i=1:K
    mu = mus(:,i);
    covar = covs(:,:,i);
    [lambda,psi,T] = factoran(covar,1,'xtype','cov');
    for j=1:3
      D(j,j) = sqrt(covar(j,j));
      Lambda = D * lambda;
    end
    sp = mu - 2 * Lambda;
    ep = mu + 2 * Lambda;
    for j=1:3
      fprintf(fw,'%.6e\t',sp(j,1));
    end
    for j=1:3
      fprintf(fw,'%.6e\t',ep(j,1));
    end
    fprintf(fw,'\n');
  end
  fclose(fw);
end
