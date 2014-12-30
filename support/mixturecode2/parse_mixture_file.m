% K: # of components
% D: dimensionality
% mixture: mixture file
function [mus,covs] = parse_mixture_file(K,D,mixture)

fr = fopen(mixture,'r');
all_lines = textscan(fr, '%[^\n]', K);  % cell
fclose(fr);
for i=1:K
  line = all_lines{1}{i};
  [ans1,ans2] = strsplit(line,{'[mu]:','[cov]:','(',')','\t',' ',',',''});
  w_str = ans1(1);
  mat = cell2mat(w_str);
  w = str2num(mat);

  count = 1;
  mu = [];
  for j=1:D
    mu_str = ans1(j+count);
    mat = cell2mat(mu_str);
    val = str2num(mat);
    mu = [mu; val];
  end

  count = D + 2;
  covar = [];
  for j=1:D
    row = [];
    for k = 1:D
      cov_str = ans1(count);
      mat = cell2mat(cov_str);
      val = str2num(mat);
      row = [row val];
      count = count+1;
    end
    covar = [covar; row];
  end

  mus(:,i) = mu;
  covs(:,:,i) = covar;
end
