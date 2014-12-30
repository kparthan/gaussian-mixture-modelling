function [mus,covs] = parse_bivariate(K,mixture)

fr = fopen(mixture,'r');
all_lines = textscan(fr, '%[^\n]', K);  % cell
fclose(fr);
  for i=1:K
    line = all_lines{1}{i};
    [ans1,ans2] = strsplit(line,{'[mu]:','[cov]:','(',')','\t',' ',',',''});
    w_str = ans1(1);
    mat = cell2mat(w_str);
    w = str2num(mat);

    mu1_str = ans1(2);
    mat = cell2mat(mu1_str);
    mu1 = str2num(mat);
    mu2_str = ans1(3);
    mat = cell2mat(mu2_str);
    mu2 = str2num(mat);

    mu = [mu1; mu2];

    cov11_str = ans1(4);
    mat = cell2mat(cov11_str);
    cov11 = str2num(mat);
    cov12_str = ans1(5);
    mat = cell2mat(cov12_str);
    cov12 = str2num(mat);
    cov21 = cov12;
    cov22_str = ans1(7);
    mat = cell2mat(cov22_str);
    cov22 = str2num(mat);

    covar = [cov11 cov12; cov21 cov22];

    mus(:,i) = mu;
    covs(:,:,i) = covar;
  end
end

