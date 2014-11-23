% original = 1, if true mixture is known and 'mixture_density.dat' is present
%          = 0, otherwise
function [] = compare_densities_univariate(original_present)

  if (original_present == 1)
    M = load('../sampled_data/mixture_density.dat');
    x = M(:,1);
    density = M(:,2);
    fig = figure();
    scatter(x,density,2,'cdata',M(:,2));
    xlabel('X');
    ylabel('Y');
    savefig(fig,'../figs/mixture_density.fig');
  end

  M = load('../sampled_data/inferred_mixture_1_density.dat');
  x = M(:,1);
  density = M(:,2);
  fig = figure();
  scatter(x,density,2,'cdata',M(:,2));
  xlabel('X');
  ylabel('Y');
  savefig(fig,'../figs/inferred_mixture_1_density.fig');

  M = load('../sampled_data/inferred_mixture_2_density.dat');
  x = M(:,1);
  density = M(:,2);
  fig = figure();
  scatter(x,density,2,'cdata',M(:,2));
  xlabel('X');
  ylabel('Y');
  savefig(fig,'../figs/inferred_mixture_2_density.fig');

end
