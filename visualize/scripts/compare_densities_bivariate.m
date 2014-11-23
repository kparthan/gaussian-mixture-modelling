% original = 1, if true mixture is known and 'mixture_density.dat' is present
%          = 0, otherwise
function [] = compare_densities_bivariate(original_present)

  if (original_present == 1)
    M = load('../sampled_data/mixture_density.dat');
    x = M(:,1);
    y = M(:,2);
    density = M(:,3);
    fig = figure();
    scatter3(x,y,density,2,'cdata',M(:,3));
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    savefig(fig,'../figs/mixture_density.fig');
  end

  M = load('../sampled_data/inferred_mixture_1_density.dat');
  x = M(:,1);
  y = M(:,2);
  density = M(:,3);
  fig = figure();
  scatter3(x,y,density,2,'cdata',M(:,3));
  xlabel('X');
  ylabel('Y');
  zlabel('Z');
  savefig(fig,'../figs/inferred_mixture_1_density.fig');

  M = load('../sampled_data/inferred_mixture_2_density.dat');
  x = M(:,1);
  y = M(:,2);
  density = M(:,3);
  fig = figure();
  scatter3(x,y,density,2,'cdata',M(:,3));
  xlabel('X');
  ylabel('Y');
  zlabel('Z');
  savefig(fig,'../figs/inferred_mixture_2_density.fig');

end
