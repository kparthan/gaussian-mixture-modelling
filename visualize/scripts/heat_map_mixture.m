% plots the mixture density
function [] = heat_map_mixture()

M = load('../sampled_data/mixture_density.dat');
d = size(M,2)-1;

fig = figure();
xlabel('X');
x = M(:,1);
if (d == 1)
  density = M(:,2);
  scatter(x,density,2,'cdata',density);
  ylabel('Y');
elseif (d == 2)
  y = M(:,2);
  density = M(:,3);
  scatter3(x,y,density,2,'cdata',density);
  ylabel('Y');
  zlabel('Z');
elseif (d == 3)
  y = M(:,2);
  z = M(:,3);
  density = M(:,4);
  fig = figure();
  scatter3(x,y,z,2,'cdata',density);
  ylabel('Y');
  zlabel('Z');
end
savefig(fig,'../figs/mixture_density.fig');

