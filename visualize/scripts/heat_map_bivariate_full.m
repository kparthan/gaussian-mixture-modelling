%% redundant -- don't use
% heat map whole region
function [] = heat_map_bivariate_full()

M = load('../sampled_data/probability_density.dat');
x = M(:,1);
y = M(:,2);
density = M(:,3);
fig = figure();
scatter3(x,y,density,2,'cdata',M(:,3));
xlabel('X');
ylabel('Y');
zlabel('Z');
savefig(fig,'../figs/mixture_density_full.fig');

