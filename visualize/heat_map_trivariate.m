function [] = heat_map_trivariate(K)

M = load('mixture_density.dat');
x = M(:,1);
y = M(:,2);
z = M(:,3);
density = M(:,4);
fig = figure();
scatter3(x,y,z,2,'cdata',density);
xlabel('X');
ylabel('Y');
zlabel('Z');
savefig(fig,'mixture_density.fig');

