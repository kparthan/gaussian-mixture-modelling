function [] = heat_map_bivariate(K)

M = load('mixture_density.dat');
x = M(:,1);
y = M(:,2);
density = M(:,3);
fig = figure();
scatter3(x,y,density,2,'cdata',M(:,3));
xlabel('X');
ylabel('Y');
zlabel('Z');
savefig(fig,'mixture_density.fig');

hold off;

% plot the sampled data
fig = figure();
for k = 1:K
   data_file = strcat('comp',num2str(k),'_density.dat');
   M = load(data_file);
   x = M(:,1);
   y = M(:,2);
   density = M(:,3);
   colors = rand(1,3);
   plot3(x,y,density,'.','Color',colors);
   hold on;
end  
xlabel('X');
ylabel('Y');
zlabel('Z');
savefig(fig,'individual_components.fig');

