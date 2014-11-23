% Korig is original number of components
% Kinf is inferred number of components
% mixture is the path to the inferred mixture
function [] = heat_map_bivariate(Korig,Kinf,mixture)

% read the mixture
[mus,covs] = parse_bivariate(Kinf,mixture);

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
hold on;
for i=1:Kinf
  elipsnorm(mus(:,i),covs(:,:,i),2);
end

hold off;

% plot the sampled data
fig = figure();
for k = 1:Korig
   data_file = strcat('../sampled_data/comp',num2str(k),'_density.dat');
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
savefig(fig,'../figs/individual_components.fig');

