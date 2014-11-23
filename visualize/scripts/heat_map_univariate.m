% Korig is original number of components
% Kinf is inferred number of components
% mixture is the path to the inferred mixture
function [] = heat_map_univariate(Korig,Kinf,mixture)

% read the mixture
%[mus,sigmas] = parse_univariate(Kinf,mixture);

M = load('../sampled_data/mixture_density.dat');
x = M(:,1);
density = M(:,2);
fig = figure();
scatter(x,density,2,'cdata',M(:,2));
xlabel('X');
ylabel('Y');
savefig(fig,'../figs/mixture_density.fig');
hold on;
%for i=1:Kinf
%  elipsnorm(mus(:,i),covs(:,:,i),2);
%end

hold off;

% plot the sampled data
fig = figure();
for k = 1:Korig
   data_file = strcat('../sampled_data/comp',num2str(k),'_density.dat');
   M = load(data_file);
   x = M(:,1);
   density = M(:,2);
   colors = rand(1,3);
   plot(x,density,'.','Color',colors);
   hold on;
end  
xlabel('X');
ylabel('Y');
savefig(fig,'../figs/individual_components.fig');

