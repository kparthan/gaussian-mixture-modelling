% Visualization of bivariate clusters in XY-plane only
function [] = visualize2D(K)

hold on;
%c(1,:) = [1 0.5 0];
%c(2,:) = [1 0 1];
%c(3,:) = [0 1 1];
%c(4,:) = [1 0 0];
%c(5,:) = [0 1 0];
%c(6,:) = [0 0 1];
%c(7,:) = [0.5 0.5 0.5];
%c(8,:) = [0.5 0.8 0.8];
%c(9,:) = [0.25 0.25 0.25];

% plot the sampled data
for k = 1:K
   %k
   data_file = strcat('../sampled_data/comp',num2str(k),'.dat');
   M = load(data_file);
   x = M(:,1);
   y = M(:,2);
   colors = rand(1,3);
   plot(x,y,'.','Color',colors);
end  
xlabel('X');
ylabel('Y');

% create legend
N = [1:K];
legend_cell = [cellstr(num2str(N','%d'))];
legend(legend_cell);

end
