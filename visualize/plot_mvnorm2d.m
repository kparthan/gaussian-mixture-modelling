function [] = plot_mvnrm2d(data_file)

% plot the sampled data
M = load(data_file);
x = M(:,1);
y = M(:,2);
%colors = rand(1,3);
plot(x,y,'.','Color',[0 0 1]);
grid on;

xlabel('X');
ylabel('Y');

end
