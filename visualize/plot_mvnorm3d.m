function [] = plot_mvnrm3d(data_file)

% plot the sampled data
M = load(data_file);
x = M(:,1);
y = M(:,2);
z = M(:,3);
%colors = rand(1,3);
plot3(x,y,z,'.','Color',[0 0 1]);
grid on;

xlabel('X');
ylabel('Y');
zlabel('Z');

end
