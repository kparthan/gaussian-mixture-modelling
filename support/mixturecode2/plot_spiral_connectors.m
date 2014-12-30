function [] = plot_spiral_connectors()

  iterations = 50;
  folder1 = '../../experiments/infer_components/exp_spiral/';
  folder2 = './exp_spiral/';

  data_folder = strcat(folder1,'data/');
  conn1_folder = strcat(folder1,'connectors/');
  conn2_folder = strcat(folder2,'connectors/');


  for iter=1:iterations
    iter
    iter_str = int2str(iter);

    data_file = strcat(data_folder,'spiral_iter_',iter_str,'.dat');

    fig = figure();
    set(gcf,'visible','off');
    hold on;
    plot_spiral_data(data_file);
    conn_file = strcat(conn1_folder,'connectors_iter_',iter_str);
    points = load(conn_file);
    num_lines = size(points,1);
    starts = points(:,1:3);
    ends = points(:,4:6);
    for i=1:num_lines
      sp = starts(i,:);
      ep = ends(i,:);
      draw_line(sp,ep);
    end
    output_file = strcat(folder1,'plots/ans_iter_',iter_str,'.fig');
    savefig(fig,output_file);
    hold off;

    fig = figure();
    set(gcf,'visible','off');
    hold on;
    plot_spiral_data(data_file);
    conn_file = strcat(conn2_folder,'connectors_iter_',iter_str);
    points = load(conn_file);
    num_lines = size(points,1);
    starts = points(:,1:3);
    ends = points(:,4:6);
    for i=1:num_lines
      sp = starts(i,:);
      ep = ends(i,:);
      draw_line(sp,ep);
    end
    output_file = strcat(folder2,'plots/ans_iter_',iter_str,'.fig');
    savefig(fig,output_file);
    hold off;

  end % iter loop 

end

% plot the spiral data
function [] = plot_spiral_data(data_file)

  set(gcf,'visible','off');
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

% draw a line between sp and ep
function [] = draw_line(sp,ep)

  set(gcf,'visible','off');
  x = [sp(1) ep(1)];
  y = [sp(2) ep(2)];
  z = [sp(3) ep(3)];
  line(x,y,z,'Color','r','LineWidth',3);

end
