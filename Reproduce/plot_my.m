clear;%清楚workspace的变量，防止同名变量干扰
clc;%清楚workspace的命令显示。
%----------------------------------------------------------------填放数据--------------------------------------------------------------------

% NTN TDL-D, N=256, QPSK, nu_max = 2, Δf=15kHz，max_iter=10



MMSE = [
% SNR        BER       ave_iter1    ave_iter2   error_blocks 
 0     0.17556641              0              0           1200
 3     0.10036458              0              0           1200
 6    0.041937708              0              0           1200
 9    0.011696949              0              0           1200
12   0.0018312731              0              0           1200
15  0.00021135719              0              0           1200
18  2.4077816e-05              0              0           1200
21  2.2798917e-06              0              0           1200
24  2.1931769e-07              0              0            224


];

WMRC_DFE = [
% SNR        BER       ave_iter1    ave_iter2   error_blocks 
 0     0.18105143         4.8075              0           1200
 3     0.10085612      4.7058333              0           1200
 6     0.04129672      4.3449132              0           1200
 9   0.0087740916      3.6191038              0           1200
12   0.0012256431      2.9932553              0           1200
15  0.00023777909      2.5951622              0           1200
18  8.4348771e-05      2.4003398              0           1200
21  4.7730989e-05      2.3259074              0           1200
24   3.600798e-05      2.2894823              0           1200
27  3.4467931e-05      2.2727789              0           1200
30  3.0575392e-05      2.2656023              0           1199
 
 
 

];


MRC_TD = [
% SNR        BER       ave_iter1    ave_iter2   error_blocks 
0     0.17557129      4.6558333              0           1200
 3      0.1003776         5.4325              0           1200
 6    0.041934456      6.2805995              0           1200
 9    0.011119963      6.9684211              0           1200
12   0.0018286748      7.3941988              0           1200
15  0.00022827188      7.5904696              0           1200
18   2.782611e-05      7.6909686              0           1200
21  4.8689132e-06      7.7396359              0           1200
24   1.258665e-06      7.7658459              0            430
     


];


MP = [
% SNR        BER       ave_iter1    ave_iter2   error_blocks 
 0      0.1833138             10              0           1200
 3     0.10449219             10              0           1200
 6      0.0450339             10              0           1200
 9     0.01325151      9.9641288              0           1200
12   0.0022520787      8.1139013              0           1200
18  0.00014117367      3.6351357              0           1180
21  8.981248e-05      3.245763             0           532


];



soft_CD = [
% SNR        BER       ave_iter1    ave_iter2   error_blocks 
0     0.17491048          8.455              0           1200
 3    0.099423828      9.5566667              0           1200
 6    0.036929461      9.5643154              0           1200
 9   0.0069195492      7.1052043              0           1200
12  0.00060726713      3.1214133              0           1198
15  3.204242e-05      1.376719             0           350



];

soft_noCD = [
% SNR        BER       ave_iter1    ave_iter2   error_blocks 
0     0.1760266      7.017949             0           780
3     0.1005265      8.182609             0           460
6    0.03752219      8.318182             0           436
9   0.0067593661      5.8988636              0           1191
12  0.00062282758      2.6333683              0           1198
15  8.49185e-06      1.31107            0            7
18  1.23585e-06      1.05293            0           12
 

];


PCG_MPA_AVrEP1e7 = [
% SNR        BER       ave_iter1    ave_iter2   error_blocks 
 0     0.17539551      4.2458333      24.524167           1200
 3    0.098792318      4.7766667        24.3875           1200
 6    0.037297394      4.3477178      18.044813           1200
 9   0.0072272592      2.9940547      13.830559           1200
12  0.00071721466      2.2176485      12.653428           1200
15  4.1248543e-05      2.0236642      12.256909            826
 


];


TS_MRC_CD = [
% SNR        BER       ave_iter1    ave_iter2   error_blocks 
 0     0.18042318      4.9991667         4.8525           1200
 3     0.10272135      4.9991667          4.815           1200
 6    0.038872814              5      4.1769547           1200
 9   0.0083842606              5      2.7960408           1200
12  0.00083954036              5       1.486631           1200
15  6.4514833e-05      4.9999748      1.0879498           1200
18  6.9595811e-06       4.999981      1.0196171            201



];


TS_MRC_noCD = [
% SNR        BER       ave_iter1    ave_iter2   error_blocks 
0     0.17840007              5      3.8691667           1200
 3     0.10394694              5         3.7575           1200
 6     0.04041379              5           3.25           1200
 9   0.0097138049              5      2.5780164           1200
12   0.0011336074              5      2.1208356           1200
15  8.3958433e-05      4.9999261      2.0170506           1200
18  7.0368908e-06      4.9999711      2.0027004            310

];






%----------------------------------------------------------------开始画图--------------------------------------------------------------------

width = 8;%设置图宽，这个不用改
height = 7*0.75;%设置图高，这个不用改
fontsize = 18;%设置图中字体大小
linewidth = 2;%设置线宽，一般大小为2，好看些。1是默认大小
marksize = 12;%标记的大小，按照个人喜好设置

%-------------------Figure1----------------------
h1 = figure(1);
%这是用于裁剪figure的。需要把fig.m文件放在一个文件夹中
fig(h1, 'units','inches','width',width, 'height', height, 'font','Times New Roman','fontsize',fontsize);
% axes1 = axes('Parent',h1,...
%     'Position',[0.091785714285714 0.1099604743083 0.390357142857143 0.858418972332016]);%left bottom width heigh
%颜色集合，这是默认的八种颜色，颜色的数量可以更改0.101785714285714 0.1699604743083 0.380357142857143 0.798418972332016
ColorSet = [
1       0       0      % 红色
0       1       0      % 绿色
0       0       1      % 蓝色
1       0       1      % 洋红色
0       1       1      % 青色
1       0.5     0      % 橙色
0.5     0       0.5  % 紫色
0.6     0.3     0   % 棕色
0       0.5     0   % 深绿色

0       0       0
0       0       0
1       0.75    0.8 % 粉色
0.5     0       0   % 深红色
0.5     0.5     0   % 橄榄绿
0       0       0.5 % 深蓝色
0       0       0   % 黑色
0.5     0.5     0.5 % 灰色
    ];
%设置循环使用的颜色集合
set(gcf, 'DefaultAxesColorOrder', ColorSet);
%纵坐标对数域，如果不需要对数改为plot
semilogy(MMSE(:,1), MMSE(:,2),'-o',... 
    WMRC_DFE(:,1), WMRC_DFE(:,2), '-s',... 
    MRC_TD(:,1), MRC_TD(:,2), '-h',... 
    MP(:,1), MP(:,2), '-*',... 
    soft_CD(:,1), soft_CD(:,2), '-x',... 
    soft_noCD(:,1), soft_noCD(:,2), '-^',...  
    PCG_MPA_AVrEP1e7(:,1), PCG_MPA_AVrEP1e7(:,2), '-<',... 
    TS_MRC_CD(:,1), TS_MRC_CD(:,2), '->',... 
    TS_MRC_noCD(:,1), TS_MRC_noCD(:,2), '-p');

grid on;
legend('MMSE', ...
    'WMRC-DFE', ...
    'MRC-TD', ...
    'MP', ... 
    'IMMSE-C-soft', ...
    'IMMSE-D-soft', ...
    'PCG-MPA-AVrEP 1e-7', ...
    'TS-MRC-C', ...
    'TS-MRC-D');
xlabel('SNR (dB)');%横坐标标号
ylabel('BER');%纵坐标标号
% title('K=10db');
set(get(gca,'Children'),'linewidth',linewidth,'markersize', marksize);%设置图中线宽,标记大小
set(gca, 'XTick', 0:3:30);%设置横坐标
set(gca, 'YTick', [1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1]);%设置横坐标
axis([3 24 8e-7 1e-1]);%横纵坐标范围
%-------------------------Figure1----------------------------
function h = fig(varargin)
% FIG - Creates a figure with a desired size, no white-space, and several other options.
%
%       All Matlab figure options are accepted. 
%       FIG-specific options of the form FIG('PropertyName',propertyvalue,...) 
%       can be used to modify the default behavior, as follows:
%
%       -'units'    : preferred unit for the width and height of the figure 
%                      e.g. 'inches', 'centimeters', 'pixels', 'points', 'characters', 'normalized' 
%                      Default is 'centimeters'
%
%       -'width'    : width of the figure in units defined by 'units'
%                      Default is 14 centimeters
%                      Note: For IEEE journals, one column wide standard is
%                      8.5cm (3.5in), and two-column width standard is 17cm (7 1/16 in)
%
%       -'height'   : height of the figure in units defined by 'units'
%                      Specifying only one dimension sets the other dimension
%                      to preserve the figure's default aspect ratio. 
%
%       -'font'     : The font name for all the texts on the figure, including labels, title, legend, colorbar, etc.
%                      Default is 'Times New Roman' 
%
%       -'fontsize' : The font size for all the texts on the figure, including labels, title, legend, colorbar, etc.
%                      Default is 14pt
%       
%       -'border'   : Thin white border around the figure (compatible with export_fig -nocrop) 
%                      'on', 'off'
%                      Default is 'off' 
%
%   FIG(H) makes H the current figure. 
%   If figure H does not exist, and H is an integer, a new figure is created with
%   handle H.
%
%   FIG(H,...) applies the properties to the figure H.
%
%   H = FIG(...) returns the handle to the figure created by FIG.
%
%
% Example 1:
%   fig
%
% Example 2:
%   h=fig('units','inches','width',7,'height',2,'font','Helvetica','fontsize',16)
%
%
% Copyright   2012 Reza Shirvany,  matlab.sciences@neverbox.com 
% Source: 	 http://www.mathworks.com/matlabcentral/fileexchange/30736
% Updated:   05/14/2012
% Version:   1.6.5 
%







% default arguments
width=14;
font='Times New Roman';
fontsize=14; 
units='centimeters';
bgcolor='w';
sborder='off';
flag='';
Pindex=[];

%%%%%%%%%%% process optional arguments
optargin = size(varargin,2);
if optargin>0

% check if a handle is passed in
if isscalar(varargin{1})
    flag=[flag '1'];
    i=2;
     if ishghandle(varargin{1})==1
        flag=[flag 'i'];
    end
else
    i=1;
end

% get the property values
while (i <= optargin)
    if (strcmpi(varargin{i}, 'border'))
        if (i >= optargin)
            error('Property value required for: %s', num2str(varargin{i}));
        else
            sborder = varargin{i+1};flag=[flag 'b'];
            i = i + 2;
        end
    elseif (strcmpi(varargin{i}, 'width'))
        if (i >= optargin)
            error('Property value required for: %s', num2str(varargin{i}));
        else
            width = varargin{i+1};flag=[flag 'w'];
            i = i + 2;
        end
    elseif (strcmpi(varargin{i}, 'height'))
        if (i >= optargin)
            error('Property value required for: %s', num2str(varargin{i}));
        else
            height = varargin{i+1};flag=[flag 'h'];
            i = i + 2;
        end
    elseif (strcmpi(varargin{i}, 'font'))
        if (i >= optargin)
            error('Property value required for: %s', num2str(varargin{i}));
        else
            font = varargin{i+1};flag=[flag 'f'];
            i = i + 2;
        end
    elseif (strcmpi(varargin{i}, 'fontsize'))
        if (i >= optargin)
            error('Property value required for: %s', num2str(varargin{i}));
        else
           fontsize = varargin{i+1};flag=[flag 's'];
            i = i + 2;
        end
    elseif (strcmpi(varargin{i}, 'units'))
        if (i >= optargin)
            error('Property value required for: %s', num2str(varargin{i}));
        else
            units = varargin{i+1};flag=[flag 'u'];
            i = i + 2;
        end
    elseif (strcmpi(varargin{i}, 'color'))
        if (i >= optargin)
            error('Property value required for: %s', num2str(varargin{i}));
        else
            bgcolor = varargin{i+1};flag=[flag 'c'];
            i = i + 2;
        end
    else
        %other figure properties
        if (i >= optargin)
            error('A property value is missing.');
        else
        Pindex = [Pindex i i+1];
        i = i + 2;
        end
    end

end

end

% We use try/catch to handle errors
try

% creat a figure with a given (or new) handle
if length(strfind(flag,'1'))==1
    s=varargin{1};
    if ishandle(s)==1
    set(0, 'CurrentFigure', s);
    else 
        figure(s);
    end
else
    s=figure;
end

flag=[flag 's'];

% set other figure properties
if ~isempty(Pindex)
    set(s,varargin{Pindex});
end


% set the background color
set(s, 'color',bgcolor);

% set the font and font size
set(s, 'DefaultTextFontSize', fontsize); 
set(s, 'DefaultAxesFontSize', fontsize); 
set(s, 'DefaultAxesFontName', font);
set(s, 'DefaultTextFontName', font);

%%%%%%%%%%% set the figure size
% set the root unit
old_units=get(0,'Units');
set(0,'Units',units);

% get the screen size
scrsz = get(0,'ScreenSize');

% set the root unit to its default value
set(0,'Units',old_units);

% set the figure unit
set(s,'Units',units);

% get the figure's position
pos = get(s, 'Position');
old_pos=pos;
aspectRatio = pos(3)/pos(4);

% set the width and height of the figure
if length(strfind(flag,'w'))==1 && length(strfind(flag,'h'))==1 
    pos(3)=width;
    pos(4)=height;
elseif ~contains(flag,'h')
    pos(3)=width;
    pos(4) = width/aspectRatio;
elseif ~contains(flag,'w') && length(strfind(flag,'h'))==1
    pos(4)=height;
    pos(3)=height*aspectRatio; 
end

% make sure the figure stays in the middle of the screen
diff=old_pos-pos;

 if diff(3)<0
 pos(1)=old_pos(1)+diff(3)/2;
     if pos(1)<0
         pos(1)=0;
     end
 end
 if diff(4)<0
 pos(2)=old_pos(2)+diff(4);
    if pos(2)<0
         pos(2)=0;
     end
 end
 
% warning if the given width (or height) is greater than the screen size
if pos(3)>scrsz(3)
warning(['Maximum width (screen width) is reached! width=' num2str(scrsz(3)) ' ' units]);
end

if pos(4)>scrsz(4)
warning(['Maximum height (screen height) is reached! height=' num2str(scrsz(4)) ' ' units]);
end

% apply the width, height, and position to the figure
set(s, 'Position', pos);
if strcmpi(sborder, 'off')
    set(s,'DefaultAxesLooseInset',[0,0,0,0]);
end

    

% handle errors
catch ME
    if ~contains(flag,'i') && contains(flag,'s')
    close(s);
    end
   error(ME.message)
end

s=figure(s);
% return handle if caller requested it.
  if (nargout > 0)
        h =s;
  end
%
% That's all folks!
%
%flag/1iwhfsucsb
end