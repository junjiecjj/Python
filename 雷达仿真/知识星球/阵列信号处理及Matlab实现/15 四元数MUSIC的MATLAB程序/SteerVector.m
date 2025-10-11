%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%**�������֣�����ʸ���ӵ�������Ϣ��
%**���ߣ�    ����
%**���ڣ�    2006-6-10
%**�޸��ˣ�
%**���ڣ�      
%**������    ����Q_MUSIC����
%**         {(1+i(rou)exp[j(fai)])*exp[-j(thita)]}*beita*exp[j(alfa)]    
%**         �˴�����beita��afa��Ϊ1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function output = SteerVector(N, thita, rou, fai, beita, alfa)
    
    a = zeros(N,4);
    c = PolarSource(rou, fai);
   for i = 1:N
        b = [cos(-(i-1)*thita),0,sin(-(i-1)*thita),0]*beita;
        a(i,:) = hpc(c,b);
        d = [cos(alfa),0,sin(alfa),0];
% d = [cos((i-1)*alfa),0,sin((i-1)*alfa),0];
        a(i,:) = hpc(a(i,:),d);
  end
    
  output = a;
    
    
    
        