function varargout = fig(varargin)
%���ߣ�ţ��ǿ
%�汾��1.0.0
% ���ʱ�䣺2014.02.19
% FIG MATLAB code for fig.fig
%      FIG, by itself, creates a new FIG or raises the existing
%      singleton*.
%
%      H = FIG returns the handle to a new FIG or the handle to
%      the existing singleton*.
%
%      FIG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in FIG.M with the given input arguments.
%
%      FIG('Property','Value',...) creates a new FIG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before fig_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to fig_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help fig

% Last Modified by GUIDE v2.5 19-Feb-2014 08:14:13

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @fig_OpeningFcn, ...
                   'gui_OutputFcn',  @fig_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before fig is made visible.
function fig_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to fig (see VARARGIN)

% Choose default command line output for fig
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes fig wait for user response (see UIRESUME)
% uiwait(handles.figure1);

%���ñ���ͼƬ
ha=axes('units','normalized','position',[0 0 1 1]); 
uistack(ha,'down') 
II=imread('radar.jpg'); 
image(II) 
colormap gray 
set(ha,'handlevisibility','off','visible','off'); 



% --- Outputs from this function are returned to the command line.
function varargout = fig_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;





% 
% --- Executes on button press in startbutton.
function startbutton_Callback(hObject, eventdata, handles)
% hObject    handle to startbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)




% clear all
% clc
% close all

%��ȡ���书��
Pt=str2double(get(handles.Ptinput,'String'));

%��ȡ����Ƶ��
Fc=str2double(get(handles.Fcinput,'String'))*1e6;

%��ȡ������
Tp=str2double(get(handles.Tpinput,'String'))*1e-6;

%��ȡ�����ظ�Ƶ��
Fr=1e3*[str2double(get(handles.Frinput1,'String')) str2double(get(handles.Frinput2,'String')) str2double(get(handles.Frinput3,'String'))];

%��ȡ����
B=1e6*str2double(get(handles.Binput,'String'));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%    �״�ϵͳ�������    %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c=3e8;                           % ����
k=1.38e-23;                      % ������������

% Pt=20e3;                         % ���书�ʡ�W��

% Fc=1e9;                          % ����Ƶ�ʡ�Hz��
Wavelength=c/Fc;                 % ����������m��

% Tp=8e-6;                        % �����ȡ�΢�롿
% Fr=[8e3 11e3 13e3];                         % �����ظ�Ƶ�ʡ�Hz��

% B=10e6;                           % ����Hz��
Fs=20e6;                         % �����ʡ�Hz��
F=10^(6.99/10);                     % ����ϵ��
K=B/Tp;                          % ��Ƶ�ʡ�Hz��
Tr=1./Fr;% �����ظ����ڡ��롿
R_T=Tr*c/2;%���ģ������

Delta_t=1/Fs;                    % ʱ�������ʱ�������롿
vv=Fr*Wavelength/2;  %���ģ���ٶ�
D=5;                             % ���߿׾���m��
Ae=1*pi*(D/2)^2;                 % ������Ч�����m^2��
% G=4*pi*Ae/Wavelength^2;          % ��������
G=10^(32/10);
BeamWidth=0.88*Wavelength/D;     % ����3dB������ȡ�deg��
BeamShift=0.8*BeamWidth/2;         % A��B��������������ļнǡ�deg��
Theta0=30*pi/180;                % ���������ʼָ�򡾶ȡ�
Wa=0;2*pi/1;                       % ���߲���ת�١�rad/sec��

Num_Tr_CPI=64;                      % CPI������


R_set=[70e3,7e3,10e3];          % Ŀ����롾m��         
RCS=[1,1,1];                 % Ŀ��ƽ������ɢ��������m^2��   
Theta_target_set=30.1*pi/180; % Ŀ�귽λ�ǡ�deg��
V_set=[2100,1000,900];                % Ŀ���ٶȡ�m/s�� 

for a=1:length(Fr)
    
   R_A(a)=mod(R_set(1),R_T(a));%�ж��Ƿ����ģ��
end
for a=1:length(Fr)
    
   v_A(a)=mod(V_set(1),vv(a));
end    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%      ���������ź�     %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s=lfm(Pt,Tp,Fr,B,Fs,G,Num_Tr_CPI);

figure
s_plot(s);
title('�״﷢���ź�')
xlabel('time [sec]')
ylabel('magnitude [v]')
print(gcf,'-dbitmap','G:\202404\20240428\PD_radar\�ҵĴ���\����ͼƬ\�״﷢���ź�.jpg')   % ����Ϊpng��ʽ��ͼƬ��


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%      Ŀ��ز�     %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[s_A s_B] = target(G,Fc,Fs,Fr,Num_Tr_CPI,Theta0,Wa,BeamWidth,s,R_set,V_set,RCS,Theta_target_set);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%      ����������     %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[s_A s_B] = nose(s_A,s_B,k,B,F);

figure
subplot(2,1,1)
s_plot(s_A);
title('Aͨ���ز��ź�')
xlabel('time [sec]')
ylabel('magnitude [v]')

subplot(2,1,2)
s_plot(s_B);
title('Bͨ���ز��ź�')
xlabel('time [sec]')
ylabel('magnitude [v]')

print(gcf,'-dbitmap','G:\202404\20240428\PD_radar\�ҵĴ���\����ͼƬ\�״�ز��ź�.jpg')   % ����Ϊpng��ʽ��ͼƬ��
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%      �Ͳ������    %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[s_Sigma s_Delta] =sigma_delta(s_A,s_B);


figure
subplot(2,1,1)
s_plot(s_Sigma);
title('��ͨ���ز��ź�')
xlabel('time [sec]')
ylabel('magnitude [v]')

subplot(2,1,2)
s_plot(s_Delta);
title('��ͨ���ز��ź�')
xlabel('time [sec]')
ylabel('magnitude [v]')
print(gcf,'-dbitmap','G:\202404\20240428\PD_radar\�ҵĴ���\����ͼƬ\�Ͳ���ƻز��ź�.jpg')   % ����Ϊpng��ʽ��ͼƬ��



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  ƥ���˲�������ѹ��)  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%]
[s_Sigma_rc s_Delta_rc] = match(s_Sigma,s_Delta,Tr,Fs,K,Num_Tr_CPI);

figure
s_plot(s_Sigma_rc);
title('��ͨ��ƥ���˲����')
xlabel('time [sec]')
ylabel('magnitude [v]')


print(gcf,'-dbitmap','G:\202404\20240428\PD_radar\�ҵĴ���\����ͼƬ\ƥ���˲����.jpg')   % ����Ϊpng��ʽ��ͼƬ��




%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  �������˲���������ۣ�  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[ S_Sigma_a S_Delta_a] =mtd(s_Sigma_rc,s_Delta_rc,Tr,Fs,Num_Tr_CPI );

S_Sigma_abs=cell(1,3);
S_Delta_abs=cell(1,3);
for m=1:length(Fr)
  S_Sigma_abs{1,m}=abs(S_Sigma_a{1,m});
  S_Delta_abs{1,m}=abs(S_Delta_a{1,m});  
end


figure

s_plot(S_Sigma_abs);
title('��ͨ��MTD���')
xlabel('time [sec]')
ylabel('magnitude [v]')


print(gcf,'-dbitmap','G:\202404\20240428\PD_radar\�ҵĴ���\����ͼƬ\MTD���.jpg')   % ����Ϊpng��ʽ��ͼƬ��





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%   CFAR�����龯��⣩  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[ S_out] = CFAR(S_Sigma_a,Num_Tr_CPI );

figure
s_plot(S_out);

title('��ͨ��CFAR���')
xlabel('time [sec]')

print(gcf,'-dbitmap','G:\202404\20240428\PD_radar\�ҵĴ���\����ͼƬ\CFAR���.jpg')   % ����Ϊpng��ʽ��ͼƬ��



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  Ŀ��ȷ�������롢�����ղ���   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[s_R s_D Target_R Target_D target_num Target_Range_all Target_Doppler_all ] = measure(S_out,S_Sigma_a,Num_Tr_CPI ,Fs,Tp,Fr,Wavelength);

figure
subplot(2,1,1)
s_plot_B(s_R);
title('�������Ķ�λ����')
xlabel('time [sec]')
ylabel('���� [m]')

subplot(2,1,2)
s_plot_B(s_D);
title('�ٶ����Ķ�λ����')
xlabel('time [sec]')
ylabel('�ٶ� [m/s]')

print(gcf,'-dbitmap','G:\202404\20240428\PD_radar\�ҵĴ���\����ͼƬ\����������ն���.jpg')   % ����Ϊpng��ʽ��ͼƬ��



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  �����ģ��   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
v_TT=Fr*Wavelength/2;
R_TT=Tr*c/2;
for m=1:length(Fr)
    v_aa(m)=mod(V_set(1),v_TT(m));
    R_aa(m)=mod(R_set(1),R_TT(m));
end


R_am=[Target_R{1,1}(1),Target_R{1,2}(1),Target_R{1,3}(1)];


[ R] = R_ambity(Fr,R_am );
R_plot=[zeros(1,2e3) R zeros(1,500)];

figure
subplot(2,1,1)
plot(R_plot,'LineWidth',2)
title('�����ģ�����')
ylabel('���� [m]')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  ���ٶ�ģ��   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
V_am=[Target_D{1,1}(1),Target_D{1,2}(1),Target_D{1,3}(1)];

 [V] =  V_ambity( Fr,V_am );

V_plot=[zeros(1,2e3) V zeros(1,500)];
subplot(2,1,2)
plot(V_plot,'LineWidth',2)
title('���ٶ�ģ�����')
ylabel('�ٶ� [m/s]')

print(gcf,'-dbitmap','G:\202404\20240428\PD_radar\�ҵĴ���\����ͼƬ\��ģ�����.jpg')   % ����Ϊpng��ʽ��ͼƬ

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%       ��������       %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [angle_aa angle_result ] =angel_measure(S_Sigma_a,S_Delta_a,BeamWidth,BeamShift,Target_Range_all,Target_Doppler_all,Theta0,G);
[angle_result ] =angel_measure(S_Sigma_a,S_Delta_a,BeamWidth,BeamShift,Target_Range_all,Target_Doppler_all,Theta0,G);
[angle_aa ] =angel_measure(S_Sigma_a,S_Delta_a,BeamWidth,BeamShift,Target_Range_all,Target_Doppler_all,Theta0,G);
figure 
s_plot_B(angle_result);
title('�Ͳ�ͨ����ǽ��')
xlabel('time [sec]')
ylabel('�Ƕ� [��]')
print(gcf,'-dbitmap','G:\202404\20240428\PD_radar\�ҵĴ���\����ͼƬ\��ǽ��.jpg')   % ����Ϊpng��ʽ��ͼƬ��
figure
angle_s=(angle_aa{1,1}(1)+angle_aa{1,2}(1)+angle_aa{1,3}(1))/3;
angle_plot=[zeros(1,2e3) angle_s zeros(1,500)];
plot(angle_plot,'LineWidth',2)
title('��Ƶ���ȡƽ��')
ylabel('�Ƕ� [��]')

print(gcf,'-dbitmap','G:\202404\20240428\PD_radar\�ҵĴ���\����ͼƬ\��Ƶ���ȡƽ��.jpg')   % ����Ϊpng��ʽ��ͼƬ

aaa=1;




% --- Executes during object creation, after setting all properties.


% --- Executes on button press in Title.



function Ptinput_Callback(hObject, eventdata, handles)
% hObject    handle to Ptinput (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Ptinput as text
%        str2double(get(hObject,'String')) returns contents of Ptinput as a double


% --- Executes during object creation, after setting all properties.
function Ptinput_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Ptinput (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function Title_CreateFcn(hObject, eventdata, handles)
% hObject    handle to title (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
axis off;
imshow(imread('title.png'));
% Hint: place code in OpeningFcn to populate title



function Fcinput_Callback(hObject, eventdata, handles)
% hObject    handle to Fcinput (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Fcinput as text
%        str2double(get(hObject,'String')) returns contents of Fcinput as a double


% --- Executes during object creation, after setting all properties.
function Fcinput_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Fcinput (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Tpinput_Callback(hObject, eventdata, handles)
% hObject    handle to Tpinput (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Tpinput as text
%        str2double(get(hObject,'String')) returns contents of Tpinput as a double


% --- Executes during object creation, after setting all properties.
function Tpinput_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Tpinput (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Frinput1_Callback(hObject, eventdata, handles)
% hObject    handle to Frinput1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Frinput1 as text
%        str2double(get(hObject,'String')) returns contents of Frinput1 as a double


% --- Executes during object creation, after setting all properties.
function Frinput1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Frinput1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Binput_Callback(hObject, eventdata, handles)
% hObject    handle to Binput (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Binput as text
%        str2double(get(hObject,'String')) returns contents of Binput as a double


% --- Executes during object creation, after setting all properties.
function Binput_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Binput (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Frinput2_Callback(hObject, eventdata, handles)
% hObject    handle to Frinput2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Frinput2 as text
%        str2double(get(hObject,'String')) returns contents of Frinput2 as a double


% --- Executes during object creation, after setting all properties.
function Frinput2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Frinput2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Frinput3_Callback(hObject, eventdata, handles)
% hObject    handle to Frinput3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Frinput3 as text
%        str2double(get(hObject,'String')) returns contents of Frinput3 as a double


% --- Executes during object creation, after setting all properties.
function Frinput3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Frinput3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
