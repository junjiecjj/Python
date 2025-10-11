function varargout = twodimensional(varargin)
% TWODIMENSIONAL M-file for twodimensional.fig
%      TWODIMENSIONAL, by itself, creates a new TWODIMENSIONAL or raises the existing
%      singleton*.
%
%      H = TWODIMENSIONAL returns the handle to a new TWODIMENSIONAL or the handle to
%      the existing singleton*.
%
%      TWODIMENSIONAL('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in TWODIMENSIONAL.M with the given input arguments.
%
%      TWODIMENSIONAL('Property','Value',...) creates a new TWODIMENSIONAL or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before twodimensional_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to twodimensional_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Copyright 2002-2003 The MathWorks, Inc.

% Edit the above text to modify the response to help twodimensional

% Last Modified by GUIDE v2.5 16-May-2012 13:07:13

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @twodimensional_OpeningFcn, ...
                   'gui_OutputFcn',  @twodimensional_OutputFcn, ...
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


% --- Executes just before twodimensional is made visible.
function twodimensional_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to twodimensional (see VARARGIN)

% Choose default command line output for twodimensional
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes twodimensional wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = twodimensional_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;






% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

derad = pi/180;
radeg = 180/pi;
twpi = 2*pi;
kelm =  str2num(get(handles.kelm_input,'String'));               % 
dd = 0.5;               % 
d=-(kelm-1)/2*dd:dd:(kelm-1)/2*dd;     % 
iwave = 3;              % number of DOA
theta1 = [10 20 30];  
theta2 = [15 25 35];% DOA
snr = str2num(get(handles.snr_input,'String'));              % input SNR (dB)
n = str2num(get(handles.n_input,'String'));                % 
A0=exp(j*twpi*d.'*(sin(theta1*derad).*cos(theta2*derad)))/sqrt(kelm);
A1=exp(j*twpi*d.'*(sin(theta1*derad).*sin(theta2*derad)))/sqrt(kelm);%%%% direction matrix
S=randn(iwave,n)
X0=[];
for im=1:kelm
      X0=[X0;A0*diag(A1(im,:))*S];
end
for inn=1:50
X=awgn(X0,snr,'measured');
L=iwave;
J1=eye(kelm-1,kelm);
J2=flipud(fliplr(J1));
Q=qq(kelm);
Y=kron(Q',Q')*X;
Q0=qq(kelm-1);
K1=real(Q0'*J2*Q);
K2=imag(Q0'*J2*Q);
I=eye(kelm);
Ku1=kron(I,K1);
Ku2=kron(I,K2);
Kv1=kron(K1,I);
Kv2=kron(K2,I);
E=[real(Y),imag(Y)];
Ey=E*E'/n;
[V,D]=eig(Ey);
EVAs =diag(D).';
[EVAs,I0] = sort(EVAs);
EVAs=fliplr(EVAs);
EVs=fliplr(V(:,I0));
Es=EVs(:,1:L);
fiu=pinv(Ku1*Es)*Ku2*Es;
fiv=pinv(Kv1*Es)*Kv2*Es;
F=fiu+j*fiv;
[VV,DD]=eig(F);
EVA = diag(DD).';
u=2*atan(real(EVA))/pi;
v=2*atan(imag(EVA))/pi;
theta10=asin(sqrt(u.^2+v.^2))*radeg;
theta20=atan(v./u)*radeg;
try
    delete(allchild(handles.plotarea))
end
axes(handles.plotarea)
axis([5 40 0 50])
h=plot(theta10,theta20,'k*');hold on;
end
 
xlabel('·½Î»½Ç/(¡ã)')
ylabel('Ñö½Ç/(¡ã )')
title('ESPRIT  ')
% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

derad = pi/180;
radeg = 180/pi;
twpi = 2*pi;
kelm =str2num(get(handles.kelm_input,'String'));;  % the number of array¨®
kelm2 =str2num(get(handles.kelm_input,'String'));;  % the number of array¨®
dd = 0.5;  % space between array
d=0:dd:(kelm-1)*dd;
d2=0:dd:(kelm2-1)*dd;
M = kelm;	  % number of antennas
N = kelm2;
J=str2num(get(handles.n_input,'String'));;       % the length of transmit signal
F=3;        % the number of users
alpha1 = [10;20; 30;]; 	% angles of arrival of each ray of each signal [degrees]
alpha1=alpha1(1:F);
theta1=alpha1.';
alpha2 = [15; 25; 35;]; 	% angles of arrival of each ray of each signal [degrees]
alpha2=alpha2(1:F);
theta2=alpha2.';
A1=exp(-1i*twpi*d.'*(sin(theta1*derad).*sin(theta2*derad)));
A2=exp(-1i*twpi*d2.'*(sin(theta1*derad).*cos(theta2*derad)));     % direction matrix
data0=randn(F,J)+1i*randn(F,J);
MM=khatri_rao(A2,A1);
X= MM*data0;
snr = str2num(get(handles.snr_input,'String'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%snr
clear Y ; Y=awgn(X,snr,'measured');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RD-MUSIC
clear est1 est2;
 R = Y*Y'/J;
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 2D-Music
 [SP]=doa_music(Y,F,d,d2);

try
    delete(allchild(handles.plotarea))
end

axes(handles.plotarea)
axis([5 40 0 50])
h=contour([0:90],[0:90],SP);
%set(h,'Linewidth',2);
grid on
xlabel('·½Î»½Ç/(¡ã)')
ylabel('Ñö½Ç/(¡ã )')

title('MUSIC ')



function snr_input_Callback(hObject, eventdata, handles)
% hObject    handle to snr_input (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of snr_input as text
%        str2double(get(hObject,'String')) returns contents of snr_input as a double


% --- Executes during object creation, after setting all properties.
function snr_input_CreateFcn(hObject, eventdata, handles)
% hObject    handle to snr_input (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function kelm_input_Callback(hObject, eventdata, handles)
% hObject    handle to kelm_input (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of kelm_input as text
%        str2double(get(hObject,'String')) returns contents of kelm_input as a double


% --- Executes during object creation, after setting all properties.
function kelm_input_CreateFcn(hObject, eventdata, handles)
% hObject    handle to kelm_input (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function n_input_Callback(hObject, eventdata, handles)
% hObject    handle to n_input (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of n_input as text
%        str2double(get(hObject,'String')) returns contents of n_input as a double


% --- Executes during object creation, after setting all properties.
function n_input_CreateFcn(hObject, eventdata, handles)
% hObject    handle to n_input (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
delete(gcf);
frame;

% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
try
    delete(allchild(handles.plotarea))
end




% --- Executes on key press over snr_input with no controls selected.
function snr_input_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to snr_input (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


