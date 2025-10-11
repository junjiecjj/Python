function varargout = onedimensional(varargin)
% ONEDIMENSIONAL M-file for onedimensional.fig
%      ONEDIMENSIONAL, by itself, creates a new ONEDIMENSIONAL or raises the existing
%      singleton*.
%
%      H = ONEDIMENSIONAL returns the handle to a new ONEDIMENSIONAL or the handle to
%      the existing singleton*.
%
%      ONEDIMENSIONAL('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in ONEDIMENSIONAL.M with the given input arguments.
%
%      ONEDIMENSIONAL('Property','Value',...) creates a new ONEDIMENSIONAL or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before onedimensional_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to onedimensional_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Copyright 2002-2003 The MathWorks, Inc.

% Edit the above text to modify the response to help onedimensional

% Last Modified by GUIDE v2.5 26-Mar-2012 20:32:15

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @onedimensional_OpeningFcn, ...
                   'gui_OutputFcn',  @onedimensional_OutputFcn, ...
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


% --- Executes just before onedimensional is made visible.
function onedimensional_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to onedimensional (see VARARGIN)

% Choose default command line output for onedimensional
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes onedimensional wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = onedimensional_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
derad = pi/180;
radeg = 180/pi;
twpi = 2*pi;
kelm = str2num(get(handles.kelm_input,'String'));             % ¡§
dd = 0.5;  % space between array
d=0:dd:(kelm-1)*dd; 
iwave = 3;              % number of DOA
theta = [10 30 60] ; % 
pw= [1 0.8 0.7 ]'  ; %power

nv=ones(1,kelm);        % normalized noise variance
SNR = str2num(get(handles.snr_input,'String'));  % input SNR (dB)
SNR0= 10^(SNR/10);
n = str2num(get(handles.n_input,'String'));  % ¡§
A=exp(-j*twpi*d.'*sin(theta*derad)); % direction matrix
K=length(d);
cr=zeros(K,K);
L=length(theta);
%randn('state',12345);
data=randn(L,n);
data=sign(data);
%data(1,:)=data(4,:);
twpi = 2.0 * pi;
derad = pi / 180.0;
s = diag(pw)*data;
A1=exp(-j*twpi*d.'*sin([0:0.2:90]*derad));
%% generate sensor outputs
received_signal0 = A*s;% 
received_signal=received_signal0;
cx = received_signal + diag(sqrt(nv/SNR0/2))*(randn(K,n)+j*randn(K,n));% x=AS+n  
%cx = received_signal;
received_signal1=cx;
Rxx=received_signal1*received_signal1'/n;

%%%%%%%%%
%%Propagator Method 
G=Rxx(:,1:iwave);
H=Rxx(:,iwave+1:end);
P=inv(G'*G)*G'*H;
Q=[P',-diag(ones(1,kelm-iwave))];

for iang = 1:361
        angle1(iang)=(iang-181)/2;
        phim=derad*angle1(iang);
        a=exp(-j*twpi*d*sin(phim)).';
        SP(iang)=1/(a'*Q'*Q*a);
end
SP=abs(SP);
SPmax=max(SP);
SP=10*log10(SP/SPmax);
try
    delete(allchild(handles.plotarea))
end
axes(handles.plotarea)
h=plot(hObject,[0 0 30 60],angle1,SP,'-k');
set(h,'Linewidth',2)
grid on
axis([-90 90 -60 0])
xlabel('angle (degree)')
ylabel('magnitude (dB)')
set(handles.plotarea, 'XTick',[-90:30:90])
title('Propagator Method ')

grid on  
 


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
derad = pi/180;        % deg -> rad
radeg = 180/pi;
twpi = 2*pi;
kelm =str2num(get(handles.kelm_input,'String'));              % ¡ì
dd = 0.5;               % space 
d=0:dd:(kelm-1)*dd;     % 
iwave = 3;              % number of DOA
theta =[10 30 60]; % 
SNR = str2num(get(handles.snr_input,'String'));  % input SNR (dB)
n = str2num(get(handles.n_input,'String'));              % 
A=exp(-j*twpi*d.'*sin(theta*derad));%%%% direction matrix
S=randn(iwave,n);
X=A*S;
X1=awgn(X,SNR,'measured');
Rxx=X1*X1'/n;
InvS=inv(Rxx); %%%%
[EV,D]=eig(Rxx);%%%% 
EVA=diag(D)';
[EVA,I]=sort(EVA);
EVA=fliplr(EVA);
EV=fliplr(EV(:,I));

% MUSIC
for iang = 1:361
        angle1(iang)=(iang-181)/2;
        phim=derad*angle1(iang);
        a=exp(-j*twpi*d*sin(phim)).';
        L=iwave;    
        En=EV(:,L+1:kelm);
        SP(iang)=(a'*a)/(a'*En*En'*a);
end
   
% 
SP=abs(SP);
SPmax=max(SP);
SP=10*log10(SP/SPmax);

try
    delete(allchild(handles.plotarea))
end

axes(handles.plotarea)
h=plot(hObject,[0 0 50 60],angle1,SP,'-k');
set(h,'Linewidth',2)
grid on
axis([-90 90 -60 0])
xlabel('angle (degree)')
ylabel('magnitude (dB)')
title('MUSIC')
set(handles.plotarea, 'XTick',[-90:30:90])
grid on  

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

radeg = 180/pi;
derad=1/radeg;
twpi = 2*pi;
kelm = str2num(get(handles.kelm_input,'String'));               % ¡§
dd = 0.5;               % 
d=0:dd:(kelm-1)*dd;     % 
iwave = 3;              % number of DOA
theta = [10 30 60] ; % 
pw= [1.0  1.0  1];      % power
cb = [ 1 0 0
       0 1 0
       0 0 1];          % source relation 

nv=ones(1,kelm);        % normalized noise variance
n =  str2num(get(handles.n_input,'String'));                % 
A=exp(-j*twpi*d.'*sin(theta*derad));%%%% direction matrix
for iter=1:30
S=randn(iwave,n);
SNR0= str2num(get(handles.snr_input,'String')); % input SNR (dB)

X0=A*S;
X=awgn(X0,SNR0,'measured');
Rxx=X*X'/n;
[EV,D]=eig(Rxx);
EVA=diag(D)'; [EVA,I]=sort(EVA);
EVA=fliplr(EVA); EV=fliplr(EV(:,I));

% TLS-ESPRIT            
estimates=(tls_esprit(dd,Rxx,iwave));
disp('TLS-ESPRIT')
disp('angles'),disp(estimates(1,:))
doaes(:,iter)=sort(estimates(1,:));
end
try
    delete(allchild(handles.plotarea))
end

axes(handles.plotarea)

axis([0 30 0 70])
plot([1:30],doaes(1,1:30),'k*');hold on
plot([1:30],doaes(2,1:30),'k*');hold on
plot([1:30],doaes(3,1:30),'k*');
xlabel('experiment')
ylabel('DOA estimation')
title('ESPRIT')
set(handles.plotarea, 'XTick',[0:10:30])

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
derad = pi/180;        % deg -> rad
radeg = 180/pi;
twpi = 2*pi;
kelm = str2num(get(handles.kelm_input,'String'));               % ¡§
dd = 0.5;               % space 
d=0:dd:(kelm-1)*dd;     % 
iwave = 3;              % number of DOA
theta =[10 30 60]; % 
SNR = str2num(get(handles.snr_input,'String'));  % input SNR (dB)
n = str2num(get(handles.n_input,'String'));                % 
A=exp(-j*twpi*d.'*sin(theta*derad));%%%% direction matrix
S=randn(iwave,n);
X=A*S;
X1=awgn(X,SNR,'measured');
Rxx=X1*X1'/n;



for iang = 1:361
        angle1(iang)=(iang-181)/2;
        phim=derad*angle1(iang);
        a=exp(-j*twpi*d*sin(phim)).';
      
       
        SP(iang)=(a'*a)/(a'*inv(Rxx)*a);
end
   
% 
SP=abs(SP);
SPmax=max(SP);
SP=10*log10(SP/SPmax);
try
    delete(allchild(handles.plotarea))
end
axes(handles.plotarea)
h=plot(hObject,[0 0 30 60],angle1,SP,'-k');
set(h,'Linewidth',2)
grid on
axis([-90 90 -60 0])
xlabel('angle (degree)')
ylabel('magnitude (dB)')
title('Capon')
set(handles.plotarea, 'XTick',[-90:30:90])
grid on  

% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
derad = pi/180;
radeg = 180/pi;
twpi = 2*pi;
kelm = str2num(get(handles.kelm_input,'String'));            % ¡§
dd = 0.5;               % 
d=0:dd:(kelm-1)*dd;     % 
iwave = 3;             % number of DOA
theta = [10 30 60];  % 
SNR = str2num(get(handles.snr_input,'String'));  % input SNR (dB)   % input SNR (dB)
n =200;                % 
A=exp(-j*twpi*d.'*(sin(theta*derad)));
for iter=1:10
S=randn(iwave,n);
X0=A*S;
X=awgn(X0,SNR,'measured');
Rxx=X*X';
InvS=inv(Rxx); %%%%
[EVx,Dx]=eig(Rxx);%%%% 
EVAx=diag(Dx)';
[EVAx,Ix]=sort(EVAx);
EVAx=fliplr(EVAx);
EVx=fliplr(EVx(:,Ix));
% 
% Root-MUSIC
Unx=EVx(:,iwave+1:kelm);
syms z
pz = z.^([0:kelm-1]');
pz1 = (z^(-1)).^([0:kelm-1]);
fz = z.^(kelm-1)*pz1*Unx*Unx'*pz;
a = sym2poly(fz);
zx = roots(a);
rx=zx.';
[as,ad]=(sort(abs((abs(rx)-1))));
DOAest(iter,:)=asin(sort(-imag(log(rx(ad([1,3,5]))))/pi))*180/pi;
end
try
    delete(allchild(handles.plotarea))
end

axes(handles.plotarea)
axis([0 10 0 70])
plot([1:10],DOAest(1:10,1),'k*');hold on
plot([1:10],DOAest(1:10,2),'k*');hold on
plot([1:10],DOAest(1:10,3),'k*');

xlabel('experiment')
ylabel('DOA estimation')
title('ROOT-MUSIC')
set(handles.plotarea, 'XTick',[0:1:10])

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



function iwave_input_Callback(hObject, eventdata, handles)
% hObject    handle to iwave_input (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of iwave_input as text
%        str2double(get(hObject,'String')) returns contents of iwave_input as a double


% --- Executes during object creation, after setting all properties.
function iwave_input_CreateFcn(hObject, eventdata, handles)
% hObject    handle to iwave_input (see GCBO)
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


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
try
    delete(allchild(handles.plotarea))
end
% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
delete(gcf);
frame;



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




% --- Executes on mouse press over axes background.
function plotarea_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to plotarea (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)




% --- Executes on button press in pushbutton1.
function pushbutton10_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton2.
function pushbutton11_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton3.
function pushbutton12_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton4.
function pushbutton13_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton5.
function pushbutton14_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


