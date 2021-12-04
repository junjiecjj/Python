import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import leastsq

plt.figure(figsize=(12,9))
x=np.linspace(0,10,11)
y=np.sin(x)
ax=plt.plot()

plt.plot(x,y,'ro')

xnew=np.linspace(0,10,101)
for kind in ['nearest','zero','linear','quadratic',7]:
    f=interpolate.interp1d(x,y,kind=kind)
    ynew=f(xnew)
    plt.plot(xnew,ynew,label=str(kind))
    
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(loc='lower right')
plt.show()

'''
x=np.linspace(0,2*np.pi+np.pi/4,10)
y=np.sin(x)

x_new=np.linspace(0,2*np.pi+np.pi/4,100)
f_linear=interpolate.interp1d(x,y,kind='cubic')

tck=interpolate.splrep(x,y)
y_bspline=interpolate.splev(x_new,tck)

plt.xlabel('安培/A',fontproperties='SimHei',fontsize=20)
plt.ylabel('伏特/A',fontproperties='SimHei',fontsize=20)

plt.plot(x,y,'o',label='origin data')
#plt.plot(x_new,f_linear(x_new),'r',label='linear')
plt.plot(x_new,y_bspline,'b',label='B-spline')

plt.legend()
plt.show()
'''

'''
plt.figure(figsize=(9,9))
x=np.linspace(0,10,1000)
X=np.array([8.19,2.72,6.39,8.71,4.7,2.66,3.78])
Y=np.array([7.01,2.78,6.47,6.71,4.1,4.23,4.05])

def f(p):
    k,b=p
    return(Y-(k*X+b))
    
r=leastsq(f,[1,0])
k,b=r[0]
print('k=',k,'b=',b)

plt.scatter(X,Y,s=100,alpha=1.0,marker='o',label='data point')

y=k*x+b

ax=plt.gca()
ax.set_xlabel(...,fontsize=20)
ax.set_ylabel(...,fontsize=20)

plt.plot(x,y,color='r',linewidth=2,linestyle=':',markersize=20,label='line')
plt.legend(loc=0,numpoints=1)
leg=plt.gca().get_legend()
ltext=leg.get_texts()
plt.setp(ltext,fontsize='xx-large')

plt.xlabel('安培/A',fontproperties='SimHei',fontsize=20)
plt.ylabel('伏特/A',fontproperties='SimHei',fontsize=20)

plt.xlim(0,x.max()*1.1)
plt.ylim(0,y.max()*1.1)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(loc='best')
plt.show()
'''

'''
此段内容在以下网址中
https://www.yiibai.com/scipy/scipy_interpolate.html

x=np.linspace(0,4,20)
y=np.cos(x**2/3+4)
print(x,y)
plt.plot(x,y,'o')
plt.show()

f1=interpolate.interp1d(x,y,kind='linear')
f2=interpolate.interp1d(x,y,kind='cubic')
xnew=np.linspace(0,4,8)

plt.plot(x,y,'o',xnew,f1(xnew),'-',xnew,f2(xnew),'--')
plt.legend(['data','linear','cubic','nearst'],loc='best')
plt.show()

x = np.linspace(-3,3,50)
y = np.exp(-x**2)+0.1*np.random.randn(50)

plt.plot(x,y,'ro',ms=4)
plt.show()

spl=UnivariateSpline(x,y)
xs=np.linspace(-3,3,1000)

plt.plot(xs,spl(xs),'g',lw=3)
plt.show()

plt.plot(x,y,'ro',xs,spl(xs),'g')
plt.show()

spl.set_smoothing_factor(0.5)
plt.plot(xs,spl(xs),'b',lw=3)
plt.show()

plt.plot(x,y,'ro',xs,spl(xs),'b')
plt.show()

x1=np.linspace(1,4096,1024)
x_new=np.linspace(1,4096,4096)

tck=interpolate.splrep(x1,data)
y_bspline=interpolate.splev(x_new,tck)
'''

