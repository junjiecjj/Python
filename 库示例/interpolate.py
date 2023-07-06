import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import leastsq
# 记录差值拟合函数的用法
################################################
#  https://zhuanlan.zhihu.com/p/28149195

plt.figure(figsize=(12,9))
x=np.linspace(0,10,11)
y=np.sin(x)
ax=plt.plot()

plt.plot(x,y,'ro')

xnew=np.linspace(0,10,101)
#最近邻，0阶，线性、二次插值等插值方法
for kind in ['nearest','zero','linear','quadratic',7]:
    f=interpolate.interp1d(x,y,kind=kind)
    ynew=f(xnew)
    plt.plot(xnew,ynew,label=str(kind))
    
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(loc='lower right')
plt.show()
################################################


x=np.linspace(0,2*np.pi+np.pi/4,10)
y=np.sin(x)

#立方插值
x_new=np.linspace(0,2*np.pi+np.pi/4,100)
f_linear=interpolate.interp1d(x,y,kind='cubic')

#B-spline插值
tck=interpolate.splrep(x,y)
y_bspline=interpolate.splev(x_new,tck)

plt.xlabel('安培/A',fontproperties='SimHei',fontsize=20)
plt.ylabel('伏特/A',fontproperties='SimHei',fontsize=20)

plt.plot(x,y,'o',label='origin data')
#plt.plot(x_new,f_linear(x_new),'r',label='linear')
plt.plot(x_new,y_bspline,'b',label='B-spline')

plt.legend()
plt.show()
################################

#拟合
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
#######################################


#此段内容在以下网址中
#https://www.yiibai.com/scipy/scipy_interpolate.html

x=np.linspace(0,4,20)
y=np.cos(x**2/3+4)
print(x,y)
plt.plot(x,y,'o')
plt.show()

f1=interpolate.interp1d(x,y,kind='linear')#线性插值
f2=interpolate.interp1d(x,y,kind='cubic')#三次方插值
xnew=np.linspace(0,4,8)

plt.plot(x,y,'o',xnew,f1(xnew),'-',xnew,f2(xnew),'--')
plt.legend(['linear','nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next'],loc='best')
plt.show()

x = np.linspace(-3,3,50)
y = np.exp(-x**2)+0.1*np.random.randn(50)

plt.plot(x,y,'ro',ms=4)
plt.show()

spl=UnivariateSpline(x,y)#拟合
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

#############################
'''
interp1d也可以对多维数组进行插值，即dat数组的第一行为y1,第二行为y2,第三行为y3
以x为横坐标，y1,y2,y3分别为纵坐标，绘出三条曲线，然后对三条曲线分别插值
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

x=np.arange(0,np.pi,0.5)
y=np.sin(x)
z=np.cos(x)
k=x

dat=np.zeros((3,len(x)))

dat[0]=y
dat[1]=z
dat[2]=k
fig=plt.figure()
ax1=fig.add_subplot(211)
ax1.plot(x,y,'-',x,z,'o',x,k,'*')
f = interp1d(x,dat,kind='linear',fill_value="extrapolate")

new_x=np.arange(0,np.pi,0.1)
y=f(new_x)
ax2=fig.add_subplot(212)
ax2.plot(new_x,y[0],'-',new_x,y[1],'o',new_x,y[2],'*')
