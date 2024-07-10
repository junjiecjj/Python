




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Bk1_Ch25_01

# 导入包
import numpy as np
import matplotlib.pyplot as plt

from sympy import init_printing
init_printing("mathjax")


# 定义符号变量
from sympy import symbols
# 从sympy中导入symbols

x, y = symbols('x y')
# 用sympy.symbols (简做symbols) 定义x和y两个符号变量

# 定义解析式：
f1 = x + y
# f1 # 打印结果


f1 + 1

f2 = x**2 - y**2
f2 # 打印结果

f2 + 2*x + 1

f2 * x

f2 ** x

# 将字符串转化为符号表达式
str_expression = 'x**3 + x**2 + x + 1'
str_expression

from sympy import sympify
str_2_sym = sympify(str_expression)
str_2_sym



################################ 分式
# 定义分式
from sympy import Rational
Rational(1, 2)



x + Rational(1, 2)


x * Rational(1, 2)

from sympy import sqrt, simplify
1 / (sqrt(2) + 1)

float(1 / (sqrt(2) + 1))
simplify(1 / (sqrt(2) + 1))


################################ 假设条件

from sympy import symbols
k = symbols('k', integer=True)
x, y, z = symbols('x,y,z', real=True)




################################ 符号数字

from sympy import S
S(1)


1/S(5)

float(1/S(5))


from sympy import pi
pi
float(pi)

int(pi)


from sympy import exp
exp(2)

sqrt(2)

expression_sqrt = sqrt(2) + 1


expression_sqrt.evalf()

import sympy as sym

sym.isprime(3)


####################### 因式分解、展开#
from sympy import expand, factor
f2

f2_factored= factor(f2)
f2_factored

f2_b_factored = (x - y)**2*(x + y)*(x - 1)
f2_b_factored

expand(f2_b_factored)

# 将符号表达式转化为str
str(f2_b_factored)

str(expand(f2_b_factored))

expr_x_y = x**4 - x**3*y - x**3 - x**2*y**2 + x**2*y + x*y**3 + x*y**2 - y**3
from sympy import factor_list
factor_list(expr_x_y)


#################### 判断两个多项式是否等价

from sympy import simplify
simplify(f2 - f2_factored)

# 第二种方法
f2_factored.equals(f2)

# 化简
simplify((x**3 + x**2 - x - 1)/(x**2 + 2*x + 1))

### 合并同类项
from sympy import collect

expr = x*y + x - 3 + 2*x**2 - z*x**2 + x**3
collect(expr, x)

## 带下角标的变量
x1, x2 = symbols('x1 x2')

f4 = x1 + x2
f4

f5 = (x1 - x2)**5
f5

f5_expanded = expand(f5)
f5_expanded

########################### 提取系数
from sympy import Poly
# 导入sympy.Poly，简做Poly

# 将x2看成常数
f5_expanded_x1 = Poly(f5_expanded, x1)
f5_expanded_x1.coeffs()

# 将x1看成常数
f5_expanded_x2 = Poly(f5_expanded, x2)
f5_expanded_x2.coeffs()

# 将x1、x2都看成是变量
f5_expanded_x1x2 = Poly(f5_expanded, [x1, x2])
coefficients = f5_expanded_x1x2.coeffs()
coefficients

type(coefficients)
# 判断结果类型


# 判断元素类型
type(coefficients[0])


# 将数值转化为浮点数float
coefficients_float = eval(str(coefficients))
# 先将coefficients 转化为str，再转化为浮点数


import matplotlib.pyplot as plt
plt.stem(coefficients_float)


##################### 解等式
from sympy import solve
from sympy import solveset,Eq


equality = x**2 - 4
solve(equality, x)


solveset(Eq(x**2, 1), x)
solveset(Eq(x**2 - 1, 0), x)


a,b,c = symbols("a,b,c", real=True)
# 定义等式 a*x**2+b*x = -c
equation_2 = Eq(a*x**2+b*x+c, 0)
solve(equation_2, x)



#################### 阶乘、组合数、Gamma函数
from sympy import factorial

##### 阶乘
factorial(5)


factorial(x)


#### 组合数
from sympy import binomial
binomial(5, 4)

factorial(5)/factorial(4)
factorial(6)/factorial(3)



# Gamma函数¶
from sympy import gamma
gamma(5)


######################## 函数
from sympy import exp, cos, sin, sqrt
f_gaussian_x1 = exp(-x1**2)
f_gaussian_x1


f_sin_x1 = sin(x1)
f_sin_x1


from sympy import log
log(x*y)

#### log展开
# 假设条件
from sympy import expand_log
x, y = symbols('x y', positive=True)
expand_log(log(x*y))

expand_log(log(x**2))


### 三角函数展开
from sympy import expand_trig
expr = sin(2*x) + cos(2*x)
expand_trig(expr)


## Gamma函数
from sympy import gamma
gamma(x)
simplify(gamma(x)/gamma(x - 2))


# 将符号变量替换成具体数值
f1

f1_x_to_2 = f1.subs(x, 2)
f1_x_to_2
# x = 1赋值运算对符号变量没有影响


# 同时将x、y替换成数值
f1.subs([(x, 2), (y, 4)])


f1.evalf(subs={x: 2})


f1.evalf(subs={x: 2, y: 4})


# 也可以用subs() 将变量替换成其他变量、表达式
f1.subs(x, x**2)

f1.subs(x, sin(x))


f1_ = f1.subs([(x, sin(x)), (y, -exp(-y**2))])
# 进一步替换
f1_.subs(sin(x),exp(x))


# 将符号函数转化为Python函数

# 一元函数
f_gaussian_x1
from sympy import lambdify
f_gaussian_x1_fcn = lambdify(x1, f_gaussian_x1)
f_gaussian_x1_fcn


import numpy as np
x1_array = np.linspace(-3,3,100)

f_x1_array = f_gaussian_x1_fcn(x1_array)

plt.plot(x1_array, f_x1_array)



# 二元函数
f_gaussian_x1x2 = exp(-x1**2 - x2**2)
f_gaussian_x1x2

f_gaussian_x1x2_fcn = lambdify([x1,x2],f_gaussian_x1x2)
f_gaussian_x1x2_fcn

xx1,xx2 = np.meshgrid(np.linspace(-3,3,100),np.linspace(-3,3,100))


ff = f_gaussian_x1x2_fcn(xx1,xx2)
plt.contourf(xx1,xx2,ff, levels = 20, cmap = 'RdYlBu_r')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot_wireframe(xx1,xx2,ff)





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SymPy线性代数
## 矩阵

from sympy import Matrix
A = Matrix([[1, 2, 3], [3, 2, 1]])
A

from sympy import shape
shape(A)

A.T

# 增加一行
Matrix([A, (0, 0, -1)])

# 指定形状
Matrix(2, 3, [1, 2, 3, 4, 5, 6])


### 行向量、列向量
a = Matrix([1, 2, 3])
a

a.T

Matrix([[1, 2, 3]])

##### 索引、切片
A[0, 0]
A[-1,-1]
A[0,:]
A[:,0]

A.row(0)
# A.row_del(0) 删除指定行
# 插入指定行 M.row_insert(1, Matrix([[0, 4]]))
A.row(-1)

A.col(0)
# A.col_del(0) 删除指定列
# M.col_insert(0, Matrix([1, -2]))

A.col(-1)

A[0:2, 0:2]



### 产生特殊矩阵
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt

eye(4)

m = Matrix(2, 2, [0, 1, 1, 2])
m.is_symmetric()

zeros(2)


zeros(2, 5)
ones(3)

ones(3,5)


from sympy import eye
eye(3)

A = eye(3)
A.is_diagonal() # 判断矩阵 A 是否为对角阵 (diagonal matrix)。


from sympy import zeros
zeros(3, 3)

from sympy import ones
ones(3, 2)

from sympy import diag
diag(1, 2, 3)



### 上三角
m = Matrix(2, 2, [1, 2, 0, 1])
m.is_upper

### 上三角
from sympy import ones
A = ones(3)
A.upper_triangular()

### 下三角
from sympy import ones
A = ones(4)
A.lower_triangular()


m = Matrix(2, 3, lambda i, j: 1)
m

m.reshape(1, 6)

m.reshape(3, 2)

from sympy import Matrix, symbols
A = Matrix(2, 2, symbols('a:d'))
A

A.rot90(-2)

## 基本运算
from sympy import Matrix

A = Matrix([[1, 3], [-2, 3]])
B = Matrix([[0, 3], [0, 7]])
A + B # 加法
A - B # 减法
3*A   # 标量乘矩阵
A.multiply_elementwise(B) # 逐项积
A * B # 矩阵乘法
A @ B

### 加法
A + B

# 逐项积
A = Matrix([[0, 1, 2], [3, 4, 5]])
B = Matrix([[1, 10, 100], [100, 10, 1]])
A.multiply_elementwise(B)

## 矩阵乘法
B = Matrix([0, 1, 1])

A*B

Matrix_2x2 = Matrix([[1.25, -0.75],
                     [-0.75, 1.25]])
Matrix_2x2


# 求逆
Matrix_2x2**-1
Matrix_2x2.inv()

# 求转置
Matrix_2x2.T

# 将符号矩阵转化为浮点数numpy矩阵
import numpy as np

np.array(Matrix_2x2).astype(np.float64)


##### 符号求逆
from sympy import symbols
a, b, c, d = symbols('a b c d')
A = Matrix([[a, b],
            [c, d]])

A.inv()

# 行列式值
A.det()

# 方阵的迹
A.trace()



##### 方程组
A = Matrix([[1, 2, 0], [3, 1, 2], [0, -1, 1]])
b = Matrix([[3], [4], [1]])
A.inv() @ b


from sympy import Symbol, S
a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
C = Matrix([[S(1) / 5, 5, 5], [3, 1, 7], [a, b, c]])
C.det()
C.inv()


A = Matrix(2, 2, [1, 2, 3, 4])
A.trace()


############ 正定性
from sympy import Matrix, symbols
from sympy.plotting import plot3d
x1, x2 = symbols('x1 x2')
x = Matrix([x1, x2])


A = Matrix([[1, 0], [0, 1]])
A.is_positive_definite
# True

x.T*A*x

f_x = x.T*A*x
f_x = f_x[0]
f_x

p = plot3d((x.T*A*x)[0, 0], (x1, -1, 1), (x2, -1, 1))



# 半正定
A = Matrix([[1, -1], [-1, 1]])
A.is_positive_definite
# False
A.is_positive_semidefinite
# True


# 负定
A = Matrix([[-1, 0], [0, -1]])
A.is_negative_definite
# True
A.is_negative_semidefinite
# True
A.is_indefinite
# False


# 半负定

# 不定
A = Matrix([[1, 2], [2, -1]])
A.is_indefinite
# True


##########
## Cholesky分解

from sympy import Matrix

A = Matrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
A.cholesky()

A.cholesky() * A.cholesky().T

## 对角化
M = Matrix([[1, 2, 0], [0, 3, 0], [2, -4, 2]])
M.is_diagonalizable()

M = Matrix([[3, -2,  4, -2], [5,  3, -3, -2], [5, -2,  2, -2], [5, -2, -3,  3]])
P, D = M.diagonalize()

## 特征值分解
M = Matrix([[3, -2,  4, -2], [5,  3, -3, -2], [5, -2,  2, -2], [5, -2, -3,  3]])
M


### 对符号数值矩阵求特征值
M.eigenvals()

M.eigenvects()
M.eigenvects()[0][2]


Matrix_2x2_abc = Matrix([[a**2, 2*a*b*c],
                         [2*a*b*c, b**2]])
Matrix_2x2_abc

Matrix_2x2_abc.eigenvals()

M.eigenvects()

####### 奇异值分解
from sympy import Matrix
A = Matrix([[0, 1],[1, 1],[1, 0]])
# 奇异值分解
U, S, V = A.singular_value_decomposition()

U.T @ U

V.T @ V

M = A @ A.T

M.eigenvects()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 可视化正定性
# 导入包
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify, expand, simplify

# 定义可视化函数
def visualize(xx1,xx2,f2_array):
    fig = plt.figure(figsize=(6,3))
    # 左子图，三维
    ax_3D = fig.add_subplot(1, 2, 1, projection='3d')
    ax_3D.plot_wireframe(xx1, xx2, f2_array, rstride=10, cstride=10, color = [0.8,0.8,0.8], linewidth = 0.25)
    ax_3D.contour(xx1, xx2, f2_array, levels = 12, cmap = 'RdYlBu_r')

    ax_3D.set_xlabel('$x_1$'); ax_3D.set_ylabel('$x_2$')
    ax_3D.set_zlabel('$f(x_1,x_2)$')
    ax_3D.set_proj_type('ortho')
    ax_3D.set_xticks([]); ax_3D.set_yticks([])
    ax_3D.set_zticks([])
    ax_3D.view_init(azim=-120, elev=30)
    ax_3D.grid(False)
    ax_3D.set_xlim(xx1.min(), xx1.max());
    ax_3D.set_ylim(xx2.min(), xx2.max())

    # 右子图，平面等高线
    ax_2D = fig.add_subplot(1, 2, 2)
    ax_2D.contour(xx1, xx2, f2_array, levels = 12, cmap = 'RdYlBu_r')

    ax_2D.set_xlabel('$x_1$'); ax_2D.set_ylabel('$x_2$')
    ax_2D.set_xticks([]); ax_2D.set_yticks([])
    ax_2D.set_aspect('equal'); ax_2D.grid(False)
    ax_2D.set_xlim(xx1.min(), xx1.max());
    ax_2D.set_ylim(xx2.min(), xx2.max())
    plt.tight_layout()

# 定义二元函数
def fcn(A, xx1, xx2):
    x1,x2 = symbols('x1 x2')
    x = np.array([[x1,x2]]).T
    f_x = x.T@A@x
    f_x = f_x[0][0]
    print(simplify(expand(f_x)))
    f_x_fcn = lambdify([x1,x2],f_x)
    ff_x = f_x_fcn(xx1,xx2)
    return ff_x

# 生成数据
x1_array = np.linspace(-2,2,201)
x2_array = np.linspace(-2,2,201)
xx1, xx2 = np.meshgrid(x1_array, x2_array)

# 不定矩阵
A = np.array([[2, 0], [0, 1]])
f2_array = fcn(A, xx1, xx2)
visualize(xx1,xx2,f2_array)






















































