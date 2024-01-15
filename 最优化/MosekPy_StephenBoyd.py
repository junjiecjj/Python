#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 23:49:50 2022

@author: jack

# 安装Mosek
pip install Mosek
conda install -c mosek mosek

手动安装：
1. 下载Mosek的安装包，然后将安装包放在家目录下；
2. 下载 mosek.lic放在1中的mosek目录下；
3. cd  /home/jack/mosek/10.0/tools/platform/linux64x86/python/3
4. python setup.py install --user
"""


import sys
import mosek

# Since the value of infinity is ignored, we define it solely
# for symbolic purposes
inf = 0.0

# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def main():
    # Create a task object
    with mosek.Task() as task:
        # Attach a log stream printer to the task
        task.set_Stream(mosek.streamtype.log, streamprinter)

        # Bound keys for constraints
        bkc = [mosek.boundkey.fx,
               mosek.boundkey.lo,
               mosek.boundkey.up]

        # Bound values for constraints
        blc = [30.0, 15.0, -inf]
        buc = [30.0, +inf, 25.0]

        # Bound keys for variables
        bkx = [mosek.boundkey.lo,
               mosek.boundkey.ra,
               mosek.boundkey.lo,
               mosek.boundkey.lo]

        # Bound values for variables
        blx = [0.0, 0.0, 0.0, 0.0]
        bux = [+inf, 10.0, +inf, +inf]

        # Objective coefficients
        c = [3.0, 1.0, 5.0, 1.0]

        # Below is the sparse representation of the A
        # matrix stored by column.
        asub = [[0, 1],
                [0, 1, 2],
                [0, 1],
                [1, 2]]
        aval = [[3.0, 2.0],
                [1.0, 1.0, 2.0],
                [2.0, 3.0],
                [1.0, 3.0]]

        numvar = len(bkx)
        numcon = len(bkc)

        # Append 'numcon' empty constraints.
        # The constraints will initially have no bounds.
        task.appendcons(numcon)

        # Append 'numvar' variables.
        # The variables will initially be fixed at zero (x=0).
        task.appendvars(numvar)

        for j in range(numvar):
            # Set the linear term c_j in the objective.
            task.putcj(j, c[j])

            # Set the bounds on variable j
            # blx[j] <= x_j <= bux[j]
            task.putvarbound(j, bkx[j], blx[j], bux[j])

            # Input column j of A
            task.putacol(j,                  # Variable (column) index.
                         asub[j],            # Row index of non-zeros in column j.
                         aval[j])            # Non-zero Values of column j.

        # Set the bounds on constraints.
         # blc[i] <= constraint_i <= buc[i]
        for i in range(numcon):
            task.putconbound(i, bkc[i], blc[i], buc[i])

        # Input the objective sense (minimize/maximize)
        task.putobjsense(mosek.objsense.maximize)

        # Solve the problem
        task.optimize()
        # Print a summary containing information
        # about the solution for debugging purposes
        task.solutionsummary(mosek.streamtype.msg)

        # Get status information about the solution
        solsta = task.getsolsta(mosek.soltype.bas)

        if (solsta == mosek.solsta.optimal):
            xx = task.getxx(mosek.soltype.bas)

            print("Optimal solution: ")
            for i in range(numvar):
                print("x[" + str(i) + "]=" + str(xx[i]))
        elif (solsta == mosek.solsta.dual_infeas_cer or
              solsta == mosek.solsta.prim_infeas_cer):
            print("Primal or dual infeasibility certificate found.\n")
        elif solsta == mosek.solsta.unknown:
            print("Unknown solution status")
        else:
            print("Other solution status")


# call the main function
try:
    main()
except mosek.Error as e:
    print("ERROR: %s" % str(e.errno))
    if e.msg is not None:
        print("\t%s" % e.msg)
        sys.exit(1)
except:
    import traceback
    traceback.print_exc()
    sys.exit(1)





#=====================================================================================
# https://blog.csdn.net/weixin_42062224/article/details/120385023
from mosek.fusion import *
def main1( ):
    # constraint的系数，每行指代一个不等式
    A = [[50.0, 31.0],
         [3.0, -2.0]]
    # cost的系数
    c = [1.0, 0.64]
    with Model('milo1') as M:
        # 1.设置变量，名称x，数量为2，限制为非负整数。greaterThan和lessThan都是包含了“=”
        x = M.variable('x', 2, Domain.integral(Domain.greaterThan(0.0)))

        # 2.设置constraints
        # 50.0 x[0] + 31.0 x[1] <= 250.0
        # 3.0 x[0] - 2.0 x[1] >= -4.0
        M.constraint('c1', Expr.dot(A[0], x), Domain.lessThan(250.0))
        M.constraint('c2', Expr.dot(A[1], x), Domain.greaterThan(-4.0))

        # 3.设置求解器terminate的条件约束，
        # 3.1 time-out时间为60（单位未知，maybe60ms，超时会返回目前suboptimal的答案）
        # Set max solution time
        M.setSolverParam('mioMaxTime', 60.0)
        # 3.2 类似精度的约束，详见FusionAPI13.4.5
        # Set max relative gap (to its default value)
        M.setSolverParam('mioTolRelGap', 1e-4)
        # Set max absolute gap (to its default value)
        M.setSolverParam('mioTolAbsGap', 0.0)

        # 4.键入目标函数cost为Maximize或者Minimize
        # Set the objective function to (c^T * x)
        M.objective('obj', ObjectiveSense.Maximize, Expr.dot(c, x))

        # Solve the problem
        M.solve()

        # x.level()返回求解情况即x值为多少的list，其余的也是求解精度
        print('[x0, x1] = ', x.level())
        print("MIP rel gap = %.2f (%f)" % (M.getSolverDoubleInfo(
        "mioObjRelGap"), M.getSolverDoubleInfo("mioObjAbsGap")))


main1()


























