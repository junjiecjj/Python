






# Bk3_Ch14_01

import numpy as np
from matplotlib import pyplot as plt

# Calculate sum of arithmetic progression sequence

def sum_AP(a, n, d):
    sum_ = (n * (a + a + (n - 1) * d)) / 2
    return sum_

a = 1    # initial term
n = 100  # number of terms
d = 1    # common difference

# Generate arithmetic progression, AP, sequence

AP_sequence = np.arange(a, a + n*d, d)
index       = np.arange(1, n + 1, 1)
print("AP sequence")
print(AP_sequence)

sum_result = sum_AP(a, n, d)
sum_result_2 = np.sum(AP_sequence)
print("Sum of AP sequence = " , sum_result)

fig, ax = plt.subplots(figsize=(20, 8))

plt.xlabel("Index, k")
plt.ylabel("Term, $a_k$")
plt.stem(index, AP_sequence)
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
plt.xlim(index.min(),index.max())
plt.ylim(0,AP_sequence.max() + 1)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

#%% cumulative sum

cumsum_AP = np.cumsum(AP_sequence)

fig, ax = plt.subplots(figsize=(20, 8))

plt.xlabel("Index, k")
plt.ylabel("Cumulative sum, $S_k$")
plt.stem(index, cumsum_AP)
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
plt.xlim(index.min(),index.max())
plt.ylim(0,cumsum_AP.max() + 1)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)







# Bk3_Ch14_02

import numpy as np
from matplotlib import pyplot as plt

a = 1    # initial term
n = 50   # number of terms
q = -1.1 # common ratio, q = 1.1, 1, 0.9, -0.9, -1, -1.1

# Generate geometric progression, GP, sequence

GP_sequence = [a*q**i for i in range(n)]
index       = np.arange(1, n + 1, 1)

fig, ax = plt.subplots()

plt.xlabel("Index, $k$")
plt.ylabel("Term, $a_k$")
plt.plot(index, GP_sequence, marker = '.',markersize = 6, linestyle = 'None')
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])



# Bk3_Ch14_03

def fib(n):
   if n <= 1:
       return (n)
   else:
       return (fib(n-1) + fib(n-2))

# Display n-term from Fibonacci sequence
n = 10  # number of terms
for i in range(n):
    print(fib(i))




# Bk3_Ch14_04

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def heatmap_sum(data,i_array,j_array,title):

    fig, ax = plt.subplots(figsize=(10, 10))


    ax = sns.heatmap(data,cmap='RdYlBu_r',
                     cbar_kws={"orientation": "horizontal"},
                     yticklabels=i_array, xticklabels=j_array,
                     ax = ax)
    ax.set_xlabel('Index, $j$')
    ax.set_ylabel('Index, $i$')

    ax.set_aspect("equal")
    plt.title(title)
    plt.yticks(rotation=0)

# Repeatability
np.random.seed(0)

m = 12 # j = 1 ~ n
n = 8  # i = 1 ~ m

j_array = np.arange(1,m + 1)
i_array = np.arange(1,n + 1)

jj, ii = np.meshgrid(j_array,i_array)

a_ij = np.random.normal(loc=0.0, scale=1.0, size=(n, m))

#%% heatmap of a_i_j

title = '$a_{i,j}$'
heatmap_sum(a_ij,i_array,j_array,title)

#%% partial summation of a_ij over i

# sum_over_i = a_ij.sum(axis = 0).reshape((1,-1))

all_1 = np.ones((8, 1))
sum_over_i = all_1.T@a_ij
# sum over row dimension

title = '$\sum_{i=1}^{n} a_{i,j}$'
heatmap_sum(sum_over_i,i_array,j_array,title)

#%% partial summation of a_ij over j

# sum_over_j = a_ij.sum(axis = 1).reshape((-1,1))

all_1 = np.ones((12, 1))
sum_over_j = a_ij@all_1
# sum over column dimension

title = '$\sum_{j=1}^{m} a_{i,j}$'
heatmap_sum(sum_over_j,i_array,j_array,title)


# Bk3_Ch14_05

from sympy import limit_seq, Sum, lambdify, factorial
from sympy.abc import n, k
import numpy as np
from matplotlib import pyplot as plt

seq_sum = Sum(1 / 2**k,(k, 0, n))
seq_sum = Sum(1 /((k + 1)*(k + 2)),(k, 0, n))
seq_sum = Sum(1 /factorial(k),(k, 0, n))

seq_limit = limit_seq(seq_sum, n)

seq_sum_fcn = lambdify(n,seq_sum)

seq_sum.evalf(subs={n: 5})

n_array = np.arange(0,100 + 1,1)

seq_sum_array = []

for n in n_array:

    seq_n = seq_sum_fcn(n)

    seq_sum_array.append(seq_n)

fig, ax = plt.subplots()

ax.plot(n_array,seq_sum_array,linestyle = 'None', marker = '.')

ax.set_xlabel('$k$')
ax.set_ylabel('Sum of sequence')
ax.set_xscale('log')
ax.set_ylim(0,3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(True, which="both", axis='x')
plt.tight_layout()
plt.show()



















# Bk3_Ch14_06

import numpy as np
import matplotlib.pyplot as plt

n_array = np.arange(1, 100 + 1, 1)

a_n_array = (-1)**(n_array + 1)/(2*n_array - 1)

a_n_cumsum = np.cumsum(a_n_array)

pi_appx = 4*a_n_cumsum

fig, ax = plt.subplots(figsize=(20, 8))

plt.xlabel("Index, k")
plt.ylabel("Approx pi")
plt.stem(n_array, pi_appx)
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
plt.axhline(y=np.pi, color='r', linestyle='-')

plt.xlim(n_array.min(),n_array.max())
plt.ylim(2.5,4.1)


plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)


































































































































































































































































