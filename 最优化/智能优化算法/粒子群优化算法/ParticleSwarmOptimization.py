

# https://mp.weixin.qq.com/s?__biz=MzAxMjg5NzQ5OQ==&mid=2247485704&idx=1&sn=a6c3bd24e65538369795d5126bf707f3&chksm=9bab94bcacdc1daa06dec8c24402e87163bb9710cef895c4f23328ff3009e1bc565d0677e131&scene=21#wechat_redirect

## https://mp.weixin.qq.com/s?__biz=MzAxMjg5NzQ5OQ==&mid=2247484458&idx=1&sn=0d4dcd52a687bc8ca233c598b919ad6f&scene=21#wechat_redirect

# https://mp.weixin.qq.com/s?__biz=MzAxMjg5NzQ5OQ==&mid=2247485767&idx=1&sn=693399f66f482991774fa82eecefbf0a&chksm=9ad157cde7055dad05e65c3b1dfe198a588f85dc5997ca6a53638b833b4a16bf89d16e36edef&mpshare=1&scene=1&srcid=0901ndDFyuLHgLOSF1NCuPyy&sharer_shareinfo=61f34979ab9dbc389c3f3f77cabab6f0&sharer_shareinfo_first=61f34979ab9dbc389c3f3f77cabab6f0&exportkey=n_ChQIAhIQI4QW%2BjrlYCgPZu%2F9GMTHSRKfAgIE97dBBAEAAAAAADciJVCLvn4AAAAOpnltbLcz9gKNyK89dVj0wUx%2FGLUSIeg2%2BOPfQWwhSq%2By9%2BDdWndiZH6CHta3bfLZuRJesr2zRRRlpgH6KDJ7pLv7q4iWCXtI7Noie5LCvEWaXOeLTcIwqz%2BGLEeFSYWRt2Tcj7c9gkRmdg8AD%2FuC54QjBnl3H0zfy5Pw8XCqssi6jv4W6NaefSU%2FU8z0pgimR8sri1dCDr7%2FG678SAHAZqjlFevwQwP1OsoiYtnkrdSTbmePei34HaXNPyZFhRDADhPScTTJEiiATmFbqpokApke%2FWrv930pwB%2B5eUh9D20vHrEhpL%2FzoiqgZLnh1DAJgKdg%2B%2F80IXK9P%2Fcby%2FWSktU8olLp%2FJ4T&acctmode=0&pass_ticket=w9Z8%2BPhs7M7bx2nQIhtEqzjaA5TtmdVUkrs3%2B%2FgnaujOJVQlQWAXspRPy0MvD8bi&wx_header=0#rd

## 多目标粒子群优化算法，附完整代码直接免费获取
import random
import matplotlib.pyplot as plt
# 定义目标函数
def objective_function1(x):
    return x**2

def objective_function2(x):
    return (x - 2)**2
# 评估适应度函数
def evaluate_fitness(particle):
    return [objective_function1(particle), objective_function2(particle)]
# 判断解A是否支配解B
def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))
# 非支配排序函数
def non_dominated_sorting(solutions):
    non_dominated = []
    for solution in solutions:
        if not any(dominates(other, solution) for other in solutions):
            non_dominated.append(solution)
    return non_dominated

# 更新归档函数
def update_archive(particles, archive):
    combined_solutions = particles + archive
    non_dominated_solutions = non_dominated_sorting(combined_solutions)
    archive_size = 50  # 设定归档大小
    new_archive = non_dominated_solutions[:archive_size]
    return new_archive

# 初始化粒子群和归档
particles = [random.random() * 5 for _ in range(10)]
archive = []

# 评估并更新归档
for particle in particles:
    fitness = evaluate_fitness(particle)
    archive = update_archive([fitness], archive)

# 提取归档中的解
archive_x = [sol[0] for sol in archive]
archive_y = [sol[1] for sol in archive]

# 绘制图示
plt.figure(figsize=(10, 6))
plt.scatter(archive_x, archive_y, color='red', label='Non-dominated Solutions in Archive')
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.title('External Archive in MOPSO')
plt.legend()
plt.grid(True)
plt.show()
















































































































































































































































































































































































































