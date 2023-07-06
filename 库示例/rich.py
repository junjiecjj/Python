#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 20:50:24 2022
https://mp.weixin.qq.com/s?__biz=MzIxNjM4NDE2MA==&mid=2247514417&idx=3&sn=3d8de379a146011fe400184c85f20c67&chksm=978b2efea0fca7e8c7c3513b6dd5cb42a7f442e8754e9feb98328b8c35b228f57086409edf2d&mpshare=1&scene=1&srcid=0109t2apXtxwVlHQVMRxjnnv&sharer_sharetime=1647688492358&sharer_shareid=0d5c82ce3c8b7c8f30cc9a686416d4a8&exportkey=AZHrML3e2R1RglRGWJDfUyQ%3D&acctmode=0&pass_ticket=HtUZpbsBpQ0mIk%2BoPK4K47uZExu7fNHfVfHlbsTOmMv74UV9HL0sgd84sBqhifTg&wx_header=0#rd
@author: jack
"""

from rich.console import Console
from rich.table import Column, Table

console = Console()


# 自定义 Console 控制台输出
console.print("Hello", "World!")
console.print("Hello", "World!", style="bold red")

# Rich 的 Print 功能
print("Hello, [bold magenta]World[/bold magenta]!", ":vampire:", locals())
console.print("Where there is a [bold cyan]Will[/bold cyan] there [u]is[/u] a [i]way[/i].")


# Console 控制台记录
test_data = [
    {"jsonrpc": "2.0", "method": "sum", "params": [None, 1, 2, 4, False, True], "id": "1",},
    {"jsonrpc": "2.0", "method": "notify_hello", "params": [7]},
    {"jsonrpc": "2.0", "method": "subtract", "params": [42, 23], "id": "2"},
]

def test_log():
    enabled = False
    context = {
        "foo": "bar",
    }
    movies = ["Deadpool", "Rise of the Skywalker"]
    console.log("Hello from", console, "!")
    console.log(test_data, log_locals=True)


test_log()

# 表情符号
console.print(":smiley: :vampire: :pile_of_poo: :thumbs_up: :raccoon:")

# 7.表格
table = Table(show_header=True, header_style="bold magenta")
table.add_column("Date", style="dim", width=12)
table.add_column("Title")
table.add_column("Production Budget", justify="right")
table.add_column("Box Office", justify="right")
table.add_row(
    "Dev 20, 2019", "Star Wars: The Rise of Skywalker", "$275,000,000", "$375,126,118"
)
table.add_row(
    "May 25, 2018",
    "[red]Solo[/red]: A Star Wars Story",
    "$275,000,000",
    "$393,151,347",
)
table.add_row(
    "Dec 15, 2017",
    "Star Wars Ep. VIII: The Last Jedi",
    "$262,000,000",
    "[bold]$1,332,539,889[/bold]",
)

console.print(table)


# Rich 可以渲染多个不闪烁的进度条形图，以跟踪长时间运行的任务。
# 基本用法：用 track 函数调用程序并迭代结果。下面是一个例子：
import time
def do_step(index):
    for i in range(index):
        time.sleep(0.01)
        
from rich.progress import track
for step in track(range(20)):
    do_step(step)

from rich.progress import track
for i in track(range(3), description='Processing...'):
    time.sleep(1)


from rich.progress import Progress
with Progress() as progress:
    
        task1 = progress.add_task('[red]Downloading...', total=1000)
        task2 = progress.add_task('[green]Processing...', total=1000)
        task3 = progress.add_task('[cyan]Cooking...', total=1000)
    
        while not progress.finished:
            progress.update(task1, advance=5)
            progress.update(task2, advance=3)
            progress.update(task3, advance=9)
            time.sleep(0.02)
            

# rich.panel.Panel的实例为一个被线条包围的可渲染对象            
from rich.panel import Panel
console.print(Panel('[blue]This is a panel[/]'))

# 9.按列输出数据
# import os
# import sys

# from rich import print
# from rich.columns import Columns

# directory = os.listdir(sys.argv[1])
# print(Columns(directory))






#10.Markdown
#Rich 可以呈现markdown，相当不错的将其格式显示到终端。
#为了渲染 markdown，请导入 Markdown 类，将其打印到控制台。例子如下：
from rich.console import Console
from rich.markdown import Markdown

console = Console()
with open("/home/jack/公共的/MarkDown/FPGA.md") as readme:
    markdown = Markdown(readme.read())
console.print(markdown)

# 11.语法突出显示
# Rich 使用 pygments 库来实现语法高亮显示。用法类似于渲染 markdown。构造一个 Syntax 对象并将其打印到控制台。下面是一个例子：

from rich.console import Console
from rich.syntax import Syntax

my_code = '''
def iter_first_last(values: Iterable[T]) -&gt; Iterable[Tuple[bool, bool, T]]:
    """Iterate and generate a tuple with a flag for first and last value."""
    iter_values = iter(values)
    try:
        previous_value = next(iter_values)
    except StopIteration:
        return
    first = True
    for value in iter_values:
        yield first, False, previous_value
        first = False
        previous_value = value
    yield first, True, previous_value
'''
syntax = Syntax(my_code, "python", theme="solarized", line_numbers=True)
console = Console()
console.print(syntax)


#12.错误回溯(traceback)
#Rich 可以渲染漂亮的错误回溯日志，比标准的 Python 回溯更容易阅读，并能显示更多代码。
#你可以将 Rich 设置为默认的回溯处理程序，这样所有异常都将由 Rich 为你呈现。
#下面是在 OSX（与 Linux 类似）上的外观：

try:
    1/0
except:
        console.print_exception()




console.print([1, 2, 3])
console.print("[blue underline]Looks like a link")
console.print(locals())
console.print("FOO", style="white on blue")



console.rule("[bold red]Chapter 2")
console.rule("[bold red]Chapter 2", align='left')



console.log('hello world')











