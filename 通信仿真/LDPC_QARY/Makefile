


#  ①变量定义
CC=g++			# 定义变量，表示使用g++编译器
# SRCS=main.cpp a.cpp	  # 表示项目中所需要的源文件，有更多直接在后面添加，换行用反斜杠

DIR_SRC=.


#把./source中的所有.c文件拿出来赋给变量SRC_PATH，包括.c文件的路径
SRC_PATH=$(wildcard ${DIR_SRC}/*.cpp)

# 去掉SRC的路径，然后把所有.c文件后缀名，改为.o，最后把这些.o文件赋给变量
OBJS = $(patsubst %.cpp,%.o,$(notdir ${SRC_PATH}))


# SRCS=AWGN.cpp BCHCodec.cpp BinaryLDPCCodec.cpp Blockfading.cpp BPSKModem.cpp  CodedModulationSystemSimulator.cpp EbitsinLDPC_simulator.cpp FiniteField2.cpp RandNum.cpp RanXORLDPCCodec.cpp RepetitionCodec.cpp SourceSink.cpp   Utility.cpp Time.cpp
# OBJS=$(SRCS:.cpp=.o)	#  表示目标依赖文件.o，其又依赖于SRCS中的cpp文件

EXEC=run		# 表示生成的可执行文件
CFLAGS=-Wall -O3 -g	# 配置编译器，-Wall表示输出警告信息，-O表示编译优化，-g表示编译debug版本
LDLIBS += -L./usrLib -lm


#  ②目标文件的编译规则
$(EXEC):$(OBJS)		# 规则入口->目标文件：引入依赖文件,等价于mytest:main.o a.o
	$(CC) $(CFLAGS) -o $(EXEC) $(OBJS) $(LDLIBS)
# 等价于g++ -o mytest main.o a.o;注意：命令前要缩进tab

# ③目标依赖文件的编译规则
%o:%cpp			#缺省规则：等价于%o:%cpp，表示OBJS中所有的.o文件与SRCS中同名.cpp文件的依赖关系（OBJS中有多个.o，下面命令就需要执行多次）
	$(CC) $(CFLAGS)   -c $^ -o $@	# 等价于g++ -o main.o -c main.cpp和g++ -o a.o -c a.cpp


#  ④清除命令
# 注意：此处clean不是文件，而是一个动作
clean:
	rm -rf $(OBJS)  #删除依赖文件
	# rm -rf *.o      #删除依赖文件

cleanall:
	rm -rf $(OBJS)  #删除依赖文件
	rm -rf ${EXEC}



all:
	@echo ${SRC_PATH}
	@echo ${OBJS}
	@echo ${OBJ_PATH}




# 一些常见的自动化变量说明如下：
# 变量		  解析
# $0		 当前脚本的文件名。
# $n（n≥1）	 传递给脚本或函数的参数。n 是一个数字，表示第几个参数。例如，第一个参数是 $1，第二个参数是 $2>。
# $#		传递给脚本或函数的参数个数。
# $*		传递给脚本或函数的所有参数。
# $@		表示目标文件。
# $?		上个命令的退出状态，或函数的返回值。
# $$		当前 Shell 进程 ID。对于 Shell 脚本，就是这些脚本所在的进程 ID。
# $^			所有依赖文件的集合，用空格分开，如果其中有重复的依赖文件，则只保留一个。比如依赖文件有三个: 1.c, 2.c, 1.c，那么$^拿到的只有 1.c 和 2.c 两个文件，舍弃了一个1.c
# $<			依赖文件中的第一个文件。如果依赖文件是以 “%” 定义的，那么 $< 就是依赖文件的集合。一般不会有重复的依赖文件，所以上面的 Makefile，用 $^ 替换掉 $< 也是可以的
# $+			与 $^ 一样，只是如果有重复的依赖文件，不会舍弃重复的依赖文件
# $@			目标文件的集合

# wildcard、notdir、patsubst的意思:
# wildcard : 扩展通配符
# notdir :   去除路径
# patsubst ：替换通配符
