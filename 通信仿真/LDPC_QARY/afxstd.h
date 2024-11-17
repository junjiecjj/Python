#pragma once

#ifndef AFXSTD_H
#define AFXSTD_H

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif // !_CRT_SECURE_NO_WARNINGS

#include <stdio.h>
//#include <tchar.h>

// TODO: 在此处引用程序需要的其他头文件
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include "FiniteField2.h"

#define PI 3.14159265358979
#define DOUBLE_POS_MAX 1.0e300
#define DOUBLE_POS_MIN 1.0e-11
#define INT_POS_MAX 1<<30
#define MAXNUM 1.0e300
#define SMALLPROB DOUBLE_POS_MIN
using namespace std;

#endif // !AFXSTD_H
