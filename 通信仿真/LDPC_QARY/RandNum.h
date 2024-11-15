#pragma once

#ifndef RANDNUM_H
#define RANDNUM_H

#include "afxstd.h"

class CRandNum  
{
public:
	CRandNum();
	~CRandNum();

};

class CLCRandNum  
{
public:
	CLCRandNum();
	~CLCRandNum();

	void SetSeed(int flag);
	void SetState(long int s);
	void PrintState(FILE *fp);
	long int getState();
	double Uniform();
	void Normal(double *nn, int len_nn);

private:
    long int state;

    static const int A;
    static const long M;
    static const int Q;
    static const int R;
};


class CWHRandNum  
{
public:
	CWHRandNum();
	~CWHRandNum();

	void SetSeed(int flag);
	void SetState(int x, int y, int z);
	void PrintState(FILE *fp);
	void getState(int *x, int *y, int *z);
	double Uniform();
	void Normal(double *nn, int len_nn);

private:
    int X, Y, Z;
};

#endif // !RANDNUM_H