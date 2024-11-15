// RandNum.h: interface for the CRandNum class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_RANDNUM_H__72FEDEAF_A51C_4724_9A53_5F2F5E00CED4__INCLUDED_)
#define AFX_RANDNUM_H__72FEDEAF_A51C_4724_9A53_5F2F5E00CED4__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Designed by Xiao Ma (maxiao@mail.sysu.edu.cn), Sun Yat-sen University.
//This program can only be employed for academic research.
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CRandNum  
{
public:
	CRandNum();
	virtual ~CRandNum();

};

//////////////////////////////////////////////////////////////////////
/*
The following generator employs the linear-congruential method,
and specifically uses a choice of multiplier that was proposed
as a standard by Stephen K. Park et al. in "Technical correspondence,"
Communications of the ACM36(1993), number 7, 108-110
*/
/////////////////////////////////////////////////////////////////////
class CLCRandNum  
{
public:
	CLCRandNum();
	virtual ~CLCRandNum();

	void SetSeed(int flag);
	void PrintState(FILE *fp);
	double Uniform();
	void Normal(double *nn, int len_nn);

private:
    long int state;

    static const int A;
    static const long M;
    static const int Q;
    static const int R;
};

/////////////////////////////////////////////////////////////
//The following generator employs the Wichman-Hill algorithm
/////////////////////////////////////////////////////////////

class CWHRandNum  
{
public:
	CWHRandNum();
	virtual ~CWHRandNum();

	void SetSeed(int flag);
	void PrintState(FILE *fp);
	double Uniform();
	void Normal(double *nn, int len_nn);

private:
    int X, Y, Z;
};


#endif // !defined(AFX_RANDNUM_H__72FEDEAF_A51C_4724_9A53_5F2F5E00CED4__INCLUDED_)
