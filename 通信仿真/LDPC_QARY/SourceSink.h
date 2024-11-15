#pragma once

#ifndef SOURCESINK_H
#define SOURCESINK_H

#include "afxstd.h"

class CSourceSink
{
public:
	CSourceSink();
	~CSourceSink();

	inline double TolBlk() 
		{ return m_num_tot_blk; }
	inline double TolBit() 
		{ return m_num_tot_bit; }
	inline int ErrBlk() 
		{ return m_num_err_blk; }
	inline int ErrBit() 
		{ return m_num_err_bit; }
	inline double BER()
		{ return m_num_err_bit / m_num_tot_bit; }
	inline double FER() 
		{ return m_num_err_blk / m_num_tot_blk; }
	void GetBitStr(int *uu, int len);
	void GetSymStr(int *uu, int qary, int len);
	void ClrCnt();
	int CntErr(int *uu, int *uu_hat, int len, int accumulator);

private:
	double m_num_tot_blk;
	double m_num_tot_bit;
	int m_num_err_blk;
	int m_num_err_bit;
};

#endif // !SOURCESINK_H