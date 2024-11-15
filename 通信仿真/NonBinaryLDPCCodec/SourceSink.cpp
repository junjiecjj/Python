// SourceSink.cpp: implementation of the CSourceSink class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "SourceSink.h"

#include "RandNum.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Programmed by Xiao Ma (maxiao@mail.sysu.edu.cn), Sun Yat-sen University.
//This program can only be employed for academic research.
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

extern CLCRandNum rndGen0;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CSourceSink::CSourceSink()
{

}

CSourceSink::~CSourceSink()
{

}

/////////////////////////////////////////////////////////////////////////////
void CSourceSink::GetBitStr(int *uu, int len)
{
	int t; 

	for (t = 0; t < len; t++)
	{
		uu[t] = (rndGen0.Uniform() < 0.5?0:1);
//	    uu[t] = 0;
	}
	
	return;
}

/////////////////////////////////////////////////////////////////////////////
void CSourceSink::GetSymStr(int *uu, int qary, int len)
{
	int t; 

	for (t = 0; t < len; t++)
	{
		uu[t] = qary;
		while (uu[t] == qary)
		{
			uu[t] = (int)(qary * rndGen0.Uniform());
		}
//	    uu[t] = 0;
	}
	
	return;
}


//////////////////////////////////////////////////////////////////////////////////
void CSourceSink::ClrCnt()
{
	m_num_tot_blk = 0;
	m_num_tot_bit = 0;
	m_num_err_blk = 0;
	m_num_err_bit = 0;
	
	return;
}

//////////////////////////////////////////////
void CSourceSink::CntErr(int *uu, int *uu_hat, int len, int accumulator)
{
	int t;

	m_temp_err = 0;
	for (t = 0; t < len; t++)
	{
		if (uu_hat[t] != uu[t])
		{
			m_temp_err++;
		}
	}

	if (accumulator == 1)
	{
		if (m_temp_err > 0)
		{
			m_num_err_bit += m_temp_err;
			m_num_err_blk += 1;
		}
		
		m_num_tot_blk += 1.0;
		m_num_tot_bit += len;

		m_ber = m_num_err_bit / m_num_tot_bit;
		m_fer = m_num_err_blk / m_num_tot_blk;
	}

	return;
}

///////////////////////////////////////////////////
void CSourceSink::PrintResult(FILE *fp)
{
	fprintf(fp, "\n tot_blk = %d: err_blk = %d: err_bit = %d: ber = %12.10f", 
		(int)m_num_tot_blk, m_num_err_blk, m_num_err_bit, m_ber);
	
	return;
}