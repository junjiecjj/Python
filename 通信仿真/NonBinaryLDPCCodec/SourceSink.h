// SourceSink.h: interface for the CSourceSink class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_SOURCESINK_H__361A1355_49CD_4B5C_BD7F_0E4C5D0F773B__INCLUDED_)
#define AFX_SOURCESINK_H__361A1355_49CD_4B5C_BD7F_0E4C5D0F773B__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Designed by Xiao Ma (maxiao@mail.sysu.edu.cn), Sun Yat-sen University.
//This program can only be employed for academic research.
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class CSourceSink  
{
public:
	CSourceSink();
	virtual ~CSourceSink();
	
	void GetBitStr(int *uu, int len);
	void GetSymStr(int *uu, int qary, int len);
	void ClrCnt();
	void CntErr(int *uu, int *uu_hat, int len, int accumulator);
	void PrintResult(FILE *fp);

public:
	double m_num_tot_blk;
	double m_num_tot_bit;
	int m_num_err_blk;
	int m_num_err_bit;
	int m_temp_err;

	double m_ber;
	double m_fer;
};

#endif // !defined(AFX_SOURCESINK_H__361A1355_49CD_4B5C_BD7F_0E4C5D0F773B__INCLUDED_)
