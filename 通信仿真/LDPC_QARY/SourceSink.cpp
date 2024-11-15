#include "SourceSink.h"
#include "RandNum.h"

extern CLCRandNum rndGen0;

CSourceSink:: CSourceSink() {
	ClrCnt();
}


CSourceSink:: ~CSourceSink() {
}


void CSourceSink:: GetBitStr(int *uu, int len) {
	for (int t = 0; t < len; t++)
		uu[t] = (rndGen0.Uniform() < 0.5 ? 0 : 1);
		// uu[t] = 0;
}


void CSourceSink:: GetSymStr(int *uu, int qary, int len) {
	for (int t = 0; t < len; t++) {
		uu[t] = qary;
		while (uu[t] == qary)
			uu[t] = (int)(qary * rndGen0.Uniform());
			// uu[t] = 0;
	}
}


void CSourceSink:: ClrCnt() {
	m_num_tot_blk = 0;
	m_num_tot_bit = 0;
	m_num_err_blk = 0;
	m_num_err_bit = 0;
}


int CSourceSink:: CntErr(int *uu, int *uu_hat, int len, int accumulator) {
	int m_temp_err = 0;
	for (int t = 0; t < len; t++) {
		if (uu_hat[t] != uu[t])
			m_temp_err++;
	}

	if (accumulator == 1) {
		if (m_temp_err > 0) {
			m_num_err_bit += m_temp_err;
			m_num_err_blk += 1;
		}

		m_num_tot_blk += 1.0;
		m_num_tot_bit += len;
	}

	if (m_temp_err > 0)
		return 1;
	else
		return 0;
}