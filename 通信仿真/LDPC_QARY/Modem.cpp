#include "afxstd.h"
#include "Modem.h"
#include "Utility.h"

Modem::Modem() {
	m_num_signal = 0;
	m_dim_signal = 0;
	m_constellation = NULL;
}

Modem::~Modem() {
	if (m_constellation != NULL) {
		delete[]m_constellation;
		m_constellation = NULL;
	}
}

void Modem::BPSK_Setup() {
	m_num_signal = 2;
	m_dim_signal = 1;
	m_constellation = new double[m_num_signal];

	m_constellation[0] = +1;
	m_constellation[1] = -1;
}

void Modem::BPSK_Free() {
	if (m_constellation != NULL) {
		delete[]m_constellation;
		m_constellation = NULL;
	}
}

void Modem::BPSK_Modulation(int *uu, double *xx, int len_uu) {
	for (int i = 0; i < len_uu; ++i) {
		xx[i] = m_constellation[uu[i]];
	}
}

void Modem::BPSK_Hard_DeModulation(double *yy, int *uu_hat, int len_yy) {
	for (int i = 0; i < len_yy; ++i) {
		if (yy[i] > 0)
			uu_hat[i] = 0;
		else
			uu_hat[i] = 1;
	}
}

void Modem::BPSK_Soft_DeModulation(double *yy, double var, double *p0_cc, int len_yy) {
	for (int i = 0; i < len_yy; ++i) {  //prob(x|y)
		if (yy[i] > 0)
			p0_cc[i] = 1.0 / (1.0 + exp(-2.0 * yy[i] / var));
		else
			p0_cc[i] = 1.0 - 1.0 / (1.0 + exp(2.0 * yy[i] / var));
	}
	ProbClip(p0_cc, len_yy);
}