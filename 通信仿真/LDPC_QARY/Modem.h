#pragma once

#ifndef MODEM_H
#define MODEM_H

class Modem
{
public:
	Modem();
	~Modem();

	void BPSK_Setup();//初始化参数，读取文件数据
	void BPSK_Free();//释放空间
	void BPSK_Modulation(int *uu, double *xx, int len_uu);//调制
	void BPSK_Hard_DeModulation(double *yy, int *uu_hat, int len_yy);//解调，硬判决
	void BPSK_Soft_DeModulation(double *yy, double var, double *p0_cc, int len_yy);//解调，软判决

private:
	int m_num_signal;//星座点数
	int m_dim_signal;//星座维数
	double *m_constellation;//星座
};

#endif // !MODEM_H



















