#pragma once

#ifndef MODEM_H
#define MODEM_H

class Modem
{
public:
	Modem();
	~Modem();

	void BPSK_Setup();//��ʼ����������ȡ�ļ�����
	void BPSK_Free();//�ͷſռ�
	void BPSK_Modulation(int *uu, double *xx, int len_uu);//����
	void BPSK_Hard_DeModulation(double *yy, int *uu_hat, int len_yy);//�����Ӳ�о�
	void BPSK_Soft_DeModulation(double *yy, double var, double *p0_cc, int len_yy);//��������о�

private:
	int m_num_signal;//��������
	int m_dim_signal;//����ά��
	double *m_constellation;//����
};

#endif // !MODEM_H



















