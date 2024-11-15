#pragma once

// ������ʾУ������в�Ϊ��Ľڵ�
typedef struct Edge
{
	int m_row_no;		// �б�
	int m_col_no;		// �б�

	double m_alpha[2];	// ǰ��
	double m_beta[2];	// ����
	double m_v2c[2];	// �����ڵ㴫�͸�У��ڵ����Ϣ
	double m_c2v[2];	// У��ڵ㴫�͸������ڵ����Ϣ

	struct Edge *left;
	struct Edge *right;
	struct Edge *up;
	struct Edge *down;
} Edge;


class CNonBinaryLDPCCodec
{
public:
	CNonBinaryLDPCCodec(void);
	~CNonBinaryLDPCCodec(void);
public:
	void Malloc(char *filename);
	void Free();
	void Encoder();
	void Decoder();
	void SPADecoder();		// �ͻ������㷨
	void FFTDecoder();		// ���ٸ���Ҷ�任�㷨
	void MinSumDecoder();	// Min-sum�㷨

private:
	void FFT();			// ִ�и���Ҷ�任
	void IFFT();		// ����Ҷ���任

private:
	//parity-check matrix
	int m_num_row;		// У�������������У�鷽�̸���
	int m_num_col;		// У������������������ڵ����
	int **m_decH;		// 
	int **m_encH;		// ���ɾ���

	//code parameters
	int m_codedim;		// ����ǰ�볤
	int m_codelen;		// ������볤
	int m_codechk;		// ���У��λ���� = ������볤 - ����ǰ�볤
	double m_coderate;	// ����
	int m_encoder_active;	// �Ƿ���б��룬0��ʾ�����룬1��ʾ����

	int m_Q_num;		// ��ʾ�Ǽ�����
};
