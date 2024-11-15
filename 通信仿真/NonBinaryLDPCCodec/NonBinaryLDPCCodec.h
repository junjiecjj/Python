#pragma once

// 用来表示校验矩阵中不为零的节点
typedef struct Edge
{
	int m_row_no;		// 行标
	int m_col_no;		// 列标

	double m_alpha[2];	// 前向
	double m_beta[2];	// 后向
	double m_v2c[2];	// 变量节点传送给校验节点的信息
	double m_c2v[2];	// 校验节点传送给变量节点的信息

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
	void SPADecoder();		// 和积译码算法
	void FFTDecoder();		// 快速傅里叶变换算法
	void MinSumDecoder();	// Min-sum算法

private:
	void FFT();			// 执行傅里叶变换
	void IFFT();		// 傅里叶反变换

private:
	//parity-check matrix
	int m_num_row;		// 校验矩阵行数，即校验方程个数
	int m_num_col;		// 校验矩阵列数，即变量节点个数
	int **m_decH;		// 
	int **m_encH;		// 生成矩阵

	//code parameters
	int m_codedim;		// 编码前码长
	int m_codelen;		// 编码后码长
	int m_codechk;		// 码的校验位长度 = 编码后码长 - 编码前码长
	double m_coderate;	// 码率
	int m_encoder_active;	// 是否进行编码，0表示不编码，1表示编码

	int m_Q_num;		// 表示是几进制
};
