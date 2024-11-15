#pragma once

#ifndef LDPCQARYCODEC_H
#define LDPCQARYCODEC_H

#include "afxstd.h"
#include "RandNum.h"
#include "Mapper.h"


class Edge
{
public:
    int m_row_no;
	int m_col_no;

	double *m_alpha;
	double *m_beta;
	double *m_e2s;
	double *m_s2e;
    double *m_e2h;
    double *m_h2e;
    double *m_s2h;
    double *m_h2s;

    int pai_element;

    static int m;
    static int q_ary;
    static double *curr;
    static double *ar;
    static double *curr_array;

    struct Edge *up;
	struct Edge *down;
	struct Edge *left;
	struct Edge *right;

    Edge();
    Edge(int row, int col, const int& pai_e);
    ~Edge();

    void set(int row, int col, const int& pai_e);

    void Permutation(int direction);  // 0 represent permutation from variable node to check node 
	                                  // 1 represent permutation from check node to variable node

	void Transform(int direction);  // 0 represent forward transform
	                                // 1 represent inverse transform

	void Norm(int direction);  // 0 represent normilize the vector v2c
	                           // 1 represent normilize the vector c2v
    
    void HadamardTransform(double *data_in, double *data_out, int stage);

    static void init(int q_ary, int degree);
	
};


class CLDPCQARYCodec
{
public:
    CLDPCQARYCodec();
    CLDPCQARYCodec(string filename, int max_iterations, string mapping_name);
    ~CLDPCQARYCodec();

    int m_q_ary;
    int m_degree;
    int sym_satisfied_num;
    CModem m_modem;

    void Malloc(string filename, string mapping_name);
    void Free();
    void getParam(int *len_uu, int *len_cc, int *len_xx, int *q_ary, int *degree, int *blk_dim, int *blk_len);
    void encoder(int *uu, int *sym);
    void transfer2qary(double *yy, double **channel_for_spa);
    int qary_decoder(double **channel_for_spa, int *uu_hat);
    void transfer2binary(int *uu_q, int *uu_b);
    
private:
    // code parameters
    int m_codedim;
	int m_codelen;
    int m_codechk;

    // generator matrix
    int m_num_row;
    int m_num_col;

    // block parameters
    int m_blk_dim;
    int m_blk_len;
    int m_blk_rownum;
    int m_blk_colnum;
    int **m_seq_table;
    int *m_blk_parity;

    // SPA parameters
    int m_max_iter;    
    double *temp_fwd;
    double *temp_bwd;
    double *m_prob_H2S;
    double *m_prob_H2E;
    double *m_prob_temp;

    int *m_uu_hat;
    int *m_cc_hat;
    double **m_cc_soft;
    bool m_success;

    // graph
    Edge *m_row_head;
    Edge *m_col_head;
    Edge *m_row_link;
    Edge *m_col_link;

    // matrix
    int **m_matrixG;
    int **m_matrixH;
    int **m_matrixHb;
    int **m_matrixB;

    void load_matrixG(string filename);
    void load_matrixH(string filename);
    void partition_matrixH();
    void Malloc_graph(int len_uu, int len_pp);
    void Free_graph();

    void InitMsgPart();
    void process_Enode(double *prob_in1, double *prob_in2, double *prob_out);/*process "=" node*/
    void process_Snode(double *prob_in1, double *prob_in2, double *prob_out);/*process "+" node*/
    void process_Hnode(int Pb_row_no, int Pb_col_no, double *prob_in, double *prob_out, int direction);/*process "H" node, direction=0: -->--, direction=1: --<--*/
    int SoftInSoftOutDecoderPartition(double **U2E, int *cc_hat);/*for partitioned matrix H*/

};


#endif // !LDPCQARYCODEC_H