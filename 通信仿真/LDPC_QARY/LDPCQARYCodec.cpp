#include "afxstd.h"
#include "Utility.h"
#include "RandNum.h"
#include "LDPCQARYCodec.h"

using namespace std;

CFiniteField2 GF;
extern CWHRandNum rndGen2;

int Edge:: m = 3;
int Edge:: q_ary = 8;
double *Edge:: curr = 0;
double *Edge:: ar = 0;
double *Edge:: curr_array = 0;


Edge:: Edge() {
    m_alpha = new double[q_ary];
    m_beta = new double[q_ary];
    m_e2s = new double[q_ary];
    m_s2e = new double[q_ary];
    m_e2h = new double[q_ary];
    m_h2e = new double[q_ary];
    m_s2h = new double[q_ary];
    m_h2s = new double[q_ary];

    pai_element = int(1);
}


Edge:: Edge(int row, int col, const int& pai_e) {
    m_row_no = row;
    m_col_no = col;

    m_alpha = new double[q_ary];
    m_beta = new double[q_ary];
    m_e2s = new double[q_ary];
    m_s2e = new double[q_ary];
    m_e2h = new double[q_ary];
    m_h2e = new double[q_ary];
    m_s2h = new double[q_ary];
    m_h2s = new double[q_ary];

    pai_element = pai_e;
}


Edge:: ~Edge() {
    delete[]m_alpha;
    delete[]m_beta;
    delete[]m_e2s;
    delete[]m_s2e;
    delete[]m_e2h;
    delete[]m_h2e;
    delete[]m_s2h;
    delete[]m_h2s;
}


void Edge:: init(int q_ary, int degree) {
    q_ary = q_ary;
    m = degree;
    curr = new double[q_ary];
    ar = new double[q_ary];
    curr_array = new double[q_ary];
}


void Edge:: set(int row, int col, const int& pai_e) {
    m_row_no = row;
    m_col_no = col;
    pai_element = pai_e;
}


void Edge:: Permutation(int direction) {
	int gf(0);
	int gf_curr(0);

	if (direction == 0)
		gf = pai_element;
	else
		gf = GF.Div(1, pai_element);

	for (int i = 0 ; i < q_ary ; i++) {
		gf_curr = GF.Mult(gf, i);
		if (direction == 0)
			curr[gf_curr] = m_e2s[i];
		else
			curr[gf_curr] = m_s2e[i];
	}
	
	for (int i = 0 ; i < q_ary ; i++) {
		if (direction == 0)
			m_e2s[i] = curr[i];
		else
			m_s2e[i] = curr[i];
	}
}


void Edge:: Transform(int direction) {
	if (direction == 0) {
		HadamardTransform(m_e2s, ar, m);
		for(int i = 0; i < q_ary; i++)
			m_e2s[i] = ar[i];
	} else {
		HadamardTransform(m_s2e, ar, m);
		for (int i = 0; i < q_ary; i++)
			m_s2e[i] = ar[i];
	}
}


void Edge:: Norm(int direction) {
	double curr_sum = 0.0;

	if (direction == 0) {
		for (int i = 0; i < q_ary ; i++)
			curr_sum += m_e2s[i];
	} else {
		for (int i = 0; i < q_ary; i++)
			curr_sum += m_s2e[i];
	}

	if (direction == 0) {
		for (int i = 0; i < q_ary; i++) {
			m_e2s[i] /= curr_sum ;
			if (m_e2s[i] < SMALLPROB)
				m_e2s[i] = SMALLPROB;
			if (m_e2s[i] > 1 - SMALLPROB)
				m_e2s[i] = 1 - SMALLPROB;
		}
	} else {
		for (int i = 0; i < q_ary; i++) {
			m_s2e[i] /= curr_sum;
			if (m_s2e[i] < SMALLPROB)
				m_s2e[i] = SMALLPROB;
			if (m_s2e[i] > 1 - SMALLPROB)
				m_s2e[i] = 1 - SMALLPROB;
		}
	}

	curr_sum = 0.0;
//To make sure that the sum is euqal to 1.0.
	if (direction == 0) {
		for (int i = 0; i < q_ary; i++)
			curr_sum += m_e2s[i];
	} else {
		for (int i = 0; i < q_ary; i++)
			curr_sum += m_s2e[i];
	}

	if (direction == 0) {
		for (int i = 0; i < q_ary; i++)
			m_e2s[i] /= curr_sum ;
	} else {
		for (int i = 0; i < q_ary; i++)
			m_s2e[i] /= curr_sum;
	}

}


void Edge:: HadamardTransform(double *data_in , double *data_out , int stage) {
	int num = 1<<stage, curr_dist = 0, block = 0, block_row_num = 0;
	
    for (int i = 0; i < num; i++)
		curr_array[i] = data_in[i];
    
	for (int i = 0; i < stage; i++) {
		curr_dist = 1<<i;
		block = num>>(i + 1);
		block_row_num = 1<<(i + 1);
		for (int j = 0; j < block; j++) {
			for (int s = 0; s < curr_dist; s++) {
				data_out[j * block_row_num + s] = curr_array[j * block_row_num + s] + curr_array[j * block_row_num + curr_dist + s];
				data_out[j * block_row_num + s + curr_dist] = curr_array[j * block_row_num + s] - curr_array[j * block_row_num + curr_dist + s];
			}
		}

		for (int j = 0; j < num; j++)
			curr_array[j] = data_out[j];
	}
	/*for(int j = 0 ; j < num ; j ++)
	{
		  data_out[j] /= (double)(num>>1);
	}*/
}




CLDPCQARYCodec:: CLDPCQARYCodec() {

}


CLDPCQARYCodec:: ~CLDPCQARYCodec() {

}


CLDPCQARYCodec:: CLDPCQARYCodec(string filename, int max_iterations, string mapping_name) {
    m_max_iter = max_iterations;
    Malloc(filename, mapping_name);
}


void CLDPCQARYCodec:: load_matrixG(string filename) {
    int i, j;

    FILE *fp = fopen(filename.c_str(), "r");
	if (fp == NULL) {
		printf("can not open file: %s\n", filename);
		system("pause");
		exit(1);
	}

    for (i = 0; i < m_codedim; i++) {
        for (j = 0; j < m_codelen; j++)
            fscanf(fp, "%d", &m_matrixG[i][j]);
    }

    return;
}


void CLDPCQARYCodec:: load_matrixH(string filename) {
    int i, j;

    FILE *fp = fopen(filename.c_str(), "r");
	if (fp == NULL) {
		printf("can not open file: %s\n", filename);
		system("pause");
		exit(1);
	}

    for (i = 0; i < m_codedim; i++) {
        for (j = 0; j < m_codelen; j++)
            fscanf(fp, "%d", &m_matrixH[i][j]);
    }

    return;
}


void CLDPCQARYCodec:: partition_matrixH() {
    int i, j, s, t, x, y;
    bool flag;

    for (i = 0; i < m_blk_rownum; i++) {
        for (j = 0; j < m_blk_colnum; j++) {
            flag = false;
            for (s = 0; s < m_blk_dim; s++) {
                for (t = 0; t < m_blk_len; t++) {
                    x = i * m_blk_dim + s;
                    y = j * m_blk_len + t;
                    if (m_matrixH[x][y] == 1) {
                        m_matrixHb[i][j] = 1;
                        flag = true;
                        break;
                    } else {
                        m_matrixHb[i][j] = 0;
                    }
                }
                if (flag == true) break;
            }
        }
    }

#if 0
    /* test */
    FILE *fp;
    fp = fopen("matrixHb.txt", "w+");
    for (i = 0; i < m_blk_rownum; i++) {
        for (j = 0; j < m_blk_colnum; j++)
            fprintf(fp, "%d\t", m_matrixHb[i][j]);
        fprintf(fp, "\n");
    }
    fclose(fp);
#endif

    return;
}


void CLDPCQARYCodec:: Malloc_graph(int len_uu, int len_pp) {
    int row_no, col_no;
    Edge *temp_edge;
    
    m_num_row = len_uu;
    m_num_col = len_pp;

    m_row_head = new Edge[m_num_row];
    m_col_head = new Edge[m_num_col];

    for (row_no = 0; row_no < m_num_row; row_no++) {
		(m_row_head + row_no)->m_row_no = row_no;
		(m_row_head + row_no)->m_col_no = -1;
		(m_row_head + row_no)->left = m_row_head + row_no;
		(m_row_head + row_no)->right = m_row_head + row_no;
		(m_row_head + row_no)->up = m_row_head + row_no;
		(m_row_head + row_no)->down = m_row_head + row_no;
	}

	for (col_no = 0; col_no < m_num_col; col_no++) {
		(m_col_head + col_no)->m_row_no = -1;
		(m_col_head + col_no)->m_col_no = col_no;
		(m_col_head + col_no)->left = m_col_head + col_no;
		(m_col_head + col_no)->right = m_col_head + col_no;
		(m_col_head + col_no)->up = m_col_head + col_no;
		(m_col_head + col_no)->down = m_col_head + col_no;
	}

    for (row_no = 0; row_no < m_num_row; row_no++) {
        for (col_no = 0; col_no < m_num_col; col_no++) {
            if (m_matrixHb[row_no][col_no] == 1) {  // caution!
                temp_edge = new Edge;
                temp_edge->m_row_no = row_no;
                temp_edge->m_col_no = col_no;
                temp_edge->right = (m_row_head + row_no)->right;
                (m_row_head + row_no)->right = temp_edge;
                temp_edge->left = m_row_head + row_no;
                (temp_edge->right)->left = temp_edge;
                temp_edge->down = (m_col_head + col_no)->down;
                (m_col_head + col_no)->down = temp_edge;
                temp_edge->up = m_col_head + col_no;
                (temp_edge->down)->up = temp_edge;
            }
        }
    }

    return;
}


void CLDPCQARYCodec:: Free_graph() {
	Edge *p_edge;

	for (int i = 0; i < m_num_row; i++) {
		while ((m_row_head+i)->right->m_col_no != -1){
			p_edge = (m_row_head+i)->right;
			(m_row_head+i)->right = p_edge->right;
			delete p_edge;
		}
	}

	delete []m_row_head;
	delete []m_col_head;

	return;
}


void CLDPCQARYCodec:: Malloc(string filename, string mapping_name) { 
    char temp_str[80];
    char Gmatrix_file_name[80];
    char Hmatrix_file_name[80];

    //===========start read LDPC file===============//
	FILE *fp = fopen(filename.c_str(), "r");
	if (fp == NULL) {
		printf("can not open file: %s\n", filename);
		system("pause");
		exit(1);
	}

    fscanf(fp, "%s", temp_str);
	fscanf(fp, "%d", &m_codedim);

    fscanf(fp, "%s", temp_str);
	fscanf(fp, "%d", &m_codelen);

    fscanf(fp, "%s", temp_str);
	fscanf(fp, "%s", Gmatrix_file_name);

    fscanf(fp, "%s", temp_str);
	fscanf(fp, "%s", Hmatrix_file_name);

    fscanf(fp, "%s", temp_str);
	fscanf(fp, "%d ", &m_max_iter);

    fscanf(fp, "%s", temp_str);
	fscanf(fp, "%d ", &m_q_ary);

    fscanf(fp, "%s", temp_str);
	fscanf(fp, "%d ", &m_blk_dim);

    fscanf(fp, "%s", temp_str);
	fscanf(fp, "%d ", &m_blk_len);

    m_degree = 1;
    while ((1<<m_degree) < m_q_ary)
        m_degree++;
    
    m_seq_table = new int*[m_q_ary];
    for (int q = 0; q < m_q_ary; q++)
        m_seq_table[q] = new int[m_degree];

    fscanf(fp, "%s", temp_str);
    for (int q = 0; q < m_q_ary; q++)
        for (int m = 0; m < m_degree; m++)
            fscanf(fp, "%d", &m_seq_table[q][m]);
    
    fclose(fp);
    //============end read LDPC file================//

    //============Allocation============//
    m_codechk = m_codelen - m_codedim;
    m_blk_rownum = m_codechk / m_blk_dim;
    m_blk_colnum = m_codelen / m_blk_len;
    m_blk_parity = new int[m_blk_dim];

    temp_fwd = new double[m_q_ary];
    temp_bwd = new double[m_q_ary];
    m_prob_H2S = new double[m_q_ary];
    m_prob_H2E = new double[m_q_ary];
    m_prob_temp = new double[m_q_ary];

    m_uu_hat = new int[m_blk_rownum];
    m_cc_hat = new int[m_blk_colnum];
    m_cc_soft = new double*[m_blk_colnum];
    for (int i = 0; i < m_blk_colnum; i++)
        m_cc_soft[i] = new double[m_q_ary];

    m_matrixG = new int*[m_codedim];
    for (int i = 0; i < m_codedim; i++)
        m_matrixG[i] = new int[m_codelen];

    m_matrixH = new int*[m_codechk];
    for (int i = 0; i < m_codechk; i++)
        m_matrixH[i] = new int[m_codelen];
    
    m_matrixHb = new int*[m_blk_rownum];
    for (int i = 0; i < m_blk_rownum; i++)
        m_matrixHb[i] = new int[m_blk_colnum];
    
    m_matrixB = new int*[m_blk_dim];
    for (int i = 0; i < m_blk_dim; i++)
        m_matrixB[i] = new int[m_blk_len];

    //============Mapping============//
    m_modem.Malloc(mapping_name.c_str());
    if (m_modem.m_num_signal != m_q_ary) {
        printf("\nCardinality of signal set and order of finite field don't match.\n");
		system("pause");
        exit(1);
    }

    //============NormalGraph============//
    load_matrixG(Gmatrix_file_name);
    load_matrixH(Hmatrix_file_name);
    partition_matrixH();
    Malloc_graph(m_blk_rownum, m_blk_colnum);

}


void CLDPCQARYCodec:: Free() {
    Free_graph();
    if (m_seq_table != NULL) {
        for (int q = 0; q < m_degree; q++)
            delete[]m_seq_table[q];
        delete[]m_seq_table;
        m_seq_table = NULL;
    }
    if (m_blk_parity != NULL) {
        delete[]m_blk_parity;
        m_blk_parity = NULL;
    }
    if (temp_fwd != NULL) {
        delete[]temp_fwd;
        temp_fwd = NULL;
    }
    if (temp_bwd != NULL) {
        delete[]temp_bwd;
        temp_bwd = NULL;
    }
    if (m_prob_H2S != NULL) {
        delete[]m_prob_H2S;
        m_prob_H2S = NULL;
    }
    if (m_prob_H2E != NULL) {
        delete[]m_prob_H2E;
        m_prob_H2E = NULL;
    }
    if (m_prob_temp != NULL) {
        delete[]m_prob_temp;
        m_prob_temp = NULL;
    }
    if (m_uu_hat != NULL) {
        delete[]m_uu_hat;
        m_uu_hat = NULL;
    }
    if (m_cc_hat != NULL) {
        delete[]m_cc_hat;
        m_cc_hat = NULL;
    }
    if (m_cc_soft != NULL) {
        for (int i = 0; i < m_blk_rownum+m_blk_colnum; i++)
            delete[]m_cc_soft[i];
        delete[]m_cc_soft;
        m_cc_soft = NULL;
    }
    if (m_matrixG != NULL) {
        for (int i = 0; i < m_codedim; i++)
            delete[]m_matrixG[i];
        delete[]m_matrixG;
        m_matrixG = NULL;
    }
    if (m_matrixH != NULL) {
        for (int i = 0; i < m_codechk; i++)
            delete[]m_matrixH[i];
        delete[]m_matrixH;
        m_matrixH = NULL;
    }
    if (m_matrixHb != NULL) {
        for (int i = 0; i < m_blk_rownum; i++)
            delete[]m_matrixHb[i];
        delete[]m_matrixHb;
        m_matrixHb = NULL;
    }
    if (m_matrixB != NULL) {
        for (int i = 0; i < m_blk_dim; i++)
            delete[]m_matrixB[i];
        delete[]m_matrixB;
        m_matrixB = NULL;
    }
}


void CLDPCQARYCodec:: getParam(int *len_uu, int *len_cc, int *len_xx, int *q_ary, int *degree, int *blk_dim, int *blk_len) {
    (*len_uu) = m_codedim;
	(*len_cc) = m_codelen;
    (*len_xx) = m_codelen;
    (*q_ary) = m_q_ary;
    (*degree) = m_degree;
    (*blk_dim) = m_blk_dim;
    (*blk_len) = m_blk_len;
}


void CLDPCQARYCodec:: encoder(int *uu, int *cc) {
    //codeword = [parity_check_bits information_bits]
    
    for (int i = m_codechk; i < m_codelen; i++)
        cc[i] = uu[i-m_codechk];
    
    for (int i = 0; i < m_codechk; i++) {
        cc[i] = 0;
        for (int j = m_codechk; j < m_codelen; j++)
            cc[i] ^= (cc[j] & m_matrixG[i][j]);
    }
    
    return;
}


void CLDPCQARYCodec:: transfer2qary(double *m_yy_pr0, double **channel_for_spa) {
    int i, m, q;
    double temp_sum;

    for (i = 0; i < m_codelen; i+=m_degree) {
        temp_sum = 0.0;
        for (q = 0; q < m_q_ary; q++) {
            m_prob_temp[q] = 1.0;
            for (m = 0; m < m_degree; m++) {
                if (m_seq_table[q][m] == 0)
                    m_prob_temp[q] *= m_yy_pr0[i+m];
                else
                    m_prob_temp[q] *= (1.0-m_yy_pr0[i+m]);
            }
        }

        for (q = 0; q < m_q_ary; q++)
            temp_sum += m_prob_temp[q];
        for (q = 0; q < m_q_ary; q++)
            channel_for_spa[i/m_degree][q] = (m_prob_temp[q]/temp_sum);
    }

    return;
}


void CLDPCQARYCodec:: transfer2binary(int *cc_q, int *uu_b) {
    int i, j;
    for (i = m_blk_rownum; i < m_blk_colnum; i++)
        for (j = 0; j < m_degree; j++)
            uu_b[(i-m_blk_rownum)*m_degree+j] = m_seq_table[cc_q[i]][j];

    return;
}


void CLDPCQARYCodec:: InitMsgPart() {
	Edge *p_edge;
    double m_init_prob = 1.0 / m_q_ary;

	for (int i = 0; i < m_num_col; i++) {
		p_edge = (m_col_head+i)->down;
		while (p_edge->m_row_no != -1) {
            for (int q = 0; q < m_q_ary; q++) {
                p_edge->m_e2s[q] = m_init_prob;
                p_edge->m_s2e[q] = m_init_prob;
                p_edge->m_e2h[q] = m_init_prob;
                p_edge->m_h2e[q] = m_init_prob;
                p_edge->m_s2h[q] = m_init_prob;
                p_edge->m_h2s[q] = m_init_prob;
            }
			p_edge = p_edge->down;
		}
	}

	return;
}


int CLDPCQARYCodec:: qary_decoder(double **yy, int *cc_hat) {
    double iter = 0.0;
    iter += (double)SoftInSoftOutDecoderPartition(yy, cc_hat);
    return iter;
}


int CLDPCQARYCodec:: SoftInSoftOutDecoderPartition(double **U2E, int *cc_hat) {

    InitMsgPart();
    Edge *p_edge;

    int i, j, q, iter, parity_check;
    double temp0_sum, temp1_sum, temp_prob;

    for (iter = 0; iter < m_max_iter; iter++) {
    
// E node ==========>>> H node
        for (j = 0; j < m_num_col; j++) {
            
            // forward
                p_edge = (m_col_head+j)->down;
                for (q = 0; q < m_q_ary; q++)
                    p_edge->m_alpha[q] = U2E[j][q];
                
                while (p_edge->m_row_no != -1) {
                    process_Enode(p_edge->m_alpha, p_edge->m_h2e, temp_fwd);
                    for (q = 0; q < m_q_ary; q++)
                        p_edge->down->m_alpha[q] = temp_fwd[q];
                    p_edge = p_edge->down;
                }

            // decision
                temp_prob = -1.0;
                for (q = 0; q < m_q_ary; q++) {
                    if (temp_prob < (m_col_head+j)->m_alpha[q]) {
                        temp_prob = (m_col_head+j)->m_alpha[q];
                        cc_hat[j] = q;
                    }
                }

            // backward
                p_edge = (m_col_head+j)->up;
                for (q = 0; q < m_q_ary; q++)
                    p_edge->m_beta[q] = 1.0;
                
                while (p_edge->m_row_no != -1) {
                    process_Enode(p_edge->m_alpha, p_edge->m_beta, temp_bwd);
                    for (q = 0; q < m_q_ary; q++)
                        p_edge->m_e2h[q] = temp_bwd[q];
                    process_Enode(p_edge->m_beta, p_edge->m_h2e, temp_bwd);
                    for (q = 0; q < m_q_ary; q++)
                        p_edge->up->m_beta[q] = temp_bwd[q];
                    p_edge = p_edge->up;
                }
        
        }

    // termination
        m_success = 1;
		for (i = 0; i < m_num_row; i++) {
			parity_check = 0;
			p_edge = (m_row_head+i)->right;
			while (p_edge->m_col_no != -1) {
				parity_check ^= cc_hat[p_edge->m_col_no];
				p_edge = p_edge->right;
			}
			if (parity_check != 0) {
				m_success = 0;
				break;
			}
		}
		if (m_success == 1) break;


// H node ==========>>> S node
        for (j = 0; j < m_num_col; j++) {
            p_edge = (m_col_head+j)->down;
            while (p_edge->m_row_no != -1) {
                process_Hnode(p_edge->m_row_no, j, p_edge->m_e2h, m_prob_H2S, 0);
                for (q = 0; q < m_q_ary; q++)
                    p_edge->m_h2s[q] = m_prob_H2S[q];
                p_edge = p_edge->down;
            }
        }

// S node ==========>>> H node
        for (i = 0; i < m_num_row; i++) {
            
        // forward
            p_edge = (m_row_head+i)->right;
            p_edge->m_alpha[0] = 1.0;
            for (q = 1; q < m_q_ary; q++)
                p_edge->m_alpha[q] = 0.0;

            while (p_edge->m_col_no != -1) {
                process_Snode(p_edge->m_alpha, p_edge->m_h2s, temp_fwd);
                for (q = 0; q < m_q_ary; q++)
                    p_edge->right->m_alpha[q] = temp_fwd[q];
                p_edge = p_edge->right;
            }

        // backward
            p_edge = (m_row_head+i)->left;
            p_edge->m_beta[0] = 1.0;
            for (q = 1; q < m_q_ary; q++)
                p_edge->m_beta[q] = 0.0;
            
            while (p_edge->m_col_no != -1) {
                process_Snode(p_edge->m_alpha, p_edge->m_beta, temp_bwd);
                for (q = 0; q < m_q_ary; q++)
                    p_edge->m_s2h[q] = temp_bwd[q];
                process_Snode(p_edge->m_beta, p_edge->m_h2s, temp_bwd);
                for (q = 0; q < m_q_ary; q++)
                    p_edge->left->m_beta[q] = temp_bwd[q];
                p_edge = p_edge->left;
            }

        }

// H node ==========>>> E node
        for (i = 0; i < m_num_row; i++) {
            p_edge = (m_row_head+i)->right;
            while (p_edge->m_col_no != -1) {
                process_Hnode(i, p_edge->m_col_no, p_edge->m_s2h, m_prob_H2E, 1);
                for (q = 0; q < m_q_ary; q++)
                    p_edge->m_h2e[q] = m_prob_H2E[q];
                p_edge = p_edge->right;
            }
        }

    }

    return iter + (iter<m_max_iter);
}


void CLDPCQARYCodec:: process_Enode(double *prob_in1, double *prob_in2, double *prob_out) {
    int q;

    for (q = 0; q < m_q_ary; q++) 
        prob_out[q] = 0.0;

    for (q = 0; q < m_q_ary; q++)
        prob_out[q] = prob_in1[q] * prob_in2[q];
    
    double temp_sum = 0.0;
    for (q = 0; q < m_q_ary; q++)
        temp_sum += prob_out[q];
    for (q = 0; q < m_q_ary; q++)
        prob_out[q] /= temp_sum;

    for (q = 0; q < m_q_ary; q++) {
        if (prob_out[q] < SMALLPROB)
            prob_out[q] = SMALLPROB;
        if (prob_out[q] > 1 - SMALLPROB)
            prob_out[q] = 1 - SMALLPROB;
    }

    return;
}


void CLDPCQARYCodec:: process_Snode(double *prob_in1, double *prob_in2, double *prob_out) {
    int i, j, q;

    for (q = 0; q < m_q_ary; q++) 
        prob_out[q] = 0.0;

    for (i = 0; i < m_q_ary; i++)
        for (j = 0; j < m_q_ary; j++)
            prob_out[i^j] += prob_in1[i] * prob_in2[j];

    double temp_sum = 0.0;
    for (q = 0; q < m_q_ary; q++)
        temp_sum += prob_out[q];
    for (q = 0; q < m_q_ary; q++)
        prob_out[q] /= temp_sum;

    for (q = 0; q < m_q_ary; q++) {
        if (prob_out[q] < SMALLPROB)
            prob_out[q] = SMALLPROB;
        if (prob_out[q] > 1 - SMALLPROB)
            prob_out[q] = 1 - SMALLPROB;
    }

    return;
}


void CLDPCQARYCodec:: process_Hnode(int Pb_row_no, int Pb_col_no, double *prob_in, double *prob_out, int direction) {
    int i, j, q, m, temp;

    for (i = 0; i < m_q_ary; i++) 
        prob_out[i] = 0.0;
    
    for (i = 0; i < m_blk_dim; i++)
        for (j = 0; j < m_blk_len; j++)
            m_matrixB[i][j] = m_matrixH[Pb_row_no*m_blk_dim+i][Pb_col_no*m_blk_len+j];

    for (q = 0; q < m_q_ary; q++) {
        for (i = 0; i < m_blk_dim; i++) {
            temp = 0;
            for (m = 0; m < m_degree; m++)
                temp += (m_matrixB[i][m] * m_seq_table[q][m]);
            m_blk_parity[i] = temp % 2;
        }
        temp = 0;
        for (j = m_blk_len-1; j >= 0; j--)
            temp += (m_blk_parity[j] << (m_blk_len-j-1));

        if (direction == 0)  // E ---> H ---> S
            prob_out[temp] += prob_in[q];
        else                 // E <--- H <--- S
            prob_out[q] = prob_in[temp];

    }

    double temp_sum = 0.0;
    for (q = 0; q < m_q_ary; q++)
        temp_sum += prob_out[q];
    for (q = 0; q < m_q_ary; q++)
        prob_out[q] /= temp_sum;

    for (q = 0; q < m_q_ary; q++) {
        if (prob_out[q] < SMALLPROB)
            prob_out[q] = SMALLPROB;
        if (prob_out[q] > 1 - SMALLPROB)
            prob_out[q] = 1 - SMALLPROB;
    }

    return;
}