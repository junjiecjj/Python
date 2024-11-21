#include "afxstd.h"
#include "Mapper.h"
#include "Utility.h"


CModem:: CModem() {}
CModem:: ~CModem() {}



void CModem:: Malloc(const char *filename) {
    char temp_str[80];
    char mark[80];

    //===========start read Modulation_chart file===============//
    // sprintf(filename, "Modulation_chart.txt");
	FILE *fp = fopen(filename, "r");
	if (fp == NULL) {
		printf("can not open file: %s\n", filename);
		system("pause");
		exit(1);
	}

    sprintf(mark, "***MappingChart***");
    while (strcmp(temp_str, mark) != 0)
        fscanf(fp, "%s", temp_str);

    fscanf(fp, "%s", temp_str);
    fscanf(fp, "%d", &m_num_signal); // 8
    fscanf(fp, "%s", temp_str);
    fscanf(fp, "%d", &m_len_signal); // 3
    fscanf(fp, "%s", temp_str);

    m_signal_set = new double*[m_num_signal];  // 8 x 3
    for (int i = 0; i < m_num_signal; i++)
        m_signal_set[i] = new double[m_len_signal];

    m_Es = 0.0;
    for (int i = 0; i < m_num_signal; i++) {
        for (int j = 0; j < m_len_signal; j++) {
            fscanf(fp, "%lf", &m_signal_set[i][j]);
            m_Es += m_signal_set[i][j] * m_signal_set[i][j];
        }
    }
    fclose(fp);
    m_Es /= m_num_signal;

    m_sym_prob = new double[m_num_signal];

    return;
}


void CModem:: Free() {
    delete[]m_signal_set;
    delete[]m_sym_prob;
    return;
}


void CModem:: Mapping(int *cc, double *xx, int len_symseq) {
    for (int i = 0; i < len_symseq; i++) {
        for (int j = 0; j < m_len_signal; j++)
            xx[i*m_len_signal+j] = m_signal_set[cc[i]][j];
    }
}


void CModem:: Demapping(double *yy, double *sym_prob, double sigma, int len_signalseq) {
    int i, j, q;
    double sum, sqr_sum;

    for (i = 0; i < len_signalseq; i++) {  // compute prob(x=0|y);
        for (q = 0; q < m_num_signal; q++) {
            sqr_sum = 0.0;
            for (j = 0; j < m_len_signal; j++)
                sqr_sum += (yy[i*m_len_signal+j]-m_signal_set[q][j]) * (yy[i*m_len_signal+j]-m_signal_set[q][j]);
            m_sym_prob[q] = exp(-0.5 * sqr_sum / sigma);
        }
        ProbClip(m_sym_prob, m_num_signal);
        sum = 0.0;
        for (q = 0; q < m_num_signal; q++)
            sum += m_sym_prob[q];
        for (q = 0; q < m_num_signal; q++)
            sym_prob[i*m_num_signal+q] = m_sym_prob[q] / sum;
    }

    return;
}


void CModem:: Demapping(double *yy, double **sym_prob, double sigma, int len_signalseq) {
    int i, j, q;
	double sum, sqr_sum;

	for (i = 0; i < len_signalseq; i++) { // compute prob(x=0|y);
		for (q = 0; q < m_num_signal; q++) {
			sqr_sum = 0.0;
			for (j = 0; j < m_len_signal; j++)
				sqr_sum += (yy[i*m_len_signal+j]-m_signal_set[q][j]) * (yy[i*m_len_signal+j]-m_signal_set[q][j]);
			m_sym_prob[q] = sqr_sum / (sigma*sigma);
		}
		//find the minimum distance
		sqr_sum = m_sym_prob[0];
		for (q = 1; q < m_num_signal; q++) {
			if (m_sym_prob[q] < sqr_sum)
				sqr_sum = m_sym_prob[q];
		}
		//norm the probability
		for (q = 0; q < m_num_signal; q++) {
			m_sym_prob[q] -= sqr_sum;
			if(m_sym_prob[q] > 40)
				m_sym_prob[q] = 40;
		}
		//compute the probability and norm the probability
		sqr_sum = 0.0;
		for (q = 0; q < m_num_signal; q++) {
			m_sym_prob[q] = exp(-0.5 * m_sym_prob[q]);
			sqr_sum += m_sym_prob[q];
		}
		for (q = 0; q < m_num_signal; q++) {
			sym_prob[i][q] = m_sym_prob[q] / sqr_sum;
		}
		ProbClip(m_sym_prob, m_num_signal);
		sum = 0.0;
		for (q = 0; q < m_num_signal; q++)
			sum +=  m_sym_prob[q];
		for (q = 0; q < m_num_signal; q++)
			sym_prob[i][q] = m_sym_prob[q] / sum;
	}

	return;
}