#pragma once

#ifndef MAPPER_H
#define MAPPER_H

class CModem
{
public:
    CModem();
    ~CModem();

    int m_num_signal;
    int m_len_signal;
    double **m_signal_set;
    double *m_sym_prob;
    double m_Es;

    void Malloc(const char* filename);
    void Free();
    void Mapping(int *cc, double *xx, int len_symseq);
    void Demapping(double *yy, double *sym_prob, double sigma, int len_symseq);
    void Demapping(double *yy, double **sym_prob, double sigma, int len_symseq);

};

#endif // !MAPPER_H