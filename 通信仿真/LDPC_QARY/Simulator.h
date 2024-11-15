#pragma once

#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "afxstd.h"
#include "SourceSink.h"
#include "Modem.h"
#include "Mapper.h"
#include "Channel.h"
#include "LDPCQARYCodec.h"


class Simulator
{

public:
    Simulator();
    ~Simulator();

    void Simulate();

private:
    double m_min_snr;
    double m_max_snr;
    double m_inc_snr;

    int m_max_blk_err;
    int m_max_blk_num;

    int m_q_ary;
    int m_degree;
    
    int m_len_uu;
    int m_len_cc;
    int m_len_xx;
    int m_blk_dim;
    int m_blk_len;
    int m_len_cc_qary;

    int *m_uu;
    int *m_uu_hat;
    int *m_pp;
    int *m_pp_hat;
    int *m_cc;
    int *m_cc_hat;
    int *m_sym;
    int *m_sym_hat;

    double *m_xx;
    double *m_yy;
    double *m_yy_pr0;
    double **channel_for_spa;

    CSourceSink *m_sourcesink;
    Channel *m_channel;
    CLDPCQARYCodec *m_codec;
    // CModem *m_modem;/*for q_ary*/
    Modem *m_modem;

    void StartSimulator();
    void EndSimulator();

};

#endif // !SIMULATOR_H