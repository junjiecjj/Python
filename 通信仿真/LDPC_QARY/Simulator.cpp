#include "Simulator.h"
#include "RandNum.h"

using namespace std;


#define SHOW_PROCESS

CLCRandNum rndGen0;
CWHRandNum rndGen1;
CWHRandNum rndGen2;


Simulator:: Simulator() {
    m_min_snr = 0;
	m_max_snr = 0;
	m_inc_snr = 0;

	m_max_blk_err = 0;
	m_max_blk_num = 0;

	m_q_ary = 0;
	m_degree = 0;

	m_len_uu = 0;
	m_len_cc = 0;
    m_len_xx = 0;
	m_blk_dim = 0;
	m_blk_len = 0;
	m_len_cc_qary = 0;

	m_uu = NULL;
    m_uu_hat = NULL;
	m_pp = NULL;
	m_pp_hat = NULL;
	m_cc = NULL;
	m_cc_hat = NULL;
	m_sym = NULL;
	m_sym_hat = NULL;
	m_xx = NULL;
	m_yy = NULL;
	m_yy_pr0 = NULL;
	channel_for_spa = NULL;

	m_sourcesink = NULL;
	m_channel = NULL;
	m_codec = NULL;
	m_modem = NULL;
}


Simulator:: ~Simulator(){
    if (m_uu != NULL) {
		delete[]m_uu;
		m_uu = NULL;
	}
	if (m_uu_hat != NULL) {
		delete[]m_uu_hat;
		m_uu_hat = NULL;
	}
	if (m_pp != NULL) {
		delete[]m_pp;
		m_pp = NULL;
	}
	if (m_pp_hat != NULL) {
		delete[]m_pp_hat;
		m_pp_hat = NULL;
	}
	if (m_cc != NULL) {
		delete[]m_cc;
		m_cc = NULL;
	}
	if (m_cc_hat != NULL) {
		delete[]m_cc_hat;
		m_cc_hat = NULL;
	}
	if (m_sym != NULL) {
		delete[]m_sym;
		m_sym = NULL;
	}
	if (m_sym_hat != NULL) {
		delete[]m_sym_hat;
		m_sym_hat = NULL;
	}
	if (m_xx != NULL) {
		delete[]m_xx;
		m_xx = NULL;
	}
	if (m_yy != NULL) {
		delete[]m_yy;
		m_yy = NULL;
	}
	if (m_yy_pr0 != NULL) {
		delete[]m_yy_pr0;
		m_yy_pr0 = NULL;
	}
	if (channel_for_spa != NULL) {
		for (int i = 0; i < m_len_cc_qary; i++)
			delete[]channel_for_spa[i];
		delete[]channel_for_spa;
		channel_for_spa = NULL;
	}
    if (m_sourcesink != NULL) {
		delete m_sourcesink;
		m_sourcesink = NULL;
	}
	if (m_channel != NULL) {
		delete m_channel;
		m_channel = NULL;
	}
	if (m_codec != NULL) {
		delete m_codec;
		m_codec = NULL;
	}
	if (m_modem != NULL) {
		delete m_modem;
		m_modem = NULL;
	}
}

void Simulator:: StartSimulator() {
    int setup_no;
    char filename[80];
    char temp_str[80];
    char code_file_name[80];
	char mapping_file_name[80];
    FILE *fp;
    setup_no = 0;

    sprintf(filename, "Setup_of_LDPCBlockCode_AWGN%d.txt", setup_no);

    //===========start read setup file===============//
    if ((fp = fopen(filename, "r")) == NULL) {
        printf("\nCan't open the %s file!\n", filename);
    }

    fscanf(fp, "%s", temp_str);
	fscanf(fp, "%lf", &m_min_snr);

	fscanf(fp, "%s", temp_str);
	fscanf(fp, "%lf", &m_max_snr);

	fscanf(fp, "%s", temp_str);
	fscanf(fp, "%lf", &m_inc_snr);

	fscanf(fp, "%s", temp_str);
	fscanf(fp, "%d", &m_max_blk_err);

	fscanf(fp, "%s", temp_str);
	fscanf(fp, "%d", &m_max_blk_num);

	fscanf(fp, "%s", temp_str);
	fscanf(fp, "%s", code_file_name);

	fscanf(fp, "%s", temp_str);
	fscanf(fp, "%s", mapping_file_name);

	fclose(fp);
    //============end read setup file================//

    //=================start malloc==================//
	m_modem = new Modem; //
	m_modem->BPSK_Setup(); //
	m_channel = new Channel;
	m_sourcesink = new CSourceSink();
	m_codec = new CLDPCQARYCodec;
    m_codec->Malloc(code_file_name, mapping_file_name);
    m_codec->getParam(&m_len_uu, &m_len_cc, &m_len_xx, &m_q_ary, &m_degree, &m_blk_dim, &m_blk_len);
	m_uu = new int[m_len_uu];
	m_uu_hat = new int[m_len_uu];
	m_cc = new int[m_len_cc];
	m_cc_hat = new int[m_len_cc];
	m_sym = new int[m_len_cc/m_degree];
	m_sym_hat = new int[m_len_cc/m_degree];
	m_pp = new int[(m_len_cc-m_len_uu)/m_degree];
	m_pp_hat = new int[(m_len_cc-m_len_uu)/m_degree];
	m_xx = new double[m_len_xx];
	m_yy = new double[m_len_xx];
	m_yy_pr0 = new double[m_len_xx];
	m_len_cc_qary = (m_len_uu/m_blk_dim) + (m_len_cc-m_len_uu)/m_blk_len;
	channel_for_spa = new double*[m_len_cc_qary];
	for (int i = 0; i < m_len_cc_qary; i++)
		channel_for_spa[i] = new double[m_q_ary];
    //==================end malloc===================//

    //==============print information================//
	fp = fopen("snrferber.txt", "a+");
	if (fp == NULL) {
		printf("open file error\n");
		system("pause");
		exit(1);
	}
	fprintf(fp, "\n\n%%%%%%============================================================================\n");
	fclose(fp);
	//============end print information==============//

    return;
}

void Simulator:: EndSimulator() {
	m_codec->Free();
    if (m_uu != NULL) {
		delete[]m_uu;
		m_uu = NULL;
	}
	if (m_uu_hat != NULL) {
		delete[]m_uu_hat;
		m_uu_hat = NULL;
	}
	if (m_pp != NULL) {
		delete[]m_pp;
		m_pp = NULL;
	}
	if (m_pp_hat != NULL) {
		delete[]m_pp_hat;
		m_pp_hat = NULL;
	}
	if (m_cc != NULL) {
		delete[]m_cc;
		m_cc = NULL;
	}
	if (m_cc_hat != NULL) {
		delete[]m_cc_hat;
		m_cc_hat = NULL;
	}
	if (m_sym != NULL) {
		delete[]m_sym;
		m_sym = NULL;
	}
	if (m_sym_hat != NULL) {
		delete[]m_sym_hat;
		m_sym_hat = NULL;
	}
	if (m_xx != NULL) {
		delete[]m_xx;
		m_xx = NULL;
	}
	if (m_yy != NULL) {
		delete[]m_yy;
		m_yy = NULL;
	}
	if (m_yy_pr0 != NULL) {
		delete[]m_yy_pr0;
		m_yy_pr0 = NULL;
	}
	if (channel_for_spa != NULL) {
		for (int i = 0; i < m_len_cc_qary; i++)
			delete[]channel_for_spa[i];
		delete[]channel_for_spa;
		channel_for_spa = NULL;
	}

    if (m_sourcesink != NULL) {
		delete m_sourcesink;
		m_sourcesink = NULL;
	}
	if (m_channel != NULL) {
		delete m_channel;
		m_channel = NULL;
	}
	if (m_codec != NULL) {
		delete m_codec;
		m_codec = NULL;
	}
	if (m_modem != NULL) {
		delete m_modem;
		m_modem = NULL;
	}
}

void Simulator:: Simulate() {
    FILE *fp;
    StartSimulator();
    for (double snr = m_min_snr; snr < m_max_snr + 1.0*m_inc_snr; snr += m_inc_snr) {
        fp = fopen("snrferber.txt", "a+");
		if (fp == NULL) {
			printf("open file error\n");
			system("pause");
			exit(1);
		}
        double var = pow(10, -0.1 * snr);
		m_sourcesink->ClrCnt();
        m_channel->AWGN_Initial(var);

        int err = 0;
		double iter = 0.0;
		while (err < m_max_blk_err && m_sourcesink->TolBlk() < m_max_blk_num) {
			m_sourcesink->GetBitStr(m_uu, m_len_uu);
			m_codec->encoder(m_uu, m_cc);
			m_modem->BPSK_Modulation(m_cc, m_xx, m_len_cc);
			m_channel->AWGN(m_xx, m_yy, m_len_xx);
			m_modem->BPSK_Soft_DeModulation(m_yy, var, m_yy_pr0, m_len_cc);

			m_codec->transfer2qary(m_yy_pr0, channel_for_spa);
			iter += (double)m_codec->qary_decoder(channel_for_spa, m_sym_hat);
            m_codec->transfer2binary(m_sym_hat, m_uu_hat);

			m_sourcesink->CntErr(m_uu, m_uu_hat, m_len_uu, 1);
			err = m_sourcesink->ErrBlk();

#ifdef SHOW_PROCESS
            if ((int)(m_sourcesink->TolBlk()) % 1000 == 0) {
                printf("temp = (%le  %le  %le)    %%%%ebn = %f    err = (%d/%d  %d/%d)\n",
                m_sourcesink->BER(),
                m_sourcesink->FER(),
                iter / m_sourcesink->TolBlk(),
                snr,
                (int)m_sourcesink->ErrBit(),
				(int)m_sourcesink->TolBit(),
                (int)m_sourcesink->ErrBlk(),
				(int)m_sourcesink->TolBlk());
            }
#endif
        }
        fprintf(fp, "%f  %le  %le  %le   %%%%err = (%d/%d  %d/%d)\n", snr, m_sourcesink->BER(), m_sourcesink->FER(), iter / m_sourcesink->TolBlk(),
			(int)m_sourcesink->ErrBit(), (int)m_sourcesink->TolBit(), (int)m_sourcesink->ErrBlk(), (int)m_sourcesink->TolBlk());
		fclose(fp);
    }
    EndSimulator();
}


int main() {
    rndGen0.SetSeed(-1);
	rndGen1.SetSeed(-1);
	rndGen2.SetSeed(-1);

	Simulator *theSim = new Simulator();
	theSim->Simulate();
	delete theSim;
	system("pause");

	return 0;
}