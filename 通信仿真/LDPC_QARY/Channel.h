#pragma once
#ifndef CHANNEL_H
#define CHANNEL_H

#include "afxstd.h"

class Channel {
public:
	Channel();
	~Channel();
	void AWGN_Initial(double var);
	void AWGN(double *xx, double *yy, int len);
private:
	double m_var;
	double m_sigma;
};

#endif // !CHANNEL_H
