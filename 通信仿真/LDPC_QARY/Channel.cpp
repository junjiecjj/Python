#include "Channel.h"
#include "Utility.h"
#include "RandNum.h"

extern CWHRandNum rndGen1;

Channel::Channel() {
	
}

Channel::~Channel() {

}

void Channel::AWGN_Initial(double var) {
	m_var = var;
	m_sigma = sqrt(var);
}

void Channel::AWGN(double *xx, double *yy, int len) {
	rndGen1.Normal(yy, len);
	for (int i = 0; i < len; i++)
		yy[i] = xx[i] + m_sigma * yy[i];
}