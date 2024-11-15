#include "afxstd.h"
#include "Utility.h"

void ProbClip(double *xx, int len_xx) {
	for (int i = 0; i < len_xx; i++) {
		if (xx[i] < SMALLPROB)
			xx[i] = SMALLPROB;
		else if (xx[i] > 1.0 - SMALLPROB)
			xx[i] = 1.0 - SMALLPROB;
	}
}


int BitDotProd(int a, int b, int len)
{
	int temp = a & b;
	int prod = 0;
	for (int i = 0; i < len; i++)
		prod += (temp >> i) % 2;
	return (prod % 2);
}