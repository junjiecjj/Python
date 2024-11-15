//////////////////////////////////////////////////////////////////////////////////
// Designed by chunhua shi (springing18@163.com).Sun Yat-sen University.
// This program can only be employed for academic research.
///////////////////////////////////////////////////////////////////////////////////

#ifndef _UTILITY_H
#define _UTILITY_H
//
#define PI 3.14159265358979
#define e  2.71828
#define SMALLPROB 1.0e-20
#define DBL_MAX 1.0e+300
#define MAX_NUM 65536


void ProbClip(double *xx, int len_xx);

void Dec2Bin(int d, int *b, int len_b);
void SeqDec2Bin(int *bin, int *dec, int len_dec, int len_symbol);
int Bin2Dec(int *bin, int len_bin);
void SeqBin2Dec(int *bin, int *dec, int len_dec, int len_symbol);

int CompareMetric(double *A, double *B,int len);
int BitDotProd(int a, int b, int len);
int Systemizer(int num_row, int num_col, int **H, int **sysH, int *pai);

int min(int x, int y);
int max(int x, int y);

void BubbleSort(double *value, int *index, int len);

//
#endif