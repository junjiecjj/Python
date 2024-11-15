//////////////////////////////////////////////////////////////////////////////////
// Designed by chunhua shi (springing18@163.com).Sun Yat-sen University.
// This program can only be employed for academic research.
///////////////////////////////////////////////////////////////////////////////////

//
#include "stdafx.h"
#include "Utility.h"
//

void ProbClip(double *xx, int len_xx)
{
	int i;

	for (i = 0; i < len_xx; i++){
		if (xx[i] < SMALLPROB)
			xx[i] = SMALLPROB;
		else if (xx[i] > 1.0 - SMALLPROB)
			xx[i] = 1.0 - SMALLPROB;
	}

	return;
}
///////////////////////////////////////////////////////////
//binary (b[0] b[1] b[2] ......), where b[0] is the MSB, b[end-1] is the LSB
//in other words, b is the binary expression of dec number d;
//            d     --> b = (b[0] b[1] b[2])  len_b = 3   
//for example d = 0 --> b = (0 0 0)
//            d = 1 --> b = (0 0 1)
//            d = 2 --> b = (0 1 0)   
//            d = 3 --> b = (0 1 1)
//            d = 4 --> b = (1 0 0)   
//            d = 5 --> b = (1 0 1)
//            d = 6 --> b = (1 1 0)   
//            d = 7 --> b = (1 1 1)
//            d = 8 --> b = (0 0 0)   

//十进制转化为二进制表示
void Dec2Bin(int d, int *b, int len_b)
{
	int i;

	for (i = 0; i < len_b; i++)
		b[len_b-i-1] = (d >> i) % 2;

	return;
}

// 二进制转化为十进制表示
int Bin2Dec(int *bin, int len_bin)
{
	int i = 0;

	int dec  = 0;
	for(i = 0; i < len_bin; i++)
	{
		dec  = dec  * 2 + bin[i];
	}

	return dec;
}

void SeqDec2Bin(int *bin, int *dec, int len_dec, int len_symbol)
{
	int i, j, t;

	t = 0;
	for (i = 0; i < len_dec; i++){
		for (j = 0; j < len_symbol; j++){
			bin[t] = (dec[i] >> (len_symbol-1-j)) & 1; 
			t++;
		}
	}

	return;
}

//二进制转化十进制
void SeqBin2Dec(int *bin, int *dec, int len_dec, int len_symbol)
{
	int i, j, t;

	t = 0;
	for (i = 0; i < len_dec; i++){
		dec[i] = 0;
		for (j = 0; j < len_symbol; j++){
			dec[i] = (dec[i] << 1) ^ bin[t];
			t++;
		}
	}

	return;
}

//比较两个数组是否相同
int CompareMetric(double *A, double *B,int len)
{
	int i;
	int flag;

	flag = 1;
	for(i = 0; i < len ;i++){
		if(A[i] != B[i]){
			flag = 0;
			break;
		}
	}
	return flag;
}


int BitDotProd(int a, int b, int len)
{
	int i; 
	int temp;
	int prod;

	temp = a & b;

	prod = 0;
	for (i = 0; i < len; i++)
		prod += (temp >> i) % 2;
	
	return (prod % 2);
}

int Systemizer(int num_row, int num_col, int **H, int **sysH, int *pai)
{
	int i, j, ii, jj, m, n;
	int flag;
	int temp;
	int rank;

	for (j = 0; j < num_col; j++)
		pai[j] = j;//pai[0] means that the first column (0th column) of sysH comes from the pai[0]-th of H

//copy H to sysH;
	for (i = 0; i < num_row; i++){
		for (j = 0; j < num_col; j++)
			sysH[i][j] = H[i][j];
	}

//initialize the rank
	rank = 0;
//find the most left_up non_zero element
	for (i = 0; i < num_row; i++){
		flag = 0;
		for (jj = i; jj < num_col; jj++){
			for (ii = i; ii < num_row; ii++){
				if (sysH[ii][jj] != 0){
					flag = 1;
					break;
				}
			}
			if (flag == 1){
				rank++;
				break;
			}
		}

		if (flag == 0)
			break;
		else{
			//swap to ensure the (1,1) elment is non_zero
			//swap i and ii row
			if (ii != i){
				for (n = 0; n < num_col; n++){
					temp = sysH[i][n];
					sysH[i][n] = sysH[ii][n];
					sysH[ii][n] = temp;
				}
			}
//swap i and jj col
			if (jj != i){
				temp = pai[i];
				pai[i] = pai[jj];
				pai[jj] = temp;

				for (m = 0; m < num_row; m++){
					temp = sysH[m][i];
					sysH[m][i] = sysH[m][jj];
					sysH[m][jj] = temp;
				}
			}
//elimination
			for (m = 0; m < num_row; m++){
				if (m != i && sysH[m][i] == 1){
					for (n = 0; n < num_col; n++)
						sysH[m][n] ^= sysH[i][n];
				}
			}
		}
	}

	return rank;
}

/////////////////////////////////////////////////////////////
int min(int x, int y)
{
	return x < y? x : y;
}

int max(int x, int y)
{
	return x > y? x : y;
}

////////////////////////////////////////////////////////////////
void BubbleSort(double *value, int *index, int len)
{
	int i, j;
	int cnt;
	int itemp;
	double vtemp;

	for (i = 0; i < len; i++)
		index[i] = i;

	for (i = 0; i < len-1; i++){
		cnt = 0;
		for (j = len-1; j > i; j--){
			if (value[j] > value[j-1]){
				vtemp = value[j-1];
				value[j-1] = value[j];
				value[j] = vtemp;
				
				itemp = index[j-1];
				index[j-1] = index[j];
				index[j] = itemp;

				cnt++;
			}
		}
		if (cnt == 0)
			break;
	}


	return;
}