#include "mex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#pragma warning(disable:4996)

int			**T;
int			H_ColNum;

void		Init(char * fname);
void		ReadMatrix();
void		MakeGeneratorMatrix();
void		MakeFactorGraph();
void		End();

/*************************
 output terms
*************************/
/* G & H parameters */
int			InfoLen;                  //number of row of PG
int			SystematicParityCheckNum; //number of column of PG
int			CodeLen;                  //number of column of H
int			ParityCheckNum;           //number of row of H
char		**PG=NULL;                //parity part the generator matrix, InfoLen X SystematicParityCheckNum;
/* factor graph parameter */
int			*RowFirstElementIndex;    //(ParityCheckNum+1) X 1, RowFirstElementIndex[i] = number of non-zero elements in row 0 to i-1;
int			*ColFirstElementIndex;    //(CodeLen+1) X 1, ColFirstElementIndex[i] = number of non-zero elements in col 0 to i-1;
int			*Chk2VarInterleaver;      //NonzeroNumber X 1,
int			*NonzeroCol;              //NonzeroNumber X 1, storing the column index for each nonzero elements
/* decoder parameters */
int			LDPC_Itnum;               //decoder iteration number
char		IfReset[8];               //
char		IfAIter[8];               //
char		IfDispl[8];               //
int			It_Gap;                   //
int			NonzeroNumber;            //number of nonzero elements in the parity-check matrix
char		HFileName[1024];          //the file name for the parity-check matrix
char		IfFastEncode[1024];       //'Yes' for not performing Gaussian ELIMINATION, 'No' for performing Gaussian ELIMINATION.
//double	**pr;                     //(NUser*NonzeroNumber) X 1, storing messages between check & variable nodes
//double    **lr;                     //(NUser*NonzeroNumber) X 1, storing messages between check & variable nodes

void Init(char * fname)
{
	int i;
	char tempChar[256];

	FILE *file = fopen(fname, "r");
	if( !file )
	{
		mexErrMsgIdAndTxt("LDPCInit:Init",
                "Error -- Cannot read the input file --> %s\n", fname);
		system("pause");
		exit (-1);
	}
	else
	{
		fscanf(file, "%s", tempChar);		fscanf(file, "%d", &ParityCheckNum);
		fscanf(file, "%s", tempChar);		fscanf(file, "%d", &CodeLen);
		fscanf(file, "%s", tempChar);		fscanf(file, "%d", &H_ColNum);
		fscanf(file, "%s", tempChar);		fscanf(file, "%d", &LDPC_Itnum);
		fscanf(file, "%s", tempChar);		fscanf(file, "%s", &IfReset);
		fscanf(file, "%s", tempChar);		fscanf(file, "%s", &IfAIter);
		fscanf(file, "%s", tempChar);		fscanf(file, "%s", &IfDispl);
		fscanf(file, "%s", tempChar);		fscanf(file, "%d", &It_Gap);
		fscanf(file, "%s", tempChar);		fscanf(file, "%s", &HFileName);
		fscanf(file, "%s", tempChar);		fscanf(file, "%s", &IfFastEncode);
		fclose( file );
	}

	T = (int**)malloc(ParityCheckNum * sizeof(*T));
	for (i = 0; i < ParityCheckNum; i++)
    {
        T[i] = (int*)malloc(H_ColNum * sizeof(*(T[i])));
    }

	ReadMatrix();
	if(IfFastEncode[0] == 'N' && IfFastEncode[1] == 'o')
	{
		MakeGeneratorMatrix();
	}
	else
	{
		InfoLen = CodeLen - ParityCheckNum;
		SystematicParityCheckNum = ParityCheckNum;
	}
	MakeFactorGraph();//Factor graph may be changed in MakeGeneratorMatrix.
}

void ReadMatrix()
{ 
	int i,j;

	FILE *LDPCReadChkMtrx = fopen(HFileName, "r");
	if( !LDPCReadChkMtrx )
	{
		mexErrMsgIdAndTxt("LDPCInit:ReadMatrix",
                "Can not read parity check matrix!");
		system("pause");
	}

	// read check matrix from file
	if( ParityCheckNum ) 
	{   
		for( i = 0;i < ParityCheckNum; i++ )
		{  
			for( j = 0;j < H_ColNum; j++ )
			{  
				fscanf(LDPCReadChkMtrx, "%d", T[i]+j);
				T[i][j]--; 
			}         
		}      
	}
	else 
		T = NULL;
	fclose(LDPCReadChkMtrx);
}

void MakeFactorGraph()
{
	int i,j,k;
	int row,col;
	int sum;
	int *row_weight = (int *)malloc(ParityCheckNum * sizeof(*row_weight));
	int *col_weight = (int *)malloc(CodeLen        * sizeof(*col_weight));
	
	//count row weight, col weight and non-zero element numbers
	for(i = 0; i < ParityCheckNum; i++)
	{
		row_weight[i] = 0;
	}
	for(i = 0; i < CodeLen; i++)
	{
		col_weight[i] = 0;
	}	
	NonzeroNumber = 0;
	for(i = 0; i < ParityCheckNum; i++ )
	{
		for(j = 0;j < H_ColNum; j++ )
	    {
			if( T[i][j] >= 0 )
			{
				row  =i;
			    col = T[i][j];
				
				row_weight[row]++;
				col_weight[col]++;
				NonzeroNumber++;
			}
		}
	}
	
	RowFirstElementIndex = (int *)malloc((ParityCheckNum+1) * sizeof(*RowFirstElementIndex));
	for(sum = i = 0; i < ParityCheckNum; i++ )
	{
		RowFirstElementIndex[i] = sum; //number of non-zero elements in row 0 to i-1;
		sum += row_weight[i];
	}
	RowFirstElementIndex[ParityCheckNum] = sum;
	
	ColFirstElementIndex = (int *)malloc((CodeLen+1) * sizeof(*ColFirstElementIndex));
	for(sum = i = 0; i < CodeLen; i++ )
	{
		ColFirstElementIndex[i] = sum; //number of non-zero elements in col 0 to i-1;
		sum += col_weight[i];
	}
	ColFirstElementIndex[CodeLen] = sum;
	
	for(i = 0; i < CodeLen; i++)
	{
		col_weight[i] = 0;
	}
	k = 0;
	Chk2VarInterleaver = (int *)malloc(NonzeroNumber * sizeof(*Chk2VarInterleaver));
	NonzeroCol         = (int *)malloc(NonzeroNumber * sizeof(*NonzeroCol));
	for(i = 0; i < ParityCheckNum; i++ )
	{
		for(j = 0;j < H_ColNum; j++ )
	    {
			if( T[i][j] >= 0 )
			{
				row  =i;
			    col = T[i][j];
				
				Chk2VarInterleaver[ColFirstElementIndex[col] + col_weight[col]] = k;
				NonzeroCol[k] = col;
				col_weight[col]++;
				k++;
			}
		}
	}
	
	FILE *fo = fopen("H_permute.txt", "w");
	for (i = 0; i < ParityCheckNum; i++)
	{
		for (j = 0; j < H_ColNum; j++)
		{
			fprintf(fo, "%d ", T[i][j]);
		}
		fprintf(fo, "\n");
	}
	fclose(fo);

	free(row_weight);
	free(col_weight);
}

void End()
{
	int i;

	for (i = 0; i < ParityCheckNum; i++)
	{
		free(T[i]);
	}
	free(T);
	
	if(PG != NULL)
	{
		for (i = 0; i < InfoLen; i++)
		{
			free(PG[i]);
		}
		free(PG);
	}	
	free(RowFirstElementIndex);
	free(ColFirstElementIndex);
	free(Chk2VarInterleaver);
	free(NonzeroCol);
}

typedef unsigned long int UNCHAR;
#define BIT_PER_UNIT    (sizeof(UNCHAR) * 8)
#define GetBytePos(col) ((col)/(BIT_PER_UNIT))
#define GetBitMask(col) (1 << ((BIT_PER_UNIT-1)-(col)%(BIT_PER_UNIT)))

void MakeGeneratorMatrix()
{
	int i, j, col, row;
	UNCHAR **tempH, **ColumnPermuteH;
	UNCHAR *tmp_ptr;
	int *ColIndex;
	int bytePos;
	UNCHAR bitMask;
	int tmp;
	int nonzero_row;
	int nonzero_col;
	int BitPackCodeLen = CodeLen / BIT_PER_UNIT + (int)((CodeLen%BIT_PER_UNIT)>0);

	printf("\nMakeGeneratorMatrix: READING parity-check matrix!");
	ColIndex = (int *)malloc(CodeLen * sizeof(*ColIndex));
	for (i = 0; i < CodeLen; i++)
	{
		ColIndex[i] = i;
	}
	tempH = (UNCHAR**)malloc(ParityCheckNum * sizeof(*tempH));
	ColumnPermuteH = (UNCHAR**)malloc(ParityCheckNum * sizeof(*ColumnPermuteH));
	for (i = 0; i < ParityCheckNum; i++)
	{
		tempH[i] = (UNCHAR*)malloc(BitPackCodeLen * sizeof(*(tempH[i])));
		ColumnPermuteH[i] = (UNCHAR*)malloc(BitPackCodeLen * sizeof(*(ColumnPermuteH[i])));
		//init to zeros
		for (j = 0; j < BitPackCodeLen; j++)
		{
			tempH[i][j] = 0;
			ColumnPermuteH[i][j] = 0;
		}
		for (j = 0; j < H_ColNum; j++)
		{
			if (T[i][j] >= 0)
			{
				col = T[i][j];
				bytePos = GetBytePos(col);
				bitMask = GetBitMask(col);
				tempH[i][bytePos] |= bitMask;
				ColumnPermuteH[i][bytePos] |= bitMask;
			}
		}
	}
	//output("tempH.txt", tempH, ParityCheckNum, CodeLen);
	//output("ColumnPermuteH.txt", ColumnPermuteH, ParityCheckNum, CodeLen);

	printf("\nMakeGeneratorMatrix: GAUSSIAN ELIMINATION of parity-check matrix!");
	//systematic H
	for (i = 0; i < ParityCheckNum; i++)
	{
		//find the non-zero element in column [i, CodeLen-1] & row [i, ParityCheckNum-1].
		nonzero_row = -1;
		nonzero_col = -1;
		for (j = i; (j < CodeLen) && (nonzero_col == -1); j++)
		{
			bytePos = GetBytePos(ColIndex[j]);
			bitMask = GetBitMask(ColIndex[j]);
			for (row = i; row < ParityCheckNum; row++)
			{
				if ((tempH[row][bytePos] & bitMask) != 0)
				{
					nonzero_row = row;
					nonzero_col = ColIndex[j];
					break;
				}
			}
		}

		//a nonzero row & col is found
		// sway row 'i' with row 'nonzero_row' for tempH
		// & sway column 'i' with column 'nonzero_col' for tempH & ColumnPermuteH
		if (nonzero_row != -1 && nonzero_col != -1)
		{
			//we only need to swap the row pointer
			if (nonzero_row != i)
			{
				tmp_ptr = tempH[i];
				tempH[i] = tempH[nonzero_row];
				tempH[nonzero_row] = tmp_ptr;
			}

			//we do not actually swap the column elments,
			//we only swap their indexes
			if (nonzero_col != i)
			{
				if (i % 1000 == 0)
				{
					printf("\nGaussian elimination: swap row %d with %d and column %d with %d!", i, nonzero_row, i, nonzero_col);
				}
				tmp = ColIndex[i];
				ColIndex[i] = ColIndex[nonzero_col];
				ColIndex[nonzero_col] = tmp;
			}
		}
		else if (((nonzero_row == -1) && (nonzero_col != -1)) || ((nonzero_row != -1) && (nonzero_col = -1)))
		{
			printf("\n\nGaussian elimination: (nonzero_row = %d && nonzero_col = %d) is not possible!\n", nonzero_row, nonzero_col);
			printf("There must be something wrong in the Gaussian elimination process!\n\n");
			system("pause");
			exit(1);
		}
		else
		{
			// all remaining rows are zeros, the elimination process is done
			break;
		}

		//adding the new row i to all other row in range [0, ParityCheckNum-1]
		//that has a nonzero element in i-th position in that row.
		bytePos = GetBytePos(ColIndex[i]);//the columns may be swapped
		bitMask = GetBitMask(ColIndex[i]);
		for (row = 0; row < ParityCheckNum; row++)
		{
			if (((tempH[row][bytePos] & bitMask) != 0) && (row != i))
			{
				for (j = bytePos; j < BitPackCodeLen; j++)// elements in columns [0, i-1] have been all zeros for all rows with index larger than i-1.
				{
					tempH[row][j] ^= tempH[i][j];
				}
			}
		}
		if (i % 1000 == 1)
		{
			printf("\nGaussian elimination: Finishing %d rows!", i + 1);
		}
	}//end for systematic H

	printf("\n\nMakeGeneratorMatrix: SETTING COLUMN PERMUTED parity-check matrix!");
	//set T using column permute H
	for (i = 0; i < ParityCheckNum; i++)
	{
		//set intial to -1
		for (j = 0; j < H_ColNum; j++)
		{
			T[i][j] = -1;
		}

		//set column index for T
		tmp = 0;//count the number of nonzero elements in a col
		for (j = 0; j < CodeLen; j++)
		{
			bytePos = GetBytePos(ColIndex[j]);
			bitMask = GetBitMask(ColIndex[j]);
			if ((ColumnPermuteH[i][bytePos] & bitMask) != 0)
			{
				T[i][tmp] = j;
				tmp++;
			}
		}
		if (tmp > H_ColNum)
		{
			printf("\n\nSetting T: nonzero_cnt(=%d) > H_ColNum(=%d)\n", tmp, H_ColNum);
			printf("This is not possible!\nThere must be something wrong in the Gaussian elimination process!\n\n");
			system("pause");
			exit(1);
		}
	}

	printf("\n\nMakeGeneratorMatrix: FINDING RANK of parity-check matrix!");
	//count non-zero rows in the systematic parity-check matrix H
	SystematicParityCheckNum = 0;
	for (i = 0; i < ParityCheckNum; i++)
	{
		for (j = 0; j < BitPackCodeLen; j++)
		{
			if (tempH[i][j] != 0)
			{
				SystematicParityCheckNum++;
				break;
			}
		}
	}

	printf("\n\nMakeGeneratorMatrix: SETTING GENERATOR matrix!\n");
	//set the true infomation length
	InfoLen = CodeLen - SystematicParityCheckNum;

	//set generator matrix
	PG = (char**)malloc(InfoLen * sizeof(*PG));
	for (i = 0; i < InfoLen; i++)
	{
		PG[i] = (char*)malloc(SystematicParityCheckNum * sizeof(*(PG[i])));
		for (j = 0; j < SystematicParityCheckNum; j++)
		{
			bytePos = GetBytePos(ColIndex[i + SystematicParityCheckNum]);
			bitMask = GetBitMask(ColIndex[i + SystematicParityCheckNum]);
			PG[i][j] = (char)((tempH[j][bytePos] & bitMask) != 0);
		}
	}
	//output("tempH_sys.txt", tempH, ParityCheckNum, CodeLen);
	//output("ColumnPermuteH_sys.txt", ColumnPermuteH, ParityCheckNum, CodeLen);
	//output("PG.txt", PG, InfoLen, 32);

	//free
	for (i = 0; i < ParityCheckNum; i++)
	{
		free(tempH[i]);
		free(ColumnPermuteH[i]);
	}
	free(tempH);
	free(ColumnPermuteH);

	free(ColIndex);
}

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{    
    int NUser;
	char filename[256]; //set-up file name
	
    const char *field_names[] = {"NUser",
								 "InfoLen",
								 "SystematicParityCheckNum",
								 "PG",
								 "CodeLen",
								 "ParityCheckNum",
								 "LDPC_Itnum",
								 "IfReset",
								 "IfAIter",
								 "IfDispl",
								 "It_Gap",
								 "RowFirstElementIndex",
								 "ColFirstElementIndex",
								 "NonzeroNumber",
								 "Chk2VarInterleaver",
								 "NonzeroCol",
								 "pr",
								 "lr",
								 "HFileName",
								 "IfFastEncode"};
	int NUMBER_OF_FIELDS = (sizeof(field_names)/sizeof(*field_names));
	mwSize dims[] = {1, 1};
	
	mxArray *field_value;
	INT32_T *data;
	int i, j, k;
			
	/*****************************************************************
	 Check for proper number of input and  output arguments
	*****************************************************************/
    if (nrhs !=2 || nlhs > 1) {
        mexErrMsgIdAndTxt("LDPCInit:ErrorInputArg",
                "Usage: %s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s",
                "LDPCstruct = LDPCInit(NUser, LDPC_param);\n",
                "Input:\n",
                " NUser:     an integer > 0, specifying number of users in the system.\n",
                " LDPC_param: a string, specifying the file name containing the LDPC parity-check matrix. in the file 'LDPC_param', There is a setting specifying the file name of parity-check matrix.\n",
                "Output:\n",
                " LDPCstruct: a struct containing the following fields. Except the field 'NUser', all other fields are specified by the content in the file 'LDPC_param'. You should not change all these fields.\n",
                "             'NUser':                    an integer > 0, equal to the first input argument.\n",
                "             'InfoLen':                  an integer > 0, number of information bits of the LDPC code.\n",
				"             'SystematicParityCheckNum': an integer > 0, number of parity-check of the LDPC code.\n",
                "             'PG':                       an InfoLen x SystematicParityCheckNum matrix of type mxINT32_CLASS that only contains 0 and 1, the parity part of the systematic generator matrix of the LDPC code.\n",
                "             'CodeLen':                  an integer > 0, codeword length in bits of the LDPC code.\n",
                "             'ParityCheckNum':           an integer > 0, number of rows of the LDPC parity-check matrix.\n",
                "             'LDPC_Itnum':               an integer > 0, number of iteration number for the LDPC decoder.\n",
                "             'IfReset':                  a string. If = 'Yes', reset the internal message parameters 'pr' and 'lr' every time calling the function 'LDPCDecode'. If ~= 'Yes', no message reset is performed.\n",
                "             'IfAIter':                  a string. If = 'Yes', iteration will stop whenever the parity-check is passed during the decoding iteration. If ~= 'Yes', decoding iteration will alway perform 'LDPC_ItNum' times.\n",
                "             'IfDispl':                  a string. If = 'Yes', output the number of error parity-check equations every 'It_Gap' iteration. If ~= 'Yes', no any output.\n",
                "             'It_Gap':                   an integer > 0. a interval for output message.\n",
                "             'RowFirstElementIndex':     a (ParityCheckNum+1) x 1 matrix of type mxINT32_CLASS, internal parameter for storing the parity-check matrix. Do not change it.\n",
                "             'ColFirstElementIndex':     a (CodeLen+1) x 1        matrix of type mxINT32_CLASS, internal parameter for storing the parity-check matrix. Do not change it.\n",
                "             'NonzeroNumber':            an integer > 0, internal parameter for storing the parity-check matrix. Do not change it.\n",
                "             'Chk2VarInterleaver':       a NonzeroNumber x 1     matrix of type mxINT32_CLASS,, internal parameter for storing the parity-check matrix. Do not change it.\n",
                "             'NonzeroCol':               a NonzeroNumber x 1     matrix of type mxINT32_CLASS,, internal parameter for storing the parity-check matrix. Do not change it.\n",
                "             'pr':                       a NonzeroNumber x NUser matrix of type mxDOUBLE_CLASS,, internal parameter for decoding. Do not change it.\n",
                "             'lr':                       a NonzeroNumber x NUser matrix of type mxDOUBLE_CLASS,, internal parameter for decoding. Do not change it.\n",
				"             'HFileName':                a string. Containing the file name of the parity-check matrix.\n",
				"             'IfFastEncode':             a string. 'Yes' for not performing Gaussian ELIMINATION, 'No' for performing Gaussian ELIMINATION.\n");
    }
	
    NUser = ((int)mxGetScalar(prhs[0]));
	if(mxGetNumberOfDimensions(prhs[0]) != 2 ||
       mxGetM(prhs[0]) != 1 || mxGetN(prhs[0]) != 1 || NUser <= 0){
		mexErrMsgIdAndTxt("LDPCInit:ErrorInputArg",
                "The first input argument 'NUser' should be an integer > 0!");
	}
	
	if(mxGetNumberOfDimensions(prhs[1]) != 2 ||
	  (mxGetM(prhs[1]) > 1 && mxGetN(prhs[1]) > 1) ||
	  (mxGetClassID(prhs[1]) != mxCHAR_CLASS)){
		mexErrMsgIdAndTxt("LDPCInit:ErrorInputArg",
                "The second input argument 'LDPC_param' should be a string!");
	}
	
	/*****************************************************************
	 Read set-up file and construct factor graph & generator matrix
	*****************************************************************/        
	mxGetString(prhs[1], filename, 256);
	Init(filename);
    
	/*****************************************************************
	 Create output struct
	*****************************************************************/
    plhs[0] = mxCreateStructArray(2, dims, NUMBER_OF_FIELDS, field_names);

	/*****************************************************************
	 Set fields of the struct
	*****************************************************************/
	/* set field 'NUser'*/
    field_value = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    *((INT32_T*)mxGetData(field_value)) = (INT32_T)NUser;
	mxSetField(plhs[0],0,"NUser",field_value);
	
	/* set field 'InfoLen'*/
    field_value = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    *((INT32_T*)mxGetData(field_value)) = (INT32_T)InfoLen;
	mxSetField(plhs[0],0,"InfoLen",field_value);
	
	/* set field 'SystematicParityCheckNum'*/
    field_value = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    *((INT32_T*)mxGetData(field_value)) = (INT32_T)SystematicParityCheckNum;
	mxSetField(plhs[0],0,"SystematicParityCheckNum",field_value);
	
	/* set field 'PG'*/
	if(PG != NULL)
	{
	k = sizeof(INT32_T)*8;
	field_value = mxCreateNumericMatrix(InfoLen/k+1, SystematicParityCheckNum,
            mxINT32_CLASS, mxREAL);
	data = (INT32_T*)mxGetData(field_value);
	for(i = 0; i < SystematicParityCheckNum; i++)
	{
		for(j = 0; j < (InfoLen/k+1); j++)
		{
			data[j] = 0;
		}		
		for(j = 0; j < InfoLen; j++)
		{
			if((INT32_T)(PG[j][i])==1)
			{
				data[j/k] |= (1<<(j%k));
			}
		}
		data = data + (InfoLen/k+1);
	}
	}
	else
	{
		field_value = NULL;
	}
	mxSetField(plhs[0],0,"PG",field_value);
		
	/* set field 'CodeLen'*/
    field_value = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    *((INT32_T*)mxGetData(field_value)) = (INT32_T)CodeLen;
	mxSetField(plhs[0],0,"CodeLen",field_value);
	
	/* set field 'ParityCheckNum'*/
    field_value = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    *((INT32_T*)mxGetData(field_value)) = (INT32_T)ParityCheckNum;
	mxSetField(plhs[0],0,"ParityCheckNum",field_value);
	
	/* set field 'LDPC_Itnum'*/
    field_value = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    *((INT32_T*)mxGetData(field_value)) = (INT32_T)LDPC_Itnum;
	mxSetField(plhs[0],0,"LDPC_Itnum",field_value);
	
	/* set field 'IfReset'*/
	mxSetField(plhs[0],0,"IfReset",mxCreateString(IfReset));
		
	/* set field 'IfAIter'*/
	mxSetField(plhs[0],0,"IfAIter",mxCreateString(IfAIter));
	
	/* set field 'IfDispl'*/
	mxSetField(plhs[0],0,"IfDispl",mxCreateString(IfDispl));
	
	/* set field 'It_Gap'*/
	field_value = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
	*((INT32_T*)mxGetData(field_value)) = (INT32_T)It_Gap;
	mxSetField(plhs[0],0,"It_Gap",field_value);
	
	/* set field 'NonzeroNumber'*/
	field_value = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
	*((INT32_T*)mxGetData(field_value)) = (INT32_T)NonzeroNumber;
	mxSetField(plhs[0], 0, "NonzeroNumber", field_value);
	
	/* set field 'RowFirstElementIndex'*/
    field_value = mxCreateNumericMatrix(ParityCheckNum+1, 1,
            mxINT32_CLASS, mxREAL);
    data = (INT32_T*)mxGetData(field_value);
    for ( i = 0; i <= ParityCheckNum; i++ )
	{
        data[i] = (INT32_T)(RowFirstElementIndex[i]);
    }
	mxSetField(plhs[0],0,"RowFirstElementIndex",field_value);
	
	/* set field 'ColFirstElementIndex'*/
    field_value = mxCreateNumericMatrix(CodeLen+1, 1,
            mxINT32_CLASS, mxREAL);
    data = (INT32_T*)mxGetData(field_value);
    for ( i = 0; i <= CodeLen; i++ )
	{
        data[i] = (INT32_T)(ColFirstElementIndex[i]);
    }
	mxSetField(plhs[0],0,"ColFirstElementIndex",field_value);
	
	/* set field 'Chk2VarInterleaver'*/
    field_value = mxCreateNumericMatrix(NonzeroNumber, 1,
            mxINT32_CLASS, mxREAL);
    data = (INT32_T*)mxGetData(field_value);
    for ( i = 0; i < NonzeroNumber; i++)
	{
        data[i] = (INT32_T)(Chk2VarInterleaver[i]);
    }
	mxSetField(plhs[0],0,"Chk2VarInterleaver",field_value);
	
	/* set field 'NonzeroCol'*/
	field_value = mxCreateNumericMatrix(NonzeroNumber, 1,
            mxINT32_CLASS, mxREAL);
	data = (INT32_T*)mxGetData(field_value);
	for (i = 0; i < NonzeroNumber; i++)
	{
		data[i] = (INT32_T)(NonzeroCol[i]);
	}
	mxSetField(plhs[0], 0, "NonzeroCol", field_value);

	/* set field 'pr'*/
	mxSetField(plhs[0],0,"pr", mxCreateDoubleMatrix(NonzeroNumber, NUser, mxREAL));
	
	/* set field 'lr'*/
	mxSetField(plhs[0],0,"lr", mxCreateDoubleMatrix(NonzeroNumber, NUser, mxREAL));
	
	/* set field 'HFileName'*/
	mxSetField(plhs[0],0,"HFileName",mxCreateString(HFileName));
	
	/* set field 'IfFastEncode'*/
	mxSetField(plhs[0],0,"IfFastEncode",mxCreateString(IfFastEncode));
	
	/*****************************************************************
	 Free temporal memory
	******************************************************************/
	End();
}
