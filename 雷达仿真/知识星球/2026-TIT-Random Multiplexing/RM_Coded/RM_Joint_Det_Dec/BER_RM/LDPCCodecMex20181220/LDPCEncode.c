#include "mex.h"

//const char *field_names[] = { "NUser",
//"InfoLen",
//"SystematicParityCheckNum",
//"PG",
//"CodeLen",
//"ParityCheckNum",
//"LDPC_Itnum",
//"IfReset",
//"IfAIter",
//"IfDispl",
//"It_Gap",
//"RowFirstElementIndex",
//"ColFirstElementIndex",
//"NonzeroNumber",
//"Chk2VarInterleaver",
//"NonzeroCol",
//"pr",
//"lr" };

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
	INT32_T InfoLen;
	INT32_T SystematicParityCheckNum;
	INT32_T *PGorigin, *PG;
	INT32_T PGrow;
	INT32_T	BitPerPack;
	INT32_T	Log2BitPerPack;
	INT32_T PackedInfoLen;
	INT32_T CodeLen;

	INT32_T *info, *codeword;
	size_t  dim, row, col;
	INT32_T i, j, nuser, n, b;

    /*****************************************************************
	 Check input/output argument number and input type
	*****************************************************************/
	if (nrhs != 2 || nlhs > 1 || !mxIsStruct(prhs[0]) ||
		mxGetClassID(prhs[1]) != mxINT32_CLASS)
	{
		mexErrMsgIdAndTxt("LDPCEncode:ErrorInputArg",
                "Usage: %s%s%s%s%s%s",
                "codeword = LDPCEncode(LDPCstruct, data);\n",
                "Input:\n"
                " LDPCstruct: a struct set using the function 'LDPCInit'.\n",
                " data:       an InfoLen x N matrix of type mxINT32_CLASS, 'InfoLen' is specified in LDPCstruct, 'N' can be any integer > 0",
                "Output:\n",
                " codeword:   an CodeLen x N matrix of type mxINT32_CLASS, 'CodeLen' is specified in LDPCstruct, 'N' is equal to the column of 'data'");
	}

	/*****************************************************************
	 Prepare input argument & check validity
	*****************************************************************/
	InfoLen                  = *((INT32_T*)mxGetData(mxGetFieldByNumber(prhs[0], 0, 1)));
	SystematicParityCheckNum = *((INT32_T*)mxGetData(mxGetFieldByNumber(prhs[0], 0, 2)));
    PGorigin                 = (INT32_T*)mxGetData(mxGetFieldByNumber(prhs[0], 0, 3));
	PGrow                    = (INT32_T)mxGetM(mxGetFieldByNumber(prhs[0], 0, 3));
	BitPerPack               = sizeof(INT32_T)*8;
	Log2BitPerPack           = 0;
	while ((1<<Log2BitPerPack) < BitPerPack)
	{
		Log2BitPerPack++;
	}
    CodeLen                  = *((INT32_T*)mxGetData(mxGetFieldByNumber(prhs[0], 0, 4)));
	if ((1<<Log2BitPerPack) != BitPerPack)
	{
		mexErrMsgIdAndTxt("LDPCEncode:ErrorInputArg",
                "(1<<Log2BitPerPack) != BitPerPack");
	}
    
    info = (INT32_T*)mxGetData(prhs[1]);
	dim  = mxGetNumberOfDimensions(prhs[1]);
	row  = mxGetM(prhs[1]);
    col  = mxGetN(prhs[1]);
	if (dim > 2 || dim <=0 || row != InfoLen)
	{
		mexErrMsgIdAndTxt("LDPCEncode:ErrorInputArg",
                "The second input argument should be a vector or a matrix with row size equal to 'InfoLen'");
	}
    
	/*****************************************************************
	 Prepare output argument
	*****************************************************************/
    if(nlhs == 0 || plhs[0] == NULL ||
       mxGetM(plhs[0]) != CodeLen || mxGetN(plhs[0]) != col ||
       mxGetClassID(plhs[0]) != mxINT32_CLASS || mxIsComplex(plhs[0]))
    {
        if(plhs[0] != NULL)
        {
            mxFree(plhs[0]);
        }
        plhs[0] = mxCreateNumericMatrix(CodeLen, col, mxINT32_CLASS, mxREAL);
    }
    codeword = (INT32_T*)mxGetData(plhs[0]);
    
    /*****************************************************************
     Enocde each column of 'data'
    *****************************************************************/
    for (nuser = 0; nuser < col; nuser++)
    {
		PG = PGorigin;
        //encode, parity bits are the first 'SystematicParityCheckNum' bits of the codeword
        for (j = 0; j < SystematicParityCheckNum; j++)
        {			
            codeword[j] = 0;
			for(i = 0, n = 0; (n < PGrow) && (i < InfoLen); n++)
			{
				for (b = 0; (b < BitPerPack) && (i < InfoLen); b++, i++)
				{
					codeword[j] ^= info[i] & (PG[n] >> b);//PG is stored column by column;
				}
			}
			PG = PG + PGrow;
        }

        //systematic bits are the last 'InfoLen' bits of the codeword	
        for (j = InfoLen - 1, n = CodeLen - 1; j >= 0; j--, n--)
        {
            codeword[n] = info[j];
        }

        info     = info     + InfoLen;
        codeword = codeword + CodeLen;
    }
}
