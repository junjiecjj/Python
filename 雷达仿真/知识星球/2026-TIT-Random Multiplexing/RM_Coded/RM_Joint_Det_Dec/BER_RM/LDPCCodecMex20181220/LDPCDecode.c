#include "mex.h"
#include <string.h>
#include <math.h>

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

inline double mini(double x, double y)
{
	if (x > y) return y;
	else return x;
}

inline double Basic_Calculate(double x,double y)
{	
	double z = mini((double)fabs(x), (double)fabs(y));
	z = x * y > 0 ? z : -z;
	z = (double)(z + log(1 + exp(-fabs(x + y))) - log(1 + exp(-fabs(x - y))));
	return z;
}

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
	/*****************************************************************
	 Check input/output argument number and input type
	*****************************************************************/
	if (nrhs != 3 || nlhs > 1 || !mxIsStruct(prhs[0]) ||
		!mxIsDouble(prhs[2]) ) {
		mexErrMsgIdAndTxt("LDPCDecode:ErrorInputArg",
                "Usage: %s%s%s%s%s%s%s%s%s%s",
                "LDPCresult = LDPCDecode(LDPCstruct, nuser, lratio);\n",
                "Input:\n",
                " LDPCstruct: a struct set using the function 'LDPCInit(NUser, HFileName)'.\n",
                " nuser:      a 1 x N matrix specifying the users to be decoded, N <= NUser. Each element should be an integer between 1 and NUser and be different from each other. NUser is specified in LDPCstruct.\n",
                " lratio:     an CodeLen x N matrix of type mxDOUBLE_CLASS, CodeLen is specified in LDPCstruct, N is the length of the parameter 'nuser'. Each column of 'lratio' is an LLR vector corresponding to the codeword bits.\n",
                "Output:\n",
                " LDPCresult: a struct containing the following fields.\n",
                "             'DecodedCodeword': a CodeLen x LDPC_ItNum x N array of type mxINT32_CLASS, CodeLen and LDPC_ItNum are specified in LDPCstruct, N is the length of the parameter 'nuser'. DecodedCodeword(:,i,j) is the decoded codeword at the i-th iteration for codeword j, the decoded infomation bits are the last 'InfoLen' elements in DecodedCodeword(:,i,j).\n",
                "             'AppLLR':          a CodeLen x N matrix of type mxDOUBLE_CLASS, CodeLen is specified in LDPCstruct, N is the length of the parameter 'nuser'. Each column is the APP LLR corresponding to 'lratio'\n",
                "             'ItNum':           a N x 1 matrix of type mxDOUBLE_CLASS, N is the length of the parameter 'nuser'. Number of iterations excuted for each codeword.\n");
    }

	/*****************************************************************
	 Prepare input argument
	*****************************************************************/
	size_t  NUser   = (size_t)(*((INT32_T*)mxGetData(mxGetFieldByNumber(prhs[0], 0, 0))));
	//INT32_T InfoLen = *((INT32_T*)mxGetData(mxGetFieldByNumber(prhs[0], 0, 1)));
	INT32_T N       = *((INT32_T*)mxGetData(mxGetFieldByNumber(prhs[0], 0, 4)));
	INT32_T M       = *((INT32_T*)mxGetData(mxGetFieldByNumber(prhs[0], 0, 5)));

	INT32_T LDPC_Itnum = *((INT32_T*)mxGetData(mxGetFieldByNumber(prhs[0], 0, 6)));
	char    IfReset    = *mxArrayToString(mxGetFieldByNumber(prhs[0], 0, 7));
	char    IfAIter    = *mxArrayToString(mxGetFieldByNumber(prhs[0], 0, 8));
	char    IfDispl    = *mxArrayToString(mxGetFieldByNumber(prhs[0], 0, 9));
	INT32_T It_Gap     = *((INT32_T*)mxGetData(mxGetFieldByNumber(prhs[0], 0, 10)));

	INT32_T *RowFirstElementIndex = (INT32_T*)mxGetData(mxGetFieldByNumber(prhs[0], 0, 11));
	INT32_T *ColFirstElementIndex = (INT32_T*)mxGetData(mxGetFieldByNumber(prhs[0], 0, 12));
	INT32_T NonzeroNumber         = *((INT32_T*)mxGetData(mxGetFieldByNumber(prhs[0], 0, 13)));
	INT32_T *Chk2VarInterleaver   = (INT32_T*)mxGetData(mxGetFieldByNumber(prhs[0], 0, 14));
	INT32_T *NonzeroCol           = (INT32_T*)mxGetData(mxGetFieldByNumber(prhs[0], 0, 15));
	double  *PR0                  = mxGetPr(mxGetFieldByNumber(prhs[0], 0, 16));
	double  *LR0                  = mxGetPr(mxGetFieldByNumber(prhs[0], 0, 17));

	double *nuserIndex            = mxGetPr(prhs[1]);
	size_t  nuser_num             = mxGetN(prhs[1]);
    
	double *lratio                = mxGetPr(prhs[2]);

	if (mxGetM(prhs[1]) != 1 || mxGetNumberOfDimensions(prhs[1]) > 2 ||
        mxGetM(prhs[2]) != N || mxGetN(prhs[2]) != nuser_num || nuser_num > NUser)
	{
		mexErrMsgIdAndTxt("LDPCDecode:ErrorInputArg",
                "The dimensions of the second or the third input arguments is not matched!\n");
        mexErrMsgIdAndTxt("LDPCDecode:ErrorInputArg",
                "Usage: %s%s%s%s%s%s%s%s%s%s",
                "LDPCresult = LDPCDecode(LDPCstruct, nuser, lratio);\n",
                "Input:\n",
                " LDPCstruct: a struct set using the function 'LDPCInit(NUser, HFileName)'.\n",
                " nuser:      a 1 x N matrix specifying the users to be decoded. Each element should be an integer between 1 and NUser. NUser is specified in LDPCstruct.\n",
                " lratio:     an CodeLen x N matrix of type mxDOUBLE_CLASS, CodeLen is specified in LDPCstruct, N is the length of the parameter 'nuser'. Each column of 'lratio' is an LLR vector corresponding to the codeword bits.\n",
                "Output:\n",
                " LDPCresult: a struct containing the following fields.\n",
                "             'DecodedCodeword': a CodeLen x LDPC_ItNum x N array of type mxINT32_CLASS, CodeLen and LDPC_ItNum are specified in LDPCstruct, N is the length of the parameter 'nuser'. DecodedCodeword(:,i,j) is the decoded codeword at the i-th iteration for codeword nuser(j), the decoded infomation bits are the last 'InfoLen' elements in DecodedCodeword(:,i,j).\n",
                "             'AppLLR':          a CodeLen x N matrix of type mxDOUBLE_CLASS, CodeLen is specified in LDPCstruct, N is the length of the parameter 'nuser'. AppLLR(:,j) is the APP LLR corresponding to lratio(:,j) of codeword for user(j).\n",
                "             'ItNum':           a 1 x N matrix of type mxDOUBLE_CLASS, N is the length of the parameter 'nuser'. Number of iterations excuted for the codeword of user(j).\n");
    }
    
    for(size_t i = 0; i < nuser_num; i++)
    {
        if(nuserIndex[i] > NUser || nuserIndex[i] <= 0)
        {
            mexErrMsgIdAndTxt("LDPCDecode:ErrorArgRange",
                    "The user index = %d is out-of range!\n", i);
        }            
        for(size_t j = i+1; j < nuser_num; j++)
        {
            if(nuserIndex[j] == nuserIndex[i])
            {
                mexErrMsgIdAndTxt("LDPCDecode:ErrorArgRange",
                        "There exists two identical user indeces in the second input argument!\n");
            }
        }
    }

	/*****************************************************************
	 Prepare output argument
	*****************************************************************/
    if(nlhs == 0 || plhs[0] == NULL ||  !mxIsStruct(plhs[0]) ||
       mxGetNumberOfFields(plhs[0]) != 3 ||
       strcmp(mxGetFieldNameByNumber(plhs[0],0),"DecodedCodeword") != 0 ||
       strcmp(mxGetFieldNameByNumber(plhs[0],1),"AppLLR") != 0 ||
       strcmp(mxGetFieldNameByNumber(plhs[0],2),"ItNum") != 0 ||
	   strcmp(mxGetFieldNameByNumber(plhs[0],3),"ErrParityNum") != 0 ||
       mxGetM(mxGetFieldByNumber(plhs[0],0,1)) != N ||
       mxGetN(mxGetFieldByNumber(plhs[0],0,1)) != nuser_num)
    {
        if(plhs[0] != NULL)
        {
            mxFree(plhs[0]);
        }
        
        const char *field_names[] = {"DecodedCodeword", "AppLLR", "ItNum", "ErrParityNum"};
        int NUMBER_OF_FIELDS = (sizeof(field_names) / sizeof(*field_names));
        mwSize dims[] = { 1, 1 };
        plhs[0] = mxCreateStructArray(2, dims, NUMBER_OF_FIELDS, field_names);	
        /* set field 'DecodedCodeword' */
        mwSize dc_dims[3];
        dc_dims[0] = N;
        dc_dims[1] = LDPC_Itnum;
        dc_dims[2] = nuser_num;
        mxSetFieldByNumber(plhs[0], 0, 0, mxCreateNumericArray(3, dc_dims, mxINT32_CLASS, mxREAL));
        /* set field 'AppLLR' */
        mxSetFieldByNumber(plhs[0], 0, 1, mxCreateDoubleMatrix(N, nuser_num, mxREAL));
        /* set field 'ItNum' */
        mxSetFieldByNumber(plhs[0], 0, 2, mxCreateDoubleMatrix(1, nuser_num, mxREAL));
        /* set field 'ErrParityNum' */
        mxSetFieldByNumber(plhs[0], 0, 3, mxCreateDoubleMatrix(1, nuser_num, mxREAL));
    }
    INT32_T *DecodedCodeword = (INT32_T *)mxGetData(mxGetFieldByNumber(plhs[0], 0, 0));
    double  *applr           = mxGetPr(mxGetFieldByNumber(plhs[0], 0, 1));
    double  *it_num          = mxGetPr(mxGetFieldByNumber(plhs[0], 0, 2));
	double  *err_parity_num  = mxGetPr(mxGetFieldByNumber(plhs[0], 0, 3));
    
	/*****************************************************************
	 Decode, from apriori to aposteriori LLR
	*****************************************************************/
	INT32_T n, i, cnt, parity;
	INT32_T j, e, f;
    INT32_T nuser;
	size_t  nuser_id;
	double  pr, dl;
	double  *PR;
	double  *LR;
	    
    for(nuser_id = 0; nuser_id < nuser_num; nuser_id++)
    {
        nuser = (INT32_T)(nuserIndex[nuser_id]) - 1; //in C++, index begins from 0;
        PR              = PR0 + NonzeroNumber * nuser;
        LR              = LR0 + NonzeroNumber * nuser;

        /***********************************/
        /** Reset inner message if needed **/
        /***********************************/
        if (IfReset == 'Y')
        {
            for (i = 0; i < NonzeroNumber; i++)
            {
                PR[i] = 0.0;
                LR[i] = 0.0;
            }
        }
        for (n = 0 ; n < LDPC_Itnum; n++)
        {
			/***********************************/
			/**  Set iteration number         **/
			/***********************************/
			*it_num = (double)(n + 1);

            /***********************************/
            /**  Vertical step                **/
            /**  Recompute likelihood ratios. **/
            /***********************************/

            for (j = 0; j < N; j++)
            {
                pr = 0;

                for (f = ColFirstElementIndex[j]; f < ColFirstElementIndex[j+1]; f++)
                {
                    e = Chk2VarInterleaver[f];
                    PR[e] = pr;
                    pr += LR[e];
                }

                pr = lratio[j];
                for (f = ColFirstElementIndex[j + 1] - 1; f >= ColFirstElementIndex[j]; f--)
                {
                    e = Chk2VarInterleaver[f];
                    PR[e] += pr;
                    pr += LR[e];
                }
            }

            /***********************************/
            /**  Horizational step            **/
            /**  Compute likelihood ratios    **/
            /**  using f-function.            **/
            /***********************************/

            for (i = 0; i < M; i++)
            {
                dl = -1e308;

                for (e = RowFirstElementIndex[i]; e < RowFirstElementIndex[i+1]; e++)
                {
                    LR[e] = dl;
                    dl = Basic_Calculate(dl, PR[e]);
                }

                dl = -1e308;

                for (e = RowFirstElementIndex[i + 1] - 1; e >= RowFirstElementIndex[i]; e--)
                {
                    LR[e] = Basic_Calculate(dl, LR[e]);
                    dl = Basic_Calculate(PR[e], dl);
                }
            }



            /***********************************/
            /**  Vertical step                **/
            /**  Compute app likelihood ratios**/
            /***********************************/

            for (j = 0; j < N; j++)
            {
                pr = 0;

                for (f = ColFirstElementIndex[j]; f < ColFirstElementIndex[j+1]; f++)
                {
                    e = Chk2VarInterleaver[f];
                    pr += LR[e];
                }
                applr[j] = pr + lratio[j];
            }

            /***********************************/
            /**  hard decision                **/
            /**  using a posteriori LLR       **/
            /**  Systematic bits are the last **/
            /** 'InfoLen' bits of the codeword**/
            /***********************************/
            for (i = 0; i < N; i++)
            {
                DecodedCodeword[i] = (applr[i] > 0) ? 0 : 1;
            }

            /***********************************/
            /**  parity-check                 **/
            /**  for early stop               **/
            /***********************************/
            cnt = 0;
            for( i = 0; (i < M) && (cnt == 0); i++ )
            {
                parity = 0;
                for ( e = RowFirstElementIndex[i]; e < RowFirstElementIndex[i + 1]; e++)
                {
                    parity ^= DecodedCodeword[NonzeroCol[e]];//binary addition is XOR
                }
                cnt += parity;
            }

            /***********************************/
            /**  check for early stop         **/
            /**  AND                          **/
            /**  Set remaining results        **/
            /***********************************/
            if(n%It_Gap==0 && IfDispl == 'Y')
                printf("%d		%d\n", n, cnt);
            if (cnt == 0 && IfAIter == 'Y')
            {
                INT32_T *tmpCodeword = DecodedCodeword;
                for (j = n + 1; j < LDPC_Itnum; j++)
                {
                    DecodedCodeword = DecodedCodeword + N;
                    for (i = 0; i < N; i++)
                    {
                        DecodedCodeword[i] = tmpCodeword[i];
                    }
                }
				n = LDPC_Itnum - 1;
            }

            /***********************************/
            /** Increase the pointer by N     **/
            /** for next result               **/
            /***********************************/
            DecodedCodeword = DecodedCodeword + N;
        }//end for LDPC_Itnum
		err_parity_num[nuser_id] = (double)cnt;
		
		//increase pointer
		//DecodedCodeword = DecodedCodeword + N * LDPC_Itnum * nuser_id;//the pointer has been increased in the iteration
		lratio = lratio + N;
		applr  = applr  + N;
		it_num = it_num + 1;
    }// end for userIndex
}
