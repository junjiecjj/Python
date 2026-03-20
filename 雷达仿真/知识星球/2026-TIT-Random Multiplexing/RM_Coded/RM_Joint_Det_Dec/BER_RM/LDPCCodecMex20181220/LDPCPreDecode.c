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
    /*****************************************************************
	 Check input/output argument number and input type
	*****************************************************************/
   if (nrhs != 1 || nlhs != 0 || !mxIsStruct(prhs[0])) {
         mexErrMsgIdAndTxt("LDPCPreDecode:ErrorInputArg",
                 "Usage: LDPCPreDecode(LDPCstruct);\n%s%s%s%s",
                 "Input:\n",
                 " LDPCstruct: a struct set using the function 'LDPCInit(NUser, HFileName)'.\n",
                 "Output:\n",
                 " There is not output parameter for this function. This function only initializes the internal message 'pr' and 'lr' in to 0.\n");
    }
	
    /*****************************************************************
	 Set pr and pl to all zero
	*****************************************************************/
	double  *pr  = mxGetPr(mxGetFieldByNumber(prhs[0], 0, 16));
	double  *lr  = mxGetPr(mxGetFieldByNumber(prhs[0], 0, 17));
	mxArray *ptr = mxGetFieldByNumber(prhs[0], 0, 16);
	size_t  len  = mxGetN(ptr) * mxGetM(ptr);
	for (size_t j = 0; j < len; j++)
	{
		pr[j] = 0.0;
		lr[j] = 0.0;
	}
}
