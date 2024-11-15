#pragma once

#ifndef FINITEFIELD2_H
#define FINITEFIELD2_H


class CFiniteField2  
{
public:
	CFiniteField2();
	virtual ~CFiniteField2();
	
	int m_len_element;
	int m_num_element;
	int *m_gen_poly;
	int *Exp;
	int *Log;
	int **Vec;

	int Add(int alpha, int beta);
	int Mult(int alpha, int beta);
	int Div(int alpha, int beta);
	int Pow(int alpha, int exp);
	int VecInd(int alpha, int index);

	void Malloc(int m);
	void Free();

	void VecAdd(int *alpha, int *beta, int *gamma, int len);
	void VecMult(int *alpha, int *beta, int *gamma, int len);
	void ScalarMult(int alpha, int *beta, int *gamam, int len);
	int PolyEvaluate(int x, int *poly, int deg);

	int PolyMult(int *A, int deg_A, int *B, int deg_B, int *C);
	int PolyAdd(int *A, int deg_A, int *B, int deg_B, int *C);

private:
	static const int PrimitivePolynomial[17][6];
};

#endif // !FINITEFIELD2_H
