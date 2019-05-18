#include "math.h"
#include "errno.h"
#ifndef fint
#ifndef Long
#include "arith.h"	/* for Long */
#ifndef Long
#define Long long
#endif
#endif
#define fint Long
#endif
#ifndef real
#define real double
#endif
#ifdef __cplusplus
extern "C" {
#endif
 real acosh_(real *);
 real asinh_(real *);
 real acoshd_(real *, real *);
 real asinhd_(real *, real *);
 void in_trouble(char *, real);
 void in_trouble2(char *, real, real);
 void domain_(char *, real *, fint);
 void zerdiv_(real *);
 fint auxcom_[1] = { 0 /* nlc */ };
 fint funcom_[6] = {
	100 /* nvar */,
	1 /* nobj */,
	0 /* ncon */,
	0 /* nzc */,
	0 /* densejac */,

	/* objtype (0 = minimize, 1 = maximize) */

	0 };

 real boundc_[1+200+0] /* Infinity, variable bounds, constraint bounds */ = {
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80};

 real x0comn_[100] = {
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0. };

 real
feval0_(fint *nobj, real *x)
{
	real v[4];


  /***  objective ***/

	v[0] = x[1] - x[0];
	v[1] = 1. + v[0];
	v[0] = x[0] * x[0];
	v[2] = v[1] - v[0];
	v[1] = v[2] * v[2];
	v[2] = 100. * v[1];
	v[1] = x[2] - x[1];
	v[0] = 1. + v[1];
	v[1] = x[1] * x[1];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[3] - x[2];
	v[0] = 1. + v[3];
	v[3] = x[2] * x[2];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[4] - x[3];
	v[0] = 1. + v[1];
	v[1] = x[3] * x[3];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[5] - x[4];
	v[0] = 1. + v[3];
	v[3] = x[4] * x[4];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[6] - x[5];
	v[0] = 1. + v[1];
	v[1] = x[5] * x[5];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[7] - x[6];
	v[0] = 1. + v[3];
	v[3] = x[6] * x[6];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[8] - x[7];
	v[0] = 1. + v[1];
	v[1] = x[7] * x[7];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[9] - x[8];
	v[0] = 1. + v[3];
	v[3] = x[8] * x[8];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[10] - x[9];
	v[0] = 1. + v[1];
	v[1] = x[9] * x[9];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[11] - x[10];
	v[0] = 1. + v[3];
	v[3] = x[10] * x[10];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[12] - x[11];
	v[0] = 1. + v[1];
	v[1] = x[11] * x[11];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[13] - x[12];
	v[0] = 1. + v[3];
	v[3] = x[12] * x[12];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[14] - x[13];
	v[0] = 1. + v[1];
	v[1] = x[13] * x[13];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[15] - x[14];
	v[0] = 1. + v[3];
	v[3] = x[14] * x[14];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[16] - x[15];
	v[0] = 1. + v[1];
	v[1] = x[15] * x[15];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[17] - x[16];
	v[0] = 1. + v[3];
	v[3] = x[16] * x[16];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[18] - x[17];
	v[0] = 1. + v[1];
	v[1] = x[17] * x[17];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[19] - x[18];
	v[0] = 1. + v[3];
	v[3] = x[18] * x[18];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[20] - x[19];
	v[0] = 1. + v[1];
	v[1] = x[19] * x[19];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[21] - x[20];
	v[0] = 1. + v[3];
	v[3] = x[20] * x[20];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[22] - x[21];
	v[0] = 1. + v[1];
	v[1] = x[21] * x[21];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[23] - x[22];
	v[0] = 1. + v[3];
	v[3] = x[22] * x[22];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[24] - x[23];
	v[0] = 1. + v[1];
	v[1] = x[23] * x[23];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[25] - x[24];
	v[0] = 1. + v[3];
	v[3] = x[24] * x[24];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[26] - x[25];
	v[0] = 1. + v[1];
	v[1] = x[25] * x[25];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[27] - x[26];
	v[0] = 1. + v[3];
	v[3] = x[26] * x[26];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[28] - x[27];
	v[0] = 1. + v[1];
	v[1] = x[27] * x[27];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[29] - x[28];
	v[0] = 1. + v[3];
	v[3] = x[28] * x[28];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[30] - x[29];
	v[0] = 1. + v[1];
	v[1] = x[29] * x[29];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[31] - x[30];
	v[0] = 1. + v[3];
	v[3] = x[30] * x[30];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[32] - x[31];
	v[0] = 1. + v[1];
	v[1] = x[31] * x[31];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[33] - x[32];
	v[0] = 1. + v[3];
	v[3] = x[32] * x[32];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[34] - x[33];
	v[0] = 1. + v[1];
	v[1] = x[33] * x[33];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[35] - x[34];
	v[0] = 1. + v[3];
	v[3] = x[34] * x[34];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[36] - x[35];
	v[0] = 1. + v[1];
	v[1] = x[35] * x[35];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[37] - x[36];
	v[0] = 1. + v[3];
	v[3] = x[36] * x[36];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[38] - x[37];
	v[0] = 1. + v[1];
	v[1] = x[37] * x[37];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[39] - x[38];
	v[0] = 1. + v[3];
	v[3] = x[38] * x[38];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[40] - x[39];
	v[0] = 1. + v[1];
	v[1] = x[39] * x[39];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[41] - x[40];
	v[0] = 1. + v[3];
	v[3] = x[40] * x[40];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[42] - x[41];
	v[0] = 1. + v[1];
	v[1] = x[41] * x[41];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[43] - x[42];
	v[0] = 1. + v[3];
	v[3] = x[42] * x[42];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[44] - x[43];
	v[0] = 1. + v[1];
	v[1] = x[43] * x[43];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[45] - x[44];
	v[0] = 1. + v[3];
	v[3] = x[44] * x[44];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[46] - x[45];
	v[0] = 1. + v[1];
	v[1] = x[45] * x[45];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[47] - x[46];
	v[0] = 1. + v[3];
	v[3] = x[46] * x[46];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[48] - x[47];
	v[0] = 1. + v[1];
	v[1] = x[47] * x[47];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[49] - x[48];
	v[0] = 1. + v[3];
	v[3] = x[48] * x[48];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[50] - x[49];
	v[0] = 1. + v[1];
	v[1] = x[49] * x[49];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[51] - x[50];
	v[0] = 1. + v[3];
	v[3] = x[50] * x[50];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[52] - x[51];
	v[0] = 1. + v[1];
	v[1] = x[51] * x[51];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[53] - x[52];
	v[0] = 1. + v[3];
	v[3] = x[52] * x[52];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[54] - x[53];
	v[0] = 1. + v[1];
	v[1] = x[53] * x[53];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[55] - x[54];
	v[0] = 1. + v[3];
	v[3] = x[54] * x[54];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[56] - x[55];
	v[0] = 1. + v[1];
	v[1] = x[55] * x[55];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[57] - x[56];
	v[0] = 1. + v[3];
	v[3] = x[56] * x[56];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[58] - x[57];
	v[0] = 1. + v[1];
	v[1] = x[57] * x[57];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[59] - x[58];
	v[0] = 1. + v[3];
	v[3] = x[58] * x[58];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[60] - x[59];
	v[0] = 1. + v[1];
	v[1] = x[59] * x[59];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[61] - x[60];
	v[0] = 1. + v[3];
	v[3] = x[60] * x[60];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[62] - x[61];
	v[0] = 1. + v[1];
	v[1] = x[61] * x[61];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[63] - x[62];
	v[0] = 1. + v[3];
	v[3] = x[62] * x[62];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[64] - x[63];
	v[0] = 1. + v[1];
	v[1] = x[63] * x[63];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[65] - x[64];
	v[0] = 1. + v[3];
	v[3] = x[64] * x[64];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[66] - x[65];
	v[0] = 1. + v[1];
	v[1] = x[65] * x[65];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[67] - x[66];
	v[0] = 1. + v[3];
	v[3] = x[66] * x[66];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[68] - x[67];
	v[0] = 1. + v[1];
	v[1] = x[67] * x[67];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[69] - x[68];
	v[0] = 1. + v[3];
	v[3] = x[68] * x[68];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[70] - x[69];
	v[0] = 1. + v[1];
	v[1] = x[69] * x[69];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[71] - x[70];
	v[0] = 1. + v[3];
	v[3] = x[70] * x[70];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[72] - x[71];
	v[0] = 1. + v[1];
	v[1] = x[71] * x[71];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[73] - x[72];
	v[0] = 1. + v[3];
	v[3] = x[72] * x[72];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[74] - x[73];
	v[0] = 1. + v[1];
	v[1] = x[73] * x[73];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[75] - x[74];
	v[0] = 1. + v[3];
	v[3] = x[74] * x[74];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[76] - x[75];
	v[0] = 1. + v[1];
	v[1] = x[75] * x[75];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[77] - x[76];
	v[0] = 1. + v[3];
	v[3] = x[76] * x[76];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[78] - x[77];
	v[0] = 1. + v[1];
	v[1] = x[77] * x[77];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[79] - x[78];
	v[0] = 1. + v[3];
	v[3] = x[78] * x[78];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[80] - x[79];
	v[0] = 1. + v[1];
	v[1] = x[79] * x[79];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[81] - x[80];
	v[0] = 1. + v[3];
	v[3] = x[80] * x[80];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[82] - x[81];
	v[0] = 1. + v[1];
	v[1] = x[81] * x[81];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[83] - x[82];
	v[0] = 1. + v[3];
	v[3] = x[82] * x[82];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[84] - x[83];
	v[0] = 1. + v[1];
	v[1] = x[83] * x[83];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[85] - x[84];
	v[0] = 1. + v[3];
	v[3] = x[84] * x[84];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[86] - x[85];
	v[0] = 1. + v[1];
	v[1] = x[85] * x[85];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[87] - x[86];
	v[0] = 1. + v[3];
	v[3] = x[86] * x[86];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[88] - x[87];
	v[0] = 1. + v[1];
	v[1] = x[87] * x[87];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[89] - x[88];
	v[0] = 1. + v[3];
	v[3] = x[88] * x[88];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[90] - x[89];
	v[0] = 1. + v[1];
	v[1] = x[89] * x[89];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[91] - x[90];
	v[0] = 1. + v[3];
	v[3] = x[90] * x[90];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[92] - x[91];
	v[0] = 1. + v[1];
	v[1] = x[91] * x[91];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[93] - x[92];
	v[0] = 1. + v[3];
	v[3] = x[92] * x[92];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[94] - x[93];
	v[0] = 1. + v[1];
	v[1] = x[93] * x[93];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[95] - x[94];
	v[0] = 1. + v[3];
	v[3] = x[94] * x[94];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[96] - x[95];
	v[0] = 1. + v[1];
	v[1] = x[95] * x[95];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[97] - x[96];
	v[0] = 1. + v[3];
	v[3] = x[96] * x[96];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];
	v[1] = x[98] - x[97];
	v[0] = 1. + v[1];
	v[1] = x[97] * x[97];
	v[3] = v[0] - v[1];
	v[0] = v[3] * v[3];
	v[3] = 100. * v[0];
	v[2] += v[3];
	v[3] = x[99] - x[98];
	v[0] = 1. + v[3];
	v[3] = x[98] * x[98];
	v[1] = v[0] - v[3];
	v[0] = v[1] * v[1];
	v[1] = 100. * v[0];
	v[2] += v[1];

	return v[2];
}

 void
ceval0_(real *x, real *c)
{}
#ifdef __cplusplus
	}
#endif

#include <stdio.h>
#include <stdlib.h>

main(int argc, char **argv)
{

FILE *file_input;
real *x_input, f_val, *c_val;
double *input_values;
fint objective_number;
int i;

x_input = malloc (100 * sizeof(real));
input_values = malloc (100 * sizeof(double));
c_val = malloc (0 * sizeof(real));

file_input = fopen("input.in","r");
for (i=0; i < 100; i++)
    fscanf(file_input, "%lf" ,&input_values[i]);

fclose(file_input);
for (i=0; i < 100; i++)
 {
    x_input[i] = input_values[i];
 }

f_val = feval0_(&objective_number, x_input);

FILE *output_out;
output_out = fopen ("output.out","w");
fprintf(output_out,"%30.15f\n",f_val);
fclose(output_out);

}
