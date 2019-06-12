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
	50 /* nvar */,
	1 /* nobj */,
	0 /* ncon */,
	0 /* nzc */,
	0 /* densejac */,

	/* objtype (0 = minimize, 1 = maximize) */

	0 };

 real boundc_[1+100+0] /* Infinity, variable bounds, constraint bounds */ = {
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

 real x0comn_[50] = {
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1.,
		-1. };

 real
feval0_(fint *nobj, real *x)
{
	real v[3];


  /***  objective ***/

	v[0] = x[1] * x[1];
	v[1] = x[0] - v[0];
	v[0] = v[1] * v[1];
	v[1] = 31.359999999999996 * v[0];
	v[0] = -1. + x[1];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[2] * x[2];
	v[0] = x[1] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 92.16 * v[2];
	v[1] += v[0];
	v[0] = -1. + x[2];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[3] * x[3];
	v[0] = x[2] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 31.359999999999996 * v[2];
	v[1] += v[0];
	v[0] = -1. + x[3];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[4] * x[4];
	v[0] = x[3] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 49. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[4];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[5] * x[5];
	v[0] = x[4] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 23.04 * v[2];
	v[1] += v[0];
	v[0] = -1. + x[5];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[6] * x[6];
	v[0] = x[5] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 81. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[6];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[7] * x[7];
	v[0] = x[6] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 23.04 * v[2];
	v[1] += v[0];
	v[0] = -1. + x[7];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[8] * x[8];
	v[0] = x[7] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 16. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[8];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[9] * x[9];
	v[0] = x[8] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 19.360000000000003 * v[2];
	v[1] += v[0];
	v[0] = -1. + x[9];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[10] * x[10];
	v[0] = x[9] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 36. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[10];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[11] * x[11];
	v[0] = x[10] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 40.96000000000001 * v[2];
	v[1] += v[0];
	v[0] = -1. + x[11];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[12] * x[12];
	v[0] = x[11] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 25. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[12];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[13] * x[13];
	v[0] = x[12] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 25. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[13];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[14] * x[14];
	v[0] = x[13] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 23.04 * v[2];
	v[1] += v[0];
	v[0] = -1. + x[14];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[15] * x[15];
	v[0] = x[14] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 23.04 * v[2];
	v[1] += v[0];
	v[0] = -1. + x[15];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[16] * x[16];
	v[0] = x[15] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 31.359999999999996 * v[2];
	v[1] += v[0];
	v[0] = -1. + x[16];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[17] * x[17];
	v[0] = x[16] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 4. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[17];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[18] * x[18];
	v[0] = x[17] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 4. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[18];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[19] * x[19];
	v[0] = x[18] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 25. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[19];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[20] * x[20];
	v[0] = x[19] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 51.84 * v[2];
	v[1] += v[0];
	v[0] = -1. + x[20];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[21] * x[21];
	v[0] = x[20] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 9. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[21];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[22] * x[22];
	v[0] = x[21] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 25. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[22];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[23] * x[23];
	v[0] = x[22] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 31.359999999999996 * v[2];
	v[1] += v[0];
	v[0] = -1. + x[23];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[24] * x[24];
	v[0] = x[23] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 40.96000000000001 * v[2];
	v[1] += v[0];
	v[0] = -1. + x[24];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[25] * x[25];
	v[0] = x[24] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 64. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[25];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[26] * x[26];
	v[0] = x[25] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 16. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[26];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[27] * x[27];
	v[0] = x[26] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 40.96000000000001 * v[2];
	v[1] += v[0];
	v[0] = -1. + x[27];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[28] * x[28];
	v[0] = x[27] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 25. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[28];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[29] * x[29];
	v[0] = x[28] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 121. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[29];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[30] * x[30];
	v[0] = x[29] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 25. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[30];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[31] * x[31];
	v[0] = x[30] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 25. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[31];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[32] * x[32];
	v[0] = x[31] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 25. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[32];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[33] * x[33];
	v[0] = x[32] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 144. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[33];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[34] * x[34];
	v[0] = x[33] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 36. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[34];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[35] * x[35];
	v[0] = x[34] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 64. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[35];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[36] * x[36];
	v[0] = x[35] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 25. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[36];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[37] * x[37];
	v[0] = x[36] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 31.359999999999996 * v[2];
	v[1] += v[0];
	v[0] = -1. + x[37];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[38] * x[38];
	v[0] = x[37] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 51.84 * v[2];
	v[1] += v[0];
	v[0] = -1. + x[38];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[39] * x[39];
	v[0] = x[38] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 36. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[39];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[40] * x[40];
	v[0] = x[39] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 77.44000000000001 * v[2];
	v[1] += v[0];
	v[0] = -1. + x[40];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[41] * x[41];
	v[0] = x[40] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 31.359999999999996 * v[2];
	v[1] += v[0];
	v[0] = -1. + x[41];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[42] * x[42];
	v[0] = x[41] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 36. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[42];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[43] * x[43];
	v[0] = x[42] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 25. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[43];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[44] * x[44];
	v[0] = x[43] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 64. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[44];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[45] * x[45];
	v[0] = x[44] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 36. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[45];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[46] * x[46];
	v[0] = x[45] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 25. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[46];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[47] * x[47];
	v[0] = x[46] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 31.359999999999996 * v[2];
	v[1] += v[0];
	v[0] = -1. + x[47];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[48] * x[48];
	v[0] = x[47] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 5.76 * v[2];
	v[1] += v[0];
	v[0] = -1. + x[48];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[49] * x[49];
	v[0] = x[48] - v[2];
	v[2] = v[0] * v[0];
	v[0] = 36. * v[2];
	v[1] += v[0];
	v[0] = -1. + x[49];
	v[2] = v[0] * v[0];
	v[1] += v[2];

	return v[1];
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

x_input = malloc (50 * sizeof(real));
input_values = malloc (50 * sizeof(double));
c_val = malloc (0 * sizeof(real));

file_input = fopen("input.in","r");
for (i=0; i < 50; i++)
    fscanf(file_input, "%lf" ,&input_values[i]);

fclose(file_input);
for (i=0; i < 50; i++)
 {
    x_input[i] = input_values[i];
 }

f_val = feval0_(&objective_number, x_input);

FILE *output_out;
output_out = fopen ("output.out","w");
fprintf(output_out,"%30.15f\n",f_val);
fclose(output_out);

}
