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
	4 /* nvar */,
	1 /* nobj */,
	0 /* ncon */,
	0 /* nzc */,
	0 /* densejac */,

	/* objtype (0 = minimize, 1 = maximize) */

	0 };

 real boundc_[1+8+0] /* Infinity, variable bounds, constraint bounds */ = {
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80,
		-1.7e80,
		1.7e80};

 real x0comn_[4] = {
		2.7,
		90.,
		1500.,
		10. };

 real
feval0_(fint *nobj, real *x)
{
	real v[4];


  /***  objective ***/

	v[0] = x[0] * x[0];
	v[1] = -7.391 + v[0];
	v[0] = v[1] / 7.391;
	v[1] = v[0] * v[0];
	v[0] = 10000. * v[1];
	v[1] = x[0] * x[0];
	v[2] = x[1] * x[1];
	v[3] = 0.000428 * v[2];
	v[1] += v[3];
	v[3] = x[2] * x[2];
	v[2] = 1.83184e-07 * v[3];
	v[1] += v[2];
	v[2] = x[3] * x[3];
	v[3] = 0.000428 * v[2];
	v[2] = 1. + v[3];
	if (v[2] == 0.) {
	zerdiv_(&v[2]);	}
	v[3] = v[1] / v[2];
	v[1] = -11.18 + v[3];
	v[3] = v[1] / 11.18;
	v[1] = v[3] * v[3];
	v[3] = 10000. * v[1];
	v[0] += v[3];
	v[3] = x[0] * x[0];
	v[1] = x[1] * x[1];
	v[2] = 0.001 * v[1];
	v[3] += v[2];
	v[2] = x[2] * x[2];
	v[1] = 1.e-06 * v[2];
	v[3] += v[1];
	v[1] = x[3] * x[3];
	v[2] = 0.001 * v[1];
	v[1] = 1. + v[2];
	if (v[1] == 0.) {
	zerdiv_(&v[1]);	}
	v[2] = v[3] / v[1];
	v[3] = -16.44 + v[2];
	v[2] = v[3] / 16.44;
	v[3] = v[2] * v[2];
	v[2] = 10000. * v[3];
	v[0] += v[2];
	v[2] = x[0] * x[0];
	v[3] = x[1] * x[1];
	v[1] = 0.00161 * v[3];
	v[2] += v[1];
	v[1] = x[2] * x[2];
	v[3] = 2.5921e-06 * v[1];
	v[2] += v[3];
	v[3] = x[3] * x[3];
	v[1] = 0.00161 * v[3];
	v[3] = 1. + v[1];
	if (v[3] == 0.) {
	zerdiv_(&v[3]);	}
	v[1] = v[2] / v[3];
	v[2] = -16.2 + v[1];
	v[1] = v[2] / 16.2;
	v[2] = v[1] * v[1];
	v[1] = 10000. * v[2];
	v[0] += v[1];
	v[1] = x[0] * x[0];
	v[2] = x[1] * x[1];
	v[3] = 0.00209 * v[2];
	v[1] += v[3];
	v[3] = x[2] * x[2];
	v[2] = 4.3681e-06 * v[3];
	v[1] += v[2];
	v[2] = x[3] * x[3];
	v[3] = 0.00209 * v[2];
	v[2] = 1. + v[3];
	if (v[2] == 0.) {
	zerdiv_(&v[2]);	}
	v[3] = v[1] / v[2];
	v[1] = -22.2 + v[3];
	v[3] = v[1] / 22.2;
	v[1] = v[3] * v[3];
	v[3] = 10000. * v[1];
	v[0] += v[3];
	v[3] = x[0] * x[0];
	v[1] = x[1] * x[1];
	v[2] = 0.00348 * v[1];
	v[3] += v[2];
	v[2] = x[2] * x[2];
	v[1] = 1.21104e-05 * v[2];
	v[3] += v[1];
	v[1] = x[3] * x[3];
	v[2] = 0.00348 * v[1];
	v[1] = 1. + v[2];
	if (v[1] == 0.) {
	zerdiv_(&v[1]);	}
	v[2] = v[3] / v[1];
	v[3] = -24.02 + v[2];
	v[2] = v[3] / 24.02;
	v[3] = v[2] * v[2];
	v[2] = 10000. * v[3];
	v[0] += v[2];
	v[2] = x[0] * x[0];
	v[3] = x[1] * x[1];
	v[1] = 0.00525 * v[3];
	v[2] += v[1];
	v[1] = x[2] * x[2];
	v[3] = 2.7562500000000002e-05 * v[1];
	v[2] += v[3];
	v[3] = x[3] * x[3];
	v[1] = 0.00525 * v[3];
	v[3] = 1. + v[1];
	if (v[3] == 0.) {
	zerdiv_(&v[3]);	}
	v[1] = v[2] / v[3];
	v[2] = -31.32 + v[1];
	v[1] = v[2] / 31.32;
	v[2] = v[1] * v[1];
	v[1] = 10000. * v[2];
	v[0] += v[1];

	return v[0];
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

x_input = malloc (4 * sizeof(real));
input_values = malloc (4 * sizeof(double));
c_val = malloc (0 * sizeof(real));

file_input = fopen("input.in","r");
for (i=0; i < 4; i++)
    fscanf(file_input, "%lf" ,&input_values[i]);

fclose(file_input);
for (i=0; i < 4; i++)
 {
    x_input[i] = input_values[i];
 }

f_val = feval0_(&objective_number, x_input);

FILE *output_out;
output_out = fopen ("output.out","w");
fprintf(output_out,"%30.15f\n",f_val);
fclose(output_out);

}
