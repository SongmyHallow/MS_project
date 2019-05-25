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
	8 /* nvar */,
	1 /* nobj */,
	0 /* ncon */,
	0 /* nzc */,
	0 /* densejac */,

	/* objtype (0 = minimize, 1 = maximize) */

	0 };

 real boundc_[1+16+0] /* Infinity, variable bounds, constraint bounds */ = {
		1.7e80,
		2.5,
		1.7e80,
		0.,
		4.1,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.5,
		4.,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		4.3};

 real x0comn_[8] = {
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5 };

 real
feval0_(fint *nobj, real *x)
{
	real v[3], rv;


  /***  objective ***/

	v[0] = x[0] * x[0];
	v[1] = 0.5 * v[0];
	v[0] = x[1] * x[1];
	v[2] = 0.5 * v[0];
	v[1] += v[2];
	v[2] = x[2] * x[2];
	v[0] = 0.5 * v[2];
	v[1] += v[0];
	v[0] = x[3] * x[3];
	v[2] = 0.5 * v[0];
	v[1] += v[2];
	v[2] = x[4] * x[4];
	v[0] = 0.5 * v[2];
	v[1] += v[0];
	v[0] = x[5] * x[5];
	v[2] = 0.5 * v[0];
	v[1] += v[2];
	v[2] = x[6] * x[6];
	v[0] = 0.5 * v[2];
	v[1] += v[0];
	v[0] = x[7] * x[7];
	v[2] = 0.5 * v[0];
	v[1] += v[2];
	rv = v[1] + x[0];
	rv += 2.*x[4];
	rv += -x[7];

	return rv;
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

x_input = malloc (8 * sizeof(real));
input_values = malloc (8 * sizeof(double));
c_val = malloc (0 * sizeof(real));

file_input = fopen("input.in","r");
for (i=0; i < 8; i++)
    fscanf(file_input, "%lf" ,&input_values[i]);

fclose(file_input);
for (i=0; i < 8; i++)
 {
    x_input[i] = input_values[i];
 }

f_val = feval0_(&objective_number, x_input);

FILE *output_out;
output_out = fopen ("output.out","w");
fprintf(output_out,"%30.15f\n",f_val);
fclose(output_out);

}