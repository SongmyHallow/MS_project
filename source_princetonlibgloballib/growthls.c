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
	3 /* nvar */,
	1 /* nobj */,
	0 /* ncon */,
	0 /* nzc */,
	0 /* densejac */,

	/* objtype (0 = minimize, 1 = maximize) */

	0 };

 real boundc_[1+6+0] /* Infinity, variable bounds, constraint bounds */ = {
		1.7e80,
		-1.7e80,
		1.7e80,
		-15.,
		15.,
		-15.,
		15.};

 real x0comn_[3] = {
		10.,
		0.,
		0. };

 real
feval0_(fint *nobj, real *x)
{
	real v[4];


  /***  objective ***/

	v[0] = 2.079441542 * x[2];
	v[1] = x[1] + v[0];
	v[0] = 2.079442 * v[1];
	v[1] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[0] * v[1];
	v[1] = -8. + v[0];
	v[0] = 2.079441542 * x[2];
	v[2] = x[1] + v[0];
	v[0] = 2.079442 * v[2];
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[0] * v[2];
	v[2] = -8. + v[0];
	v[0] = v[1] * v[2];
	v[1] = 2.19722457733622 * x[2];
	v[2] = x[1] + v[1];
	v[1] = 2.19722457733622 * v[2];
	v[2] = exp(v[1]);
	if (errno) in_trouble("exp",v[1]);
	v[1] = x[0] * v[2];
	v[2] = -8.4305 + v[1];
	v[1] = 2.19722457733622 * x[2];
	v[3] = x[1] + v[1];
	v[1] = 2.19722457733622 * v[3];
	v[3] = exp(v[1]);
	if (errno) in_trouble("exp",v[1]);
	v[1] = x[0] * v[3];
	v[3] = -8.4305 + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 2.302585093 * x[2];
	v[2] = x[1] + v[1];
	v[1] = 2.302585 * v[2];
	v[2] = exp(v[1]);
	if (errno) in_trouble("exp",v[1]);
	v[1] = x[0] * v[2];
	v[2] = -9.5294 + v[1];
	v[1] = 2.302585093 * x[2];
	v[3] = x[1] + v[1];
	v[1] = 2.302585 * v[3];
	v[3] = exp(v[1]);
	if (errno) in_trouble("exp",v[1]);
	v[1] = x[0] * v[3];
	v[3] = -9.5294 + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 2.397895273 * x[2];
	v[2] = x[1] + v[1];
	v[1] = 2.397895 * v[2];
	v[2] = exp(v[1]);
	if (errno) in_trouble("exp",v[1]);
	v[1] = x[0] * v[2];
	v[2] = -10.4627 + v[1];
	v[1] = 2.397895273 * x[2];
	v[3] = x[1] + v[1];
	v[1] = 2.397895 * v[3];
	v[3] = exp(v[1]);
	if (errno) in_trouble("exp",v[1]);
	v[1] = x[0] * v[3];
	v[3] = -10.4627 + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 2.48490665 * x[2];
	v[2] = x[1] + v[1];
	v[1] = 2.484907 * v[2];
	v[2] = exp(v[1]);
	if (errno) in_trouble("exp",v[1]);
	v[1] = x[0] * v[2];
	v[2] = -12. + v[1];
	v[1] = 2.48490665 * x[2];
	v[3] = x[1] + v[1];
	v[1] = 2.484907 * v[3];
	v[3] = exp(v[1]);
	if (errno) in_trouble("exp",v[1]);
	v[1] = x[0] * v[3];
	v[3] = -12. + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 2.564949357 * x[2];
	v[2] = x[1] + v[1];
	v[1] = 2.564949 * v[2];
	v[2] = exp(v[1]);
	if (errno) in_trouble("exp",v[1]);
	v[1] = x[0] * v[2];
	v[2] = -13.0205 + v[1];
	v[1] = 2.564949357 * x[2];
	v[3] = x[1] + v[1];
	v[1] = 2.564949 * v[3];
	v[3] = exp(v[1]);
	if (errno) in_trouble("exp",v[1]);
	v[1] = x[0] * v[3];
	v[3] = -13.0205 + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 2.63905733 * x[2];
	v[2] = x[1] + v[1];
	v[1] = 2.639057 * v[2];
	v[2] = exp(v[1]);
	if (errno) in_trouble("exp",v[1]);
	v[1] = x[0] * v[2];
	v[2] = -14.5949 + v[1];
	v[1] = 2.63905733 * x[2];
	v[3] = x[1] + v[1];
	v[1] = 2.639057 * v[3];
	v[3] = exp(v[1]);
	if (errno) in_trouble("exp",v[1]);
	v[1] = x[0] * v[3];
	v[3] = -14.5949 + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 2.708050201 * x[2];
	v[2] = x[1] + v[1];
	v[1] = 2.70805 * v[2];
	v[2] = exp(v[1]);
	if (errno) in_trouble("exp",v[1]);
	v[1] = x[0] * v[2];
	v[2] = -16.1078 + v[1];
	v[1] = 2.708050201 * x[2];
	v[3] = x[1] + v[1];
	v[1] = 2.70805 * v[3];
	v[3] = exp(v[1]);
	if (errno) in_trouble("exp",v[1]);
	v[1] = x[0] * v[3];
	v[3] = -16.1078 + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 2.77258872223978 * x[2];
	v[2] = x[1] + v[1];
	v[1] = 2.772589 * v[2];
	v[2] = exp(v[1]);
	if (errno) in_trouble("exp",v[1]);
	v[1] = x[0] * v[2];
	v[2] = -18.0596 + v[1];
	v[1] = 2.77258872223978 * x[2];
	v[3] = x[1] + v[1];
	v[1] = 2.772589 * v[3];
	v[3] = exp(v[1]);
	if (errno) in_trouble("exp",v[1]);
	v[1] = x[0] * v[3];
	v[3] = -18.0596 + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 2.890371758 * x[2];
	v[2] = x[1] + v[1];
	v[1] = 2.890372 * v[2];
	v[2] = exp(v[1]);
	if (errno) in_trouble("exp",v[1]);
	v[1] = x[0] * v[2];
	v[2] = -20.4569 + v[1];
	v[1] = 2.890371758 * x[2];
	v[3] = x[1] + v[1];
	v[1] = 2.890372 * v[3];
	v[3] = exp(v[1]);
	if (errno) in_trouble("exp",v[1]);
	v[1] = x[0] * v[3];
	v[3] = -20.4569 + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 2.99573227355399 * x[2];
	v[2] = x[1] + v[1];
	v[1] = 2.995732 * v[2];
	v[2] = exp(v[1]);
	if (errno) in_trouble("exp",v[1]);
	v[1] = x[0] * v[2];
	v[2] = -24.25 + v[1];
	v[1] = 2.99573227355399 * x[2];
	v[3] = x[1] + v[1];
	v[1] = 2.995732 * v[3];
	v[3] = exp(v[1]);
	if (errno) in_trouble("exp",v[1]);
	v[1] = x[0] * v[3];
	v[3] = -24.25 + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 3.218875825 * x[2];
	v[2] = x[1] + v[1];
	v[1] = 3.218876 * v[2];
	v[2] = exp(v[1]);
	if (errno) in_trouble("exp",v[1]);
	v[1] = x[0] * v[2];
	v[2] = -32.9863 + v[1];
	v[1] = 3.218875825 * x[2];
	v[3] = x[1] + v[1];
	v[1] = 3.218876 * v[3];
	v[3] = exp(v[1]);
	if (errno) in_trouble("exp",v[1]);
	v[1] = x[0] * v[3];
	v[3] = -32.9863 + v[1];
	v[1] = v[2] * v[3];
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

x_input = malloc (3 * sizeof(real));
input_values = malloc (3 * sizeof(double));
c_val = malloc (0 * sizeof(real));

file_input = fopen("input.in","r");
for (i=0; i < 3; i++)
    fscanf(file_input, "%lf" ,&input_values[i]);

fclose(file_input);
for (i=0; i < 3; i++)
 {
    x_input[i] = input_values[i];
 }

f_val = feval0_(&objective_number, x_input);

FILE *output_out;
output_out = fopen ("output.out","w");
fprintf(output_out,"%30.15f\n",f_val);
fclose(output_out);

}