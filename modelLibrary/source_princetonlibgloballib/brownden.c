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
		25.,
		5.,
		-5.,
		-1. };

 real
feval0_(fint *nobj, real *x)
{
	real v[4];


  /***  objective ***/

	v[0] = 0.2 * x[1];
	v[1] = x[0] + v[0];
	v[0] = -1.2214027581601699 + v[1];
	v[1] = v[0] * v[0];
	v[0] = 0.19866933079506122 * x[3];
	v[2] = x[2] + v[0];
	v[0] = -0.9800665778412416 + v[2];
	v[2] = v[0] * v[0];
	v[0] = v[1] + v[2];
	v[1] = v[0] * v[0];
	v[0] = 0.4 * x[1];
	v[2] = x[0] + v[0];
	v[0] = -1.4918246976412703 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 0.3894183423086505 * x[3];
	v[3] = x[2] + v[0];
	v[0] = -0.9210609940028851 + v[3];
	v[3] = v[0] * v[0];
	v[0] = v[2] + v[3];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = 0.6 * x[1];
	v[0] = x[0] + v[2];
	v[2] = -1.8221188003905089 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 0.5646424733950354 * x[3];
	v[3] = x[2] + v[2];
	v[2] = -0.8253356149096783 + v[3];
	v[3] = v[2] * v[2];
	v[2] = v[0] + v[3];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = 0.8 * x[1];
	v[2] = x[0] + v[0];
	v[0] = -2.225540928492468 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 0.7173560908995228 * x[3];
	v[3] = x[2] + v[0];
	v[0] = -0.6967067093471654 + v[3];
	v[3] = v[0] * v[0];
	v[0] = v[2] + v[3];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = x[0] + x[1];
	v[0] = -2.718281828459045 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 0.8414709848078965 * x[3];
	v[3] = x[2] + v[0];
	v[0] = -0.5403023058681398 + v[3];
	v[3] = v[0] * v[0];
	v[0] = v[2] + v[3];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = 1.2 * x[1];
	v[0] = x[0] + v[2];
	v[2] = -3.3201169227365472 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 0.9320390859672263 * x[3];
	v[3] = x[2] + v[2];
	v[2] = -0.3623577544766736 + v[3];
	v[3] = v[2] * v[2];
	v[2] = v[0] + v[3];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = 1.4 * x[1];
	v[2] = x[0] + v[0];
	v[0] = -4.055199966844674 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 0.9854497299884601 * x[3];
	v[3] = x[2] + v[0];
	v[0] = -0.16996714290024104 + v[3];
	v[3] = v[0] * v[0];
	v[0] = v[2] + v[3];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = 1.6 * x[1];
	v[0] = x[0] + v[2];
	v[2] = -4.953032424395116 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 0.9995736030415051 * x[3];
	v[3] = x[2] + v[2];
	v[2] = 0.029199522301288815 + v[3];
	v[3] = v[2] * v[2];
	v[2] = v[0] + v[3];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = 1.8 * x[1];
	v[2] = x[0] + v[0];
	v[0] = -6.049647464412946 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 0.9738476308781951 * x[3];
	v[3] = x[2] + v[0];
	v[0] = 0.2272020946930871 + v[3];
	v[3] = v[0] * v[0];
	v[0] = v[2] + v[3];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = 2. * x[1];
	v[0] = x[0] + v[2];
	v[2] = -7.38905609893065 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 0.9092974268256817 * x[3];
	v[3] = x[2] + v[2];
	v[2] = 0.4161468365471424 + v[3];
	v[3] = v[2] * v[2];
	v[2] = v[0] + v[3];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = 2.2 * x[1];
	v[2] = x[0] + v[0];
	v[0] = -9.025013499434122 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 0.8084964038195901 * x[3];
	v[3] = x[2] + v[0];
	v[0] = 0.5885011172553458 + v[3];
	v[3] = v[0] * v[0];
	v[0] = v[2] + v[3];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = 2.4 * x[1];
	v[0] = x[0] + v[2];
	v[2] = -11.0231763806416 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 0.675463180551151 * x[3];
	v[3] = x[2] + v[2];
	v[2] = 0.7373937155412454 + v[3];
	v[3] = v[2] * v[2];
	v[2] = v[0] + v[3];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = 2.6 * x[1];
	v[2] = x[0] + v[0];
	v[0] = -13.463738035001692 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 0.5155013718214642 * x[3];
	v[3] = x[2] + v[0];
	v[0] = 0.8568887533689473 + v[3];
	v[3] = v[0] * v[0];
	v[0] = v[2] + v[3];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = 2.8 * x[1];
	v[0] = x[0] + v[2];
	v[2] = -16.444646771097045 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 0.3349881501559051 * x[3];
	v[3] = x[2] + v[2];
	v[2] = 0.9422223406686581 + v[3];
	v[3] = v[2] * v[2];
	v[2] = v[0] + v[3];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = 3. * x[1];
	v[2] = x[0] + v[0];
	v[0] = -20.08553692318767 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 0.1411200080598672 * x[3];
	v[3] = x[2] + v[0];
	v[0] = 0.9899924966004454 + v[3];
	v[3] = v[0] * v[0];
	v[0] = v[2] + v[3];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = 3.2 * x[1];
	v[0] = x[0] + v[2];
	v[2] = -24.532530197109356 + v[0];
	v[0] = v[2] * v[2];
	v[2] = -0.058374143427580086 * x[3];
	v[3] = x[2] + v[2];
	v[2] = 0.9982947757947531 + v[3];
	v[3] = v[2] * v[2];
	v[2] = v[0] + v[3];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = 3.4 * x[1];
	v[2] = x[0] + v[0];
	v[0] = -29.964100047397007 + v[2];
	v[2] = v[0] * v[0];
	v[0] = -0.2555411020268312 * x[3];
	v[3] = x[2] + v[0];
	v[0] = 0.9667981925794611 + v[3];
	v[3] = v[0] * v[0];
	v[0] = v[2] + v[3];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = 3.6 * x[1];
	v[0] = x[0] + v[2];
	v[2] = -36.59823444367798 + v[0];
	v[0] = v[2] * v[2];
	v[2] = -0.44252044329485246 * x[3];
	v[3] = x[2] + v[2];
	v[2] = 0.896758416334147 + v[3];
	v[3] = v[2] * v[2];
	v[2] = v[0] + v[3];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = 3.8 * x[1];
	v[2] = x[0] + v[0];
	v[0] = -44.701184493300815 + v[2];
	v[2] = v[0] * v[0];
	v[0] = -0.6118578909427189 * x[3];
	v[3] = x[2] + v[0];
	v[0] = 0.7909677119144168 + v[3];
	v[3] = v[0] * v[0];
	v[0] = v[2] + v[3];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = 4. * x[1];
	v[0] = x[0] + v[2];
	v[2] = -54.598150033144236 + v[0];
	v[0] = v[2] * v[2];
	v[2] = -0.7568024953079282 * x[3];
	v[3] = x[2] + v[2];
	v[2] = 0.6536436208636119 + v[3];
	v[3] = v[2] * v[2];
	v[2] = v[0] + v[3];
	v[0] = v[2] * v[2];
	v[1] += v[0];

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
