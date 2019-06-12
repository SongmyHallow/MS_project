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
		0.,
		50.,
		-1.7e80,
		1.7e80};

 real x0comn_[8] = {
		19.264,
		-1.7302,
		40.794,
		0.83021,
		3.709,
		-0.17723,
		10.,
		1. };

 real
feval0_(fint *nobj, real *x)
{
	real v[4];


  /***  objective ***/

	v[0] = -x[1];
	v[0] += x[0];
	v[0] += x[2];
	v[1] = -x[3];
	v[0] += v[1];
	v[0] += x[4];
	v[1] = -x[5];
	v[0] += v[1];
	v[0] += x[7];
	v[1] = 83.57418 - v[0];
	v[0] = v[1] * v[1];
	v[1] = 1.0000000000000009 * x[1];
	v[1] += x[0];
	v[2] = 1.0000000000000027 * x[2];
	v[1] += v[2];
	v[2] = 1.0000000000000062 * x[3];
	v[1] += v[2];
	v[2] = 1.000000000000011 * x[4];
	v[1] += v[2];
	v[2] = 1.0000000000000173 * x[5];
	v[1] += v[2];
	v[2] = 2.4674000736160004 * x[6];
	v[3] = -v[2];
	v[2] = exp(v[3]);
	if (errno) in_trouble("exp",v[3]);
	v[3] = x[7] * v[2];
	v[1] += v[3];
	v[3] = 81.007654 - v[1];
	v[1] = v[3] * v[3];
	v[0] += v[1];
	v[1] = 0.5802466620760969 * x[1];
	v[1] += x[0];
	v[3] = -0.3266276222990955 * x[2];
	v[1] += v[3];
	v[3] = -0.9592958372379013 * x[3];
	v[1] += v[3];
	v[3] = -0.7866287927024783 * x[4];
	v[1] += v[3];
	v[3] = 0.046418374720775146 * x[5];
	v[1] += v[3];
	v[3] = 1.949550365169 * x[6];
	v[2] = -v[3];
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[7] * v[3];
	v[1] += v[2];
	v[2] = 18.983286 - v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 0.38888959624442854 * x[1];
	v[1] += x[0];
	v[2] = -0.6975297638656907 * x[2];
	v[1] += v[2];
	v[2] = -0.9314137327208286 * x[3];
	v[1] += v[2];
	v[2] = -0.026904457042947638 * x[4];
	v[1] += v[2];
	v[2] = 0.9104880058476136 * x[5];
	v[1] += v[2];
	v[2] = 1.7134731460089998 * x[6];
	v[3] = -v[2];
	v[2] = exp(v[3]);
	if (errno) in_trouble("exp",v[3]);
	v[3] = x[7] * v[2];
	v[1] += v[3];
	v[3] = 8.051067 - v[1];
	v[1] = v[3] * v[3];
	v[0] += v[1];
	v[1] = 0.20987610307763616 * x[1];
	v[1] += x[0];
	v[3] = -0.9119040427138908 * x[2];
	v[1] += v[3];
	v[3] = -0.5926498368087033 * x[3];
	v[1] += v[3];
	v[3] = 0.6631379662358756 * x[4];
	v[1] += v[3];
	v[3] = 0.8710034611215323 * x[5];
	v[1] += v[3];
	v[3] = 1.4926241929 * x[6];
	v[2] = -v[3];
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[7] * v[3];
	v[1] += v[2];
	v[2] = 2.044762 - v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 0.027400834407416585 * x[1];
	v[1] += x[0];
	v[2] = -0.9984983885475546 * x[2];
	v[1] += v[2];
	v[2] = -0.08212021240874423 * x[3];
	v[1] += v[2];
	v[2] = 0.9939980638641267 * x[4];
	v[1] += v[2];
	v[2] = 0.13659296510721142 * x[5];
	v[1] += v[2];
	v[2] = 1.2675044472249999 * x[6];
	v[3] = -v[2];
	v[2] = exp(v[3]);
	if (errno) in_trouble("exp",v[3]);
	v[3] = x[7] * v[2];
	v[1] += v[3];
	v[3] = -v[1];
	v[1] = v[3] * v[3];
	v[0] += v[1];
	v[1] = -0.11110997934203104 * x[1];
	v[1] += x[0];
	v[3] = -0.9753091449812269 * x[2];
	v[1] += v[3];
	v[3] = 0.3278431372439471 * x[3];
	v[1] += v[3];
	v[3] = 0.9024558565680235 * x[4];
	v[1] += v[3];
	v[3] = -0.5283868404046833 * x[5];
	v[1] += v[3];
	v[3] = 1.0966236512040002 * x[6];
	v[2] = -v[3];
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[7] * v[3];
	v[1] += v[2];
	v[2] = 1.170451 - v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = -0.38271526343196616 * x[1];
	v[1] += x[0];
	v[2] = -0.7070580542724016 * x[2];
	v[1] += v[2];
	v[2] = 0.9239190824370774 * x[3];
	v[1] += v[2];
	v[2] = -0.0001378157770514088 * x[4];
	v[1] += v[2];
	v[2] = -0.9238135940342387 * x[5];
	v[1] += v[2];
	v[2] = 0.7615442022250001 * x[6];
	v[3] = -v[2];
	v[2] = exp(v[3]);
	if (errno) in_trouble("exp",v[3]);
	v[3] = x[7] * v[2];
	v[1] += v[3];
	v[3] = 10.479881 - v[1];
	v[1] = v[3] * v[3];
	v[0] += v[1];
	v[1] = -0.6049377685964584 * x[1];
	v[1] += x[0];
	v[3] = -0.26810059225107563 * x[2];
	v[1] += v[3];
	v[3] = 0.9293061166679677 * x[3];
	v[1] += v[3];
	v[3] = -0.8562441448692452 * x[4];
	v[1] += v[3];
	v[3] = 0.10664272807399988 * x[5];
	v[1] += v[3];
	v[3] = 0.487388289424 * x[6];
	v[2] = -v[3];
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[7] * v[3];
	v[1] += v[2];
	v[2] = 25.785001 - v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = -0.7777774948355076 * x[1];
	v[1] += x[0];
	v[2] = 0.20987566294519644 * x[2];
	v[1] += v[2];
	v[2] = 0.45130436013059516 * x[3];
	v[1] += v[2];
	v[2] = -0.9119044122066285 * x[4];
	v[1] += v[2];
	v[2] = 0.9672130983804402 * x[5];
	v[1] += v[2];
	v[2] = 0.27415591280100005 * x[6];
	v[3] = -v[2];
	v[2] = exp(v[3]);
	if (errno) in_trouble("exp",v[3]);
	v[3] = x[7] * v[2];
	v[1] += v[3];
	v[3] = 44.126844 - v[1];
	v[1] = v[3] * v[3];
	v[0] += v[1];
	v[1] = -0.9012344421491146 * x[1];
	v[1] += x[0];
	v[3] = 0.6244470394316517 * x[2];
	v[1] += v[3];
	v[3] = -0.22431191631858705 * x[3];
	v[1] += v[3];
	v[3] = -0.2201317898900903 * x[4];
	v[1] += v[3];
	v[3] = 0.6210926180403503 * x[5];
	v[1] += v[3];
	v[3] = 0.121847072356 * x[6];
	v[2] = -v[3];
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[7] * v[3];
	v[1] += v[2];
	v[2] = 62.822177 - v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = -0.9753086105372786 * x[1];
	v[1] += x[0];
	v[2] = 0.902453771576314 * x[2];
	v[1] += v[2];
	v[2] = -0.7850332575231643 * x[3];
	v[1] += v[2];
	v[2] = 0.6288456196646282 * x[4];
	v[1] += v[2];
	v[2] = -0.44160383759196054 * x[5];
	v[1] += v[2];
	v[2] = 0.030461768089 * x[6];
	v[3] = -v[2];
	v[2] = exp(v[3]);
	if (errno) in_trouble("exp",v[3]);
	v[3] = x[7] * v[2];
	v[1] += v[3];
	v[3] = 77.719674 - v[1];
	v[1] = v[3] * v[3];
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
