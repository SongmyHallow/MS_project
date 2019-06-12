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
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.,
		-1.7e80,
		1000.};

 real x0comn_[100] = {
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
		0.5,
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
	real v[4];


  /***  objective ***/

	v[0] = -0.2 + x[0];
	v[1] = v[0] * v[0];
	v[0] = x[1] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[0] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -2.3265736762358173 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[2] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[1] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -2.571261565736173 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[3] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[2] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -2.841683505217273 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[4] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[3] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -3.1405459683413985 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[5] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[4] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -3.4708400710906373 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[6] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[5] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -3.8358715078609853 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[7] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[6] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -4.2392936359629445 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[8] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[7] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -4.685144039649417 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[9] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[8] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -5.1778849396159945 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[10] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[9] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -5.7224478524054785 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[11] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[10] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -6.32428294668298 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[12] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[11] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -6.989413590355792 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[13] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[12] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -7.724496634463918 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[14] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[13] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -8.536889037182739 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[15] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[14] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -9.434721494733182 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[16] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[15] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -10.426979816122316 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[17] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[16] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -11.523594856140145 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[18] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[17] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -12.735541906692216 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[19] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[18] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -14.074950541209919 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[20] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[19] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -15.5552260114983 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[21] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[20] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -17.19118341200177 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[22] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[21] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -18.99919595424884 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[23] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[22] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -20.997358835456318 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[24] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[23] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -23.205670341345076 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[25] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[24] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -25.646231995705165 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[26] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[25] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -28.34346975987453 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[27] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[26] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -31.32437849596988 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[28] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[27] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -34.618792140540094 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[29] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[28] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -38.25968229263073 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[30] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[29] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -42.28348820462931 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[31] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[30] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -46.730481478551 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[32] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[31] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -51.64516911776724 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[33] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[32] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -57.07673896805488 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[34] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[33] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -63.079552006089315 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[35] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[34] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -69.71368640237029 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[36] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[35] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -77.04553880374539 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[37] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[36] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -85.14848885336822 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[38] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[37] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -94.10363359883098 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[39] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[38] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -104.0005991386744 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[40] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[39] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -114.93843763050617 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[41] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[40] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -127.02661863828706 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[42] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[41] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -140.38612474052093 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[43] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[42] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -155.1506623645639 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[44] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[43] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -171.46799996548995 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[45] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[44] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -189.50144694245557 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[46] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[45] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -209.43148809405727 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[47] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[46] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -231.45758997085835 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[48] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[47] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -255.80019720367034 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[49] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[48] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -282.70293878751215 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[50] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[49] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -312.43506640247836 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[51] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[50] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -345.2941491750529 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[52] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[51] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -381.60905184994283 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[53] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[52] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -421.74322617897883 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[54] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[53] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -466.0983484684075 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[55] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[54] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -515.1183396903727 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[56] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[55] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -569.2938083932128 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[57] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[56] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -629.1669608767088 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[58] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[57] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -695.3370277749773 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[59] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[58] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -768.4662613580642 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[60] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[59] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -849.286563575252 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[61] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[60] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -938.6068111757731 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[62] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[61] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -1037.3209512191852 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[63] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[62] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -1146.4169479980112 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[64] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[63] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -1266.986670916444 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[65] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[64] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -1400.236822286334 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[66] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[65] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -1547.5010144095163 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[67] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[66] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -1710.2531168179612 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[68] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[67] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -1890.122007255444 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[69] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[68] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -2088.907874033485 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[70] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[69] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -2308.600232921034 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[71] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[70] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -2551.3978388869937 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[72] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[71] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -2819.730691978962 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[73] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[72] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -3116.2843575804727 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[74] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[73] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -3444.0268444519907 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[75] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[74] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -3806.23830956018 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[76] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[75] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -4206.543886991325 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[77] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[76] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -4648.9499695117065 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[78] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[77] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -5137.884305893007 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[79] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[78] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -5678.240315310237 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[80] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[79] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -6275.426062325569 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[81] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[80] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -6935.418382616192 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[82] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[81] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -7664.822701154664 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[83] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[82] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -8470.93914152217 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[84] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[83] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -9361.83558799899 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[85] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[84] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -10346.428431662116 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[86] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[85] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -11434.571808623996 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[87] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[86] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -12637.156223538901 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[88] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[87] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -13966.21754543388 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[89] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[88] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -15435.05746673138 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[90] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[89] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -17058.376631057898 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[91] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[90] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -18852.42176222642 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[92] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[91] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -20835.14826690909 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[93] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[92] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -23026.399938382176 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[94] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[93] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -25448.107559878867 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[95] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[94] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -28124.508395239136 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[96] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[95] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -31082.38876359268 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[97] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[96] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -34351.35212584394 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[98] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = x[97] / 10.;
	v[3] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = v[0] + v[3];
	v[0] = -37964.11536605882 + v[2];
	v[2] = v[0] * v[0];
	v[0] = 1.e-05 * v[2];
	v[1] += v[0];
	v[0] = x[99] / 10.;
	v[2] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = x[98] / 10.;
	v[3] = exp(v[0]);
	if (errno) in_trouble("exp",v[0]);
	v[0] = v[2] + v[3];
	v[2] = -41956.83623303702 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[1] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[2] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[3] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[4] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[5] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[6] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[7] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[8] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[9] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[10] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[11] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[12] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[13] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[14] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[15] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[16] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[17] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[18] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[19] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[20] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[21] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[22] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[23] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[24] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[25] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[26] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[27] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[28] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[29] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[30] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[31] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[32] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[33] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[34] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[35] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[36] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[37] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[38] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[39] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[40] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[41] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[42] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[43] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[44] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[45] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[46] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[47] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[48] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[49] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[50] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[51] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[52] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[53] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[54] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[55] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[56] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[57] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[58] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[59] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[60] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[61] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[62] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[63] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[64] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[65] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[66] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[67] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[68] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[69] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[70] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[71] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[72] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[73] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[74] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[75] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[76] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[77] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[78] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[79] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[80] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[81] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[82] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[83] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[84] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[85] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[86] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[87] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[88] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[89] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[90] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[91] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[92] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[93] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[94] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[95] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[96] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[97] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[98] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[99] / 10.;
	v[0] = exp(v[2]);
	if (errno) in_trouble("exp",v[2]);
	v[2] = -0.9048374180359595 + v[0];
	v[0] = v[2] * v[2];
	v[2] = 1.e-05 * v[0];
	v[1] += v[2];
	v[2] = x[0] * x[0];
	v[0] = 100. * v[2];
	v[2] = x[1] * x[1];
	v[3] = 99. * v[2];
	v[0] += v[3];
	v[3] = x[2] * x[2];
	v[2] = 98. * v[3];
	v[0] += v[2];
	v[2] = x[3] * x[3];
	v[3] = 97. * v[2];
	v[0] += v[3];
	v[3] = x[4] * x[4];
	v[2] = 96. * v[3];
	v[0] += v[2];
	v[2] = x[5] * x[5];
	v[3] = 95. * v[2];
	v[0] += v[3];
	v[3] = x[6] * x[6];
	v[2] = 94. * v[3];
	v[0] += v[2];
	v[2] = x[7] * x[7];
	v[3] = 93. * v[2];
	v[0] += v[3];
	v[3] = x[8] * x[8];
	v[2] = 92. * v[3];
	v[0] += v[2];
	v[2] = x[9] * x[9];
	v[3] = 91. * v[2];
	v[0] += v[3];
	v[3] = x[10] * x[10];
	v[2] = 90. * v[3];
	v[0] += v[2];
	v[2] = x[11] * x[11];
	v[3] = 89. * v[2];
	v[0] += v[3];
	v[3] = x[12] * x[12];
	v[2] = 88. * v[3];
	v[0] += v[2];
	v[2] = x[13] * x[13];
	v[3] = 87. * v[2];
	v[0] += v[3];
	v[3] = x[14] * x[14];
	v[2] = 86. * v[3];
	v[0] += v[2];
	v[2] = x[15] * x[15];
	v[3] = 85. * v[2];
	v[0] += v[3];
	v[3] = x[16] * x[16];
	v[2] = 84. * v[3];
	v[0] += v[2];
	v[2] = x[17] * x[17];
	v[3] = 83. * v[2];
	v[0] += v[3];
	v[3] = x[18] * x[18];
	v[2] = 82. * v[3];
	v[0] += v[2];
	v[2] = x[19] * x[19];
	v[3] = 81. * v[2];
	v[0] += v[3];
	v[3] = x[20] * x[20];
	v[2] = 80. * v[3];
	v[0] += v[2];
	v[2] = x[21] * x[21];
	v[3] = 79. * v[2];
	v[0] += v[3];
	v[3] = x[22] * x[22];
	v[2] = 78. * v[3];
	v[0] += v[2];
	v[2] = x[23] * x[23];
	v[3] = 77. * v[2];
	v[0] += v[3];
	v[3] = x[24] * x[24];
	v[2] = 76. * v[3];
	v[0] += v[2];
	v[2] = x[25] * x[25];
	v[3] = 75. * v[2];
	v[0] += v[3];
	v[3] = x[26] * x[26];
	v[2] = 74. * v[3];
	v[0] += v[2];
	v[2] = x[27] * x[27];
	v[3] = 73. * v[2];
	v[0] += v[3];
	v[3] = x[28] * x[28];
	v[2] = 72. * v[3];
	v[0] += v[2];
	v[2] = x[29] * x[29];
	v[3] = 71. * v[2];
	v[0] += v[3];
	v[3] = x[30] * x[30];
	v[2] = 70. * v[3];
	v[0] += v[2];
	v[2] = x[31] * x[31];
	v[3] = 69. * v[2];
	v[0] += v[3];
	v[3] = x[32] * x[32];
	v[2] = 68. * v[3];
	v[0] += v[2];
	v[2] = x[33] * x[33];
	v[3] = 67. * v[2];
	v[0] += v[3];
	v[3] = x[34] * x[34];
	v[2] = 66. * v[3];
	v[0] += v[2];
	v[2] = x[35] * x[35];
	v[3] = 65. * v[2];
	v[0] += v[3];
	v[3] = x[36] * x[36];
	v[2] = 64. * v[3];
	v[0] += v[2];
	v[2] = x[37] * x[37];
	v[3] = 63. * v[2];
	v[0] += v[3];
	v[3] = x[38] * x[38];
	v[2] = 62. * v[3];
	v[0] += v[2];
	v[2] = x[39] * x[39];
	v[3] = 61. * v[2];
	v[0] += v[3];
	v[3] = x[40] * x[40];
	v[2] = 60. * v[3];
	v[0] += v[2];
	v[2] = x[41] * x[41];
	v[3] = 59. * v[2];
	v[0] += v[3];
	v[3] = x[42] * x[42];
	v[2] = 58. * v[3];
	v[0] += v[2];
	v[2] = x[43] * x[43];
	v[3] = 57. * v[2];
	v[0] += v[3];
	v[3] = x[44] * x[44];
	v[2] = 56. * v[3];
	v[0] += v[2];
	v[2] = x[45] * x[45];
	v[3] = 55. * v[2];
	v[0] += v[3];
	v[3] = x[46] * x[46];
	v[2] = 54. * v[3];
	v[0] += v[2];
	v[2] = x[47] * x[47];
	v[3] = 53. * v[2];
	v[0] += v[3];
	v[3] = x[48] * x[48];
	v[2] = 52. * v[3];
	v[0] += v[2];
	v[2] = x[49] * x[49];
	v[3] = 51. * v[2];
	v[0] += v[3];
	v[3] = x[50] * x[50];
	v[2] = 50. * v[3];
	v[0] += v[2];
	v[2] = x[51] * x[51];
	v[3] = 49. * v[2];
	v[0] += v[3];
	v[3] = x[52] * x[52];
	v[2] = 48. * v[3];
	v[0] += v[2];
	v[2] = x[53] * x[53];
	v[3] = 47. * v[2];
	v[0] += v[3];
	v[3] = x[54] * x[54];
	v[2] = 46. * v[3];
	v[0] += v[2];
	v[2] = x[55] * x[55];
	v[3] = 45. * v[2];
	v[0] += v[3];
	v[3] = x[56] * x[56];
	v[2] = 44. * v[3];
	v[0] += v[2];
	v[2] = x[57] * x[57];
	v[3] = 43. * v[2];
	v[0] += v[3];
	v[3] = x[58] * x[58];
	v[2] = 42. * v[3];
	v[0] += v[2];
	v[2] = x[59] * x[59];
	v[3] = 41. * v[2];
	v[0] += v[3];
	v[3] = x[60] * x[60];
	v[2] = 40. * v[3];
	v[0] += v[2];
	v[2] = x[61] * x[61];
	v[3] = 39. * v[2];
	v[0] += v[3];
	v[3] = x[62] * x[62];
	v[2] = 38. * v[3];
	v[0] += v[2];
	v[2] = x[63] * x[63];
	v[3] = 37. * v[2];
	v[0] += v[3];
	v[3] = x[64] * x[64];
	v[2] = 36. * v[3];
	v[0] += v[2];
	v[2] = x[65] * x[65];
	v[3] = 35. * v[2];
	v[0] += v[3];
	v[3] = x[66] * x[66];
	v[2] = 34. * v[3];
	v[0] += v[2];
	v[2] = x[67] * x[67];
	v[3] = 33. * v[2];
	v[0] += v[3];
	v[3] = x[68] * x[68];
	v[2] = 32. * v[3];
	v[0] += v[2];
	v[2] = x[69] * x[69];
	v[3] = 31. * v[2];
	v[0] += v[3];
	v[3] = x[70] * x[70];
	v[2] = 30. * v[3];
	v[0] += v[2];
	v[2] = x[71] * x[71];
	v[3] = 29. * v[2];
	v[0] += v[3];
	v[3] = x[72] * x[72];
	v[2] = 28. * v[3];
	v[0] += v[2];
	v[2] = x[73] * x[73];
	v[3] = 27. * v[2];
	v[0] += v[3];
	v[3] = x[74] * x[74];
	v[2] = 26. * v[3];
	v[0] += v[2];
	v[2] = x[75] * x[75];
	v[3] = 25. * v[2];
	v[0] += v[3];
	v[3] = x[76] * x[76];
	v[2] = 24. * v[3];
	v[0] += v[2];
	v[2] = x[77] * x[77];
	v[3] = 23. * v[2];
	v[0] += v[3];
	v[3] = x[78] * x[78];
	v[2] = 22. * v[3];
	v[0] += v[2];
	v[2] = x[79] * x[79];
	v[3] = 21. * v[2];
	v[0] += v[3];
	v[3] = x[80] * x[80];
	v[2] = 20. * v[3];
	v[0] += v[2];
	v[2] = x[81] * x[81];
	v[3] = 19. * v[2];
	v[0] += v[3];
	v[3] = x[82] * x[82];
	v[2] = 18. * v[3];
	v[0] += v[2];
	v[2] = x[83] * x[83];
	v[3] = 17. * v[2];
	v[0] += v[3];
	v[3] = x[84] * x[84];
	v[2] = 16. * v[3];
	v[0] += v[2];
	v[2] = x[85] * x[85];
	v[3] = 15. * v[2];
	v[0] += v[3];
	v[3] = x[86] * x[86];
	v[2] = 14. * v[3];
	v[0] += v[2];
	v[2] = x[87] * x[87];
	v[3] = 13. * v[2];
	v[0] += v[3];
	v[3] = x[88] * x[88];
	v[2] = 12. * v[3];
	v[0] += v[2];
	v[2] = x[89] * x[89];
	v[3] = 11. * v[2];
	v[0] += v[3];
	v[3] = x[90] * x[90];
	v[2] = 10. * v[3];
	v[0] += v[2];
	v[2] = x[91] * x[91];
	v[3] = 9. * v[2];
	v[0] += v[3];
	v[3] = x[92] * x[92];
	v[2] = 8. * v[3];
	v[0] += v[2];
	v[2] = x[93] * x[93];
	v[3] = 7. * v[2];
	v[0] += v[3];
	v[3] = x[94] * x[94];
	v[2] = 6. * v[3];
	v[0] += v[2];
	v[2] = x[95] * x[95];
	v[3] = 5. * v[2];
	v[0] += v[3];
	v[3] = x[96] * x[96];
	v[2] = 4. * v[3];
	v[0] += v[2];
	v[2] = x[97] * x[97];
	v[3] = 3. * v[2];
	v[0] += v[3];
	v[3] = x[98] * x[98];
	v[2] = 2. * v[3];
	v[0] += v[2];
	v[2] = x[99] * x[99];
	v[0] += v[2];
	v[2] = -1. + v[0];
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
