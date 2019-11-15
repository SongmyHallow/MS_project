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
	30 /* nvar */,
	1 /* nobj */,
	0 /* ncon */,
	0 /* nzc */,
	0 /* densejac */,

	/* objtype (0 = minimize, 1 = maximize) */

	0 };

 real boundc_[1+60+0] /* Infinity, variable bounds, constraint bounds */ = {
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80,
		0.,
		1.7e80};

 real x0comn_[30] = {
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
		100.,
		140.,
		120.,
		20.,
		20.,
		200.,
		180.,
		20.,
		600.,
		40.,
		50.,
		30.,
		70.,
		150.,
		20. };

 real
feval0_(fint *nobj, real *x)
{
	real v[4];


  /***  objective ***/

	v[0] = 0.01 * x[15];
	v[1] = -1. + v[0];
	v[0] = 0.01 * x[15];
	v[2] = -1. + v[0];
	v[0] = v[1] * v[2];
	v[1] = 0.007143 * x[16];
	v[2] = -1. + v[1];
	v[1] = 0.007143 * x[16];
	v[3] = -1. + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 0.008333 * x[17];
	v[2] = -1. + v[1];
	v[1] = 0.008333 * x[17];
	v[3] = -1. + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 0.05 * x[18];
	v[2] = -1. + v[1];
	v[1] = 0.05 * x[18];
	v[3] = -1. + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 0.05 * x[19];
	v[2] = -1. + v[1];
	v[1] = 0.05 * x[19];
	v[3] = -1. + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 0.005 * x[20];
	v[2] = -1. + v[1];
	v[1] = 0.005 * x[20];
	v[3] = -1. + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 0.005556 * x[21];
	v[2] = -1. + v[1];
	v[1] = 0.005556 * x[21];
	v[3] = -1. + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 0.05 * x[22];
	v[2] = -1. + v[1];
	v[1] = 0.05 * x[22];
	v[3] = -1. + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 0.001667 * x[23];
	v[2] = -1. + v[1];
	v[1] = 0.001667 * x[23];
	v[3] = -1. + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 0.025 * x[24];
	v[2] = -1. + v[1];
	v[1] = 0.025 * x[24];
	v[3] = -1. + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 0.02 * x[25];
	v[2] = -1. + v[1];
	v[1] = 0.02 * x[25];
	v[3] = -1. + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 0.033333 * x[26];
	v[2] = -1. + v[1];
	v[1] = 0.033333 * x[26];
	v[3] = -1. + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 0.014286 * x[27];
	v[2] = -1. + v[1];
	v[1] = 0.014286 * x[27];
	v[3] = -1. + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 0.006667 * x[28];
	v[2] = -1. + v[1];
	v[1] = 0.006667 * x[28];
	v[3] = -1. + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 0.05 * x[29];
	v[2] = -1. + v[1];
	v[1] = 0.05 * x[29];
	v[3] = -1. + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 0.4 * x[0];
	v[2] = 0.4 * x[1];
	v[1] += v[2];
	v[2] = 0.4 * x[2];
	v[1] += v[2];
	v[2] = 0.4 * x[3];
	v[1] += v[2];
	v[2] = 0.4 * x[4];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = 0.4 * x[0];
	v[3] = 0.4 * x[1];
	v[1] += v[3];
	v[3] = 0.4 * x[2];
	v[1] += v[3];
	v[3] = 0.4 * x[3];
	v[1] += v[3];
	v[3] = 0.4 * x[4];
	v[1] += v[3];
	v[3] = -1. + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 0.4 * x[5];
	v[2] = 0.4 * x[6];
	v[1] += v[2];
	v[2] = 0.4 * x[7];
	v[1] += v[2];
	v[2] = 0.4 * x[8];
	v[1] += v[2];
	v[2] = 0.4 * x[9];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = 0.4 * x[5];
	v[3] = 0.4 * x[6];
	v[1] += v[3];
	v[3] = 0.4 * x[7];
	v[1] += v[3];
	v[3] = 0.4 * x[8];
	v[1] += v[3];
	v[3] = 0.4 * x[9];
	v[1] += v[3];
	v[3] = -1. + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 0.4 * x[10];
	v[2] = 0.4 * x[11];
	v[1] += v[2];
	v[2] = 0.4 * x[12];
	v[1] += v[2];
	v[2] = 0.4 * x[13];
	v[1] += v[2];
	v[2] = 0.4 * x[14];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = 0.4 * x[10];
	v[3] = 0.4 * x[11];
	v[1] += v[3];
	v[3] = 0.4 * x[12];
	v[1] += v[3];
	v[3] = 0.4 * x[13];
	v[1] += v[3];
	v[3] = 0.4 * x[14];
	v[1] += v[3];
	v[3] = -1. + v[1];
	v[1] = v[2] * v[3];
	v[0] += v[1];
	v[1] = 0.26373626373626374 * x[5];
	v[2] = 0.43956043956043955 * x[6];
	v[1] += v[2];
	v[2] = 0.46153846153846156 * x[7];
	v[1] += v[2];
	v[2] = 0.1978021978021978 * x[8];
	v[1] += v[2];
	v[2] = 0.3516483516483517 * x[9];
	v[1] += v[2];
	v[2] = 0.02197802197802198 * x[10];
	v[1] += v[2];
	v[2] = 0.04395604395604396 * x[12];
	v[1] += v[2];
	v[2] = 0.001098901098901099 * x[25];
	v[1] += v[2];
	v[2] = 0.001098901098901099 * x[28];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = 0.26373626373626374 * x[5];
	v[3] = 0.43956043956043955 * x[6];
	v[1] += v[3];
	v[3] = 0.46153846153846156 * x[7];
	v[1] += v[3];
	v[3] = 0.1978021978021978 * x[8];
	v[1] += v[3];
	v[3] = 0.3516483516483517 * x[9];
	v[1] += v[3];
	v[3] = 0.02197802197802198 * x[10];
	v[1] += v[3];
	v[3] = 0.04395604395604396 * x[12];
	v[1] += v[3];
	v[3] = 0.001098901098901099 * x[25];
	v[1] += v[3];
	v[3] = 0.001098901098901099 * x[28];
	v[1] += v[3];
	v[3] = -1. + v[1];
	v[1] = v[2] * v[3];
	v[2] = 10000. * v[1];
	v[0] += v[2];
	v[2] = 0.11428571428571428 * x[10];
	v[1] = 0.34285714285714286 * x[11];
	v[2] += v[1];
	v[1] = 0.22857142857142856 * x[12];
	v[2] += v[1];
	v[1] = 0.005714285714285714 * x[25];
	v[2] += v[1];
	v[1] = 0.005714285714285714 * x[26];
	v[2] += v[1];
	v[1] = 0.005714285714285714 * x[28];
	v[2] += v[1];
	v[1] = -1. + v[2];
	v[2] = 0.11428571428571428 * x[10];
	v[3] = 0.34285714285714286 * x[11];
	v[2] += v[3];
	v[3] = 0.22857142857142856 * x[12];
	v[2] += v[3];
	v[3] = 0.005714285714285714 * x[25];
	v[2] += v[3];
	v[3] = 0.005714285714285714 * x[26];
	v[2] += v[3];
	v[3] = 0.005714285714285714 * x[28];
	v[2] += v[3];
	v[3] = -1. + v[2];
	v[2] = v[1] * v[3];
	v[1] = 10000. * v[2];
	v[0] += v[1];
	v[1] = 0.10443864229765012 * x[0];
	v[2] = 0.2506527415143603 * x[1];
	v[1] += v[2];
	v[2] = 0.06266318537859007 * x[2];
	v[1] += v[2];
	v[2] = 0.18798955613577023 * x[3];
	v[1] += v[2];
	v[2] = 0.2924281984334204 * x[4];
	v[1] += v[2];
	v[2] = 0.20887728459530025 * x[6];
	v[1] += v[2];
	v[2] = 0.2193211488250653 * x[7];
	v[1] += v[2];
	v[2] = 0.09399477806788512 * x[8];
	v[1] += v[2];
	v[2] = 0.1671018276762402 * x[9];
	v[1] += v[2];
	v[2] = 0.020887728459530026 * x[12];
	v[1] += v[2];
	v[2] = 0.0005221932114882506 * x[23];
	v[1] += v[2];
	v[2] = 0.0005221932114882506 * x[28];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = 0.10443864229765012 * x[0];
	v[3] = 0.2506527415143603 * x[1];
	v[1] += v[3];
	v[3] = 0.06266318537859007 * x[2];
	v[1] += v[3];
	v[3] = 0.18798955613577023 * x[3];
	v[1] += v[3];
	v[3] = 0.2924281984334204 * x[4];
	v[1] += v[3];
	v[3] = 0.20887728459530025 * x[6];
	v[1] += v[3];
	v[3] = 0.2193211488250653 * x[7];
	v[1] += v[3];
	v[3] = 0.09399477806788512 * x[8];
	v[1] += v[3];
	v[3] = 0.1671018276762402 * x[9];
	v[1] += v[3];
	v[3] = 0.020887728459530026 * x[12];
	v[1] += v[3];
	v[3] = 0.0005221932114882506 * x[23];
	v[1] += v[3];
	v[3] = 0.0005221932114882506 * x[28];
	v[1] += v[3];
	v[3] = -1. + v[1];
	v[1] = v[2] * v[3];
	v[2] = 10000. * v[1];
	v[0] += v[2];
	v[2] = 0.0022222222222222222 * x[15];
	v[1] = 0.0022222222222222222 * x[20];
	v[2] += v[1];
	v[1] = 0.0022222222222222222 * x[23];
	v[2] += v[1];
	v[1] = -1. + v[2];
	v[2] = 0.0022222222222222222 * x[15];
	v[3] = 0.0022222222222222222 * x[20];
	v[2] += v[3];
	v[3] = 0.0022222222222222222 * x[23];
	v[2] += v[3];
	v[3] = -1. + v[2];
	v[2] = v[1] * v[3];
	v[1] = 10000. * v[2];
	v[0] += v[1];
	v[1] = 0.7692307692307693 * x[0];
	v[2] = 0.0038461538461538464 * x[16];
	v[1] += v[2];
	v[2] = 0.0038461538461538464 * x[21];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = 0.7692307692307693 * x[0];
	v[3] = 0.0038461538461538464 * x[16];
	v[1] += v[3];
	v[3] = 0.0038461538461538464 * x[21];
	v[1] += v[3];
	v[3] = -1. + v[1];
	v[1] = v[2] * v[3];
	v[2] = 10000. * v[1];
	v[0] += v[2];
	v[2] = 1.5 * x[13];
	v[1] = 0.25 * x[14];
	v[2] += v[1];
	v[1] = 0.0125 * x[29];
	v[2] += v[1];
	v[1] = -1. + v[2];
	v[2] = 1.5 * x[13];
	v[3] = 0.25 * x[14];
	v[2] += v[3];
	v[3] = 0.0125 * x[29];
	v[2] += v[3];
	v[3] = -1. + v[2];
	v[2] = v[1] * v[3];
	v[1] = 10000. * v[2];
	v[0] += v[1];
	v[1] = 0.7164179104477612 * x[1];
	v[2] = 0.5970149253731343 * x[6];
	v[1] += v[2];
	v[2] = 0.0014925373134328358 * x[17];
	v[1] += v[2];
	v[2] = 0.0014925373134328358 * x[22];
	v[1] += v[2];
	v[2] = 0.0014925373134328358 * x[25];
	v[1] += v[2];
	v[2] = 0.0014925373134328358 * x[26];
	v[1] += v[2];
	v[2] = 0.0014925373134328358 * x[27];
	v[1] += v[2];
	v[2] = 0.0014925373134328358 * x[28];
	v[1] += v[2];
	v[2] = 0.0014925373134328358 * x[29];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = 0.7164179104477612 * x[1];
	v[3] = 0.5970149253731343 * x[6];
	v[1] += v[3];
	v[3] = 0.0014925373134328358 * x[17];
	v[1] += v[3];
	v[3] = 0.0014925373134328358 * x[22];
	v[1] += v[3];
	v[3] = 0.0014925373134328358 * x[25];
	v[1] += v[3];
	v[3] = 0.0014925373134328358 * x[26];
	v[1] += v[3];
	v[3] = 0.0014925373134328358 * x[27];
	v[1] += v[3];
	v[3] = 0.0014925373134328358 * x[28];
	v[1] += v[3];
	v[3] = 0.0014925373134328358 * x[29];
	v[1] += v[3];
	v[3] = -1. + v[1];
	v[1] = v[2] * v[3];
	v[2] = 10000. * v[1];
	v[0] += v[2];
	v[2] = 0.13793103448275862 * x[0];
	v[1] = 0.3310344827586207 * x[1];
	v[2] += v[1];
	v[1] = 0.2482758620689655 * x[3];
	v[2] += v[1];
	v[1] = 0.38620689655172413 * x[4];
	v[2] += v[1];
	v[1] = 0.27586206896551724 * x[6];
	v[2] += v[1];
	v[1] = 0.12413793103448276 * x[8];
	v[2] += v[1];
	v[1] = 0.2206896551724138 * x[9];
	v[2] += v[1];
	v[1] = 0.000689655172413793 * x[15];
	v[2] += v[1];
	v[1] = 0.000689655172413793 * x[16];
	v[2] += v[1];
	v[1] = 0.000689655172413793 * x[17];
	v[2] += v[1];
	v[1] = 0.000689655172413793 * x[18];
	v[2] += v[1];
	v[1] = 0.000689655172413793 * x[19];
	v[2] += v[1];
	v[1] = -1. + v[2];
	v[2] = 0.13793103448275862 * x[0];
	v[3] = 0.3310344827586207 * x[1];
	v[2] += v[3];
	v[3] = 0.2482758620689655 * x[3];
	v[2] += v[3];
	v[3] = 0.38620689655172413 * x[4];
	v[2] += v[3];
	v[3] = 0.27586206896551724 * x[6];
	v[2] += v[3];
	v[3] = 0.12413793103448276 * x[8];
	v[2] += v[3];
	v[3] = 0.2206896551724138 * x[9];
	v[2] += v[3];
	v[3] = 0.000689655172413793 * x[15];
	v[2] += v[3];
	v[3] = 0.000689655172413793 * x[16];
	v[2] += v[3];
	v[3] = 0.000689655172413793 * x[17];
	v[2] += v[3];
	v[3] = 0.000689655172413793 * x[18];
	v[2] += v[3];
	v[3] = 0.000689655172413793 * x[19];
	v[2] += v[3];
	v[3] = -1. + v[2];
	v[2] = v[1] * v[3];
	v[1] = 10000. * v[2];
	v[0] += v[1];
	v[1] = 0.48484848484848486 * x[1];
	v[2] = 0.5656565656565656 * x[4];
	v[1] += v[2];
	v[2] = 0.40404040404040403 * x[6];
	v[1] += v[2];
	v[2] = 0.32323232323232326 * x[9];
	v[1] += v[2];
	v[2] = 0.020202020202020204 * x[14];
	v[1] += v[2];
	v[2] = 0.00101010101010101 * x[17];
	v[1] += v[2];
	v[2] = 0.00101010101010101 * x[19];
	v[1] += v[2];
	v[2] = 0.00101010101010101 * x[22];
	v[1] += v[2];
	v[2] = 0.00101010101010101 * x[24];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = 0.48484848484848486 * x[1];
	v[3] = 0.5656565656565656 * x[4];
	v[1] += v[3];
	v[3] = 0.40404040404040403 * x[6];
	v[1] += v[3];
	v[3] = 0.32323232323232326 * x[9];
	v[1] += v[3];
	v[3] = 0.020202020202020204 * x[14];
	v[1] += v[3];
	v[3] = 0.00101010101010101 * x[17];
	v[1] += v[3];
	v[3] = 0.00101010101010101 * x[19];
	v[1] += v[3];
	v[3] = 0.00101010101010101 * x[22];
	v[1] += v[3];
	v[3] = 0.00101010101010101 * x[24];
	v[1] += v[3];
	v[3] = -1. + v[1];
	v[1] = v[2] * v[3];
	v[2] = 10000. * v[1];
	v[0] += v[2];
	v[2] = -0.8 * x[0];
	v[1] = 0.2 * x[1];
	v[2] += v[1];
	v[1] = 0.2 * x[2];
	v[2] += v[1];
	v[1] = 0.2 * x[3];
	v[2] += v[1];
	v[1] = 0.2 * x[4];
	v[2] += v[1];
	v[1] = -0.8 * x[0];
	v[3] = 0.2 * x[1];
	v[1] += v[3];
	v[3] = 0.2 * x[2];
	v[1] += v[3];
	v[3] = 0.2 * x[3];
	v[1] += v[3];
	v[3] = 0.2 * x[4];
	v[1] += v[3];
	v[3] = v[2] * v[1];
	v[2] = 2. * v[3];
	v[0] += v[2];
	v[2] = 0.2 * x[0];
	v[3] = -0.8 * x[1];
	v[2] += v[3];
	v[3] = 0.2 * x[2];
	v[2] += v[3];
	v[3] = 0.2 * x[3];
	v[2] += v[3];
	v[3] = 0.2 * x[4];
	v[2] += v[3];
	v[3] = 0.2 * x[0];
	v[1] = -0.8 * x[1];
	v[3] += v[1];
	v[1] = 0.2 * x[2];
	v[3] += v[1];
	v[1] = 0.2 * x[3];
	v[3] += v[1];
	v[1] = 0.2 * x[4];
	v[3] += v[1];
	v[1] = v[2] * v[3];
	v[2] = 2. * v[1];
	v[0] += v[2];
	v[2] = 0.2 * x[0];
	v[1] = 0.2 * x[1];
	v[2] += v[1];
	v[1] = -0.8 * x[2];
	v[2] += v[1];
	v[1] = 0.2 * x[3];
	v[2] += v[1];
	v[1] = 0.2 * x[4];
	v[2] += v[1];
	v[1] = 0.2 * x[0];
	v[3] = 0.2 * x[1];
	v[1] += v[3];
	v[3] = -0.8 * x[2];
	v[1] += v[3];
	v[3] = 0.2 * x[3];
	v[1] += v[3];
	v[3] = 0.2 * x[4];
	v[1] += v[3];
	v[3] = v[2] * v[1];
	v[2] = 2. * v[3];
	v[0] += v[2];
	v[2] = 0.2 * x[0];
	v[3] = 0.2 * x[1];
	v[2] += v[3];
	v[3] = 0.2 * x[2];
	v[2] += v[3];
	v[3] = -0.8 * x[3];
	v[2] += v[3];
	v[3] = 0.2 * x[4];
	v[2] += v[3];
	v[3] = 0.2 * x[0];
	v[1] = 0.2 * x[1];
	v[3] += v[1];
	v[1] = 0.2 * x[2];
	v[3] += v[1];
	v[1] = -0.8 * x[3];
	v[3] += v[1];
	v[1] = 0.2 * x[4];
	v[3] += v[1];
	v[1] = v[2] * v[3];
	v[2] = 2. * v[1];
	v[0] += v[2];
	v[2] = 0.2 * x[0];
	v[1] = 0.2 * x[1];
	v[2] += v[1];
	v[1] = 0.2 * x[2];
	v[2] += v[1];
	v[1] = 0.2 * x[3];
	v[2] += v[1];
	v[1] = -0.8 * x[4];
	v[2] += v[1];
	v[1] = 0.2 * x[0];
	v[3] = 0.2 * x[1];
	v[1] += v[3];
	v[3] = 0.2 * x[2];
	v[1] += v[3];
	v[3] = 0.2 * x[3];
	v[1] += v[3];
	v[3] = -0.8 * x[4];
	v[1] += v[3];
	v[3] = v[2] * v[1];
	v[2] = 2. * v[3];
	v[0] += v[2];
	v[2] = -0.8 * x[5];
	v[3] = 0.2 * x[6];
	v[2] += v[3];
	v[3] = 0.2 * x[7];
	v[2] += v[3];
	v[3] = 0.2 * x[8];
	v[2] += v[3];
	v[3] = 0.2 * x[9];
	v[2] += v[3];
	v[3] = -0.8 * x[5];
	v[1] = 0.2 * x[6];
	v[3] += v[1];
	v[1] = 0.2 * x[7];
	v[3] += v[1];
	v[1] = 0.2 * x[8];
	v[3] += v[1];
	v[1] = 0.2 * x[9];
	v[3] += v[1];
	v[1] = v[2] * v[3];
	v[2] = 2. * v[1];
	v[0] += v[2];
	v[2] = 0.2 * x[5];
	v[1] = -0.8 * x[6];
	v[2] += v[1];
	v[1] = 0.2 * x[7];
	v[2] += v[1];
	v[1] = 0.2 * x[8];
	v[2] += v[1];
	v[1] = 0.2 * x[9];
	v[2] += v[1];
	v[1] = 0.2 * x[5];
	v[3] = -0.8 * x[6];
	v[1] += v[3];
	v[3] = 0.2 * x[7];
	v[1] += v[3];
	v[3] = 0.2 * x[8];
	v[1] += v[3];
	v[3] = 0.2 * x[9];
	v[1] += v[3];
	v[3] = v[2] * v[1];
	v[2] = 2. * v[3];
	v[0] += v[2];
	v[2] = 0.2 * x[5];
	v[3] = 0.2 * x[6];
	v[2] += v[3];
	v[3] = -0.8 * x[7];
	v[2] += v[3];
	v[3] = 0.2 * x[8];
	v[2] += v[3];
	v[3] = 0.2 * x[9];
	v[2] += v[3];
	v[3] = 0.2 * x[5];
	v[1] = 0.2 * x[6];
	v[3] += v[1];
	v[1] = -0.8 * x[7];
	v[3] += v[1];
	v[1] = 0.2 * x[8];
	v[3] += v[1];
	v[1] = 0.2 * x[9];
	v[3] += v[1];
	v[1] = v[2] * v[3];
	v[2] = 2. * v[1];
	v[0] += v[2];
	v[2] = 0.2 * x[5];
	v[1] = 0.2 * x[6];
	v[2] += v[1];
	v[1] = 0.2 * x[7];
	v[2] += v[1];
	v[1] = -0.8 * x[8];
	v[2] += v[1];
	v[1] = 0.2 * x[9];
	v[2] += v[1];
	v[1] = 0.2 * x[5];
	v[3] = 0.2 * x[6];
	v[1] += v[3];
	v[3] = 0.2 * x[7];
	v[1] += v[3];
	v[3] = -0.8 * x[8];
	v[1] += v[3];
	v[3] = 0.2 * x[9];
	v[1] += v[3];
	v[3] = v[2] * v[1];
	v[2] = 2. * v[3];
	v[0] += v[2];
	v[2] = 0.2 * x[5];
	v[3] = 0.2 * x[6];
	v[2] += v[3];
	v[3] = 0.2 * x[7];
	v[2] += v[3];
	v[3] = 0.2 * x[8];
	v[2] += v[3];
	v[3] = -0.8 * x[9];
	v[2] += v[3];
	v[3] = 0.2 * x[5];
	v[1] = 0.2 * x[6];
	v[3] += v[1];
	v[1] = 0.2 * x[7];
	v[3] += v[1];
	v[1] = 0.2 * x[8];
	v[3] += v[1];
	v[1] = -0.8 * x[9];
	v[3] += v[1];
	v[1] = v[2] * v[3];
	v[2] = 2. * v[1];
	v[0] += v[2];
	v[2] = -0.8 * x[10];
	v[1] = 0.2 * x[11];
	v[2] += v[1];
	v[1] = 0.2 * x[12];
	v[2] += v[1];
	v[1] = 0.2 * x[13];
	v[2] += v[1];
	v[1] = 0.2 * x[14];
	v[2] += v[1];
	v[1] = -0.8 * x[10];
	v[3] = 0.2 * x[11];
	v[1] += v[3];
	v[3] = 0.2 * x[12];
	v[1] += v[3];
	v[3] = 0.2 * x[13];
	v[1] += v[3];
	v[3] = 0.2 * x[14];
	v[1] += v[3];
	v[3] = v[2] * v[1];
	v[2] = 2. * v[3];
	v[0] += v[2];
	v[2] = 0.2 * x[10];
	v[3] = -0.8 * x[11];
	v[2] += v[3];
	v[3] = 0.2 * x[12];
	v[2] += v[3];
	v[3] = 0.2 * x[13];
	v[2] += v[3];
	v[3] = 0.2 * x[14];
	v[2] += v[3];
	v[3] = 0.2 * x[10];
	v[1] = -0.8 * x[11];
	v[3] += v[1];
	v[1] = 0.2 * x[12];
	v[3] += v[1];
	v[1] = 0.2 * x[13];
	v[3] += v[1];
	v[1] = 0.2 * x[14];
	v[3] += v[1];
	v[1] = v[2] * v[3];
	v[2] = 2. * v[1];
	v[0] += v[2];
	v[2] = 0.2 * x[10];
	v[1] = 0.2 * x[11];
	v[2] += v[1];
	v[1] = -0.8 * x[12];
	v[2] += v[1];
	v[1] = 0.2 * x[13];
	v[2] += v[1];
	v[1] = 0.2 * x[14];
	v[2] += v[1];
	v[1] = 0.2 * x[10];
	v[3] = 0.2 * x[11];
	v[1] += v[3];
	v[3] = -0.8 * x[12];
	v[1] += v[3];
	v[3] = 0.2 * x[13];
	v[1] += v[3];
	v[3] = 0.2 * x[14];
	v[1] += v[3];
	v[3] = v[2] * v[1];
	v[2] = 2. * v[3];
	v[0] += v[2];
	v[2] = 0.2 * x[10];
	v[3] = 0.2 * x[11];
	v[2] += v[3];
	v[3] = 0.2 * x[12];
	v[2] += v[3];
	v[3] = -0.8 * x[13];
	v[2] += v[3];
	v[3] = 0.2 * x[14];
	v[2] += v[3];
	v[3] = 0.2 * x[10];
	v[1] = 0.2 * x[11];
	v[3] += v[1];
	v[1] = 0.2 * x[12];
	v[3] += v[1];
	v[1] = -0.8 * x[13];
	v[3] += v[1];
	v[1] = 0.2 * x[14];
	v[3] += v[1];
	v[1] = v[2] * v[3];
	v[2] = 2. * v[1];
	v[0] += v[2];
	v[2] = 0.2 * x[10];
	v[1] = 0.2 * x[11];
	v[2] += v[1];
	v[1] = 0.2 * x[12];
	v[2] += v[1];
	v[1] = 0.2 * x[13];
	v[2] += v[1];
	v[1] = -0.8 * x[14];
	v[2] += v[1];
	v[1] = 0.2 * x[10];
	v[3] = 0.2 * x[11];
	v[1] += v[3];
	v[3] = 0.2 * x[12];
	v[1] += v[3];
	v[3] = 0.2 * x[13];
	v[1] += v[3];
	v[3] = -0.8 * x[14];
	v[1] += v[3];
	v[3] = v[2] * v[1];
	v[2] = 2. * v[3];
	v[0] += v[2];

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

x_input = malloc (30 * sizeof(real));
input_values = malloc (30 * sizeof(double));
c_val = malloc (0 * sizeof(real));

file_input = fopen("input.in","r");
for (i=0; i < 30; i++)
    fscanf(file_input, "%lf" ,&input_values[i]);

fclose(file_input);
for (i=0; i < 30; i++)
 {
    x_input[i] = input_values[i];
 }

f_val = feval0_(&objective_number, x_input);

FILE *output_out;
output_out = fopen ("output.out","w");
fprintf(output_out,"%30.15f\n",f_val);
fclose(output_out);
}
