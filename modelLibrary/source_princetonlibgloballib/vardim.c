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
		0.99,
		0.98,
		0.97,
		0.96,
		0.95,
		0.94,
		0.9299999999999999,
		0.92,
		0.91,
		0.9,
		0.89,
		0.88,
		0.87,
		0.86,
		0.85,
		0.84,
		0.83,
		0.8200000000000001,
		0.81,
		0.8,
		0.79,
		0.78,
		0.77,
		0.76,
		0.75,
		0.74,
		0.73,
		0.72,
		0.71,
		0.7,
		0.69,
		0.6799999999999999,
		0.6699999999999999,
		0.6599999999999999,
		0.65,
		0.64,
		0.63,
		0.62,
		0.61,
		0.6,
		0.5900000000000001,
		0.5800000000000001,
		0.5700000000000001,
		0.56,
		0.55,
		0.54,
		0.53,
		0.52,
		0.51,
		0.5,
		0.49,
		0.48,
		0.47,
		0.45999999999999996,
		0.44999999999999996,
		0.43999999999999995,
		0.43000000000000005,
		0.42000000000000004,
		0.41000000000000003,
		0.4,
		0.39,
		0.38,
		0.37,
		0.36,
		0.35,
		0.33999999999999997,
		0.32999999999999996,
		0.31999999999999995,
		0.31000000000000005,
		0.30000000000000004,
		0.29000000000000004,
		0.28,
		0.27,
		0.26,
		0.25,
		0.24,
		0.22999999999999998,
		0.21999999999999997,
		0.20999999999999996,
		0.19999999999999996,
		0.18999999999999995,
		0.18000000000000005,
		0.17000000000000004,
		0.16000000000000003,
		0.15000000000000002,
		0.14,
		0.13,
		0.12,
		0.10999999999999999,
		0.09999999999999998,
		0.08999999999999997,
		0.07999999999999996,
		0.06999999999999995,
		0.06000000000000005,
		0.050000000000000044,
		0.040000000000000036,
		0.030000000000000027,
		0.020000000000000018,
		0.010000000000000009,
		0. };

 real
feval0_(fint *nobj, real *x)
{
	real v[3];


  /***  objective ***/

	v[0] = -1. + x[0];
	v[1] = v[0] * v[0];
	v[0] = -1. + x[1];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[2];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[3];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[4];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[5];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[6];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[7];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[8];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[9];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[10];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[11];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[12];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[13];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[14];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[15];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[16];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[17];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[18];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[19];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[20];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[21];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[22];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[23];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[24];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[25];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[26];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[27];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[28];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[29];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[30];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[31];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[32];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[33];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[34];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[35];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[36];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[37];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[38];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[39];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[40];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[41];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[42];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[43];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[44];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[45];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[46];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[47];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[48];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[49];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[50];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[51];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[52];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[53];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[54];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[55];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[56];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[57];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[58];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[59];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[60];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[61];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[62];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[63];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[64];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[65];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[66];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[67];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[68];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[69];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[70];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[71];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[72];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[73];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[74];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[75];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[76];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[77];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[78];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[79];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[80];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[81];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[82];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[83];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[84];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[85];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[86];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[87];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[88];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[89];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[90];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[91];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[92];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[93];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[94];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[95];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[96];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[97];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = -1. + x[98];
	v[0] = v[2] * v[2];
	v[1] += v[0];
	v[0] = -1. + x[99];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = 2. * x[1];
	v[2] += x[0];
	v[0] = 3. * x[2];
	v[2] += v[0];
	v[0] = 4. * x[3];
	v[2] += v[0];
	v[0] = 5. * x[4];
	v[2] += v[0];
	v[0] = 6. * x[5];
	v[2] += v[0];
	v[0] = 7. * x[6];
	v[2] += v[0];
	v[0] = 8. * x[7];
	v[2] += v[0];
	v[0] = 9. * x[8];
	v[2] += v[0];
	v[0] = 10. * x[9];
	v[2] += v[0];
	v[0] = 11. * x[10];
	v[2] += v[0];
	v[0] = 12. * x[11];
	v[2] += v[0];
	v[0] = 13. * x[12];
	v[2] += v[0];
	v[0] = 14. * x[13];
	v[2] += v[0];
	v[0] = 15. * x[14];
	v[2] += v[0];
	v[0] = 16. * x[15];
	v[2] += v[0];
	v[0] = 17. * x[16];
	v[2] += v[0];
	v[0] = 18. * x[17];
	v[2] += v[0];
	v[0] = 19. * x[18];
	v[2] += v[0];
	v[0] = 20. * x[19];
	v[2] += v[0];
	v[0] = 21. * x[20];
	v[2] += v[0];
	v[0] = 22. * x[21];
	v[2] += v[0];
	v[0] = 23. * x[22];
	v[2] += v[0];
	v[0] = 24. * x[23];
	v[2] += v[0];
	v[0] = 25. * x[24];
	v[2] += v[0];
	v[0] = 26. * x[25];
	v[2] += v[0];
	v[0] = 27. * x[26];
	v[2] += v[0];
	v[0] = 28. * x[27];
	v[2] += v[0];
	v[0] = 29. * x[28];
	v[2] += v[0];
	v[0] = 30. * x[29];
	v[2] += v[0];
	v[0] = 31. * x[30];
	v[2] += v[0];
	v[0] = 32. * x[31];
	v[2] += v[0];
	v[0] = 33. * x[32];
	v[2] += v[0];
	v[0] = 34. * x[33];
	v[2] += v[0];
	v[0] = 35. * x[34];
	v[2] += v[0];
	v[0] = 36. * x[35];
	v[2] += v[0];
	v[0] = 37. * x[36];
	v[2] += v[0];
	v[0] = 38. * x[37];
	v[2] += v[0];
	v[0] = 39. * x[38];
	v[2] += v[0];
	v[0] = 40. * x[39];
	v[2] += v[0];
	v[0] = 41. * x[40];
	v[2] += v[0];
	v[0] = 42. * x[41];
	v[2] += v[0];
	v[0] = 43. * x[42];
	v[2] += v[0];
	v[0] = 44. * x[43];
	v[2] += v[0];
	v[0] = 45. * x[44];
	v[2] += v[0];
	v[0] = 46. * x[45];
	v[2] += v[0];
	v[0] = 47. * x[46];
	v[2] += v[0];
	v[0] = 48. * x[47];
	v[2] += v[0];
	v[0] = 49. * x[48];
	v[2] += v[0];
	v[0] = 50. * x[49];
	v[2] += v[0];
	v[0] = 51. * x[50];
	v[2] += v[0];
	v[0] = 52. * x[51];
	v[2] += v[0];
	v[0] = 53. * x[52];
	v[2] += v[0];
	v[0] = 54. * x[53];
	v[2] += v[0];
	v[0] = 55. * x[54];
	v[2] += v[0];
	v[0] = 56. * x[55];
	v[2] += v[0];
	v[0] = 57. * x[56];
	v[2] += v[0];
	v[0] = 58. * x[57];
	v[2] += v[0];
	v[0] = 59. * x[58];
	v[2] += v[0];
	v[0] = 60. * x[59];
	v[2] += v[0];
	v[0] = 61. * x[60];
	v[2] += v[0];
	v[0] = 62. * x[61];
	v[2] += v[0];
	v[0] = 63. * x[62];
	v[2] += v[0];
	v[0] = 64. * x[63];
	v[2] += v[0];
	v[0] = 65. * x[64];
	v[2] += v[0];
	v[0] = 66. * x[65];
	v[2] += v[0];
	v[0] = 67. * x[66];
	v[2] += v[0];
	v[0] = 68. * x[67];
	v[2] += v[0];
	v[0] = 69. * x[68];
	v[2] += v[0];
	v[0] = 70. * x[69];
	v[2] += v[0];
	v[0] = 71. * x[70];
	v[2] += v[0];
	v[0] = 72. * x[71];
	v[2] += v[0];
	v[0] = 73. * x[72];
	v[2] += v[0];
	v[0] = 74. * x[73];
	v[2] += v[0];
	v[0] = 75. * x[74];
	v[2] += v[0];
	v[0] = 76. * x[75];
	v[2] += v[0];
	v[0] = 77. * x[76];
	v[2] += v[0];
	v[0] = 78. * x[77];
	v[2] += v[0];
	v[0] = 79. * x[78];
	v[2] += v[0];
	v[0] = 80. * x[79];
	v[2] += v[0];
	v[0] = 81. * x[80];
	v[2] += v[0];
	v[0] = 82. * x[81];
	v[2] += v[0];
	v[0] = 83. * x[82];
	v[2] += v[0];
	v[0] = 84. * x[83];
	v[2] += v[0];
	v[0] = 85. * x[84];
	v[2] += v[0];
	v[0] = 86. * x[85];
	v[2] += v[0];
	v[0] = 87. * x[86];
	v[2] += v[0];
	v[0] = 88. * x[87];
	v[2] += v[0];
	v[0] = 89. * x[88];
	v[2] += v[0];
	v[0] = 90. * x[89];
	v[2] += v[0];
	v[0] = 91. * x[90];
	v[2] += v[0];
	v[0] = 92. * x[91];
	v[2] += v[0];
	v[0] = 93. * x[92];
	v[2] += v[0];
	v[0] = 94. * x[93];
	v[2] += v[0];
	v[0] = 95. * x[94];
	v[2] += v[0];
	v[0] = 96. * x[95];
	v[2] += v[0];
	v[0] = 97. * x[96];
	v[2] += v[0];
	v[0] = 98. * x[97];
	v[2] += v[0];
	v[0] = 99. * x[98];
	v[2] += v[0];
	v[0] = 100. * x[99];
	v[2] += v[0];
	v[0] = -5050. + v[2];
	v[2] = v[0] * v[0];
	v[1] += v[2];
	v[2] = 2. * x[1];
	v[2] += x[0];
	v[0] = 3. * x[2];
	v[2] += v[0];
	v[0] = 4. * x[3];
	v[2] += v[0];
	v[0] = 5. * x[4];
	v[2] += v[0];
	v[0] = 6. * x[5];
	v[2] += v[0];
	v[0] = 7. * x[6];
	v[2] += v[0];
	v[0] = 8. * x[7];
	v[2] += v[0];
	v[0] = 9. * x[8];
	v[2] += v[0];
	v[0] = 10. * x[9];
	v[2] += v[0];
	v[0] = 11. * x[10];
	v[2] += v[0];
	v[0] = 12. * x[11];
	v[2] += v[0];
	v[0] = 13. * x[12];
	v[2] += v[0];
	v[0] = 14. * x[13];
	v[2] += v[0];
	v[0] = 15. * x[14];
	v[2] += v[0];
	v[0] = 16. * x[15];
	v[2] += v[0];
	v[0] = 17. * x[16];
	v[2] += v[0];
	v[0] = 18. * x[17];
	v[2] += v[0];
	v[0] = 19. * x[18];
	v[2] += v[0];
	v[0] = 20. * x[19];
	v[2] += v[0];
	v[0] = 21. * x[20];
	v[2] += v[0];
	v[0] = 22. * x[21];
	v[2] += v[0];
	v[0] = 23. * x[22];
	v[2] += v[0];
	v[0] = 24. * x[23];
	v[2] += v[0];
	v[0] = 25. * x[24];
	v[2] += v[0];
	v[0] = 26. * x[25];
	v[2] += v[0];
	v[0] = 27. * x[26];
	v[2] += v[0];
	v[0] = 28. * x[27];
	v[2] += v[0];
	v[0] = 29. * x[28];
	v[2] += v[0];
	v[0] = 30. * x[29];
	v[2] += v[0];
	v[0] = 31. * x[30];
	v[2] += v[0];
	v[0] = 32. * x[31];
	v[2] += v[0];
	v[0] = 33. * x[32];
	v[2] += v[0];
	v[0] = 34. * x[33];
	v[2] += v[0];
	v[0] = 35. * x[34];
	v[2] += v[0];
	v[0] = 36. * x[35];
	v[2] += v[0];
	v[0] = 37. * x[36];
	v[2] += v[0];
	v[0] = 38. * x[37];
	v[2] += v[0];
	v[0] = 39. * x[38];
	v[2] += v[0];
	v[0] = 40. * x[39];
	v[2] += v[0];
	v[0] = 41. * x[40];
	v[2] += v[0];
	v[0] = 42. * x[41];
	v[2] += v[0];
	v[0] = 43. * x[42];
	v[2] += v[0];
	v[0] = 44. * x[43];
	v[2] += v[0];
	v[0] = 45. * x[44];
	v[2] += v[0];
	v[0] = 46. * x[45];
	v[2] += v[0];
	v[0] = 47. * x[46];
	v[2] += v[0];
	v[0] = 48. * x[47];
	v[2] += v[0];
	v[0] = 49. * x[48];
	v[2] += v[0];
	v[0] = 50. * x[49];
	v[2] += v[0];
	v[0] = 51. * x[50];
	v[2] += v[0];
	v[0] = 52. * x[51];
	v[2] += v[0];
	v[0] = 53. * x[52];
	v[2] += v[0];
	v[0] = 54. * x[53];
	v[2] += v[0];
	v[0] = 55. * x[54];
	v[2] += v[0];
	v[0] = 56. * x[55];
	v[2] += v[0];
	v[0] = 57. * x[56];
	v[2] += v[0];
	v[0] = 58. * x[57];
	v[2] += v[0];
	v[0] = 59. * x[58];
	v[2] += v[0];
	v[0] = 60. * x[59];
	v[2] += v[0];
	v[0] = 61. * x[60];
	v[2] += v[0];
	v[0] = 62. * x[61];
	v[2] += v[0];
	v[0] = 63. * x[62];
	v[2] += v[0];
	v[0] = 64. * x[63];
	v[2] += v[0];
	v[0] = 65. * x[64];
	v[2] += v[0];
	v[0] = 66. * x[65];
	v[2] += v[0];
	v[0] = 67. * x[66];
	v[2] += v[0];
	v[0] = 68. * x[67];
	v[2] += v[0];
	v[0] = 69. * x[68];
	v[2] += v[0];
	v[0] = 70. * x[69];
	v[2] += v[0];
	v[0] = 71. * x[70];
	v[2] += v[0];
	v[0] = 72. * x[71];
	v[2] += v[0];
	v[0] = 73. * x[72];
	v[2] += v[0];
	v[0] = 74. * x[73];
	v[2] += v[0];
	v[0] = 75. * x[74];
	v[2] += v[0];
	v[0] = 76. * x[75];
	v[2] += v[0];
	v[0] = 77. * x[76];
	v[2] += v[0];
	v[0] = 78. * x[77];
	v[2] += v[0];
	v[0] = 79. * x[78];
	v[2] += v[0];
	v[0] = 80. * x[79];
	v[2] += v[0];
	v[0] = 81. * x[80];
	v[2] += v[0];
	v[0] = 82. * x[81];
	v[2] += v[0];
	v[0] = 83. * x[82];
	v[2] += v[0];
	v[0] = 84. * x[83];
	v[2] += v[0];
	v[0] = 85. * x[84];
	v[2] += v[0];
	v[0] = 86. * x[85];
	v[2] += v[0];
	v[0] = 87. * x[86];
	v[2] += v[0];
	v[0] = 88. * x[87];
	v[2] += v[0];
	v[0] = 89. * x[88];
	v[2] += v[0];
	v[0] = 90. * x[89];
	v[2] += v[0];
	v[0] = 91. * x[90];
	v[2] += v[0];
	v[0] = 92. * x[91];
	v[2] += v[0];
	v[0] = 93. * x[92];
	v[2] += v[0];
	v[0] = 94. * x[93];
	v[2] += v[0];
	v[0] = 95. * x[94];
	v[2] += v[0];
	v[0] = 96. * x[95];
	v[2] += v[0];
	v[0] = 97. * x[96];
	v[2] += v[0];
	v[0] = 98. * x[97];
	v[2] += v[0];
	v[0] = 99. * x[98];
	v[2] += v[0];
	v[0] = 100. * x[99];
	v[2] += v[0];
	v[0] = -5050. + v[2];
	v[2] = pow(v[0], 4.);
	if (errno) in_trouble2("pow",v[0],4.);
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