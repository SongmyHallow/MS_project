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
	7 /* nvar */,
	1 /* nobj */,
	0 /* ncon */,
	0 /* nzc */,
	0 /* densejac */,

	/* objtype (0 = minimize, 1 = maximize) */

	0 };

 real boundc_[1+14+0] /* Infinity, variable bounds, constraint bounds */ = {
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

 real x0comn_[7] = {
		1.,
		1.,
		1.,
		1.,
		1.,
		1.,
		1. };

 real
feval0_(fint *nobj, real *x)
{
	real v[3];


  /***  objective ***/

	v[0] = 3.2003886153690004 * x[1];
	v[0] += x[0];
	v[1] = 10.242487289383508 * x[2];
	v[0] += v[1];
	v[1] = 32.77993971400467 * x[3];
	v[0] += v[1];
	v[1] = 104.90854587318272 * x[4];
	v[0] += v[1];
	v[1] = 335.7481158674505 * x[5];
	v[0] += v[1];
	v[1] = 1074.5244476537807 * x[6];
	v[0] += v[1];
	v[1] = 78.596218 - v[0];
	v[0] = v[1] * v[1];
	v[1] = 3.046173318241 * x[1];
	v[1] += x[0];
	v[2] = 9.279171884763384 * x[2];
	v[1] += v[2];
	v[2] = 28.26596581073827 * x[3];
	v[1] += v[2];
	v[2] = 86.10303086698326 * x[4];
	v[1] += v[2];
	v[2] = 262.2847552466856 * x[5];
	v[1] += v[2];
	v[2] = 798.964823213825 * x[6];
	v[1] += v[2];
	v[2] = 65.77963 - v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 2.749172911969 * x[1];
	v[1] += x[0];
	v[2] = 7.557951699904112 * x[2];
	v[1] += v[2];
	v[2] = 20.77811608334644 * x[3];
	v[1] += v[2];
	v[2] = 57.12263389808345 * x[4];
	v[1] += v[2];
	v[2] = 157.0399977729332 * x[5];
	v[1] += v[2];
	v[2] = 431.73010797302004 * x[6];
	v[1] += v[2];
	v[2] = 43.96947 - v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 2.4674000736160004 * x[1];
	v[1] += x[0];
	v[2] = 6.088063123280245 * x[2];
	v[1] += v[2];
	v[2] = 15.021687398560534 * x[3];
	v[1] += v[2];
	v[2] = 37.06451259304481 * x[4];
	v[1] += v[2];
	v[2] = 91.45298110061994 * x[5];
	v[1] += v[2];
	v[2] = 225.65109230007235 * x[6];
	v[1] += v[2];
	v[2] = 27.038816 - v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 2.2008612609 * x[1];
	v[1] += x[0];
	v[2] = 4.843790289730338 * x[2];
	v[1] += v[2];
	v[2] = 10.660510404591088 * x[3];
	v[1] += v[2];
	v[2] = 23.46230437088591 * x[4];
	v[1] += v[2];
	v[2] = 51.63727678132754 * x[5];
	v[1] += v[2];
	v[2] = 113.64648208639483 * x[6];
	v[1] += v[2];
	v[2] = 14.6126 - v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 1.949550365169 * x[1];
	v[1] += x[0];
	v[2] = 3.8007466263305814 * x[2];
	v[1] += v[2];
	v[2] = 7.40974697327763 * x[3];
	v[1] += v[2];
	v[2] = 14.445674917563295 * x[4];
	v[1] += v[2];
	v[2] = 28.162570810648187 * x[5];
	v[1] += v[2];
	v[2] = 54.90435020799699 * x[6];
	v[1] += v[2];
	v[2] = 6.2614 - v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 1.7134731460089998 * x[1];
	v[1] += x[0];
	v[2] = 2.935990222093979 * x[2];
	v[1] += v[2];
	v[2] = 5.0307404025030324 * x[3];
	v[1] += v[2];
	v[2] = 8.620038584231452 * x[4];
	v[1] += v[2];
	v[2] = 14.77020463164203 * x[5];
	v[1] += v[2];
	v[2] = 25.30834899737637 * x[6];
	v[1] += v[2];
	v[2] = 1.53833 - v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 1.485015206544 * x[1];
	v[1] += x[0];
	v[2] = 2.205270163666919 * x[2];
	v[1] += v[2];
	v[2] = 3.274859727583151 * x[3];
	v[1] += v[2];
	v[2] = 4.863216494759521 * x[4];
	v[1] += v[2];
	v[2] = 7.221950447433498 * x[5];
	v[1] += v[2];
	v[2] = 10.72470623534599 * x[6];
	v[1] += v[2];
	v[2] = -v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 1.2870085672959999 * x[1];
	v[1] += x[0];
	v[2] = 1.6563910522933023 * x[2];
	v[1] += v[2];
	v[2] = 2.1317894750939166 * x[3];
	v[1] += v[2];
	v[2] = 2.7436313181173135 * x[4];
	v[1] += v[2];
	v[2] = 3.5310770119185992 * x[5];
	v[1] += v[2];
	v[2] = 4.544526366121197 * x[6];
	v[1] += v[2];
	v[2] = 1.188045 - v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 1.0966236512040002 * x[1];
	v[1] += x[0];
	v[2] = 1.2025834323799927 * x[2];
	v[1] += v[2];
	v[2] = 1.3187814344939863 * x[3];
	v[1] += v[2];
	v[2] = 1.4462069118348444 * x[4];
	v[1] += v[2];
	v[2] = 1.5859447040527888 * x[5];
	v[1] += v[2];
	v[2] = 1.7391844719660166 * x[6];
	v[1] += v[2];
	v[2] = 4.6841 - v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 0.7615442022250001 * x[1];
	v[1] += x[0];
	v[2] = 0.5799495719425118 * x[2];
	v[1] += v[2];
	v[2] = 0.44165723409569047 * x[3];
	v[1] += v[2];
	v[2] = 0.33634150599630275 * x[4];
	v[1] += v[2];
	v[2] = 0.25613892385910947 * x[5];
	v[1] += v[2];
	v[2] = 0.19506111242905555 * x[6];
	v[1] += v[2];
	v[2] = 16.9321 - v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 0.487388289424 * x[1];
	v[1] += x[0];
	v[2] = 0.23754734466765276 * x[2];
	v[1] += v[2];
	v[2] = 0.11577779397478062 * x[3];
	v[1] += v[2];
	v[2] = 0.056428740958652614 * x[4];
	v[1] += v[2];
	v[2] = 0.0275027075301877 * x[5];
	v[1] += v[2];
	v[2] = 0.013404497577666747 * x[6];
	v[1] += v[2];
	v[2] = 33.6988 - v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 0.27415591280100005 * x[1];
	v[1] += x[0];
	v[2] = 0.07516146452374954 * x[2];
	v[1] += v[2];
	v[2] = 0.020605959913968536 * x[3];
	v[1] += v[2];
	v[2] = 0.005649245749354861 * x[4];
	v[1] += v[2];
	v[2] = 0.0015487741250515515 * x[5];
	v[1] += v[2];
	v[2] = 0.00042460558397607826 * x[6];
	v[1] += v[2];
	v[2] = 52.3664 - v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 0.121847072356 * x[1];
	v[1] += x[0];
	v[2] = 0.014846709041728298 * x[2];
	v[1] += v[2];
	v[2] = 0.0018090280308559472 * x[3];
	v[1] += v[2];
	v[2] = 0.00022042476936973677 * x[4];
	v[1] += v[2];
	v[2] = 2.6858112822448926e-05 * x[5];
	v[1] += v[2];
	v[2] = 3.2725824164225457e-06 * x[6];
	v[1] += v[2];
	v[2] = 70.163 - v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 0.030461768089 * x[1];
	v[1] += x[0];
	v[2] = 0.0009279193151080186 * x[2];
	v[1] += v[2];
	v[2] = 2.8266062982124175e-05 * x[3];
	v[1] += v[2];
	v[2] = 8.610342553505343e-07 * x[4];
	v[1] += v[2];
	v[2] = 2.622862580317278e-08 * x[5];
	v[1] += v[2];
	v[2] = 7.989703165094106e-10 * x[6];
	v[1] += v[2];
	v[2] = 83.4221 - v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 88.3995 - x[0];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 3.2003886153690004 * x[1];
	v[2] += x[0];
	v[1] = 10.242487289383508 * x[2];
	v[2] += v[1];
	v[1] = 32.77993971400467 * x[3];
	v[2] += v[1];
	v[1] = 104.90854587318272 * x[4];
	v[2] += v[1];
	v[1] = 335.7481158674505 * x[5];
	v[2] += v[1];
	v[1] = 1074.5244476537807 * x[6];
	v[2] += v[1];
	v[1] = 78.596218 - v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 3.046173318241 * x[1];
	v[2] += x[0];
	v[1] = 9.279171884763384 * x[2];
	v[2] += v[1];
	v[1] = 28.26596581073827 * x[3];
	v[2] += v[1];
	v[1] = 86.10303086698326 * x[4];
	v[2] += v[1];
	v[1] = 262.2847552466856 * x[5];
	v[2] += v[1];
	v[1] = 798.964823213825 * x[6];
	v[2] += v[1];
	v[1] = 65.77963 - v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 2.749172911969 * x[1];
	v[2] += x[0];
	v[1] = 7.557951699904112 * x[2];
	v[2] += v[1];
	v[1] = 20.77811608334644 * x[3];
	v[2] += v[1];
	v[1] = 57.12263389808345 * x[4];
	v[2] += v[1];
	v[1] = 157.0399977729332 * x[5];
	v[2] += v[1];
	v[1] = 431.73010797302004 * x[6];
	v[2] += v[1];
	v[1] = 43.96947 - v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 2.4674000736160004 * x[1];
	v[2] += x[0];
	v[1] = 6.088063123280245 * x[2];
	v[2] += v[1];
	v[1] = 15.021687398560534 * x[3];
	v[2] += v[1];
	v[1] = 37.06451259304481 * x[4];
	v[2] += v[1];
	v[1] = 91.45298110061994 * x[5];
	v[2] += v[1];
	v[1] = 225.65109230007235 * x[6];
	v[2] += v[1];
	v[1] = 27.038816 - v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 2.2008612609 * x[1];
	v[2] += x[0];
	v[1] = 4.843790289730338 * x[2];
	v[2] += v[1];
	v[1] = 10.660510404591088 * x[3];
	v[2] += v[1];
	v[1] = 23.46230437088591 * x[4];
	v[2] += v[1];
	v[1] = 51.63727678132754 * x[5];
	v[2] += v[1];
	v[1] = 113.64648208639483 * x[6];
	v[2] += v[1];
	v[1] = 14.6126 - v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 1.949550365169 * x[1];
	v[2] += x[0];
	v[1] = 3.8007466263305814 * x[2];
	v[2] += v[1];
	v[1] = 7.40974697327763 * x[3];
	v[2] += v[1];
	v[1] = 14.445674917563295 * x[4];
	v[2] += v[1];
	v[1] = 28.162570810648187 * x[5];
	v[2] += v[1];
	v[1] = 54.90435020799699 * x[6];
	v[2] += v[1];
	v[1] = 6.2614 - v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 1.7134731460089998 * x[1];
	v[2] += x[0];
	v[1] = 2.935990222093979 * x[2];
	v[2] += v[1];
	v[1] = 5.0307404025030324 * x[3];
	v[2] += v[1];
	v[1] = 8.620038584231452 * x[4];
	v[2] += v[1];
	v[1] = 14.77020463164203 * x[5];
	v[2] += v[1];
	v[1] = 25.30834899737637 * x[6];
	v[2] += v[1];
	v[1] = 1.53833 - v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 1.485015206544 * x[1];
	v[2] += x[0];
	v[1] = 2.205270163666919 * x[2];
	v[2] += v[1];
	v[1] = 3.274859727583151 * x[3];
	v[2] += v[1];
	v[1] = 4.863216494759521 * x[4];
	v[2] += v[1];
	v[1] = 7.221950447433498 * x[5];
	v[2] += v[1];
	v[1] = 10.72470623534599 * x[6];
	v[2] += v[1];
	v[1] = -v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 1.2870085672959999 * x[1];
	v[2] += x[0];
	v[1] = 1.6563910522933023 * x[2];
	v[2] += v[1];
	v[1] = 2.1317894750939166 * x[3];
	v[2] += v[1];
	v[1] = 2.7436313181173135 * x[4];
	v[2] += v[1];
	v[1] = 3.5310770119185992 * x[5];
	v[2] += v[1];
	v[1] = 4.544526366121197 * x[6];
	v[2] += v[1];
	v[1] = 1.188045 - v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 1.0966236512040002 * x[1];
	v[2] += x[0];
	v[1] = 1.2025834323799927 * x[2];
	v[2] += v[1];
	v[1] = 1.3187814344939863 * x[3];
	v[2] += v[1];
	v[1] = 1.4462069118348444 * x[4];
	v[2] += v[1];
	v[1] = 1.5859447040527888 * x[5];
	v[2] += v[1];
	v[1] = 1.7391844719660166 * x[6];
	v[2] += v[1];
	v[1] = 4.6841 - v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 0.7615442022250001 * x[1];
	v[2] += x[0];
	v[1] = 0.5799495719425118 * x[2];
	v[2] += v[1];
	v[1] = 0.44165723409569047 * x[3];
	v[2] += v[1];
	v[1] = 0.33634150599630275 * x[4];
	v[2] += v[1];
	v[1] = 0.25613892385910947 * x[5];
	v[2] += v[1];
	v[1] = 0.19506111242905555 * x[6];
	v[2] += v[1];
	v[1] = 16.9321 - v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 0.487388289424 * x[1];
	v[2] += x[0];
	v[1] = 0.23754734466765276 * x[2];
	v[2] += v[1];
	v[1] = 0.11577779397478062 * x[3];
	v[2] += v[1];
	v[1] = 0.056428740958652614 * x[4];
	v[2] += v[1];
	v[1] = 0.0275027075301877 * x[5];
	v[2] += v[1];
	v[1] = 0.013404497577666747 * x[6];
	v[2] += v[1];
	v[1] = 33.6988 - v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 0.27415591280100005 * x[1];
	v[2] += x[0];
	v[1] = 0.07516146452374954 * x[2];
	v[2] += v[1];
	v[1] = 0.020605959913968536 * x[3];
	v[2] += v[1];
	v[1] = 0.005649245749354861 * x[4];
	v[2] += v[1];
	v[1] = 0.0015487741250515515 * x[5];
	v[2] += v[1];
	v[1] = 0.00042460558397607826 * x[6];
	v[2] += v[1];
	v[1] = 52.3664 - v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 0.121847072356 * x[1];
	v[2] += x[0];
	v[1] = 0.014846709041728298 * x[2];
	v[2] += v[1];
	v[1] = 0.0018090280308559472 * x[3];
	v[2] += v[1];
	v[1] = 0.00022042476936973677 * x[4];
	v[2] += v[1];
	v[1] = 2.6858112822448926e-05 * x[5];
	v[2] += v[1];
	v[1] = 3.2725824164225457e-06 * x[6];
	v[2] += v[1];
	v[1] = 70.163 - v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 0.030461768089 * x[1];
	v[2] += x[0];
	v[1] = 0.0009279193151080186 * x[2];
	v[2] += v[1];
	v[1] = 2.8266062982124175e-05 * x[3];
	v[2] += v[1];
	v[1] = 8.610342553505343e-07 * x[4];
	v[2] += v[1];
	v[1] = 2.622862580317278e-08 * x[5];
	v[2] += v[1];
	v[1] = 7.989703165094106e-10 * x[6];
	v[2] += v[1];
	v[1] = 83.4221 - v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 3.5202348851952103 * x[1];
	v[2] += x[0];
	v[1] = 12.392053646945335 * x[2];
	v[2] += v[1];
	v[1] = 43.6229395471875 * x[3];
	v[2] += v[1];
	v[1] = 153.56299358877118 * x[4];
	v[2] += v[1];
	v[1] = 540.5778071062007 * x[5];
	v[2] += v[1];
	v[1] = 1902.9608547375751 * x[6];
	v[2] += v[1];
	v[1] = 108.18086 - v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 3.3584069996584898 * x[1];
	v[2] += x[0];
	v[1] = 11.27889757535514 * x[2];
	v[2] += v[1];
	v[1] = 37.87912856550387 * x[3];
	v[2] += v[1];
	v[1] = 127.21353051535206 * x[4];
	v[2] += v[1];
	v[1] = 427.23481133402726 * x[5];
	v[2] += v[1];
	v[1] = 1434.8283808819715 * x[6];
	v[2] += v[1];
	v[1] = 92.733676 - v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 3.5202348851952103 * x[1];
	v[2] += x[0];
	v[1] = 12.392053646945335 * x[2];
	v[2] += v[1];
	v[1] = 43.6229395471875 * x[3];
	v[2] += v[1];
	v[1] = 153.56299358877118 * x[4];
	v[2] += v[1];
	v[1] = 540.5778071062007 * x[5];
	v[2] += v[1];
	v[1] = 1902.9608547375751 * x[6];
	v[2] += v[1];
	v[1] = 108.18086 - v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 3.3584069996584898 * x[1];
	v[2] += x[0];
	v[1] = 11.27889757535514 * x[2];
	v[2] += v[1];
	v[1] = 37.87912856550387 * x[3];
	v[2] += v[1];
	v[1] = 127.21353051535206 * x[4];
	v[2] += v[1];
	v[1] = 427.23481133402726 * x[5];
	v[2] += v[1];
	v[1] = 1434.8283808819715 * x[6];
	v[2] += v[1];
	v[1] = 92.733676 - v[2];
	v[2] = v[1] * v[1];
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

x_input = malloc (7 * sizeof(real));
input_values = malloc (7 * sizeof(double));
c_val = malloc (0 * sizeof(real));

file_input = fopen("input.in","r");
for (i=0; i < 7; i++)
    fscanf(file_input, "%lf" ,&input_values[i]);

fclose(file_input);
for (i=0; i < 7; i++)
 {
    x_input[i] = input_values[i];
 }

f_val = feval0_(&objective_number, x_input);

FILE *output_out;
output_out = fopen ("output.out","w");
fprintf(output_out,"%30.15f\n",f_val);
fclose(output_out);

}
