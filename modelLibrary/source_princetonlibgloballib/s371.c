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
	9 /* nvar */,
	1 /* nobj */,
	0 /* ncon */,
	0 /* nzc */,
	0 /* densejac */,

	/* objtype (0 = minimize, 1 = maximize) */

	0 };

 real boundc_[1+18+0] /* Infinity, variable bounds, constraint bounds */ = {
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

 real x0comn_[9] = {
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0.,
		0. };

 real
feval0_(fint *nobj, real *x)
{
	real v[4];


  /***  objective ***/

	v[0] = x[0] * x[0];
	v[1] = x[0] * x[0];
	v[2] = x[1] - v[1];
	v[1] = -1. + v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 0.06896551724137931 * x[2];
	v[2] += x[1];
	v[1] = 0.00356718192627824 * x[3];
	v[2] += v[1];
	v[1] = 0.00016400836442658574 * x[4];
	v[2] += v[1];
	v[1] = 7.069326052870074e-06 * x[5];
	v[2] += v[1];
	v[1] = 2.9252383667048584e-07 * x[6];
	v[2] += v[1];
	v[1] = 1.1768200325824143e-08 * x[7];
	v[2] += v[1];
	v[1] = 4.6377144141178885e-10 * x[8];
	v[2] += v[1];
	v[1] = 0.034482758620689655 * x[1];
	v[1] += x[0];
	v[3] = 0.0011890606420927466 * x[2];
	v[1] += v[3];
	v[3] = 4.1002091106646436e-05 * x[3];
	v[1] += v[3];
	v[3] = 1.4138652105740149e-06 * x[4];
	v[1] += v[3];
	v[3] = 4.8753972778414304e-08 * x[5];
	v[1] += v[3];
	v[3] = 1.6811714751177346e-09 * x[6];
	v[1] += v[3];
	v[3] = 5.7971430176473607e-11 * x[7];
	v[1] += v[3];
	v[3] = 1.9990148336715035e-12 * x[8];
	v[1] += v[3];
	v[3] = v[1] * v[1];
	v[1] = v[2] - v[3];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 0.13793103448275862 * x[2];
	v[1] += x[1];
	v[2] = 0.01426872770511296 * x[3];
	v[1] += v[2];
	v[2] = 0.001312066915412686 * x[4];
	v[1] += v[2];
	v[2] = 0.00011310921684592119 * x[5];
	v[1] += v[2];
	v[2] = 9.360762773455547e-06 * x[6];
	v[1] += v[2];
	v[2] = 7.531648208527451e-07 * x[7];
	v[1] += v[2];
	v[2] = 5.936274450070897e-08 * x[8];
	v[1] += v[2];
	v[2] = 0.06896551724137931 * x[1];
	v[2] += x[0];
	v[3] = 0.0047562425683709865 * x[2];
	v[2] += v[3];
	v[3] = 0.0003280167288531715 * x[3];
	v[2] += v[3];
	v[3] = 2.2621843369184238e-05 * x[4];
	v[2] += v[3];
	v[3] = 1.5601271289092577e-06 * x[5];
	v[2] += v[3];
	v[3] = 1.0759497440753502e-07 * x[6];
	v[2] += v[3];
	v[3] = 7.420343062588622e-09 * x[7];
	v[2] += v[3];
	v[3] = 5.117477974199049e-10 * x[8];
	v[2] += v[3];
	v[3] = v[2] * v[2];
	v[2] = v[1] - v[3];
	v[1] = -1. + v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 0.20689655172413793 * x[2];
	v[2] += x[1];
	v[1] = 0.03210463733650416 * x[3];
	v[2] += v[1];
	v[1] = 0.0044282258395178156 * x[4];
	v[2] += v[1];
	v[1] = 0.0005726154102824761 * x[5];
	v[2] += v[1];
	v[1] = 7.108329231092806e-05 * x[6];
	v[2] += v[1];
	v[1] = 8.579018037525802e-06 * x[7];
	v[2] += v[1];
	v[1] = 1.0142681423675825e-06 * x[8];
	v[2] += v[1];
	v[1] = 0.10344827586206896 * x[1];
	v[1] += x[0];
	v[3] = 0.01070154577883472 * x[2];
	v[1] += v[3];
	v[3] = 0.0011070564598794539 * x[3];
	v[1] += v[3];
	v[3] = 0.00011452308205649523 * x[4];
	v[1] += v[3];
	v[3] = 1.1847215385154678e-05 * x[5];
	v[1] += v[3];
	v[3] = 1.2255740053608288e-06 * x[6];
	v[1] += v[3];
	v[3] = 1.267835177959478e-07 * x[7];
	v[1] += v[3];
	v[3] = 1.3115536323718738e-08 * x[8];
	v[1] += v[3];
	v[3] = v[1] * v[1];
	v[1] = v[2] - v[3];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 0.27586206896551724 * x[2];
	v[1] += x[1];
	v[2] = 0.05707491082045184 * x[3];
	v[1] += v[2];
	v[2] = 0.010496535323301488 * x[4];
	v[1] += v[2];
	v[2] = 0.001809747469534739 * x[5];
	v[1] += v[2];
	v[2] = 0.0002995444087505775 * x[6];
	v[1] += v[2];
	v[2] = 4.820254853457569e-05 * x[7];
	v[1] += v[2];
	v[2] = 7.5984312960907486e-06 * x[8];
	v[1] += v[2];
	v[2] = 0.13793103448275862 * x[1];
	v[2] += x[0];
	v[3] = 0.019024970273483946 * x[2];
	v[2] += v[3];
	v[3] = 0.002624133830825372 * x[3];
	v[2] += v[3];
	v[3] = 0.0003619494939069478 * x[4];
	v[2] += v[3];
	v[3] = 4.992406812509625e-05 * x[5];
	v[2] += v[3];
	v[3] = 6.886078362082241e-06 * x[6];
	v[2] += v[3];
	v[3] = 9.498039120113436e-07 * x[7];
	v[2] += v[3];
	v[3] = 1.3100743613949566e-07 * x[8];
	v[2] += v[3];
	v[3] = v[2] * v[2];
	v[2] = v[1] - v[3];
	v[1] = -1. + v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 0.3448275862068966 * x[2];
	v[2] += x[1];
	v[1] = 0.08917954815695602 * x[3];
	v[2] += v[1];
	v[1] = 0.020501045553323223 * x[4];
	v[2] += v[1];
	v[1] = 0.004418328783043798 * x[5];
	v[2] += v[1];
	v[1] = 0.0009141369895952686 * x[6];
	v[2] += v[1];
	v[1] = 0.00018387813009100232 * x[7];
	v[2] += v[1];
	v[1] = 3.6232143860296026e-05 * x[8];
	v[2] += v[1];
	v[1] = 0.1724137931034483 * x[1];
	v[1] += x[0];
	v[3] = 0.02972651605231867 * x[2];
	v[1] += v[3];
	v[3] = 0.005125261388330806 * x[3];
	v[1] += v[3];
	v[3] = 0.0008836657566087596 * x[4];
	v[1] += v[3];
	v[3] = 0.00015235616493254478 * x[5];
	v[1] += v[3];
	v[3] = 2.6268304298714617e-05 * x[6];
	v[1] += v[3];
	v[3] = 4.529017982537003e-06 * x[7];
	v[1] += v[3];
	v[3] = 7.808651694029316e-07 * x[8];
	v[1] += v[3];
	v[3] = v[1] * v[1];
	v[1] = v[2] - v[3];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 0.41379310344827586 * x[2];
	v[1] += x[1];
	v[2] = 0.12841854934601665 * x[3];
	v[1] += v[2];
	v[2] = 0.035425806716142524 * x[4];
	v[1] += v[2];
	v[2] = 0.009161846564519618 * x[5];
	v[1] += v[2];
	v[2] = 0.002274665353949698 * x[6];
	v[1] += v[2];
	v[2] = 0.0005490571544016513 * x[7];
	v[1] += v[2];
	v[2] = 0.00012982632222305056 * x[8];
	v[1] += v[2];
	v[2] = 0.20689655172413793 * x[1];
	v[2] += x[0];
	v[3] = 0.04280618311533888 * x[2];
	v[2] += v[3];
	v[3] = 0.008856451679035631 * x[3];
	v[2] += v[3];
	v[3] = 0.0018323693129039236 * x[4];
	v[2] += v[3];
	v[3] = 0.0003791108923249497 * x[5];
	v[2] += v[3];
	v[3] = 7.843673634309305e-05 * x[6];
	v[2] += v[3];
	v[3] = 1.622829027788132e-05 * x[7];
	v[2] += v[3];
	v[3] = 3.357577298871997e-06 * x[8];
	v[2] += v[3];
	v[3] = v[2] * v[2];
	v[2] = v[1] - v[3];
	v[1] = -1. + v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 0.4827586206896552 * x[2];
	v[2] += x[1];
	v[1] = 0.1747919143876338 * x[3];
	v[2] += v[1];
	v[1] = 0.05625486899831893 * x[4];
	v[2] += v[1];
	v[1] = 0.016973451852941055 * x[5];
	v[2] += v[1];
	v[1] = 0.004916448122920858 * x[6];
	v[2] += v[1];
	v[1] = 0.0013845170001328855 * x[7];
	v[2] += v[1];
	v[1] = 0.00038193572417458913 * x[8];
	v[2] += v[1];
	v[1] = 0.2413793103448276 * x[1];
	v[1] += x[0];
	v[3] = 0.0582639714625446 * x[2];
	v[1] += v[3];
	v[3] = 0.014063717249579732 * x[3];
	v[1] += v[3];
	v[3] = 0.0033946903705882113 * x[4];
	v[1] += v[3];
	v[3] = 0.0008194080204868096 * x[5];
	v[1] += v[3];
	v[3] = 0.0001977881428761265 * x[6];
	v[1] += v[3];
	v[3] = 4.774196552182364e-05 * x[7];
	v[1] += v[3];
	v[3] = 1.1523922712164328e-05 * x[8];
	v[1] += v[3];
	v[3] = v[1] * v[1];
	v[1] = v[2] - v[3];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 0.5517241379310345 * x[2];
	v[1] += x[1];
	v[2] = 0.22829964328180735 * x[3];
	v[1] += v[2];
	v[2] = 0.0839722825864119 * x[4];
	v[1] += v[2];
	v[2] = 0.028955959512555824 * x[5];
	v[1] += v[2];
	v[2] = 0.00958542108001848 * x[6];
	v[1] += v[2];
	v[2] = 0.003084963106212844 * x[7];
	v[1] += v[2];
	v[2] = 0.0009725992058996158 * x[8];
	v[1] += v[2];
	v[2] = 0.27586206896551724 * x[1];
	v[2] += x[0];
	v[3] = 0.07609988109393578 * x[2];
	v[2] += v[3];
	v[3] = 0.020993070646602975 * x[3];
	v[2] += v[3];
	v[3] = 0.005791191902511165 * x[4];
	v[2] += v[3];
	v[3] = 0.00159757018000308 * x[5];
	v[2] += v[3];
	v[3] = 0.00044070901517326343 * x[6];
	v[2] += v[3];
	v[3] = 0.00012157490073745198 * x[7];
	v[2] += v[3];
	v[3] = 3.353790365171089e-05 * x[8];
	v[2] += v[3];
	v[3] = v[2] * v[2];
	v[2] = v[1] - v[3];
	v[1] = -1. + v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 0.6206896551724138 * x[2];
	v[2] += x[1];
	v[1] = 0.28894173602853745 * x[3];
	v[2] += v[1];
	v[1] = 0.11956209766698103 * x[4];
	v[2] += v[1];
	v[1] = 0.046381848232880565 * x[5];
	v[2] += v[1];
	v[1] = 0.01727324003155552 * x[6];
	v[2] += v[1];
	v[1] = 0.00625410414935631 * x[7];
	v[2] += v[1];
	v[1] = 0.002218204427357903 * x[8];
	v[2] += v[1];
	v[1] = 0.3103448275862069 * x[1];
	v[1] += x[0];
	v[3] = 0.09631391200951249 * x[2];
	v[1] += v[3];
	v[3] = 0.029890524416745258 * x[3];
	v[1] += v[3];
	v[3] = 0.009276369646576113 * x[4];
	v[1] += v[3];
	v[3] = 0.002878873338592587 * x[5];
	v[1] += v[3];
	v[3] = 0.0008934434499080443 * x[6];
	v[1] += v[3];
	v[3] = 0.0002772755534197379 * x[7];
	v[1] += v[3];
	v[3] = 8.605103381991865e-05 * x[8];
	v[1] += v[3];
	v[3] = v[1] * v[1];
	v[1] = v[2] - v[3];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 0.6896551724137931 * x[2];
	v[1] += x[1];
	v[2] = 0.35671819262782406 * x[3];
	v[1] += v[2];
	v[2] = 0.16400836442658578 * x[4];
	v[1] += v[2];
	v[2] = 0.07069326052870077 * x[5];
	v[1] += v[2];
	v[2] = 0.029252383667048597 * x[6];
	v[1] += v[2];
	v[2] = 0.011768200325824148 * x[7];
	v[1] += v[2];
	v[2] = 0.004637714414117891 * x[8];
	v[1] += v[2];
	v[2] = 0.3448275862068966 * x[1];
	v[2] += x[0];
	v[3] = 0.11890606420927469 * x[2];
	v[2] += v[3];
	v[3] = 0.041002091106646446 * x[3];
	v[2] += v[3];
	v[3] = 0.014138652105740154 * x[4];
	v[2] += v[3];
	v[3] = 0.004875397277841433 * x[5];
	v[2] += v[3];
	v[3] = 0.0016811714751177355 * x[6];
	v[2] += v[3];
	v[3] = 0.0005797143017647364 * x[7];
	v[2] += v[3];
	v[3] = 0.0001999014833671505 * x[8];
	v[2] += v[3];
	v[3] = v[2] * v[2];
	v[2] = v[1] - v[3];
	v[1] = -1. + v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 0.7586206896551724 * x[2];
	v[2] += x[1];
	v[1] = 0.431629013079667 * x[3];
	v[2] += v[1];
	v[1] = 0.2182951330517856 * x[4];
	v[2] += v[1];
	v[1] = 0.10350200274007074 * x[5];
	v[2] += v[1];
	v[1] = 0.04711125641961841 * x[6];
	v[2] += v[1];
	v[1] = 0.02084808473741734 * x[7];
	v[2] += v[1];
	v[1] = 0.009037593383708008 * x[8];
	v[2] += v[1];
	v[1] = 0.3793103448275862 * x[1];
	v[1] += x[0];
	v[3] = 0.14387633769322233 * x[2];
	v[1] += v[3];
	v[3] = 0.0545737832629464 * x[3];
	v[1] += v[3];
	v[3] = 0.02070040054801415 * x[4];
	v[1] += v[3];
	v[3] = 0.007851876069936401 * x[5];
	v[1] += v[3];
	v[3] = 0.0029782978196310483 * x[6];
	v[1] += v[3];
	v[3] = 0.001129699172963501 * x[7];
	v[1] += v[3];
	v[3] = 0.0004285065828482245 * x[8];
	v[1] += v[3];
	v[3] = v[1] * v[1];
	v[1] = v[2] - v[3];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 0.8275862068965517 * x[2];
	v[1] += x[1];
	v[2] = 0.5136741973840666 * x[3];
	v[1] += v[2];
	v[2] = 0.2834064537291402 * x[4];
	v[1] += v[2];
	v[2] = 0.1465895450323139 * x[5];
	v[1] += v[2];
	v[2] = 0.07278929132639034 * x[6];
	v[1] += v[2];
	v[2] = 0.035139657881705685 * x[7];
	v[1] += v[2];
	v[2] = 0.01661776924455047 * x[8];
	v[1] += v[2];
	v[2] = 0.41379310344827586 * x[1];
	v[2] += x[0];
	v[3] = 0.17122473246135553 * x[2];
	v[2] += v[3];
	v[3] = 0.07085161343228505 * x[3];
	v[2] += v[3];
	v[3] = 0.029317909006462778 * x[4];
	v[2] += v[3];
	v[3] = 0.01213154855439839 * x[5];
	v[2] += v[3];
	v[3] = 0.005019951125957955 * x[6];
	v[2] += v[3];
	v[3] = 0.002077221155568809 * x[7];
	v[2] += v[3];
	v[3] = 0.0008595397885112312 * x[8];
	v[2] += v[3];
	v[3] = v[2] * v[2];
	v[2] = v[1] - v[3];
	v[1] = -1. + v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 0.896551724137931 * x[2];
	v[2] += x[1];
	v[1] = 0.6028537455410227 * x[3];
	v[2] += v[1];
	v[1] = 0.36032637664520895 * x[4];
	v[2] += v[1];
	v[1] = 0.20190702139602226 * x[5];
	v[2] += v[1];
	v[1] = 0.10861205288889475 * x[6];
	v[2] += v[1];
	v[1] = 0.056802855246490924 * x[7];
	v[2] += v[1];
	v[1] = 0.029100970175542154 * x[8];
	v[2] += v[1];
	v[1] = 0.4482758620689655 * x[1];
	v[1] += x[0];
	v[3] = 0.2009512485136742 * x[2];
	v[1] += v[3];
	v[3] = 0.09008159416130224 * x[3];
	v[1] += v[3];
	v[3] = 0.040381404279204454 * x[4];
	v[1] += v[3];
	v[3] = 0.01810200881481579 * x[5];
	v[1] += v[3];
	v[3] = 0.00811469360664156 * x[6];
	v[1] += v[3];
	v[3] = 0.0036376212719427693 * x[7];
	v[1] += v[3];
	v[3] = 0.0016306578115605518 * x[8];
	v[1] += v[3];
	v[3] = v[1] * v[1];
	v[1] = v[2] - v[3];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 0.9655172413793104 * x[2];
	v[1] += x[1];
	v[2] = 0.6991676575505352 * x[3];
	v[1] += v[2];
	v[2] = 0.4500389519865514 * x[4];
	v[1] += v[2];
	v[2] = 0.2715752296470569 * x[5];
	v[1] += v[2];
	v[2] = 0.15732633993346745 * x[6];
	v[1] += v[2];
	v[2] = 0.08860908800850467 * x[7];
	v[1] += v[2];
	v[2] = 0.04888777269434741 * x[8];
	v[1] += v[2];
	v[2] = 0.4827586206896552 * x[1];
	v[2] += x[0];
	v[3] = 0.2330558858501784 * x[2];
	v[2] += v[3];
	v[3] = 0.11250973799663785 * x[3];
	v[2] += v[3];
	v[3] = 0.05431504592941138 * x[4];
	v[2] += v[3];
	v[3] = 0.026221056655577907 * x[5];
	v[2] += v[3];
	v[3] = 0.012658441144072096 * x[6];
	v[2] += v[3];
	v[3] = 0.006110971586793426 * x[7];
	v[2] += v[3];
	v[3] = 0.002950124214314068 * x[8];
	v[2] += v[3];
	v[3] = v[2] * v[2];
	v[2] = v[1] - v[3];
	v[1] = -1. + v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 1.0344827586206897 * x[2];
	v[2] += x[1];
	v[1] = 0.8026159334126042 * x[3];
	v[2] += v[1];
	v[1] = 0.5535282299397271 * x[4];
	v[2] += v[1];
	v[1] = 0.3578846314265477 * x[5];
	v[2] += v[1];
	v[1] = 0.22213528847165032 * x[6];
	v[2] += v[1];
	v[1] = 0.13404715683634072 * x[7];
	v[2] += v[1];
	v[1] = 0.07923969862246742 * x[8];
	v[2] += v[1];
	v[1] = 0.5172413793103449 * x[1];
	v[1] += x[0];
	v[3] = 0.26753864447086806 * x[2];
	v[1] += v[3];
	v[3] = 0.13838205748493176 * x[3];
	v[1] += v[3];
	v[3] = 0.07157692628530954 * x[4];
	v[1] += v[3];
	v[3] = 0.037022548078608386 * x[5];
	v[1] += v[3];
	v[3] = 0.01914959383376296 * x[6];
	v[1] += v[3];
	v[3] = 0.009904962327808428 * x[7];
	v[1] += v[3];
	v[3] = 0.005123256376452635 * x[8];
	v[1] += v[3];
	v[3] = v[1] * v[1];
	v[1] = v[2] - v[3];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 1.103448275862069 * x[2];
	v[1] += x[1];
	v[2] = 0.9131985731272294 * x[3];
	v[1] += v[2];
	v[2] = 0.6717782606912952 * x[4];
	v[1] += v[2];
	v[2] = 0.4632953522008932 * x[5];
	v[1] += v[2];
	v[2] = 0.30673347456059136 * x[6];
	v[1] += v[2];
	v[2] = 0.19743763879762202 * x[7];
	v[1] += v[2];
	v[2] = 0.12449269835515082 * x[8];
	v[1] += v[2];
	v[2] = 0.5517241379310345 * x[1];
	v[2] += x[0];
	v[3] = 0.30439952437574314 * x[2];
	v[2] += v[3];
	v[3] = 0.1679445651728238 * x[3];
	v[2] += v[3];
	v[3] = 0.09265907044017864 * x[4];
	v[2] += v[3];
	v[3] = 0.05112224576009856 * x[5];
	v[2] += v[3];
	v[3] = 0.02820537697108886 * x[6];
	v[2] += v[3];
	v[3] = 0.015561587294393853 * x[7];
	v[2] += v[3];
	v[3] = 0.008585703334837987 * x[8];
	v[2] += v[3];
	v[3] = v[2] * v[2];
	v[2] = v[1] - v[3];
	v[1] = -1. + v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 1.1724137931034482 * x[2];
	v[2] += x[1];
	v[1] = 1.0309155766944111 * x[3];
	v[2] += v[1];
	v[1] = 0.8057730944278155 * x[4];
	v[2] += v[1];
	v[1] = 0.5904371812617614 * x[5];
	v[2] += v[1];
	v[1] = 0.4153420171634459 * x[6];
	v[2] += v[1];
	v[1] = 0.2840557473704026 * x[7];
	v[2] += v[1];
	v[1] = 0.19030335784421057 * x[8];
	v[2] += v[1];
	v[1] = 0.5862068965517241 * x[1];
	v[1] += x[0];
	v[3] = 0.34363852556480373 * x[2];
	v[1] += v[3];
	v[3] = 0.20144327360695388 * x[3];
	v[1] += v[3];
	v[3] = 0.11808743625235227 * x[4];
	v[1] += v[3];
	v[3] = 0.06922366952724097 * x[5];
	v[1] += v[3];
	v[3] = 0.040579392481486086 * x[6];
	v[1] += v[3];
	v[3] = 0.02378791973052632 * x[7];
	v[1] += v[3];
	v[3] = 0.01394464260065336 * x[8];
	v[1] += v[3];
	v[3] = v[1] * v[1];
	v[1] = v[2] - v[3];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 1.2413793103448276 * x[2];
	v[1] += x[1];
	v[2] = 1.1557669441141498 * x[3];
	v[1] += v[2];
	v[2] = 0.9564967813358483 * x[4];
	v[1] += v[2];
	v[2] = 0.742109571726089 * x[5];
	v[1] += v[2];
	v[2] = 0.5527436810097767 * x[6];
	v[1] += v[2];
	v[2] = 0.40026266555880385 * x[7];
	v[1] += v[2];
	v[2] = 0.2839301667018116 * x[8];
	v[1] += v[2];
	v[2] = 0.6206896551724138 * x[1];
	v[2] += x[0];
	v[3] = 0.38525564803804996 * x[2];
	v[2] += v[3];
	v[3] = 0.23912419533396206 * x[3];
	v[2] += v[3];
	v[3] = 0.14842191434521781 * x[4];
	v[2] += v[3];
	v[3] = 0.09212394683496278 * x[5];
	v[2] += v[3];
	v[3] = 0.05718038079411483 * x[6];
	v[2] += v[3];
	v[3] = 0.03549127083772645 * x[7];
	v[2] += v[3];
	v[3] = 0.022029064657899174 * x[8];
	v[2] += v[3];
	v[3] = v[2] * v[2];
	v[2] = v[1] - v[3];
	v[1] = -1. + v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 1.3103448275862069 * x[2];
	v[2] += x[1];
	v[1] = 1.2877526753864446 * x[3];
	v[2] += v[1];
	v[1] = 1.1249333716019516 * x[4];
	v[2] += v[1];
	v[1] = 0.9212816405360811 * x[5];
	v[2] += v[1];
	v[1] = 0.7243179794559533 * x[6];
	v[2] += v[1];
	v[1] = 0.5536453521128838 * x[7];
	v[2] += v[1];
	v[1] = 0.4145521848332923 * x[8];
	v[2] += v[1];
	v[1] = 0.6551724137931034 * x[1];
	v[1] += x[0];
	v[3] = 0.42925089179548154 * x[2];
	v[1] += v[3];
	v[3] = 0.2812333429004879 * x[3];
	v[1] += v[3];
	v[3] = 0.1842563281072162 * x[4];
	v[1] += v[3];
	v[3] = 0.12071966324265888 * x[5];
	v[1] += v[3];
	v[3] = 0.0790921931589834 * x[6];
	v[1] += v[3];
	v[3] = 0.05181902310416154 * x[7];
	v[1] += v[3];
	v[3] = 0.033950394447554114 * x[8];
	v[1] += v[3];
	v[3] = v[1] * v[1];
	v[1] = v[2] - v[3];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 1.3793103448275863 * x[2];
	v[1] += x[1];
	v[2] = 1.4268727705112962 * x[3];
	v[1] += v[2];
	v[2] = 1.3120669154126863 * x[4];
	v[1] += v[2];
	v[2] = 1.1310921684592123 * x[5];
	v[1] += v[2];
	v[2] = 0.9360762773455551 * x[6];
	v[1] += v[2];
	v[2] = 0.7531648208527455 * x[7];
	v[1] += v[2];
	v[2] = 0.5936274450070901 * x[8];
	v[1] += v[2];
	v[2] = 0.6896551724137931 * x[1];
	v[2] += x[0];
	v[3] = 0.47562425683709875 * x[2];
	v[2] += v[3];
	v[3] = 0.32801672885317157 * x[3];
	v[2] += v[3];
	v[3] = 0.22621843369184247 * x[4];
	v[2] += v[3];
	v[3] = 0.15601271289092586 * x[5];
	v[2] += v[3];
	v[3] = 0.10759497440753507 * x[6];
	v[2] += v[3];
	v[3] = 0.07420343062588626 * x[7];
	v[2] += v[3];
	v[3] = 0.05117477974199053 * x[8];
	v[2] += v[3];
	v[3] = v[2] * v[2];
	v[2] = v[1] - v[3];
	v[1] = -1. + v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 1.4482758620689655 * x[2];
	v[2] += x[1];
	v[1] = 1.573127229488704 * x[3];
	v[2] += v[1];
	v[1] = 1.5188814629546108 * x[4];
	v[2] += v[1];
	v[1] = 1.3748496000882255 * x[5];
	v[2] += v[1];
	v[1] = 1.1946968938697684 * x[6];
	v[2] += v[1];
	v[1] = 1.0093128930968733 * x[7];
	v[2] += v[1];
	v[1] = 0.8352934287698262 * x[8];
	v[2] += v[1];
	v[1] = 0.7241379310344828 * x[1];
	v[1] += x[0];
	v[3] = 0.5243757431629014 * x[2];
	v[1] += v[3];
	v[3] = 0.3797203657386527 * x[3];
	v[1] += v[3];
	v[3] = 0.2749699200176451 * x[4];
	v[1] += v[3];
	v[3] = 0.19911614897829472 * x[5];
	v[1] += v[3];
	v[3] = 0.14418755615669618 * x[6];
	v[1] += v[3];
	v[3] = 0.10441167859622827 * x[7];
	v[1] += v[3];
	v[3] = 0.07560845691451014 * x[8];
	v[1] += v[3];
	v[3] = v[1] * v[1];
	v[1] = v[2] - v[3];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 1.5172413793103448 * x[2];
	v[1] += x[1];
	v[2] = 1.726516052318668 * x[3];
	v[1] += v[2];
	v[2] = 1.7463610644142848 * x[4];
	v[1] += v[2];
	v[2] = 1.6560320438411318 * x[5];
	v[1] += v[2];
	v[2] = 1.507560205427789 * x[6];
	v[1] += v[2];
	v[2] = 1.3342774231947097 * x[7];
	v[1] += v[2];
	v[2] = 1.156811953114625 * x[8];
	v[1] += v[2];
	v[2] = 0.7586206896551724 * x[1];
	v[2] += x[0];
	v[3] = 0.5755053507728893 * x[2];
	v[2] += v[3];
	v[3] = 0.4365902661035712 * x[3];
	v[2] += v[3];
	v[3] = 0.3312064087682264 * x[4];
	v[2] += v[3];
	v[3] = 0.25126003423796484 * x[5];
	v[2] += v[3];
	v[3] = 0.1906110604563871 * x[6];
	v[2] += v[3];
	v[3] = 0.14460149413932813 * x[7];
	v[2] += v[3];
	v[3] = 0.10969768520914547 * x[8];
	v[2] += v[3];
	v[3] = v[2] * v[2];
	v[2] = v[1] - v[3];
	v[1] = -1. + v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 1.5862068965517242 * x[2];
	v[2] += x[1];
	v[1] = 1.8870392390011892 * x[3];
	v[2] += v[1];
	v[1] = 1.9954897699782692 * x[4];
	v[2] += v[1];
	v[1] = 1.9782872719612155 * x[5];
	v[2] += v[1];
	v[1] = 1.8827837484872258 * x[6];
	v[2] += v[1];
	v[1] = 1.7421159971634679 * x[7];
	v[2] += v[1];
	v[1] = 1.5790608053107293 * x[8];
	v[2] += v[1];
	v[1] = 0.7931034482758621 * x[1];
	v[1] += x[0];
	v[3] = 0.6290130796670631 * x[2];
	v[1] += v[3];
	v[3] = 0.4988724424945673 * x[3];
	v[1] += v[3];
	v[3] = 0.3956574543922431 * x[4];
	v[1] += v[3];
	v[3] = 0.31379729141453766 * x[5];
	v[1] += v[3];
	v[3] = 0.2488737138804954 * x[6];
	v[1] += v[3];
	v[3] = 0.19738260066384117 * x[7];
	v[1] += v[3];
	v[3] = 0.15654482121614993 * x[8];
	v[1] += v[3];
	v[3] = v[1] * v[1];
	v[1] = v[2] - v[3];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 1.6551724137931034 * x[2];
	v[1] += x[1];
	v[2] = 2.0546967895362664 * x[3];
	v[1] += v[2];
	v[2] = 2.2672516298331216 * x[4];
	v[1] += v[2];
	v[2] = 2.3454327205170222 * x[5];
	v[1] += v[2];
	v[2] = 2.329257322444491 * x[6];
	v[1] += v[2];
	v[2] = 2.248938104429164 * x[7];
	v[1] += v[2];
	v[2] = 2.1270744633024603 * x[8];
	v[1] += v[2];
	v[2] = 0.8275862068965517 * x[1];
	v[2] += x[0];
	v[3] = 0.6848989298454221 * x[2];
	v[2] += v[3];
	v[3] = 0.5668129074582804 * x[3];
	v[2] += v[3];
	v[3] = 0.46908654410340445 * x[4];
	v[2] += v[3];
	v[3] = 0.3882095537407485 * x[5];
	v[2] += v[3];
	v[3] = 0.3212768720613091 * x[6];
	v[2] += v[3];
	v[3] = 0.26588430791280754 * x[7];
	v[2] += v[3];
	v[3] = 0.2200421858588752 * x[8];
	v[2] += v[3];
	v[3] = v[2] * v[2];
	v[2] = v[1] - v[3];
	v[1] = -1. + v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 1.7241379310344827 * x[2];
	v[2] += x[1];
	v[1] = 2.2294887039238995 * x[3];
	v[2] += v[1];
	v[1] = 2.562630694165402 * x[4];
	v[2] += v[1];
	v[1] = 2.7614554894023726 * x[5];
	v[2] += v[1];
	v[1] = 2.8566780924852124 * x[6];
	v[2] += v[1];
	v[1] = 2.873095782671909 * x[7];
	v[2] += v[1];
	v[1] = 2.830636239085625 * x[8];
	v[2] += v[1];
	v[1] = 0.8620689655172413 * x[1];
	v[1] += x[0];
	v[3] = 0.7431629013079666 * x[2];
	v[1] += v[3];
	v[3] = 0.6406576735413505 * x[3];
	v[1] += v[3];
	v[3] = 0.5522910978804745 * x[4];
	v[1] += v[3];
	v[3] = 0.4761130154142021 * x[5];
	v[1] += v[3];
	v[3] = 0.4104422546674156 * x[6];
	v[1] += v[3];
	v[3] = 0.3538295298857031 * x[7];
	v[1] += v[3];
	v[3] = 0.30502545679801985 * x[8];
	v[1] += v[3];
	v[3] = v[1] * v[1];
	v[1] = v[2] - v[3];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 1.793103448275862 * x[2];
	v[1] += x[1];
	v[2] = 2.4114149821640907 * x[3];
	v[1] += v[2];
	v[2] = 2.8826110131616716 * x[4];
	v[1] += v[2];
	v[2] = 3.230512342336356 * x[5];
	v[1] += v[2];
	v[2] = 3.475585692444632 * x[6];
	v[1] += v[2];
	v[2] = 3.635382735775419 * x[7];
	v[1] += v[2];
	v[2] = 3.7249241824693957 * x[8];
	v[1] += v[2];
	v[2] = 0.896551724137931 * x[1];
	v[2] += x[0];
	v[3] = 0.8038049940546969 * x[2];
	v[2] += v[3];
	v[3] = 0.7206527532904179 * x[3];
	v[2] += v[3];
	v[3] = 0.6461024684672713 * x[4];
	v[2] += v[3];
	v[3] = 0.5792642820741053 * x[5];
	v[2] += v[3];
	v[3] = 0.5193403908250599 * x[6];
	v[2] += v[3];
	v[3] = 0.46561552280867446 * x[7];
	v[2] += v[3];
	v[3] = 0.41744839975950127 * x[8];
	v[2] += v[3];
	v[3] = v[2] * v[2];
	v[2] = v[1] - v[3];
	v[1] = -1. + v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 1.8620689655172413 * x[2];
	v[2] += x[1];
	v[1] = 2.600475624256837 * x[3];
	v[2] += v[1];
	v[1] = 3.2281766370084872 * x[4];
	v[2] += v[1];
	v[1] = 3.7569297068633256 * x[5];
	v[2] += v[1];
	v[1] = 4.197397327667991 * x[6];
	v[2] += v[1];
	v[1] = 4.559241924880749 * x[7];
	v[2] += v[1];
	v[1] = 4.851213082631733 * x[8];
	v[2] += v[1];
	v[1] = 0.9310344827586207 * x[1];
	v[1] += x[0];
	v[3] = 0.8668252080856124 * x[2];
	v[1] += v[3];
	v[3] = 0.8070441592521218 * x[3];
	v[1] += v[3];
	v[3] = 0.7513859413726651 * x[4];
	v[1] += v[3];
	v[3] = 0.6995662212779985 * x[5];
	v[1] += v[3];
	v[3] = 0.6513202749829642 * x[6];
	v[1] += v[3];
	v[3] = 0.6064016353289666 * x[7];
	v[1] += v[3];
	v[3] = 0.5645808328924862 * x[8];
	v[1] += v[3];
	v[3] = v[1] * v[1];
	v[1] = v[2] - v[3];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 1.9310344827586208 * x[2];
	v[1] += x[1];
	v[2] = 2.796670630202141 * x[3];
	v[1] += v[2];
	v[2] = 3.6003116158924113 * x[4];
	v[1] += v[2];
	v[2] = 4.34520367435291 * x[5];
	v[1] += v[2];
	v[2] = 5.0344428778709585 * x[6];
	v[1] += v[2];
	v[2] = 5.670981632544299 * x[7];
	v[1] += v[2];
	v[2] = 6.257634904876468 * x[8];
	v[1] += v[2];
	v[2] = 0.9655172413793104 * x[1];
	v[2] += x[0];
	v[3] = 0.9322235434007136 * x[2];
	v[2] += v[3];
	v[3] = 0.9000779039731028 * x[3];
	v[2] += v[3];
	v[3] = 0.8690407348705821 * x[4];
	v[2] += v[3];
	v[3] = 0.839073812978493 * x[5];
	v[2] += v[3];
	v[3] = 0.8101402332206141 * x[6];
	v[2] += v[3];
	v[3] = 0.7822043631095585 * x[7];
	v[2] += v[3];
	v[3] = 0.7552317988644014 * x[8];
	v[2] += v[3];
	v[3] = v[2] * v[2];
	v[2] = v[1] - v[3];
	v[1] = -1. + v[2];
	v[2] = v[1] * v[1];
	v[0] += v[2];
	v[2] = 2. * x[2];
	v[2] += x[1];
	v[1] = 3. * x[3];
	v[2] += v[1];
	v[1] = 4. * x[4];
	v[2] += v[1];
	v[1] = 5. * x[5];
	v[2] += v[1];
	v[1] = 6. * x[6];
	v[2] += v[1];
	v[1] = 7. * x[7];
	v[2] += v[1];
	v[1] = 8. * x[8];
	v[2] += v[1];
	v[1] = x[0] + x[1];
	v[1] += x[2];
	v[1] += x[3];
	v[1] += x[4];
	v[1] += x[5];
	v[1] += x[6];
	v[1] += x[7];
	v[1] += x[8];
	v[3] = v[1] * v[1];
	v[1] = v[2] - v[3];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
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

x_input = malloc (9 * sizeof(real));
input_values = malloc (9 * sizeof(double));
c_val = malloc (0 * sizeof(real));

file_input = fopen("input.in","r");
for (i=0; i < 9; i++)
    fscanf(file_input, "%lf" ,&input_values[i]);

fclose(file_input);
for (i=0; i < 9; i++)
 {
    x_input[i] = input_values[i];
 }

f_val = feval0_(&objective_number, x_input);

FILE *output_out;
output_out = fopen ("output.out","w");
fprintf(output_out,"%30.15f\n",f_val);
fclose(output_out);

}