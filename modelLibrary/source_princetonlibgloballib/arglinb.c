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
	10 /* nvar */,
	1 /* nobj */,
	0 /* ncon */,
	0 /* nzc */,
	0 /* densejac */,

	/* objtype (0 = minimize, 1 = maximize) */

	0 };

 real boundc_[1+20+0] /* Infinity, variable bounds, constraint bounds */ = {
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

 real x0comn_[10] = {
		1.,
		1.,
		1.,
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

	v[0] = 2. * x[1];
	v[0] += x[0];
	v[1] = 3. * x[2];
	v[0] += v[1];
	v[1] = 4. * x[3];
	v[0] += v[1];
	v[1] = 5. * x[4];
	v[0] += v[1];
	v[1] = 6. * x[5];
	v[0] += v[1];
	v[1] = 7. * x[6];
	v[0] += v[1];
	v[1] = 8. * x[7];
	v[0] += v[1];
	v[1] = 9. * x[8];
	v[0] += v[1];
	v[1] = 10. * x[9];
	v[0] += v[1];
	v[1] = -1. + v[0];
	v[0] = v[1] * v[1];
	v[1] = 2. * x[0];
	v[2] = 4. * x[1];
	v[1] += v[2];
	v[2] = 6. * x[2];
	v[1] += v[2];
	v[2] = 8. * x[3];
	v[1] += v[2];
	v[2] = 10. * x[4];
	v[1] += v[2];
	v[2] = 12. * x[5];
	v[1] += v[2];
	v[2] = 14. * x[6];
	v[1] += v[2];
	v[2] = 16. * x[7];
	v[1] += v[2];
	v[2] = 18. * x[8];
	v[1] += v[2];
	v[2] = 20. * x[9];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 3. * x[0];
	v[2] = 6. * x[1];
	v[1] += v[2];
	v[2] = 9. * x[2];
	v[1] += v[2];
	v[2] = 12. * x[3];
	v[1] += v[2];
	v[2] = 15. * x[4];
	v[1] += v[2];
	v[2] = 18. * x[5];
	v[1] += v[2];
	v[2] = 21. * x[6];
	v[1] += v[2];
	v[2] = 24. * x[7];
	v[1] += v[2];
	v[2] = 27. * x[8];
	v[1] += v[2];
	v[2] = 30. * x[9];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 4. * x[0];
	v[2] = 8. * x[1];
	v[1] += v[2];
	v[2] = 12. * x[2];
	v[1] += v[2];
	v[2] = 16. * x[3];
	v[1] += v[2];
	v[2] = 20. * x[4];
	v[1] += v[2];
	v[2] = 24. * x[5];
	v[1] += v[2];
	v[2] = 28. * x[6];
	v[1] += v[2];
	v[2] = 32. * x[7];
	v[1] += v[2];
	v[2] = 36. * x[8];
	v[1] += v[2];
	v[2] = 40. * x[9];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 5. * x[0];
	v[2] = 10. * x[1];
	v[1] += v[2];
	v[2] = 15. * x[2];
	v[1] += v[2];
	v[2] = 20. * x[3];
	v[1] += v[2];
	v[2] = 25. * x[4];
	v[1] += v[2];
	v[2] = 30. * x[5];
	v[1] += v[2];
	v[2] = 35. * x[6];
	v[1] += v[2];
	v[2] = 40. * x[7];
	v[1] += v[2];
	v[2] = 45. * x[8];
	v[1] += v[2];
	v[2] = 50. * x[9];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 6. * x[0];
	v[2] = 12. * x[1];
	v[1] += v[2];
	v[2] = 18. * x[2];
	v[1] += v[2];
	v[2] = 24. * x[3];
	v[1] += v[2];
	v[2] = 30. * x[4];
	v[1] += v[2];
	v[2] = 36. * x[5];
	v[1] += v[2];
	v[2] = 42. * x[6];
	v[1] += v[2];
	v[2] = 48. * x[7];
	v[1] += v[2];
	v[2] = 54. * x[8];
	v[1] += v[2];
	v[2] = 60. * x[9];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 7. * x[0];
	v[2] = 14. * x[1];
	v[1] += v[2];
	v[2] = 21. * x[2];
	v[1] += v[2];
	v[2] = 28. * x[3];
	v[1] += v[2];
	v[2] = 35. * x[4];
	v[1] += v[2];
	v[2] = 42. * x[5];
	v[1] += v[2];
	v[2] = 49. * x[6];
	v[1] += v[2];
	v[2] = 56. * x[7];
	v[1] += v[2];
	v[2] = 63. * x[8];
	v[1] += v[2];
	v[2] = 70. * x[9];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 8. * x[0];
	v[2] = 16. * x[1];
	v[1] += v[2];
	v[2] = 24. * x[2];
	v[1] += v[2];
	v[2] = 32. * x[3];
	v[1] += v[2];
	v[2] = 40. * x[4];
	v[1] += v[2];
	v[2] = 48. * x[5];
	v[1] += v[2];
	v[2] = 56. * x[6];
	v[1] += v[2];
	v[2] = 64. * x[7];
	v[1] += v[2];
	v[2] = 72. * x[8];
	v[1] += v[2];
	v[2] = 80. * x[9];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 9. * x[0];
	v[2] = 18. * x[1];
	v[1] += v[2];
	v[2] = 27. * x[2];
	v[1] += v[2];
	v[2] = 36. * x[3];
	v[1] += v[2];
	v[2] = 45. * x[4];
	v[1] += v[2];
	v[2] = 54. * x[5];
	v[1] += v[2];
	v[2] = 63. * x[6];
	v[1] += v[2];
	v[2] = 72. * x[7];
	v[1] += v[2];
	v[2] = 81. * x[8];
	v[1] += v[2];
	v[2] = 90. * x[9];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 10. * x[0];
	v[2] = 20. * x[1];
	v[1] += v[2];
	v[2] = 30. * x[2];
	v[1] += v[2];
	v[2] = 40. * x[3];
	v[1] += v[2];
	v[2] = 50. * x[4];
	v[1] += v[2];
	v[2] = 60. * x[5];
	v[1] += v[2];
	v[2] = 70. * x[6];
	v[1] += v[2];
	v[2] = 80. * x[7];
	v[1] += v[2];
	v[2] = 90. * x[8];
	v[1] += v[2];
	v[2] = 100. * x[9];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 11. * x[0];
	v[2] = 22. * x[1];
	v[1] += v[2];
	v[2] = 33. * x[2];
	v[1] += v[2];
	v[2] = 44. * x[3];
	v[1] += v[2];
	v[2] = 55. * x[4];
	v[1] += v[2];
	v[2] = 66. * x[5];
	v[1] += v[2];
	v[2] = 77. * x[6];
	v[1] += v[2];
	v[2] = 88. * x[7];
	v[1] += v[2];
	v[2] = 99. * x[8];
	v[1] += v[2];
	v[2] = 110. * x[9];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 12. * x[0];
	v[2] = 24. * x[1];
	v[1] += v[2];
	v[2] = 36. * x[2];
	v[1] += v[2];
	v[2] = 48. * x[3];
	v[1] += v[2];
	v[2] = 60. * x[4];
	v[1] += v[2];
	v[2] = 72. * x[5];
	v[1] += v[2];
	v[2] = 84. * x[6];
	v[1] += v[2];
	v[2] = 96. * x[7];
	v[1] += v[2];
	v[2] = 108. * x[8];
	v[1] += v[2];
	v[2] = 120. * x[9];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 13. * x[0];
	v[2] = 26. * x[1];
	v[1] += v[2];
	v[2] = 39. * x[2];
	v[1] += v[2];
	v[2] = 52. * x[3];
	v[1] += v[2];
	v[2] = 65. * x[4];
	v[1] += v[2];
	v[2] = 78. * x[5];
	v[1] += v[2];
	v[2] = 91. * x[6];
	v[1] += v[2];
	v[2] = 104. * x[7];
	v[1] += v[2];
	v[2] = 117. * x[8];
	v[1] += v[2];
	v[2] = 130. * x[9];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 14. * x[0];
	v[2] = 28. * x[1];
	v[1] += v[2];
	v[2] = 42. * x[2];
	v[1] += v[2];
	v[2] = 56. * x[3];
	v[1] += v[2];
	v[2] = 70. * x[4];
	v[1] += v[2];
	v[2] = 84. * x[5];
	v[1] += v[2];
	v[2] = 98. * x[6];
	v[1] += v[2];
	v[2] = 112. * x[7];
	v[1] += v[2];
	v[2] = 126. * x[8];
	v[1] += v[2];
	v[2] = 140. * x[9];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 15. * x[0];
	v[2] = 30. * x[1];
	v[1] += v[2];
	v[2] = 45. * x[2];
	v[1] += v[2];
	v[2] = 60. * x[3];
	v[1] += v[2];
	v[2] = 75. * x[4];
	v[1] += v[2];
	v[2] = 90. * x[5];
	v[1] += v[2];
	v[2] = 105. * x[6];
	v[1] += v[2];
	v[2] = 120. * x[7];
	v[1] += v[2];
	v[2] = 135. * x[8];
	v[1] += v[2];
	v[2] = 150. * x[9];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 16. * x[0];
	v[2] = 32. * x[1];
	v[1] += v[2];
	v[2] = 48. * x[2];
	v[1] += v[2];
	v[2] = 64. * x[3];
	v[1] += v[2];
	v[2] = 80. * x[4];
	v[1] += v[2];
	v[2] = 96. * x[5];
	v[1] += v[2];
	v[2] = 112. * x[6];
	v[1] += v[2];
	v[2] = 128. * x[7];
	v[1] += v[2];
	v[2] = 144. * x[8];
	v[1] += v[2];
	v[2] = 160. * x[9];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 17. * x[0];
	v[2] = 34. * x[1];
	v[1] += v[2];
	v[2] = 51. * x[2];
	v[1] += v[2];
	v[2] = 68. * x[3];
	v[1] += v[2];
	v[2] = 85. * x[4];
	v[1] += v[2];
	v[2] = 102. * x[5];
	v[1] += v[2];
	v[2] = 119. * x[6];
	v[1] += v[2];
	v[2] = 136. * x[7];
	v[1] += v[2];
	v[2] = 153. * x[8];
	v[1] += v[2];
	v[2] = 170. * x[9];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 18. * x[0];
	v[2] = 36. * x[1];
	v[1] += v[2];
	v[2] = 54. * x[2];
	v[1] += v[2];
	v[2] = 72. * x[3];
	v[1] += v[2];
	v[2] = 90. * x[4];
	v[1] += v[2];
	v[2] = 108. * x[5];
	v[1] += v[2];
	v[2] = 126. * x[6];
	v[1] += v[2];
	v[2] = 144. * x[7];
	v[1] += v[2];
	v[2] = 162. * x[8];
	v[1] += v[2];
	v[2] = 180. * x[9];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 19. * x[0];
	v[2] = 38. * x[1];
	v[1] += v[2];
	v[2] = 57. * x[2];
	v[1] += v[2];
	v[2] = 76. * x[3];
	v[1] += v[2];
	v[2] = 95. * x[4];
	v[1] += v[2];
	v[2] = 114. * x[5];
	v[1] += v[2];
	v[2] = 133. * x[6];
	v[1] += v[2];
	v[2] = 152. * x[7];
	v[1] += v[2];
	v[2] = 171. * x[8];
	v[1] += v[2];
	v[2] = 190. * x[9];
	v[1] += v[2];
	v[2] = -1. + v[1];
	v[1] = v[2] * v[2];
	v[0] += v[1];
	v[1] = 20. * x[0];
	v[2] = 40. * x[1];
	v[1] += v[2];
	v[2] = 60. * x[2];
	v[1] += v[2];
	v[2] = 80. * x[3];
	v[1] += v[2];
	v[2] = 100. * x[4];
	v[1] += v[2];
	v[2] = 120. * x[5];
	v[1] += v[2];
	v[2] = 140. * x[6];
	v[1] += v[2];
	v[2] = 160. * x[7];
	v[1] += v[2];
	v[2] = 180. * x[8];
	v[1] += v[2];
	v[2] = 200. * x[9];
	v[1] += v[2];
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

x_input = malloc (10 * sizeof(real));
input_values = malloc (10 * sizeof(double));
c_val = malloc (0 * sizeof(real));

file_input = fopen("input.in","r");
for (i=0; i < 10; i++)
    fscanf(file_input, "%lf" ,&input_values[i]);

fclose(file_input);
for (i=0; i < 10; i++)
 {
    x_input[i] = input_values[i];
 }

f_val = feval0_(&objective_number, x_input);

FILE *output_out;
output_out = fopen ("output.out","w");
fprintf(output_out,"%30.15f\n",f_val);
fclose(output_out);

}
