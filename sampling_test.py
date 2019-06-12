from numpy import bitwise_xor
from sobol import *
import datetime

def sobol_test01 ( ):
  print ( '' )
  print ( 'SOBOL_TEST01'  )
  print ( '  BITWISE_XOR returns the bitwise exclusive OR of two integers.' )
  print ( '' )
  print ('     I     J     BITXOR(I,J)' )
  print ( ''  )
  seed = 123456789
  for test in range ( 0, 10 ):
    i, seed = i4_uniform ( 0, 100, seed )
    j, seed = i4_uniform ( 0, 100, seed )
    k = bitwise_xor ( i, j )
    print (' %X  %X  %X' % ( i, j, k ))
  return

def sobol_test02 ( ):
    print ( '' )
    print ( 'SOBOL_TEST02' )
    print ( '  I4_BIT_HI1 returns the location of the high 1 bit.' )
    print ( '' )
    print (  '     I     I4_BIT_HI1(I)' )
    print ( '' )

    seed = 123456789
    for test in range( 0, 10):
      [ i, seed ] = i4_uniform ( 0, 100, seed )
      j = i4_bit_hi1 ( i )
      print ( '%6d %6d' % ( i, j ) )
    return

def sobol_test03 ( ):
    print ( '' )
    print ( 'SOBOL_TEST03' )
    print ( '  I4_BIT_LO0 returns the location of the low 0 bit.' )
    print ( '' )
    print ( '     I     I4_BIT_LO0(I)' )
    print ( '' )

    seed = 123456789

    for test in range ( 0, 10 ):
      [ i, seed ] = i4_uniform ( 0, 100, seed )
      j = i4_bit_lo0 ( i )
      print ( '%6d %6d'%( i, j ) )
    return

def main():
  sobol_test01()
  sobol_test02()
  sobol_test03()
  return 

main()