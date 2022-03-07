from scipy.optimize import fsolve
import numpy as np


def equations(p, delta_L):
    ##  p = (L1, L2, W, delta_L11, delta_L21, delta_L12, delta_L22)
    (LA0, LB0) = p
    (delta_A02, delta_B02), (delta_A04, delta_B04) = delta_L
    LA2 = LA0 - delta_A02
    LB2 = LB0 - delta_B02
    LA4 = LA0 - delta_A04
    LB4 = LB0 - delta_B04
    eqn1 = 2 * delta_A02 * LA0 - 2 * delta_B02 * LB0 - (delta_A02**2 - delta_B02**2)
    eqn3 = 2 * delta_A04 * LA0 - 2 * delta_B04 * LB0 - (delta_A04**2 - delta_B04**2)
    return (eqn1, eqn3)

def findParameters(delta_L):
    (delta_A01, delta_B01), (delta_A02, delta_B02), (delta_A03, delta_B03), (delta_A04, delta_B04), (delta_A05, delta_B05) = delta_L
    delta_A13 = - delta_A01 + delta_A03
    delta_B13 = - delta_B01 + delta_B03
    delta_A15 = - delta_A01 + delta_A05
    delta_B15 = - delta_B01 + delta_B05
    LA0, LB0 = fsolve(equations, (10, 15), args=(((delta_A02, delta_B02), (delta_A04, delta_B04)), ), xtol=1.49012e-8)
    LA1, LB1 = fsolve(equations, (10, 15), args=(((delta_A13, delta_B13), (delta_A15, delta_B15)), ), xtol=1.49012e-8)
    W = np.sqrt((LA0**4 + LB0**4 - 2*LA0**2 * LB0**2 - LA1**4 - LB1**4 + 2*LA1**2 * LB1**2)/(2*LA0**2 + 2*LB0**2 - 2*LA1**2 - 2*LB1**2))
    return (LA0, LB0, W), 0

if __name__=='__main__':
    delta_L = ((-5.4178,5.0441), (-1.6437, -0.7821 ) ,(-6.1158, 3.157), (-4.4517, -2.6193), (-7.8017, 0.2091))
    (L1, L2, W), eqns = findParameters(delta_L)
    print('eqns: ',eqns)
    print('L1, L2, W :', L1, L2, W)
    #print(equations((3.60555, 6.3245), delta_L))