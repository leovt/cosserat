'''Specification of the two-rod problem

Source:
https://lcvmwww.epfl.ch/oldwebsite/lcvm_user/dna_teaching_05_06/

In particular
https://lcvmwww.epfl.ch/~lcvm/dna_teaching_05_06/exercises/sol5.pdf

#
#        |      _-o
#        | p2_-'  |
#        |_-'     | lambda
#        o        V
#   |   /
#   |p1/
#   | /
#___|/_______________
#////////////////////

The problem has one parameter and a two dimensional solution
parameter lambda is the load at the end of the second rod
the solution vector x=[p1 p2] describes the deviation from the
vertical of the first and second rod.
At the base and at the joint between the two rods a spring
is installed which tries to keep the bottom rod straight up
(center at p1=0) and the second joint straight (center at p2=p1)
'''
import cmath
from numpy import sin, cos
import numpy as np


def E(x, lam):
    """the potential energy to be minimized in an equilibrium"""
    (the, phi) = x
    return 0.5 * the**2 + 0.5 * (phi - the)**2 + lam * (cos(the) + cos(phi))

def F(x, lam):
    """the gradient of E, a solution will have F=0"""
    (the, phi) = x
    return np.array([2*the - phi - lam*sin(the), -the + phi - lam*sin(phi)])

def J(x, lam):
    """the Jacobian of F / Hessian of E"""
    (the, phi) = x
    return np.array([(2-lam*cos(the), -1), (-1, 1-lam*cos(phi))])

def F_lam(x, lam):
    """the partial derivative of F along lambda"""
    (the, phi) = x
    return np.array([-sin(the), - sin(phi)])

def draw_solution_svg(x, lam):
    """return an svg image of the solution as a string"""
    (the, phi) = x

    A = cmath.rect(100, cmath.pi/2 - the)
    B = cmath.rect(100, cmath.pi/2 - phi) + A
    C = B - 20j * lam

    D = cmath.rect(110, cmath.pi/2 - the/2)
    E = cmath.rect(110, cmath.pi/2 - phi/2) + A

    return f'''<svg viewBox="-250 -250 500 300" xmlns="http://www.w3.org/2000/svg">
<defs>
    <marker id="dot" viewBox="0 0 10 10" refX="5" refY="5"
        markerWidth="5" markerHeight="5">
      <circle cx="5" cy="5" r="2" fill="white" stroke="black" stroke-width="2"/>
    </marker>
</defs>
<rect x="-250" y="0" width="500" height="50" fill="lightgrey" />
<path d="M -250 0 250 0 M 0 30 v -180 M 0 -100 A 100 100 0 0 {int(the>0)} {A.real} {-A.imag} v -150 m 0 50 A 100 100 0 0 {int(phi>0)} {B.real} {-B.imag}"
      stroke="black" fill="none"/>
<path d="M {B.real} {-B.imag} v {50*lam} m 0 18 6 -18 -12 0 Z"
      stroke="red" fill="red"/>
<text x="{D.real}" y="{-D.imag}" dominant-baseline="middle" text-anchor="middle">θ={the:.2f}</text>
<text x="{E.real}" y="{-E.imag}" dominant-baseline="middle" text-anchor="middle">φ={phi:.2f}</text>
<text x="{B.real + 10}" y="{-B.imag + 10}" fill="red">λ={lam:.2f}</text>
<path d="M 0 0 {A.real} {-A.imag} {B.real} {-B.imag}"
      stroke="black" fill="none" stroke-width="3"
      marker-start="url(#dot)" marker-mid="url(#dot)"  marker-end="url(#dot)"/>
</svg>'''
