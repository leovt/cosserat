import numpy as np

import tworod

def newton(F, J, x0, lam):
    x = x0
    Fx = F(x, lam)
    while Fx.dot(Fx) > 1e-27:
        #print(x, Fx, Fx.dot(Fx))
        Jx = J(x, lam)
        x -= np.linalg.solve(Jx, Fx)
        Fx = F(x, lam)
    return x


class BifurcationAnalysis:
    def __init__(self, F, J, F_lam, x0, lam0):
        self.F = F
        self.J = J
        self.F_lam = F_lam
        self.branches = [Branch(self, [(x0, lam0)])]


class Branch:
    def __init__(self, prob, points):
        self.prob = prob
        self.soln = np.array([p[0] for p in points])
        self.param = np.array([p[1] for p in points])


    def extend(self, max_lam):
        F = self.prob.F
        J = self.prob.J
        F_lam = self.prob.F_lam

        new_points = []

        x0 = self.soln[-1]
        lam0 = self.param[-1]
        while abs(lam0) < max_lam:
            Fx0 = F(x0, lam0)
            assert Fx0.dot(Fx0) < 1e-20, 'given point is not a solution'
            # expansion  x=x0+eps x1 + eps**2 x2 + ...
            # dF(x)/deps at eps=0 = J(x0)*x1
            # J*x1 + F_lam*lam1 = 0
            # set lam1 = 1
            b = F_lam(x0, lam0)
            x1 = np.linalg.solve(J(x0, lam0), -b)
            eps = 0.01

            # if the det changes the sign, we skipped a bif point.
            det_low = np.linalg.det(J(x0, lam0))

            x = x0 + eps*x1
            lam = lam0 + eps
            x = newton(F, J, x, lam)

            det_high = np.linalg.det(J(x, lam))

            if det_low * det_high < 0:
                def det_eps(e):
                    x = x0 + e*x1
                    lam = lam0 + e
                    x = newton(F, J, x, lam)
                    return np.linalg.det(J(x, lam))

                eps = bisect(det_eps, 0, eps)
                xB = x0 + eps*x1
                lamB = lam0 + eps
                xB = newton(F, J, xB, lamB)
                new_points.append((xB, lamB))

                # find the branches
                A = J(xB, lamB)

                eigvals, eigvecs = np.linalg.eig(A)

                if np.sum(np.abs(eigvals) < 1e-10) != 1:
                    breakpoint()
                    assert False, 'dimension of kernel not appropriate'

                _, kernel = min(zip(np.abs(eigvals), eigvecs))

                # need to find out how to calculate the variation
                for sign in (1, -1):
                    x_branch = xB + sign * 0.1 * np.array([(5**.5-1)/2, 1.0])
                    lam_branch = lamB + 0.01 * (1/2 - 5**.5/5)
                    x_branch = newton(F, J, x_branch, lam_branch)
                    branch = Branch(self.prob, [(xB, lamB), (x_branch, lam_branch)])
                    branch.extend(max_lam)
                    self.prob.branches.append(branch)

            new_points.append((x, lam))
            x0 = x
            lam0 = lam

        self.soln = np.vstack([self.soln] + [p[0] for p in new_points])
        #print(self.param, np.array([p[1] for p in new_points]))
        self.param = np.concatenate([self.param, np.array([p[1] for p in new_points])])

class BifWidget:
    def __init__(self, bif, draw_soln_svg, xs=('soln', 1), ys=('param')):
        self.bif = bif
        self.draw_soln_svg = draw_soln_svg

    def select_data(self):
        self.draw_data = [
            np.hstack([b.param[:,np.newaxis], b.soln[:,1:2]])
        for b in self.bif.branches]

    def closest_point(self, xdata, ydata):
        def closest_point_on_branch(points):
            pdata = np.array([xdata, ydata])
            d = self.draw_data[0] - pdata
            i = np.argmin(np.sum(d*d, axis=1))
            return d[i], i

        print([(closest_point_on_branch(points), branch_index)
        for branch_index, points in enumerate(self.draw_data)])

        (dist, point), branch = min(
            (closest_point_on_branch(points), branch_index)
            for branch_index, points in enumerate(self.draw_data))

        return (branch, point, dist)

def branch_soln():
    """the given points are an eps-step off the first bifurcation point in the two rod problem"""
    sqrt5 = np.sqrt(5)
    lam0 = 0.5*(3-sqrt5)
    eps = 0.5
    lam = lam0 + eps**2 * 1/(5+sqrt5)
    the = eps * 1
    phi = eps * (2-(3-sqrt5)/2)
    #print(lam0, lam, phi, the)
    #print(np.linalg.det(J((0, 0), lam0)))
    x = newton(F, J, (the, phi), lam)
    return x, lam

def bisect(f, x0, x1, epsilon=1e-10):
    y0 = f(x0)
    y1 = f(x1)
    while True:
        assert y0*y1 < 0
        xm = (x1+x0)*0.5
        ym = f(xm)
        if abs(x1 - x0) < epsilon and abs(ym) < epsilon:
            return xm
        if y0*ym < 0:
            x1, y1 = xm, ym
        else:
            x0, y0 = xm, ym
#
# def continuation(F, x0, lam0):
#     for _ in range(20):
#         Fx0 = F(x0, lam0)
#         assert Fx0.dot(Fx0) < 1e-20, 'given point is not a solution'
#         # expansion  x=x0+eps x1 + eps**2 x2 + ...
#         # dF(x)/deps at eps=0 = J(x0)*x1
#         # J*x1 + F_lam*lam1 = 0
#         # set lam1 = 1
#
#         print(np.linalg.eigvals(J(x0, lam0)))
#
#         b = F_lam(x0, lam0)
#         x1 = np.linalg.solve(J(x0, lam0), -b)
#         eps = 0.01
#
#         # if the det changes the sign, we skipped a bif point.
#         det_low = np.linalg.det(J(x0, lam0))
#
#         x = x0 + eps*x1
#         lam = lam0 + eps
#         x = newton(F, J, x, lam)
#
#         det_high = np.linalg.det(J(x, lam))
#
#         if det_low * det_high > 0:
#             yield x, lam
#             x0 = x
#             lam0 = lam
#         else:
#             def det_eps(e):
#                 x = x0 + e*x1
#                 lam = lam0 + e
#                 x = newton(F, J, x, lam)
#                 return np.linalg.det(J(x, lam))
#
#             eps = bisect(det_eps, 0, eps)
#             x = x0 + eps*x1
#             lam = lam0 + eps
#             x = newton(F, J, x, lam)
#             yield x, lam
#
#             # find the branches
#             A = J(x, lam)
#
#             breakpoint()
#             eigvals, eigvecs = np.linalg.eig(A)
#
#             if np.sum(np.abs(eigvals) < 1e-10) != 1:
#                 assert False, 'dimension of kernel not appropriate'
#
#             _, kernel = min(zip(np.abs(eigvals), eigvecs))
#
#
#             return
#
#
#
#             return
#
# bif = BifurcationAnalysis(tworod.F, tworod.J, tworod.F_lam, (0,0), 0)
#
# print(bif.branches[0].extend(4))
#
# print(bif.branches[0].points)
