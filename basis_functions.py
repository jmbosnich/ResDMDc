"""
Basis functions

IMPORTANT: right now these are all control systems
TODO: give option for either controlled or unforced systems
"""

import autograd.numpy as np
from autograd import grad

class Observables(object):
    
    def __init__(self, choice=0, order=0, cross_terms=False):
        self.choice = choice
        self.order = order
        self.cross_terms = cross_terms

    def handmade(self, x, u):
        # note: all of these must have the state as the first two observables
        if self.choice == 1:
            lift_x = np.array([
                x[0],
                x[1],
                x[0]**2,
                (x[0]**2)*x[1]
            ])

        if self.choice == 2:
            lift_x = np.array([
                x[0],
                x[1],
                x[0]**2,
                x[1]**2,
                x[0]*x[1]
            ])

        if self.choice == 3:
            lift_x = np.array([
                x[0],
                x[1],
                x[0]**2
            ])

        if self.choice == 4:
            lift_x = np.array([
                x[0],
                x[1],
                x[0]**2,
                x[1]**2,
                x[0]*x[1],
                x[0]**3,
                x[1]**3,
                (x[0]**2)*x[1],
                (x[1]**2)*x[0]
            ])

        if self.choice == 5:
            lift_x = np.array([
                x[0],
                x[1],
                np.sin(x[0]),
                np.sin(x[1]),
                np.cos(x[0]),
                np.cos(x[1])
            ])

        if self.choice == 6:
            lift_x = np.array([
                x[0],
                x[1],
                np.exp(x[0]),
                np.exp(x[1])
            ])

        if self.choice == 7:
            # hard-coded Hermitian polynomials up
            # to order 4
            lift_x = np.array([
                x[0],
                x[1],
                x[0]**2 - 1,
                x[1]**2 - 1,
                x[0]**3 - 3*x[0],
                x[1]**3 - 3*x[1],
                x[0]**4 - 6*x[0]**2 + 3,
                x[1]**4 - 6*x[1]**2 + 3
            ])

        #NOTE: since all of our systems have scalar input,
        #we only need a single control observable

        lift = np.append(lift_x, u[0]) 

        return lift

    def poly(self, x, u):
        """
        Basis made up of polynomials up to order n
        (includes all cross terms too, at user's request)
        """

        # make polynomials
        # note: here I use lambda functions and store them in a list
        polys = []
        for n in range(self.order):
            polys.append(lambda a, n=n: a**(n+1))
        
        # evaluate polynomials at current data point
        lift_x = np.zeros(2*self.order)
        for m in range(self.order):
            lift_x[2*m] = polys[m](x[0], m)
            lift_x[1 + 2*m] = polys[m](x[1], m)

        # compute and evaluate cross terms, if desired
        if self.cross_terms:
            iter = 0
            lift_cross_terms = np.zeros(self.order**2)
            for j in range(self.order):
                for i in range(self.order):
                    lift_cross_terms[iter] = polys[j](x[0], j)*polys[i](x[1], i)
                    iter += 1
            lift_x = np.insert(lift_cross_terms, 0, lift_x)

        #append with control observable
        lift = np.append(lift_x, u[0])
        return lift
    
    # The next two functions are helper functions for defining
    # and evaluating Hermitian polynomials
    def higher_derivative(self, func, order):
        """
        Compute higher order derivatives with autograd
        """
        for i in range(order):
            func = grad(func)
        return func
    
    def make_hermite(self, x, n):
        """
        Construct the Hermitian polynomial of order n
        """
        n += 1 # because n=0 => polynomial of 1
        deriv_func = lambda v: np.exp(-0.5*v**2)
        deriv_term = self.higher_derivative(deriv_func, n)
        herm_poly = lambda a: (-1.)**n * np.exp(0.5*a**2) * deriv_term(a)
        return np.array([herm_poly(x[0]), herm_poly(x[1])])

    def hermite(self, x, u):
        """
        Basis made up of Hermitian polynomials up to order n
        (includes all cross terms too, at user's request)

        Note: refer to the markdown box below for the form
        of an nth order Hermitian polynomial
        """
        # evaluate polynomials at current point
        lift_x = np.zeros(2*self.order)
        for m in range(self.order):
            [lift_x1, lift_x2] = self.make_hermite(x, m)
            lift_x[2*m] = lift_x1
            lift_x[1 + 2*m] = lift_x2

        # compute and evaluate cross terms, if desired
        if self.cross_terms:
            iter = 0
            lift_cross_terms = np.zeros(self.order**2)
            for j in range(self.order):
                for i in range(self.order):
                    lift_cross_terms[iter] = self.make_hermite(x, j)[0]*self.make_hermite(x, i)[1]
                    iter += 1
            lift_x = np.insert(lift_cross_terms, 0, lift_x)

        #append with control observable
        lift = np.append(lift_x, u[0])
        return lift
