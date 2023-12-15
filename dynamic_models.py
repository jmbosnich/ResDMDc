"""
Describe this program

IMPORTANT: right now these are all control systems
TODO: give option for either controlled or unforced systems
"""

import autograd.numpy as np

# Classes for each specific model
class VanDerPol(object):
    """
    2D dynamics describing the van der pol osciallator
    """
    def __init__(self, eps=1.0, dt=0.001):
        self.name = "VanDerPol"
        self.eps = eps 
        self.dt = dt
        self.num_states = 2
        self.num_inputs = 1
        self.reset()

    def reset(self, state=None):
        if state is None:
            self.state = np.random.uniform(-3., 3., size=(self.num_states,))
        else:
            self.state = state.copy()
        return self.state.copy()
    
    def sample_input(self):
        return np.random.uniform(-3., 3., size=(self.num_inputs,))

    def dynamics(self, x, u):
        xdot = np.array([
            x[1],
            -x[0] + self.eps * (1 - x[0]**2) * x[1] + u[0]
        ])
        return xdot

    def step(self, u):
        k1 = self.dt * self.dynamics(self.state, u)
        k2 = self.dt * self.dynamics(self.state + k1/2., u)
        k3 = self.dt * self.dynamics(self.state + k2/2., u)
        k4 = self.dt * self.dynamics(self.state + k3, u)

        self.state = self.state + (1/6.) * (k1+2.0*k2+2.0*k3+k4)
        return self.state.copy()

class ContinuousLinearSystem(object):
    """
    2D diagonal, linear dynamics (easy easy easy)
    """
    def __init__(self, dt=0.001):
        self.name = "LinearSystem"
        self.dt = dt
        self.num_states = 2
        self.num_inputs = 1
        self.reset()
        self.evals = [np.exp(-1*dt), np.exp(-2*dt)] # eigenvalues of the discretized system

    def reset(self, state=None):
        if state is None:
            self.state = np.random.uniform(-10., 10., size=(self.num_states,))
        else:
            self.state = state.copy()
        return self.state.copy()
    
    def sample_input(self):
        return np.random.uniform(-1., 1., size=(self.num_inputs,))

    def dynamics(self, x, u):
        xdot = np.array([
            x[0] + x[1],
            -2*x[1] + u[0]
        ])
        return xdot

    def step(self, u):
        k1 = self.dt * self.dynamics(self.state, u)
        k2 = self.dt * self.dynamics(self.state + k1/2., u)
        k3 = self.dt * self.dynamics(self.state + k2/2., u)
        k4 = self.dt * self.dynamics(self.state + k3, u)

        self.state = self.state + (1/6.) * (k1+2.0*k2+2.0*k3+k4)
        return self.state.copy()
    
class DiscreteLinearSystem(object):
    """
    discrete 2D dynamics (the simplest system of all time)
    """
    def __init__(self, dt=0.001): 
        self.name = "LinearSystem"
        self.dt = dt
        self.num_states = 2
        self.num_inputs = 1
        self.reset()
        self.evals = [0.5, -0.75]

    def reset(self, state=None):
        if state is None:
            self.state = np.random.uniform(-5., 5., size=(self.num_states,))
        else:
            self.state = state.copy()
        return self.state.copy()
    
    def sample_input(self):
        return np.random.uniform(-1., 1., size=(self.num_inputs,))

    def dynamics(self, x, u):
        x_new = np.array([
            self.evals[0]*x[0] + u[0],
            self.evals[1]*x[1]
        ])
        return x_new

    def step(self, u):
        self.state = self.dynamics(self.state, u)
        return self.state.copy()
    
class SystemWithKoopmanInvariantSubspace(object):
    """
    continuous 2D system which admits a finite-dimensional Koopman invariant subspace

    Taken from "Koopman Invariant Subspaces and Finite Linear Representations of
    Nonlinear Dynamical Systems for Control" (Steve) Brunton, (Bing) Brunton, Proctor, Kutz
    """
    def __init__(self, mu=-1., lamb=0.1, dt=0.001): 
        self.name = "notlinear"
        self.dt = dt
        self.num_states = 2
        self.num_inputs = 1

        # note that we the default choices of mu and lambda
        # the system is unstable (intentionally)
        self.mu = mu
        self.lamb = lamb
        self.reset()

    def reset(self, state=None):
        if state is None:
            self.state = np.random.uniform(-5., 5., size=(self.num_states,))
        else:
            self.state = state.copy()
        return self.state.copy()

    def sample_input(self):
        return np.random.uniform(-1., 1., size=(self.num_inputs,))

    def dynamics(self, x, u):
        x_new = np.array([
            self.mu*x[0],
            self.lamb*(x[1] - x[0]**2) + u[0]
        ])
        return x_new

    def step(self, u):
        k1 = self.dt * self.dynamics(self.state, u)
        k2 = self.dt * self.dynamics(self.state + k1/2., u)
        k3 = self.dt * self.dynamics(self.state + k2/2., u)
        k4 = self.dt * self.dynamics(self.state + k3, u)

        self.state = self.state + (1/6.) * (k1+2.0*k2+2.0*k3+k4)
        return self.state.copy()
    
class SpringMassSystem(object):
    """
    Single 2nd order mass-spring system with no damping
    """
    def __init__(self, dt=0.001, k=1, m=1):
        self.name = "SpringMass"
        self.dt = dt
        self.k = k  # spring constant
        self.m = m  # mass
        self.num_states = 2
        self.num_inputs = 1
        self.reset()

    def reset(self, state=None):
        if state is None:
            self.state = np.random.uniform(-5., 5., size=(self.num_states,))
        else:
            self.state = state.copy()
        return self.state.copy()
    
    def sample_input(self):
        return np.random.uniform(-5., 5., size=(self.num_inputs,))

    def dynamics(self, x, u):
        xdot = np.array([
            x[1],
            (u[0] - self.k*x[0])/self.m
        ])
        return xdot

    def step(self, u):
        k1 = self.dt * self.dynamics(self.state, u)
        k2 = self.dt * self.dynamics(self.state + k1/2., u)
        k3 = self.dt * self.dynamics(self.state + k2/2., u)
        k4 = self.dt * self.dynamics(self.state + k3, u)

        self.state = self.state + (1/6.) * (k1+2.0*k2+2.0*k3+k4)
        return self.state.copy()