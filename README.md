# ResDMDc: Residual Dynamic Mode Decomposition with Control

Residual dynamic mode decomposition (ResDMD) offers easily computable error bounds on the eigenvalues of a data-driven Koopman operator approximation. In it's original formulation, ResDMD was applied to unforced dynamical systems, see [this paper][1]. In this project, we extend this formulation to include control, i.e. dynamical systems with a controlled external forcing term. We call this algorithm ResDMDc, which stands for ResDMD with control. Further, in order to minimize the eigenvalue errors from the onset, we employ active learning of the Koopman operator instead of naive control inputs to "randomly kick" the dynamical system.

At the current stage of this repo, there are two files:

* Item ResDMD_unforced: this is an implementation of the standard ResDMD algorithm
* Item ResDMD_forced: this is an implementation of our ResDMDc algorithm

In both of these files, there are a three (forced) systems to choose from: the Van der Pol Oscillator and two linear diagonal systems, one in continuous time and the other in discrete time. There are also a number of possible dictionaries of Koopman observable functions for the user to choose from, including a number of handmade observables, polynomial observables, and Hermitian polynomial observables.

[1]: https://www.cambridge.org/core/services/aop-cambridge-core/content/view/67F1513D7E0E182E8094ABCD3E5E94ED/S0022112022010527a_hi.pdf/residual-dynamic-mode-decomposition-robust-and-verified-koopmanism.pdf