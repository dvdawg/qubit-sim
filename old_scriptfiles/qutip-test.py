import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from qutip import * 
import qutip as qt

bloch = Bloch()
bloch.add_states(basis(2, 0))
    
bloch.make_sphere()
plt.show()