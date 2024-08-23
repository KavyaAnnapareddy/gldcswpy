import numpy as np
from gldcswpy._gldcswpy import Gld

def test_gldcswpy():
    # Example data- Chi-squared distribution 
    np.random.seed(0);
    data = np.random.chisquare(df=2, size=1000)
    # initiate class with data
    g = Gld(data)
    params = g.get_params(initial_guess=(0.64,0.1991))
    # VaR at 5%
    g.VaR(params,0.05)
    assert np.round(g.VaR(params,0.05),5) == -0.1546
    

