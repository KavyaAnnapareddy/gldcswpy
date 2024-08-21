# gldcswpy

gldcswpy is Python package that implements tools for using Generalized Lambda Distribution (GLD) in CSW Parametrization for Python.

The Generalized Lambda Distribution (GλD) is a flexible family of probability distributions that can assume a wide range of shapes, making it a valuable tool in statistical modeling. It specified by four parameters which determine location, scale and shape (skewness and kurtosis) of the distribution.

Chalabi et al (2012) introduced a new parameterization of GLD, referred to as CSW Parameterization, wherein the location and scale parameters are directly expressed as the median and interquartile range of the distribution. The two remaining shape parameters characterizing the asymmetry and steepness of the distribution are calculated numerically. 
    
This tool implements the CSW parameterization types of GLD, introduced by 
Chalabi, Y., Scott, D.J., & Wuertz, D. 2012.  It provides methods for calculating parameters of theoretical GLD based on empirical data, generating random sample, estimate Quantile based risk measures such as VaR, ES and so on.

# Installation
pip install gldcswpy

# Usage
https://github.com/KavyaAnnapareddy/using_gldcswpy

# References:
1. Chalabi, Y., Scott, D.J., & Wuertz, D. 2012. Flexible distribution modeling with the generalized lambda distribution.
2. Freimer, M., Kollia, G., Mudholkar, G.S., & Lin, C.T. 1988. A study of the generalized Tukey lambda family. Communications in Statistics-Theory and Methods, 17, 3547–3567.
3. S. Su. A discretized approach to flexibly fit generalized lambda distributions to data. Journal of Modern Applied Statistical Methods, 4(2):408–424, 2005.



