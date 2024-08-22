import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats
from scipy.optimize import minimize, fminbound

import warnings
warnings.filterwarnings('ignore')

class Gld:
    """
    Generalized Lambda Distribution in CSW Parametrization Class
    
    Generalized Lambda Distribution (GLD) is a flexible family of continuous 
    probability distributions that can assume distributions with a large range of shapes. 
    Chalabi et al [2012] introduced a new parameterization of GLD, referred to as CSW 
    Parameterization, wherein the location and scale parameters are directly expressed as 
    the median and interquartile range of the distribution. The two remaining parameters 
    characterizing the asymmetry and steepness of the distribution are calculated numerically. 
    
    This tool implements the CSW parameterization types of GLD, introduced by 
    Chalabi, Y., Scott, D.J., & Wuertz, D. 2012.  It provides methods for calculating parameters 
    of theoretical GLD based on empirical data, generating random sample, estimate 
    Quantile based risk measures such as VaR, ES and so on.
    ----------------------------------------

    GLD in CSW parameterization is a transformation of GLD in FKML form (introduced by 
    Freimer, Mudholkar, Kollia and Lin, 1988). This formulation allows for relaxed
    constraints upon support regions and for existence of moments. 
    
        Params in CSW Paramterization:
        1. Location-> median (𝜇̃)
        2. Scale -> Inter-quartile range (𝜎̃)
        3. Shape, asymmetry -> chi (𝛘)
        4. Shape, steepness -> xi (ξ)
        
    It is characterized by quantile function Q(u) and density quantile function f(u). 
    
    Q(u|𝜇̃,𝜎̃,𝛘,ξ) = 𝜇̃ + 𝜎̃ (S(u|𝛘,ξ)-S(0.5|𝛘,ξ))/(S(0.75|𝛘,ξ)-S(0.25|𝛘,ξ))
    
    f(u|𝜎̃,𝛘,ξ) =  (S(0.75|𝛘,ξ)-S(0.25|𝛘,ξ))/(𝜎̃ d/du S(u|𝛘,ξ)) where
    d/du S(u|𝛘,ξ) = u^(𝛼+𝛽−1) + (1−u)^(𝛼−𝛽−1) where
    𝛼 = 0.5 (0.5-ξ)/(sqrt(ξ(1-ξ))
    𝛽 = 0.5 (𝛘)/(sqrt(1-𝛘^2))
    
    ----------------------------------------
    References:
    1. Chalabi, Y., Scott, D.J., & Wuertz, D. 2012. Flexible distribution modeling with the generalized lambda distribution. 
    2. Freimer, M., Kollia, G., Mudholkar, G.S., & Lin, C.T. 1988. A study of the
        generalized Tukey lambda family. Communications in Statistics-Theory and Methods, 17, 3547–3567.
    3. S. Su. A discretized approach to flexibly fit generalized lambda distributions to data. Journal of Modern Applied Statistical Methods, 4(2):408–424, 2005.
    ----------------------------------------
    """
    
    def __init__(self, data):
        self._data = np.array(data).ravel()
        
    def get_params(self,initial_guess=(0.514,0.337), method='robust_moments_matching', 
                       tol = 1e-6, disp_fit= True, random_state= None,**kwargs):
        """
        Outputs parameters of estimated GLD distribution 
        for data in CSW paramterization 
        section 4, Chalabi et al 2012
        ----------------------------------------
        Parameters:
        - initial_guess : array-like
            The initial guess is for shape parameters, chi and xi (𝛘 and ξ)
            Refer to Figures 8-10, Chalabi et al 2012, for different sets of shape parameters
            initial_guess to be within domains of shape parameters 𝛘 ∈ (-1,1) and ξ ∈ (0,1) 
        - method : str
            Various methods to estimate parameters, section 4, Chalabi et al 2012:
                1. Robust Moments Matching (robust_moments_matching)
                2. Histogram Approach (histogram_approach)
            The default is robust_moments_matching
        - tol: float  
            Tolerance for termination of the selected minimization algorithm.
        - disp_fit : bool
            Plots PDF, CDF and Q-Q plot to visualize fit
        - random_state : None or int, optional
            The seed of the pseudo random number generator. The default is None.
        - bin_method : str
            Three methods for calculating number of histogram bins, section 3.5
                1. Sturges breaks (sturges)
                2. Scott breaks (scott)
                3. Freedman-Diaconis breaks (freedman-diaconis)   
        ----------------------------------------
        Output:
        - Array of location, scale, asymmetry and steepness parameters in CSW Paramterization 
            (Median, InterQuartile Range, Chi, Xi) 
        - if disp_fit is set to True , plots distribution function fit to data   

        """
        bin_method = kwargs.get('bin_method',"freedman-diaconis")
        initial_guess = np.array(initial_guess)
        # median
        median = self._pi(self._data,0.5)
        # interquartile range
        iqr = self._interquartile_range(self._data)
        # chi and xi
        if method=='robust_moments_matching':
            chi,xi = self._robust_moments_chi_xi(self._data,initial_guess, tol)
        elif method=='histogram_approach':
            chi,xi = self._histogram_approach_chi_xi(self._data, 
                                                    initial_guess,tol, bin_method)
        else:
            raise ValueError('Parameter Estimation Method not valid')

        params =  median, iqr, chi, xi
        print("CSW params: ", params)
        # Goodnesss of fit test - Kolmogorov-Smirnov test
        ks_test = stats.kstest(np.array(self._data).ravel(), self.cdf_x(params,self._data))
        print("Kolmogorov-Smirnov test : \n",ks_test)
        # plot to see how good the GLD fit is to data
        if disp_fit:
            plt.rcParams.update({'font.size': 14})
            fig,ax = plt.subplots(1,2,figsize = (12,5))
            ax[0].hist(self._data,bins=  self._hist_nbins(self._data, 'freedman-diaconis'),
                       density = True,color = 'skyblue')
            u_array = np.linspace(0.0001,0.9999,1000)
            ax[0].plot(self._Q(params,u_array),self.pdf(params,u_array),lw = 2,color = 'r')
            ax[0].set_title('PDF', fontsize=18)
            ax[0].grid()
            ax[1].plot(np.sort(self._data), np.arange(len(self._data))/len(self._data),color = 'skyblue',lw = 2)
            ax[1].plot(self._Q(params,u_array),u_array, color= 'r')
            ax[1].grid()
            ax[1].set_title('CDF', fontsize=18)
            fig.suptitle(f"GLD {method} Fit on data",fontsize=22)
            plt.tight_layout()
            plt.show()
        return params

    def generate_sample(self,params,n,m=1,random_state = None, disp_fit= True):
        """
        Generates samples of size (n,m) from a GLD distribution 
        defined by its parameters (params)
        ----------------------------------------
        Parameters:
        - params : array-like
            Parameters of GLD in CSW Parametrization. Obtained from function get_params
        - n : int
            Number of points in sample data
        - m : int, optional
            generates 2D sample array. default value of m is 1 for 1d array. 
        - random_state : None or int, optional
            The seed of the pseudo random number generator. The default is None.
        ----------------------------------------
        Output:
        Array of size (n,m)
        
        """
        # u_array random floats in the half-open interval [0.0, 1.0)
        rng = np.random.default_rng(seed=random_state)
        if m>1:
            u_array = rng.random((n,m))
        else:
            u_array = rng.random((n,))
        sample_array = np.array([self._Q(params, u) for u in u_array])
        if disp_fit:
            self._display_fit(params,sample_array)
        return sample_array 
        
    def VaR(self,params,u):
        """
        outputs Value-at-Risk at probability u
        maximum loss forecast that may happen with 
        probability u ∈ [0,1] over the holding period
        section 5.2, Chalabi et al 2012
        ----------------------------------------
        Parameters:
        - params : array-like
            Parameters of GLD in CSW Parametrization. Obtained from function get_params
        - u : float, array-like
            Lower tail probability, must be between 0 and 1.
        ----------------------------------------
        Output:
        Value at Risk with specified probability over the holding period
        Note: expect negative value/s
        """
        if self._check_params(params):
            pass
        else:
            raise ValueError('CSW Parameters are not valid')
        return -self._Q(params,u)

    def ES(self,params,u):
        """
        outputs Expected Shortfall - average VaR over the interval [0,u]
        section 5.2, Chalabi et al 2012
        
        ----------------------------------------
        Parameters:
        - params : array-like
            Parameters of GLD in CSW Parametrization. Obtained from function get_params
        - u : float, array-like
            Lower tail probability, must be between 0 and 1.
        ----------------------------------------
        Output:
        Expected Shortfall with specified probability over the holding period
        Note: expect negative value/s
        """
        if self._check_params(params):
            median, iqr, chi, xi = params
        else:
            raise ValueError('CSW Parameters are not valid')

        A = iqr/(self._s_function(3/4,chi, xi)-self._s_function(1/4,chi, xi))
        B = -1/(self._s_function(3/4,chi, xi)-self._s_function(1/4,chi, xi))
        a = self._alpha(xi)
        b = self._beta(chi)

        if chi == 0 and xi == 0.5:
            return u*(B+median+A*np.log(u))+ (A-A*u)*np.log(1-u)
        elif chi!=0 and xi==(1+chi)/2 :
            return B*u + A*(((1-u)**(1+2*a)-1)/2*a + 4*a**2) + A*u*((1/(2*a)) + np.log(u)-1) + u*median
        elif chi!=0 and xi== (1-chi)/2:
            return ((A-A*u)*np.log(1-u)) + u*(B+median) + (A*u*(-1+4*b**2+u**(2*b)))/(2*b*(1+2*b))
        else:
            return  -((u*median) + (A*u*(1+a+b))/((a+b)*(1+a+b)) - (A*u)/(a+b) + (A*u)/(a-b) +B*u + (A*((1-u)**(1+a)-(1-u)**b)*(1-u)**(-b))/((a-b)*(1+a-b)))

    def plot_pdf(self,sample_array):
        """
        Plot Probability Distribution of sample data
        ----------------------------------------
        Parameters:
        - sample_array : array-like
            Sample data generated from GLD
        ----------------------------------------
        Output:
        Figure displaying Probability Distribution of data
        
        """
        sns.histplot(sample_array, color='blue',label='GLD', kde= True)
        plt.xlabel('Data')
        plt.ylabel('Density')
        plt.title('PDF of Data')
        plt.show()
    
    def plot_cdf(self,sample_array):
        """
        Plot Cumulative Distribution of sample data
        ----------------------------------------
        Parameters:
        - sample_array : array-like
            Sample data generated from GLD
        ----------------------------------------
        Output:
        Figure displaying Cumulative Distribution of data
        
        """
        sns.histplot(sample_array, stat='proportion', cumulative=True, alpha=.4)
        sns.ecdfplot(sample_array, stat='proportion')
        plt.xlabel('Data')
        plt.ylabel('Proportion of Data')
        plt.title("CDF of Data")
        plt.show()
    
    def q_q_plot(self,sample_array):  
        """
        Generates a probability plot of sample data against the quantiles of Normal Distribution
        ----------------------------------------
        Parameters:
        - sample_array : array-like
            Sample data generated from GLD
        ----------------------------------------
        Output:
        Figure displaying Quantile-Quantile plot of data
        
        """        
        plt.figure(figsize=(12,9))
        res = stats.probplot(sample_array, dist='norm', plot=plt)
        plt.title('Q-Q Plot of Data', fontsize=16)
        plt.ylabel('Returns')
        plt.show()
    
    def cdf_x(self,params,x):
        """
        Calculates Cumulative distribution function (F_CSW) 
        of GLD at value/ array of x 
        ----------------------------------------
        Raises Value Error if csw parameters are not valid
        ----------------------------------------
        F_csw[Q_csw(u)] = u for all u∈[0,1]
        CDF at x is found numerically by 
        finding a local minimizer of the scalar function find_cdf_cost 
        in the interval 0 < u_optimal < 1 using Brent’s method
        
        ----------------------------------------
        Parameters:
        - params : array-like
            Parameters of GLD in CSW Parametrization. Obtained from function get_params
        - x : float, array-like
            A value from GLD for which probability that the random variable X is less than or equal to x
            F(x)=Pr[X≤x]
        ----------------------------------------
        Output:
        float, array-like
        Note: output between 0 and 1
        """
        def find_cdf_cost(u):
            """
            cost function that needs to be minimized
            to estimate u - lower tail probability
            """
            return np.square(self._Q(params,u)-x_arg)

        if self._check_params(params):
            median, iqr, chi, xi = params
        else:
            raise ValueError('CSW Parameters are not valid')
            
        min_val,max_val= self._params_support(params)
        x= np.array([x]).ravel()
        # start cdf result with na
        result_cdf = x*np.nan
        result_cdf[x<min_val] = 0
        result_cdf[x>max_val] = 1
        index_mask = np.argwhere(np.isnan(result_cdf)).ravel()
        for i in index_mask:
            x_arg = x[i]

            result_cdf[i] = fminbound(find_cdf_cost, x1=0,x2=1,
                                       xtol=1e-05)
        return result_cdf
    
    def pdf_x(self,params,x):
        """
        Calculates Probability distribution function (f_CSW) 
        of GLD at value/ array of x numerically
        ----------------------------------------
        Parameters:
        - params : array-like
            Parameters of GLD in CSW Parametrization. Obtained from function get_params
        - x : float, array-like
            A value from GLD for which probability that the random variable X is equal to x
            Pr[X=x]
        ----------------------------------------
        Output:
        float, array-like
        Note: output between 0 and 1
        """
        u = self.cdf_x(params,x)
        result_pdf = self.pdf(params, u)
        min_val,max_val= self._params_support(params)
        result_pdf[np.logical_or(x<min_val, x>max_val)] = 0
        return result_pdf
    
    def qdf(self,params, u):
        """
        outputs the quantile density function at u of GLD
        ----------------------------------------
        q(u) = Q'(u), derivative of Quantile function Q_csw
        domains of shape parameters 𝛘 ∈ (-1,1) and ξ ∈ (0,1)
        eq 11, Chalabi et al 2012
        ----------------------------------------
        Parameters:
        - params : array-like
            Parameters of GLD in CSW Parametrization. Obtained from function get_params
        - u : float, array-like
            Lower tail probability, must be between 0 and 1.
        ----------------------------------------
        Output:
        float, array-like
        
        """
        median, iqr, chi, xi = params
        d_du = self._first_derivative_s_function(u,chi,xi)
        qdf = iqr/(self._s_function(3/4,chi, xi)-self._s_function(1/4,chi, xi))*d_du
        return qdf 
    
    def pdf(self,params, u):
        """
        outputs the density quantile function at u of GLD
        ----------------------------------------
        fQ(u) = 1/q(u)
        domains of shape parameters 𝛘 ∈ (-1,1) and ξ ∈ (0,1)
        eq 10, Chalabi et al 2012
        ----------------------------------------
        Parameters:
        - params : array-like
            Parameters of GLD in CSW Parametrization. Obtained from function get_params
        - u : float, array-like
            Lower tail probability, must be between 0 and 1.
        ----------------------------------------
        Output:
        float, array-like
        """
        return 1/self.qdf(params, u)
    
    def fit_curve(self,initial_guess, method,tol = 1e-6,n=1000,random_state= None):
        """
        Plots the density plot of data and sample array generated from GLD
        Aids in obtaining the shape of GLD by setting the initial_guess  
        GLD parameter estimation is very sensitive to the initial guess.
        ----------------------------------------
        Parameters:
        - initial_guess : array-like
             The initial guess is for shape parameters (𝛘 and ξ)
             Refer to Figures 8-10, Chalabi et al 2012, for different sets of shape parameters
            initial_guess to be within domains of shape parameters 𝛘 ∈ (-1,1) and ξ ∈ (0,1)
        - method : str
            Various methods to estimate parameters, section 4, Chalabi et al 2012:
                1. Robust Moments Matching (robust_moments_matching)
                2. Histogram Approach (histogram_approach)
        - tol: float
            Tolerance for termination of the selected minimization algorithm.
        - n : int
            Sample size for generating sample. default is 1000.
        - random_state : None or int, optional
            The seed of the pseudo random number generator. The default is None.
        
        """
        sample_params = self.get_params(initial_guess,method,random_state,disp_fit= False)
        sample_data = self.generate_sample(sample_params,n)
    
    def _Q(self,params,u):
        """
        Quantile Distribution Function (Q_CSW) of the GLD in the CSW parameterization
        outputs the value of a random variable X such that the probability of X is less than or equal to u
        ----------------------------------------
        Raises Value Error if csw parameters are not valid
        ----------------------------------------
        P(X< Q_CSW(u)) = u ∀u∈[0,1]
        where Q_CSW(u) is the quantile function,
        X - random variable, and 
        lower tail probability value u
        eq 9 , Chalabi et al 2012
        ----------------------------------------
        Parameters:
        - params : array-like
            Parameters of GLD in CSW Parametrization. Obtained from function get_params
        - u : float, array-like
            Lower tail probability, must be between 0 and 1.
        ----------------------------------------
        Output:
        float, array-like
        
        """ 
        if self._check_params(params):
            median, iqr, chi, xi = params
        else:
            raise ValueError('CSW Parameters are not valid')
        X = median + iqr*((self._s_function(u,chi, xi)-self._s_function(1/2,chi, xi))/(self._s_function(3/4,chi, xi)-self._s_function(1/4,chi, xi)))
        return X
    
    def _check_params(self,params):
        """
        Checks if parameters are valid within CSW Parameterization 
        ----------------------------------------
        Condtions:
        1. There are 4 params - median, iqr, chi, xi
        2. chi(𝛘) ∈ (-1,1) and xi(ξ) ∈ (0,1)
        ----------------------------------------
        Parameters:
        - params : array-like
            Parameters of GLD in CSW Parametrization. Obtained from function get_params
        ----------------------------------------
        Output:
        True, False or ValueError
        """
        if len(params)!=4:
            raise ValueError('GLD has 4 parameters')            
        if not np.isfinite(params).all():
            return False
        if -1<params[2]<1 and 0<params[3]<1 :
            return True

    def _params_support(self,params):
        """
        outputs value bounds of GLD defined by params
        Minimum and Maximum possible values of GLD with specified CSW Parameters
        ----------------------------------------
        Parameters:
        - params : array-like
            Parameters of GLD in CSW Parametrization. Obtained from function get_params
        ----------------------------------------
        Output:
        Array of length 2. 
        
        """
        min_val = self._Q(params,0.00001)
        max_val = self._Q(params,0.9999)
        return min_val, max_val 

    def _pi(self,array,q,method='midpoint'):
        """
        returns q-th quantile of the array input data
        ----------------------------------------
        Parameters:
        - array : array-like
            data array for which qth quantile is calculated
        - q : float, array-like
            quantile, such as 0.25, 0.5, 0.75,0.9, etc
        - method: str  
            method to use for estimating the quantile.  
            other numpy methods available, numpy deafult -'linear'
        """
        return np.quantile(a=array, q=q, method=method)

    def _interquartile_range(self,array):
        """
        outputs interquartile range of input array
        difference between values at 75th and 25th percentile
        Interquartile range = Q(0.75) - Q(0.25)
        ----------------------------------------
        Parameters:
        - array : array-like
            data array for which interquantile range is calculated
        """
        return  self._pi(array,q=0.75)-self._pi(array,q=0.25)

    def _alpha(self,xi):
        """
        helper function for _s_function 
        𝜉 -> xi
        eq 7a ,Chalabi et al 2012
        """
        return (0.5*((0.5-xi)/(math.sqrt(xi*(1-xi)))))

    def _beta(self,chi):
        """
        helper function for _s_function 
        𝜒 -> chi 
        eq 7b , Chalabi et al 2012
        """
        return (0.5*chi/math.sqrt(1-chi**2))

    def _s_function(self,u,chi, xi):
        """
        The S function S(u|𝜒,𝜉) in terms of shape parameters 𝜒,𝜉
        probability u obtained from quantile based estimators
        used within quantile function of GLD in CSW Parametrization
        eq 8, Chalabi et al 2012
        𝜒 -> chi 
        𝜉 -> xi
        """

        if np.logical_and(u>0, u<1).all():
            if chi == 0 and xi == 0.5:
                return np.log(u) - np.log(1-u)
            elif chi!=0 and xi==(1+chi)/2 :
                return np.log(u)- (((1-u)**(2*self._alpha(xi))-1)/(2*self._alpha(xi)))
            elif chi!=0 and xi== (1-chi)/2:
                return ((u**(2*self._beta(chi))-1)/(2*self._beta(chi)))-(np.log(1-u))
            else:
                return ((u**(self._alpha(xi)+self._beta(chi))-1)/(self._alpha(xi)+self._beta(chi))) - ((1-u)**(self._alpha(xi)-self._beta(chi))-1)/(self._alpha(xi)-self._beta(chi))
        elif u==0:
            if xi < (1+chi)/2:
                return -1/(self._alpha(xi)+self._beta(chi))
            else: 
                return -np.inf
        elif u==1:
            if xi < (1-chi)/2:
                return 1/(self._alpha(xi)-self._beta(chi))
            else:
                return np.inf
        else:
            raise ValueError("u has to be in range [0,1] inclusive")
   
    def _first_derivative_s_function(self,u,chi,xi):
        """
        outputs d/du S(u|𝜒,𝜉) which is derivative of _s_function
        """
        a = self._alpha(xi)
        b = self._beta(chi)
        return (u**(a+b-1)+(1-u)**(a-b-1))

    def _robust_moments_chi_xi(self,array,initial_guess,tol):
        """
        Uses Robust Moments matching approach to estimate GLD distribution shape parameters
        ----------------------------------------
        Section 4.1 , Chalabi et al 2012
        𝛘-> asymmetry paramter proportional to λ3- λ4, FKML parameterization
        ξ -> steepness parameter proportional to λ3+ λ4, FKML parameterization
        Based on median, interquartile range, Bowley's skewness and Moor's Kurtosis
        Bounds- Sequence of (min, max) pairs for each element in x
        x0 is initial guess
        domains of shape parameters 𝛘 ∈ (-1,1) and ξ ∈ (0,1) are bounds
        ----------------------------------------
        Parameters:
        - array : array-like
            data for which chi and xi parameters are estimated
        - initial_guess : array-like
            initial_guess to be within domains of shape parameters 𝛘 ∈ (-1,1) and ξ ∈ (0,1)
        - tol : float, optional  
            Tolerance for termination of the selected minimization algorithm.
        ----------------------------------------
        Output:
        - Array of asymmetry and steepness parameters in CSW Paramterization (Chi, Xi)
        ----------------------------------------
        """
        skew= self._bowley_skewness_ratio(array)
        kurt= self._moor_kurtosis_ratio(array)

        def cost_function(sol):
            chi,xi = sol
            return np.square(self._population_skewness(chi,xi)-skew)+ np.square(self._population_kurtosis(chi,xi)-kurt)
        res = minimize(cost_function, x0=initial_guess, bounds= ((-1,1),(0,1)),
                      tol= tol)  
        return res.x

    def _histogram_approach_chi_xi(self,array, initial_guess,tol,bin_method):
        """
        Uses Histogram Approach to estimate GLD distribution shape parameters
        ----------------------------------------
        Approach: The empirical data is binned into a histogram and resulting probabilities, taken to be 
        at the midpoints of the histogram bins are fitted to the true GLD density. 
        Three methods for choosing approporiate number of bins:
        1. Sturges breaks (sturges)
        2. Scott breaks (scott)
        3. Freedman-Diaconis (freedman-diaconis)
        ----------------------------------------
        Parameters:
        - array : array-like
            data for which chi and xi parameters are estimated
        - initial_guess : array-like
            initial_guess to be within domains of shape parameters 𝛘 ∈ (-1,1) and ξ ∈ (0,1)
        - tol : float, optional  
            Tolerance for termination of the selected minimization algorithm.
        - bin_method: str
            one of the three binning methods. default method is freedman-diaconis
        ----------------------------------------
        Output:
        - Array of asymmetry and steepness parameters in CSW Paramterization (Chi, Xi) 
        """
        bin_width, bin_midpoints, proportion= self._get_hist_patches(array,bin_method)     
        median = self._pi(array,0.5)
        iqr = self._interquartile_range(array)

        def cost_function(sol):
            """
            cost function that needs to be minimized
            to estimate chi and xi 
            """
            chi,xi = sol
            params = median, iqr, chi, xi 
            GLD_proportion = []
            for i in bin_midpoints:
                # proportion of data at point i from theoretical GLD distribution
                proportion_i = self.pdf_x(params,i)*bin_width
                GLD_proportion.append(proportion_i)
            GLD_proportion = np.array(GLD_proportion).ravel()
            return (proportion*(np.square(proportion-GLD_proportion))).sum()
        res = minimize(cost_function,method="Nelder-Mead", x0=initial_guess,
                       bounds= ((-1,1),(0,1)),tol= tol)  
        return res.x
    
    def _bowley_skewness_ratio(self,array, method='midpoint'):
        """
        outputs robust skewness ratio of Bowley(1920)
        for input array a
        using numpy quantile method 
        """
        s = (self._pi(array,3/4)+self._pi(array,1/4)-2*self._pi(array,2/4))/(self._pi(array,3/4)-self._pi(array,1/4))
        return s

    def _moor_kurtosis_ratio(self,array,method='midpoint'):
        """
        outputs robust Kurtosis ratio of Moor
        for input array a
        using numpy qunatile method
        """
        k = (self._pi(array,7/8)-self._pi(array,5/8)+self._pi(array,3/8)-self._pi(array,1/8))/(self._pi(array,6/8)-self._pi(array,2/8))
        return k
    
    def _population_skewness(self,chi,xi):
        """
        outputs population robust skewness
        eq 16(a) Chalabi et al 2012
        """
        return (self._s_function(3/4,chi, xi)+self._s_function(1/4,chi, xi)-2*self._s_function(2/4,chi, xi))/(self._s_function(3/4,chi, xi)-self._s_function(1/4,chi, xi))

    def _population_kurtosis(self,chi,xi):
        """
        outputs population robust kurtosis
        eq 16(b) Chalabi et al 2012
        """
        return (self._s_function(7/8,chi, xi)-self._s_function(5/8,chi, xi)+self._s_function(3/8,chi, xi)-self._s_function(1/8,chi, xi))/(self._s_function(6/8,chi, xi)-self._s_function(2/8,chi, xi))
    
    def _hist_nbins(self,array, bin_method):
        """
        Calculates number of bins for data array
        using 3 methods for calculating number of histogram bins
        1. Sturges breaks (sturges)
        2. Scott breaks (scott)
        3. Freedman-Diaconis breaks (freedman-diaconis)
        """
        n= len(array)
        max_val = np.max(array)
        min_val = np.min(array)

        if bin_method == 'sturges':
            n_bins= np.ceil(math.log2(n+1)).astype(int)
        elif bin_method == 'scott':
            sd = np.std(array)
            if  sd ==0:
                n_bins = 1
            else:
                h= 3.49* sd*n**(1/3)
                n_bins = np.ceil((max_val-min_val)/h).astype(int)
        elif bin_method == 'freedman-diaconis':
            iqr = self._interquartile_range(array)
            if iqr == 0:
                h= stats.median_abs_deviation(array)
            else:     
                h = iqr/n**(1/3)
            n_bins = np.ceil((max_val-min_val)/h).astype(int)
        else:
            raise ValueError("Specify method to determine the number of bins")
        return n_bins
    
    def _get_hist_patches(self,array, bin_method):
        """
        helper function for histogram approach of finding CSW Parameters
        bin_method can either be sturges, scott or freedman-diaconis
        """
        n_bins = self._hist_nbins(array, bin_method)
        (counts, bins, patches) = plt.hist(array,cumulative=False, density=False, bins=n_bins)
        plt.title(f"Histogram with {n_bins} bins based on {bin_method} method")
        plt.ylabel("Counts")
        plt.xlabel("Data Range")
        proportion = counts/counts.sum()
        bin_width = bins[2]-bins[1]
        bin_midpoints = []
        for i in range(len(bins)-1):
            bin_mid = (bins[i] + bins[i+1])/2
            bin_midpoints.append(bin_mid)
        return bin_width, bin_midpoints, proportion 
    
    def _display_fit(self,params, sample_array):
        """
        Plots the density plot of data and input sample array
        ----------------------------------------
        Parameters:
        - params : array-like
            Parameters of GLD in CSW Parametrization. Obtained from function get_params
        - sample_array : array-like
            Sample data generated from GLD    
        """
        print("CSW Params: ", params)
        fig, ax = plt.subplots(figsize=(7 ,5))
        sns.kdeplot(self._data, ax= ax,color='green',label='Data', )
        sns.kdeplot(sample_array.ravel(), ax= ax, color='blue',label='GLD Sample')
        plt.title("Sample fit of GLD on data")
        plt.legend()
        plt.show()