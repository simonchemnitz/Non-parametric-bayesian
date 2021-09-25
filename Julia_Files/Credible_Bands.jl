##   Load Packages
using BayesianNonparametricStatistics, Plots
using Statistics, Distributions
using DelimitedFiles, Plots, LaTeXStrings
using NBInclude, LinearAlgebra, SparseArrays

##   Colors
lblue = RGBA(83/255, 201/255, 250/255,0.7)
dblue = RGBA(47/255, 122/255, 154/255,0.7)

function calc_d_i(func1, func2)
    # Function to calculate the distance d_i
    # Between two functions f1, f2
    """
    Parameters
    -----------------------------------------
    func1 : Function that takes 1 input
        and returns 1 input.
        Assumed to be defined on [0,1]
    
    func2 : Same as func1
    
    Calculates the distance d_i defined by
    d_i = max{|f_1(x)-f_2(x)| : x in [0,1]}
    -----------------------------------------
    
    Output : float
    """
    # x-values to evaluate the two functions
    x_values = [0.0:0.01:1;]
    
    # Calculate the difference
    diff = abs.(func1.(x_values)-func2.(x_values))
    
    # Return the supremum
    return maximum(diff)
end


function sample_distances(M, alpha, post_dist)
    # Function to return the M*(1-alpha)
    # Closest functions to the posterior mean
    # using the above metric d_i.
    """
    Parameters
    -----------------------------------------
    M : int 
        Number of samples from the posterior
    
    alpha : float
        Number specifying the 
        credible level 1-alpha
    
    post_dist : GaussianProcess{S,T}
        Posterior distribution
    -----------------------------------------
    
    Output : Array{Float64}(undef, Int(ceil(M*(1-alpha))) )
    """
    # Posterior mean
    theta_bar = mean(post_dist)
    
    ##   Matrix to contain functions and their distance
    distances = reshape([],0,2)
    
    # M times do:
    # Sample a function from posterior
    # Calculate distance
    for i in 1:M
        post_sample = rand(post_dist)
        observation = reshape([post_sample, calc_d_i(post_sample, theta_bar)],1,2)
        distances = vcat(distances, observation)
    end
    
    # Sort all samples by lowest distance
    distances = distances[sortperm(distances[:, 2]), :]
    # Treshold [M(1-a)]
    threshold = Int(ceil(M*(1-alpha)))
    
    # Return the closest functions
    return distances[1:threshold,:]
end


function cred_bands(M, alpha, post_dist)
    # Function to return upper- and lower- 
    # -band functions
    """
    Parameters
    -----------------------------------------
    M : int 
        Number of samples from the posterior
    
    alpha : float
        Number specifying the 
        credible level 1-alpha
    
    post_dist : GaussianProcess{S,T}
        Posterior distribution
    -----------------------------------------
    
    Output : (generic function with 1 method),(generic function with 1 method)
    """
    # Retrieve the [M(1-a)] closest functions
    function_matrix = sample_distances(M, alpha, post_dist)
    function_matrix = function_matrix[:,1]
    
    # Lower band
    function f_L(t)
        minimum([f.(t) for f in function_matrix]) 
    end
    # Upper Band
    function f_U(t)
        maximum([f.(t) for f in function_matrix]) 
    end
    
    # Return the upper and lower bands
    return f_L, f_U
end


