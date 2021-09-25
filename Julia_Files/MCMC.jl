##   Load Packages
using BayesianNonparametricStatistics, Plots
using Statistics, Distributions
using DelimitedFiles, Plots, LaTeXStrings
using NBInclude, LinearAlgebra, SparseArrays
##   Colors
lblue = RGBA(83/255, 201/255, 250/255,0.7)
dblue = RGBA(47/255, 122/255, 154/255,0.7)

# Refferences
# [vdM 2014], van der Muelen et al:
# Reversible jump MCMC for nonparametric drift estimation
# - for diffusion processes

##   Functions
#calculate xi_j, theta_j, w_j, mu_j


function calc_xi(j)
    # Given j calculate the matrix Xi^j
    """
    Parameters
    -----------------------------------------
    j : int
        model level j, for more details 
        - see [vdM 2014]
    -----------------------------------------
    
    Output : Array{Float64}(undef, mj, mj)
    """
    mj = 2*j-1
    return diagm([k^(-4) for k in 1.0:mj])
end

function calc_post(j,x,model, s)
    # Calculate the posterior distribution
    # for a fixed model,
    # given j, x, model and s
    """
    Parameters
    -----------------------------------------
    j : int
        model level j, for more details 
        - see [vdM 2014]
    
    x : Samplepath{S,T}
        Contains the data in form of
        samplepath timeinterval S, and samplevalues T
    
    model : SDEModel{T}
        SDEModel object, contains
        diffusion costant σ
        beginvalue, endtime T and increments Δ
        See BayesianNonparametricStatistics
            for more details
    
    s : float
        scale parameter s^2 in [vdM 2014]
    -----------------------------------------
    
    Output : GaussianProcess{S,T}
    """
    # Number of term in the basis expansion
    mj = 2*j-1
    
    # Prior distribution 
    prior_theta_j = GaussianVector(diagm([sqrt(s)*k^(-2) for k in 1.0:(mj)]))
    
    # Prior process
    prior_b = GaussianProcess(vcat([x ->1.0] , [fourier(k) for k in 1:(mj-1)]), prior_theta_j)
    
    # Return the posterior distribution
    return calculateposterior(prior_b, x, model)
end


function calc_w_matrix(post_dist)
    # Given a posterior distribution
    # return the covariance matrix
    # W_j^-1
    """
    Parameters
    -----------------------------------------
    post_dist : GaussianProcess{S,T}
        Posterior distribution
    -----------------------------------------
    
    Output : Array{Float64}(undef, n, n)
        with n being the number of basis functions
        in the GaussianProcess post_dist
    """
    # Covariance matrix Σ
    return post_dist.distribution.Σ
end


function calc_mu_vec(post_dist)
    # Given a posterior distribution
    # return the posterior mean (W_j^-1)µ^j
    """
    Parameters
    -----------------------------------------
    post_dist : GaussianProcess{S,T}
        Posterior distribution
    -----------------------------------------
    
    Output : Array{Float64}(undef, n)
        with n being the number of basis functions
        in the GaussianProcess post_dist
    """
    # Posterior mean
    return post_dist.distribution.mean
end

#sample θ_j
function sample_theta_j(post_dist)
    # Given a posterior distribution
    # sample coefficients from the posterior theta_j
    # theta_j enters in the sum 
    # sum_k=1 ^(2*j-1) theta_k*phi_k
    # For more details see [vdM 2014]
    """
    Parameters
    -----------------------------------------
    post_dist : GaussianProcess{S,T}
        Posterior distribution
    -----------------------------------------
    
    Output : Array{Float64}(undef, n)
        with n being the number of basis functions
        in the GaussianProcess post_dist
    """
    # theta_j sample
    return rand(post_dist.distribution)
end


#sample "s^2" 
function sample_s(j, theta_j)
    # Sample scale s^2 ~ IG(a,b)
    # given model number j and
    # posterior sample coefficient theta_j
    """
    Parameters
    -----------------------------------------
    j : int
        model level j, for more details 
        - see [vdM 2014]
    
    theta_j : Array{Float64}(undef, 2*j-1)
        sample values from the posterior.
        Coefficients in the function
        sum_k=1 ^(2*j-1) theta_k*phi_k
        For more details see [vdM 2014]
    -----------------------------------------
    
    Output : float
    """
    # Calculate the variance matrix Xi^j
    xi_mat = calc_xi(j)
    # Calculate a, b coefficients
    a = (5/2)+0.5*(2*j-1)
    b = (5/2) + 0.5*( transpose(theta_j)*inv(xi_mat)*theta_j )[1]
    
    # Return a sample from IG(a,b)
    return rand(InverseGamma(a,b))
end


# NOTE This method of calculating 
#p(x^T|j,s^2) is highly inefficient
#so much so that 
function cond_prob(mu,w,xi,s)
    # Given µ, W, Xi and s calculate
    # p(x^T|j,s^2)
    """
    Parameters
    -----------------------------------------
    mu : Array{Float64}(undef, 2*j-1)
        posterior mean µ
        model level j, for more details 
        - see [vdM 2014]
    
    w : Array{Float64}(undef, 2*j-1, 2*j-1,)
        covariance matrix W_j^-1
    -----------------------------------------
    
    Output : float
    """
    # Numerator
    nume = exp(0.5*(transpose(mu)*w*mu)[1])
    # Denominator
    denom = sqrt(det(s*inv(w)*xi))
    
    # Return the fraction
    # Note the regularization term 0.001
    return nume/(denom+0.001)
end

##   Main MCMC Function

function mcmc(x, model, N_iter, burn_in, max_j)
    """
    Parameters
    -----------------------------------------
    x : Samplepath{S,T}
        Contains the data in form of
        samplepath timeinterval S, and samplevalues T
    
    model : SDEModel{T}
        SDEModel object, contains
        diffusion costant σ
        beginvalue, endtime T and increments Δ
        See BayesianNonparametricStatistics
            for more details
    
    N_iter : int 
        The number of iterations/times
        to perform algorithm 1
    
    burn_in : int  
        The number of samples to
        discard as burn-in
    
    max_j : int
        The maximum number of terms
        in the sum, Σ_k=1^infty theta_k*phi_k. 
        practically we get the sum
        Σ_k=1^max_j theta_k*phi_k
    -----------------------------------------
    
    Output : Array{Float64}(undef, N_iter), Array{Float64}(undef, N_iter)
    """
    
    # Error check
    if N_iter<burn_in
        println("Error")
        println("Fewer iterations than burn in")
        InterruptException()
    end
    
    #values we can sample
    j_sample_values = [j for j in 1:max_j]
    #associated probabilities
    # p(j) proportional to (0.95)^mj
    j_prob = [(0.95)^(2*j-1) for j in 1:max_j]
    j_prob = j_prob.*(1/(sum(j_prob)))
    
    #sample j_0
    j = wsample(j_sample_values, j_prob, 1)[1]
    #j = rand(Poisson(15))
    
    #prior scale, s
    prior_s = rand(InverseGamma(5/2 , 5/2))
    
    #posterior dist in model j_0
    pdist = calc_post(j,x,model,prior_s)
    #theta_0
    thet = sample_theta_j(pdist)
    #sample s_0^2
    s = sample_s(j,thet)
    #µ_0
    mu = calc_mu_vec(pdist)
    #W_0^-1
    w = calc_w_matrix(pdist)
    #Xi_0
    xi = calc_xi(j)
    
    #lists to contain
    #µ and the model visited
    mu_lst = [mu]
    j_lst = [j]

    #perform algorithm 1
    #N_iter times
    for i in 1:N_iter
        #Sample s'
        s_prime = sample_s(j, thet)
        #Sample j'
        j_prime = max(j+wsample([0,1,-1], [0.5,1/4,1/4]),3)

        #xi matrices
        xi = calc_xi(j)
        xi_prime = calc_xi(j_prime)

        ##   Sample θ_j'
        #get posterior dist
        post_prime = calc_post(j_prime,x,model, s_prime)
        #mean vector µ, and matrix W
        mu_prime = calc_mu_vec(post_prime)
        w_prime = calc_w_matrix(post_prime)
        #sample θ_j'
        thet_prime = sample_theta_j(post_prime)

        ##  Calculate prob of transdimensional step
        #B = (cond_prob(mu_prime,w_prime,xi_prime,s))/(cond_prob(mu,w,xi,s))
        R = j_prob[j_prime]/j_prob[j]
        
        #prob of transdim step is r given by
        r = min(1, R)

        #if change = 1 we make transdim step
        change = wsample([0,1], [1-r,r])
        
        #change scale to (s')^2
        s = s_prime
        
        #if we sampled 1 we make transdim step
        if change == 1
            j = j_prime
            thet = thet_prime
            mu = mu_prime
            w = w_prime
        end
        
        #append µ and model lists
        
        #note µ' not µ
        mu_lst = cat(mu_lst, [mu_prime], dims = 2)
        j_lst = cat(j_lst , [j], dims = 2)
        
        #verbose
        print("Step: ", i," \r")
    end
    
    # Return mu_lst and j_lst
    
    return mu_lst, j_lst
end

##   Calculate the posterior mean
function mcmc_post_mean(mu_lst, N_iter, burn_in)
    """
    Parameters
    -----------------------------------------
    mu_lst : Array{Float64}(undef, 2*j-1)
        array containing each mean vector µ
        obtained in each iteration mcmc(...).
    
    N_iter : int 
        The number of iterations/times
        to perform algorithm 1
    
    burn_in : int  
        The number of samples to
        discard as burn-in
    -----------------------------------------
    
    Output : Array{Float64}(undef, max_dim)
        where max_dim is the largest dimension
        of all mean vectors
    """
    # Discard burn in observations
    mu_lst2 = mu_lst[burn_in:N_iter]
    
    # Dimension of each µ vector
    dims = [size(vec)[1] for vec in mu_lst2]
    
    # Maximum of the above dimensions
    max_dim = maximum(dims)
    
    # Add zeros to vectors of low dim
    # such that all vectors have same dimension
    # without discarding information
    adj_vecs = [vcat(vec, repeat([0],outer=max_dim-size(vec)[1])) for vec in mu_lst2] 

    # Calculate the coordinate wise mean of each
    # mean vector 
    mean_vec = sum(adj_vecs)/(size(adj_vecs)[1])
    
    # Return the mean vector
    # of type Array{Float64}(undef, max_dim)
    return mean_vec
end

function calc_mcmc_post_drift(theta)
    ##   Retrieve drift given posterior mean vec theta
    ##   b(x) = sum theta*phi(x),
    """
    Parameters
    -----------------------------------------
    theta : Array{Float64}(undef, 2*j-1)
        Posterior mean vector
    -----------------------------------------
    
    Output : (generic function with 1 method)
    """
    # Dimension of the theta vector
    dims = size(theta)[1]
    
    # Return the function b(x) = sum theta*phi(x)
    # where phi are fourier basis functions
    return sumoffunctions(vcat([x ->1.0] , [fourier(k) for  k in 1:dims-1]),theta)
end
