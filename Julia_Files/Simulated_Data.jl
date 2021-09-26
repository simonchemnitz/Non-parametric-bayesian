##   Load Packages
using BayesianNonparametricStatistics, Plots
using Statistics, Distributions
using DelimitedFiles, Plots, LaTeXStrings
using NBInclude, LinearAlgebra, SparseArrays
##   Colors
lblue = RGBA(83/255, 201/255, 250/255,0.7)
dblue = RGBA(47/255, 122/255, 154/255,0.7)

##   Defining a drift function

#drift function described in section 4 equation (13)
#in van der Meulen, Schauer & Van Zanten 2014
function drift_a(x)
    if  0<=x<2/3
        return 2/7-x-2/7*(1-3*x)*sqrt(abs(1-3*x))
    end
    if  2/3<=x<=1
        return -2/7+2/7*x
    end
end

function drift_b(x)
    #note the mod(x,1)
    #this makes the function 1 periodic
    return 12*(drift_a(mod(x,1))+0.05)
end


##   Simmulating from the SDE

# implement SDE dX_t = drift_b(X_t)dt + dW_t, 
# starting at zero till time 200.0, discretised 
# with precision 0.01.
model_sim = SDEModel(1.0,0.0,200.0,10^-2)

##   Define SDE
sde = SDE(drift_b,model_sim)

##   Sample from SDE
x_sim = rand(sde)


##   Plot path
path_plot = plot()
plot!(path_plot,x_sim.timeinterval, x_sim.samplevalues, legend = false, linecolor = dblue)
xlabel!("\$t\$")
ylabel!("\$X_t\$")

##   Direct Computation

##   Model
σ = 1.0 #diffusion
beginvalue = 0.0 #starting value
endtime = 200.0 #end time T
Δ = 200.0/20000 #increment delta

model = SDEModel(σ, beginvalue , endtime ,  Δ)

##   Prior

# number of terms in the series
# sum_k=1^J s*k^-2*Z_k*phi_k
J = 200


# distribution of theta in
# sum_j theta_j*phi_j
s = pi
distribution = GaussianVector(diagm([(s*0.5)*k^(-2) for k in 1.0:(J+1)]))

# Basis functions
# Fourier
Π = GaussianProcess(vcat([x ->1.0] , [fourier(k) for k in 1:J]), distribution)



##   Posterior

# distribution
postΠ = calculateposterior(Π, x_sim, model)

# posterior mean
post_drift_bayes = mean(postΠ)

## Plot

# x values [0,1]
xs = range(0,1,length = 500)

# plot posterior mean
bayes_post = plot(xs, post_drift_bayes.(xs), 
    linewidth = 3,
    linecolor = dblue,
    label = "Posterior Mean")
plot!(bayes_post, xs, drift_b.(xs),
    linecolor = :red,
    label = "True Drift")
xlabel!("\$X_t(rad)\$")
ylabel!("\$b(X_t)\$")

##   Sample N times from posterior and plot

#number of samples to draw
N = 100

#empty plot
sample_plot=plot()

x = 0.0:0.01:1.0

# Sample
for k in 1:N
#sample posterior
f = rand(postΠ)
    
#Plot Sample
plot!(sample_plot,x,f.(x),show=true, linecolor = dblue, alpha = 0.1, linewidth = 1, label = "")

#legend arguments
if k == 1
    plot!(sample_plot,x,f.(x),show=true, linecolor = dblue, 
            alpha = 0.1, linewidth = 1, label = "Posterior Samples")
end
end


#add the true drift to the sample plots
plot!(sample_plot,x,drift_b.(x), linecolor = RGBA(1,0,0,1), linewidth = 2, label = "True Drift",
legend=:bottomleft)
xlabel!("\$x\$")

##   Collect in one plot
l = @layout [a ; b c]
full_bayes_plot = plot(path_plot,sample_plot, bayes_post , layout = l, size = (800,600),margin=5Plots.mm)

include("MCMC.jl")

#Number of iterations to make
N_step = 5000
#number of observations to use a burn in
burn = 1000
#maximum number of terms in the sum
#sum theta_k*phi_k
j_max = 200

#perform MCMC
mu_lst, j_lst = mcmc(x_sim, model, N_step, burn, j_max)

#calculate the posterior 
#mean vector
mcmc_post = mcmc_post_mean(mu_lst, N_step, burn)

#define the function
# sum theta_k*phi_k
mcmc_post_drift = calc_mcmc_post_drift(mcmc_post)

# plot posterior mean
mcmc_post_plot = plot(xs, mcmc_post_drift.(xs), 
    linewidth = 3,
    linecolor = dblue,
    label = "Posterior Mean")
plot!(mcmc_post_plot, xs, drift_b.(xs),
    linecolor = :red,
    label = "True Drift")
xlabel!("\$X_t(rad)\$")
ylabel!("\$b(X_t)\$")

##   Plot models visited
j_lst2 = j_lst[1,:]

model_plot = plot(j_lst2, legend = false, color = dblue,)
vline!(model_plot, [1000], color = :black)
xlabel!("\$Iteration\$")
ylabel!("\$Model number\$")

##   Collect in one plot
l = @layout [a  b ]
full_mcmc_plot = plot(mcmc_post_plot,model_plot, layout = l, size = (800,300),margin=5Plots.mm)

##   Save Figures
png(full_bayes_plot, "figures\\Bayes_plot_sim")
png(full_mcmc_plot, "figures\\MCMC_plot_sim")

##   Mac save figures
try
    png(full_bayes_plot, "figures/Bayes_plot_sim")
    png(full_mcmc_plot, "figures/MCMC_plot_sim")
catch e
end

try
    png(full_bayes_plot, "/figures/Bayes_plot_sim")
    png(full_mcmc_plot, "/figures/MCMC_plot_sim")
catch e
end
