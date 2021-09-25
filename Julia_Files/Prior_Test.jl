##   Load Packages
using BayesianNonparametricStatistics, Plots
using Statistics, Distributions
using DelimitedFiles, Plots, LaTeXStrings
using NBInclude, LinearAlgebra, SparseArrays
##   Colors
lblue = RGBA(83/255, 201/255, 250/255,0.7)
dblue = RGBA(47/255, 122/255, 154/255,0.7)

##   Load Data
df = DelimitedFiles.readdlm("butane_data.txt", '\n',  ' ')
df = df[:,1]
#Subsampling to indexes
#X_1, X_1001, X_2001,...
indexes = [1:1000:4*10^6;]
df = df[indexes]

# Scale data to be 1-periodic
x_path = (df).+pi
x_path = (df).*1/(2*pi)

# Estimate sigma^2
# sum (v_i-v_i+1)^2
sig_est = zeros(3999)

for i in 1:3999
    sig_est[i] = (x_path[i+1]-x_path[i])^2
end
# Estimate T for unit diffusion
T = 1/sum(sig_est)

sigma_hat = sum(sig_est)/4.0

eta = 0.02*T^2*2*pi
alph = 3/2
s = 1/(sqrt(eta)*pi^2 )

# Change time scale for unit diffusion
time_t = [i*T for i in 0:3999]

x_data = SamplePath(time_t, x_path)

##   Model
#[0,4] nanoseconds
σ = 1.0/(2*pi)
beginvalue = 0.0
endtime = 3.9999#*10^-9
Δ = (4.0/4000)#*10^-9


model = SDEModel(σ, beginvalue , endtime ,  Δ)

# Number of terms in the series
# sum_k=1^J s*k^-2*Z_k*phi_k
J = 200

# Distribution of theta in
# sum_j theta_j*phi_j
distribution = GaussianVector(diagm([(s*0.5)*k^(-2) for k in 1.0:(J+1)]))

##   Basis functions
# Fourier
Π = GaussianProcess(vcat([x ->1.0] , [fourier(k) for k in 1:J]), distribution)


##   Posterior
postΠ = calculateposterior(Π, x_data, model)
# Posterior Mean
theta = mean(postΠ)
# x values [0,1]
x_vals = [0.01:0.001:1;]
# Plot posterior mean
bayes_post = plot(x_vals.*2*pi.-pi, theta.(x_vals.*(-1).-1), #note .*2pi to get the function on the scale of [0,2pi]
                            #instead of [0,1]
    linewidth = 3,
    linecolor = dblue,
    label = "Posterior Mean")
xlabel!("\$X_t(rad)\$")
ylabel!("\$b(X_t)\$")

multiplot = plot()

for sig in range(1/(2pi),3, length = 5)
    model = SDEModel(sig, beginvalue , endtime ,  Δ)
    for i in 1:20
        # Distribution of theta in
        # sum_j theta_j*phi_j
        distribution = GaussianVector(diagm([(i)*k^(-2) for k in 1.0:(J+1)]))

        ##   Basis functions
        # Fourier
        Π = GaussianProcess(vcat([x ->1.0] , [fourier(k) for k in 1:J]), distribution)


        ##   Posterior
        postΠ = calculateposterior(Π, x_data, model)
        # Posterior Mean
        theta = mean(postΠ)
        # x values [0,1]
        x_vals = [0.01:0.001:1;]
        # Plot posterior mean
        tmp_plot = plot(x_vals.*2*pi.-pi, theta.(x_vals.*(-1).-1), #note .*2pi to get the function on the scale of [0,2pi]
                                    #instead of [0,1]
            linewidth = 3,
            linecolor = dblue,
            label =string(i),
            linealpha = 0.3)
        png(tmp_plot, string(sig,i))
    end
end
