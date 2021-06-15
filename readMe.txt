It is expected that Julia is installed and working.
	See more at https://julialang.org/

Code is written in jupyter notebooks, add julia to jupyter notebooks via:
	using Pkg
	Pkg.add("IJulia")

Download and install Anaconda Navigator
	See more at https://docs.anaconda.com/anaconda/navigator/


The following julia packages are assumed installed and working:
	-BayesianNonparametricStatistics, 
	-Plots,
	-Statistics, 
	-Distributions,
	-DelimitedFiles, 
	-LaTeXStrings,
	-NBInclude, 
	-LinearAlgebra, 
	-SparseArrays

packages can be installed by:

using Pkg

Pkg.add("BayesianNonparametricStatistics")
Pkg.add("Plots")
Pkg.add("Statistics")
Pkg.add("Distributions")
Pkg.add("DelimitedFiles") 
Pkg.add("LaTeXStrings")
Pkg.add("NBInclude")
Pkg.add("LinearAlgebra")
Pkg.add("SparseArrays")



All notebooks and the datafile "butane_data.txt" 
are expected to be in the same folder.

In addition there should be a folder named figures, 
in case julia does not create one when saving plots.

Run main.ipynb to reproduce all figures.



Last stable versions
Julia version 1.6.1
Jupyter version 6.1.4