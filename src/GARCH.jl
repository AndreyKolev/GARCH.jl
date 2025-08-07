module GARCH
export GARCHModel, MeanModel, VarianceModel, ConditionalDistribution, fit!, diagnostics, llh, llh!, params!, ARMA, sGARCH, gjrGARCH, Normal, SkewNormal, predict, unc_mean, unc_variance, residuals, fitted, persistence, half_life, garchFit

using NLopt, Statistics, LinearAlgebra, Printf

abstract type Model end

lower_bounds(model::Model) = error("lower_bounds not implemented for $(typeof(model))")
upper_bounds(model::Model) = error("upper_bounds not implemented for $(typeof(model))")

abstract type MeanModel <: Model end
abstract type VarianceModel <: Model end
abstract type ConditionalDistribution <: Model end

include("distribution.jl")
include("mean.jl")
include("variance.jl")
include("stattests.jl")
include("deprecated.jl")

function Base.show(io::IO, model::Model)
    pq = hasproperty(model, :p) & hasproperty(model, :q) ? "($(model.p),$(model.q))" : ""
    print(io, "$(typeof(model))$pq $(params(model))")
end

"Returns a parameters of a Model as a named tuple"
params(model::Model) = NamedTuple{names(model)}((model.params...,))

"Sets the parameters of a Model in-place using a vector of parameter values."
function params!(model::Model, params::Vector{Float64})
	model.params = params
	nothing
end

mutable struct GARCHModel <: Model 
	cmean::MeanModel
	cvariance::VarianceModel
	cdist::ConditionalDistribution
end

"Returns a vector of lower bounds for the GARCH model parameters"
lower_bounds(model::GARCHModel) = [lower_bounds(model.cmean); lower_bounds(model.cvariance); lower_bounds(model.cdist)]

"Returns a vector of upper bounds for the GARCH model parameters"
upper_bounds(model::GARCHModel) = [upper_bounds(model.cmean); upper_bounds(model.cvariance); upper_bounds(model.cdist)]

"Returns a parameters of the GARCH model as a named tuple"
params(model::GARCHModel) = merge(params(model.cmean), params(model.cvariance), params(model.cdist))

"Predicts the conditional mean and variance for a GARCH model."
function predict(model::GARCHModel, x::Vector{Float64}, n::Int=10)
	μ = predict(model.cmean, x, n)
	ɛ = sim(model.cmean, x)
	σ = predict(model.cvariance, model.cdist, ɛ, n)
	(μ=μ, σ=σ)
end

"Sets the parameters of a GARCHModel in-place using a vector of parameter values."
function params!(model::GARCHModel, params::Vector{Float64})
	plen = [length(model.cmean.params), length(model.cvariance.params), length(model.cdist.params)]
	if sum(plen) ≠ length(params)
		throw(DimensionMismatch("Param vector has wrong dimension!"))
	end
	ixp = cumsum(plen)
	model.cmean.params = params[1:ixp[1]]
	model.cvariance.params = params[ixp[1]+1:ixp[2]]
	model.cdist.params = params[ixp[2]+1:end]
end

"Displays model components and parameters"
function Base.show(io::IO, model::GARCHModel)
   printstyled(io, "$(typeof(model))\n", bold=true, underline=true)
   println(io, "Mean Model: ", model.cmean)
   println(io, "GARCH Model: ", model.cvariance)
   println(io, "Distribution: ", model.cdist)
end

"Calculates log-likelihood of model using provided vector of historical observations"
function llh(model::GARCHModel, x::Vector{Float64})
	ɛ = sim(model.cmean, x)
	h = sim(model.cvariance, ɛ)
	sum(log.(pdf.(Ref(model.cdist), ɛ, 0., .√h)))  # We need to use Ref beacause of broadcasting
end

"Sets new model parameters and calculates log-likelihood using provided vector of historical observations"
function llh!(model::GARCHModel, params::Vector{Float64}, x::Vector{Float64})
	try
		params!(model, params)
		llh(model, x)
	catch
		printstyled("Error! params: $params\n", color=:red, bold=true)
	end
end

"Returns the fitted values of the conditional mean from a GARCH model"
function fitted(model::GARCHModel, x::Vector{Float64})
	ɛ = sim(model.cmean, x)
	x .- ɛ 
end

"Returns model residuals (returns standardized residuals if standardize is true, which are the residuals divided by the conditional standard deviation."
function residuals(model::GARCHModel, x::Vector{Float64}; standardize::Bool=false)
	if standardize
		ɛ = sim(model.cmean, x)
		h = sim(model.cvariance, ɛ)
		return ɛ./.√h
	end
	sim(model.cmean, x)
end

"Returns the conditional standard deviation (sigma) of the GARCH model using provided vector of historical observations"
function sigma(model::GARCHModel, x::Vector{Float64})
	ɛ = sim(model.cmean, x)
	.√sim(model.cvariance, ɛ)
end
const σ = sigma

"Calculates the unconditional mean of the GARCH model"
function unc_mean(model::GARCHModel)
	unc_mean(model.cmean)
end

"Calculates the long run unconditional variance of the GARCH model"
unc_variance(model::GARCHModel) = unc_variance(model.cvariance, model.cdist)

"Calulates the persistence of the GARCH model (measures how long volatility shocks affect future conditional variances)"
persistence(model::GARCHModel) = persistence(model.cvariance, model.cdist)

"Calulates the half life of the GARCH model (measures the time it takes for the volatility shock to decay halfway back to its long-run average level)"
half_life(model::GARCHModel) = log(0.5)/log(persistence(model))

"Optimizes the parameters of the GARCH model using maximum likelihood estimation"
function fit!(model::GARCHModel, x::Vector{Float64})
	opt = Opt(:LN_NELDERMEAD, length(params(model))) 
	lower_bounds!(opt, lower_bounds(model))
	upper_bounds!(opt, upper_bounds(model))
	max_objective!(opt, (params, grad) -> llh!(model, params, x))
	minf, minx, ret = optimize(opt, collect(params(model)))
	(llh = minf, converged = ret==:XTOL_REACHED)
end

"Estimate Hessian using central difference approximation."
function cd_hessian(params::Vector{Float64}, f)
    eps_ = max.(abs.(params.*eps(Float64)^(1/3)), 1e-9)
    n = length(params)
    H = zeros(n, n)
    function step(x, i1, i2, d1, d2)
        xc = copy(x)
        xc[i1] += d1
        xc[i2] += d2
        f(xc)
    end
    for i ∈ 1:n
        for j ∈ 1:n
            H[i,j] = (step(params, i, j, eps_[i], eps_[j]) -
                      step(params, i, j, eps_[i], -eps_[j]) -
                      step(params, i, j, -eps_[i], eps_[j]) +
                      step(params, i, j, -eps_[i], -eps_[j]))/(4eps_[i]*eps_[j])
        end
    end
    H
end

"Estimate Hessian using central difference approximation."
function cd_hessian2(model::GARCHModel, x::Vector{Float64})
    par = collect(params(model))
    x0 = deepcopy(par)
    eps = 1e-4parß
    n = length(par)
    H = zeros(n, n)
    
	function step(x, i1, i2, d1, d2)
        xc = copy(x)
        xc[i1] += d1
        xc[i2] += d2
        llh!(model, xc, x)
    end
    
	for i ∈ 1:n
        for j ∈ 1:n
            H[i,j] = (step(par, i, j, eps[i], eps[j]) -
                      step(par, i, j, eps[i], -eps[j]) -
                      step(par, i, j, -eps[i], eps[j]) +
                      step(par, i, j, -eps[i], -eps[j]))/(4eps[i]*eps[j])
        end
    end
	params!(model, x0)
    H
end

mutable struct Diagnostics
	params::NamedTuple
	hessian::Matrix
	secoef::Vector
	tval::Vector
	pval::Vector
	llh::Float64
	sigma::Vector
	ic::NamedTuple
	residuals::Vector
	jb_stat::Float64
	jb_pval::Float64
end

"Computes and returns a Diagnostics struct containing key diagnostic statistics for the GARCH model"
function diagnostics(model::GARCHModel, x::Vector{Float64})
	x0 = params(model)  # save paramters to prevent changes by llh! in subsequent llh! callls
	H = cd_hessian(collect(x0), par -> llh!(model, par, x))
	params!(model, collect(x0))  # restore settings
	det(H) ≈ 0 && @warn "Hessian is singular!"
	cvar = diag(-pinv(H))
	cvar[cvar.<0] .= NaN
	secoef = sqrt.(cvar)
	tval = collect(params(model))./secoef
	pv = pvalue_2tailed.(tval) #pnorm.(tval)
	par = params(model)
	llh_ = llh(model, x)
	ic = IC(model, x)
	res = residuals(model, x, standardize=true)
	jbstat, jbp = jbtest(res)
	Diagnostics(params(model), H, secoef, tval, pv, llh_, sigma(model, x), ic, res, jbstat, jbp)
end

"prints Diagnostics struct"
function Base.show(io::IO, diag::Diagnostics)
	printstyled("\nParameters\n", rpad("Parameter", 11), rpad("Estimate", 10), rpad("Std.Error", 10), rpad("t value", 10), rpad("Pr(>|t|)", 10), "\n", bold=true, underline=true)
	for (n, p, se, tv, pv) in zip(string.(keys(diag.params)), values(diag.params), diag.secoef, diag.tval, diag.pval)
		@printf "%s\t% 10.6f% 10.6f% 10.6f% 10.6f\n" n p se tv pv
	end
	printstyled(@sprintf("\nLogLikelihood: %.3f\n", diag.llh), bold=true)
	printstyled("\nInformation Criteria\n", bold=true, underline=true)
	@printf "%-20s % 10.4f\n" "Akaike" diag.ic.AIC
	@printf "%-20s % 10.4f\n" "Bayes" diag.ic.BIC
	@printf "%-20s % 10.4f\n" "Shibata" diag.ic.SIC
	@printf "%-20s % 10.4f\n" "Hannan-Quinn" diag.ic.HQIC
	printstyled("\nJarque-Bera test on standardized residuals\n", bold = true, underline = true)
	@printf "Statistic: % 10.4f  p-value: % 10.4f" diag.jb_stat diag.jb_pval
end

"Computes and returns a named tuple of information criteria (AIC, BIC, SIC, HQIC) for the GARCH model."
function IC(model::GARCHModel, x::Vector{Float64})
	llh_ = llh(model, x)
	n_obs = length(x)
	n_pars = length(params(model))
	AIC  = (-2llh_)/n_obs + 2n_pars/n_obs
	BIC  = (-2llh_)/n_obs + n_pars*log(n_obs)/n_obs
	SIC  = (-2llh_)/n_obs + log((n_obs + 2n_pars)/n_obs)
	HQIC = (-2llh_)/n_obs + (2n_pars*log(log(n_obs)))/n_obs
	(AIC=AIC, BIC=BIC, SIC=SIC, HQIC=HQIC)
end
end  # Module
