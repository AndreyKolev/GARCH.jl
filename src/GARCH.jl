# Julia GARCH package
# Copyright 2013 Andrey Kolev
# Distributed under MIT license (see LICENSE.md)

"Generalized Autoregressive Conditional Heteroskedastic (GARCH) models for Julia."
module GARCH

using NLopt, Distributions

export garchFit, predict

include("stattests.jl")


"Fitted GARCH model object."
type GarchFit
    data::Vector
    params::Vector
    llh::Float64
    status::Symbol
    converged::Bool
    sigma::Vector
    hessian::Array{Float64,2}
    cvar::Array{Float64,2}
    secoef::Vector
    tval::Vector
end


function Base.show(io::IO ,fit::GarchFit)
    pnorm(x) = 0.5 * (1 + erf(x / sqrt(2)))
    prt(x) = 2 * (1 - pnorm(abs(x)))
    jbstat, jbp = jbtest(fit.data./fit.sigma)

    @printf io "Fitted garch model \n"
    @printf io " * Coefficient(s):    %-15s%-15s%-15s\n" "ω" "α" "β"
    @printf io "%-22s%-15.5g%-15.5g%-15.5g\n" "" fit.params[1] fit.params[2] fit.params[3]
    @printf io " * Log Likelihood: %.5g\n" fit.llh
    @printf io " * Converged: %s\n" fit.converged
    @printf io " * Solver status: %s\n\n" fit.status
    @printf io " * Standardised Residuals Tests:\n"
    @printf io "   %-26s%-15s%-15s\n" "" "Statistic" "p-Value"
    @printf io "   %-21s%-5s%-15.5g%-15.5g\n\n" "Jarque-Bera Test" "χ²" jbstat jbp
    @printf io " * Error Analysis:\n"
    @printf io "   %-7s%-15s%-15s%-15s%-15s\n" "" "Estimate" "Std.Error" "t value" "Pr(>|t|)"
    @printf io "   %-7s%-15.5g%-15.5g%-15.5g%-15.5g\n" "ω" fit.params[1] fit.secoef[1] fit.tval[1] prt(fit.tval[1])
    @printf io "   %-7s%-15.5g%-15.5g%-15.5g%-15.5g\n" "α" fit.params[2] fit.secoef[2] fit.tval[2] prt(fit.tval[2])
    @printf io "   %-7s%-15.5g%-15.5g%-15.5g%-15.5g\n" "β" fit.params[3] fit.secoef[3] fit.tval[3] prt(fit.tval[3])
end


"Estimate Hessian using central difference approximation."
function cdHessian(params, f)
    eps = 1e-4 * params
    n = length(params)
    H = zeros(n, n)
    function step(x, i1, i2, d1, d2)
        xc = copy(x)
        xc[i1] += d1
        xc[i2] += d2
        f(xc)
    end
    for i in 1:n
        for j in 1:n
            H[i,j] = (step(params, i, j, eps[i], eps[j]) -
                      step(params, i, j, eps[i], -eps[j]) -
                      step(params, i, j, -eps[i], eps[j]) +
                      step(params, i, j, -eps[i], -eps[j])) / (4.*eps[i]*eps[j])
        end
    end
    H
end

"Simulate GARCH process."
function garchSim(ɛ²::Vector, ω, α, β)
    h = similar(ɛ²)
    h[1] = mean(ɛ²)
    for i = 2:length(ɛ²)
        h[i] = ω + α*ɛ²[i-1] + β*h[i-1]
    end
    h
end

"Normal GARCH log likelihood function."
function garchLLH(y::Vector, params::Vector)
    ɛ² = y.^2
    T = length(y)
    h = garchSim(ɛ², params...)
    -0.5*(T-1)*log(2π) - 0.5*sum(log.(h) + (y./sqrt.(h)).^2)
end

"""
    predict(fit::GarchFit, n::Integer=1)

Make n-step prediction using fitted object returned by garchFit (default step=1).

# Arguments
* `fit::GarchFit` : fitted model object returned by garchFit.
* `n::Integer` : the number of time-steps to be forecasted, by default 1 (returns scalar for n=1 and array for n>1).

# Examples

```
fit = garchFit(ret)
predict(fit, n=2)
```

"""
function predict(fit::GarchFit, n::Integer=1)
    if n < 1
        throw(ArgumentError("n shoud be >= 1 !"))
    end
    ω, α, β = fit.params
    y = fit.data
    ɛ² = y.^2
    h = garchSim(ɛ², ω, α, β)
    pred = ω + α*ɛ²[end] + β*h[end]
    if n == 1
        return sqrt(pred)
    end
    pred = [pred]
    for i in 2:n
        push!(pred, ω + (α + β)*pred[end])
    end
    sqrt.(pred)
end

"""
    garchFit(y::Vector)

Estimate parameters of the univariate normal GARCH process.

# Arguments
* `y::Vector`: univariate time-series array

# Examples

```
filename = Pkg.dir("GARCH", "test", "data", "SPY.csv")
close = Array{Float64}(readcsv(filename)[:,2])
ret = diff(log.(close))
ret = ret - mean(ret)
fit = garchFit(ret)
```

"""
function garchFit(y::Vector)
    ɛ² = y.^2
    T = length(y)
    h = zeros(T)
    
    opt = Opt(is_apple() ? (:LN_PRAXIS) : (:LN_SBPLX), 3)  # LN_SBPLX has a problem on mac currently
    lower_bounds!(opt, [1e-10, 0.0, 0.0])
    upper_bounds!(opt, [1, 0.3, 0.99])
    min_objective!(opt, (params, grad) -> -garchLLH(y, params))
    (minf, minx, ret) = optimize(opt, [1e-5, 0.09, 0.89])
    h = garchSim(ɛ², minx...)
    
    converged = minx[1] > 0 && all(minx[2:3] .>= 0) && sum(minx[2:3]) < 1.0
    H = cdHessian(minx, x -> garchLLH(y, x))
    cvar = -inv(H)
    secoef = sqrt.(diag(cvar))
    tval = minx ./ secoef
    GarchFit(y, minx, -minf, ret, converged, sqrt.(h), H, cvar, secoef, tval)
end

end  #module
