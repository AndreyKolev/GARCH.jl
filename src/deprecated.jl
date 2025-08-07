# Deprecated functions provided for compatibility that are scheduled for removal in future releases.

# Fitted GARCH model object (for compatibility with legacy versions of the package)"
struct GarchFit
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

"Estimate parameters of the univariate normal GARCH process."
function garchFit(y::Vector)
    Base.depwarn("`garchFit` is deprecated and will be removed in a future releases", :garchFit)
    model = GARCHModel(ARMA(0,0,false), sGARCH(1,1), Normal())
    status = fit!(model, y)
    diag = diagnostics(model, y)
    minx = collect(values(params(model)))
    GarchFit(y, minx, diag.llh, status.converged ? :SUCCESS : :FAILURE, status.converged, diag.sigma, diag.hessian, diagm(diag.secoef).^2, diag.secoef, diag.tval)
end

"Calculates n-step prediction using fitted object returned by garchFit (default step=1)."
function predict(fit::GarchFit, n::Integer=1)
	model = GARCHModel(ARMA(0,0,false), sGARCH(1,1), Normal())
	model.params!(fit.params)
	predict(model, fit.data, n)
end
