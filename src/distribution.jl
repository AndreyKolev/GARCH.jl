# Conditional Distributions

"Computes the error function using C standard library"
erf(x) = @ccall erf(x::Float64)::Float64

"Computes PDF of Standard Normal distribution"
stdnorm_pdf(z::Float64) = ℯ^(-z^2/2)/√(2π)

"Computes CDF of Standard Normal distribution"
stdnorm_cdf(z::Float64) = 0.5(1 + erf(z/√2))


"Calculates two-tailed p-value from a standard normal test statistic z"
pvalue_2tailed(x) = 2(1 - stdnorm_cdf(abs(x)))

"Heaviside step function"
heaviside(x, a) = (sign(x - a) + 1)/2

"Calculates PDF of Normal distribution with mean μ and standard deviation σ"
norm_pdf(x::Float64, μ::Float64, σ::Float64) = 1/(σ*√(2π))*ℯ^(-0.5*((x - μ)/σ)^2)

"Calculates CDF of Normal distribution with mean μ and standard deviation σ"
norm_cdf(x::Float64, μ::Float64=0., σ::Float64=1.) = stdnorm_cdf((x - μ)/σ)

# Conditional distribution
abstract type ConditionalDistribution <: Model end

pdf(cdist::ConditionalDistribution, x::Float64,  μ::Float64, σ::Float64) = error("pdf not implemented for $(typeof(cdist))")
cdf(cdist::ConditionalDistribution, q::Float64) = error("cdf not implemented for $(typeof(cdist))")

# Normal conditional distribution (no parameters)
mutable struct Normal <: ConditionalDistribution
	params::Vector{Float64}
	Normal() = new(Float64[])
end

"Returns a vector of lower bounds for the parameters of Normal distribution"
lower_bounds(model::Normal) = Float64[]
"Returns a vector of upper bounds for the parameters of Normal distribution"
upper_bounds(model::Normal) = Float64[]

"Returns a tuple of symbols representing the parameter names of Normal distribution (no parameters)"
names(model::Normal) = ()

"Calculates PDF of Normal distribution with mean μ and standard deviation σ"
function pdf(cdist::Normal, x::Float64, μ::Float64=0., σ::Float64=1.)
	1/(σ*√(2π))*ℯ^(-0.5*((x - μ)/σ)^2)
end

"Calculates CDF of Normal distribution with mean μ and standard deviation σ"
function cdf(cdist::Normal, x::Float64, μ::Float64=0., σ::Float64=1.)
	x == 0. && return 0.5 
	z = (x - μ)/σ
	0.5(1 + erf(z/√2))
end

# Skew normal conditional distribution with skew parameter ξ
mutable struct SkewNormal <: ConditionalDistribution
	params::Vector{Float64}
	SkewNormal() = new([1.])
end

"Returns a vector of lower bounds for the parameters of Skew Normal distribution"
lower_bounds(model::SkewNormal)  = [0.]

"Returns a vector of upper bounds for the parameters of Skew Normal distribution"
upper_bounds(model::SkewNormal)  = [10.]

"Returns a tuple of symbols representing the parameter names of Skew Normal distribution (skew parameter ξ)"
names(model::SkewNormal) = (:ξ,)


"Calculates PDF of Skew normal distribution with skew ξ, location μ and scale σ"
function pdf(cdist::SkewNormal, x::Float64, μ::Float64=0., σ::Float64=1.)
    ξ = only(cdist.params)  # ξ
    m₁ = 2/√(2π)
    σ₂ = √((1 - m₁^2)*(ξ^2 + 1/ξ^2) + 2m₁^2 - 1)
    z = σ₂*(x - μ)/σ + m₁*(ξ - 1/ξ)
    g = 2/(ξ + 1/ξ)
    dens = g*stdnorm_pdf(z/ξ^sign(z))
    dens*σ₂/σ
end

"Calculates CDF of Skew normal distribution with skew ξ, location μ and scale σ"
function cdf(cdist::SkewNormal, q::Float64, μ::Float64=0., σ::Float64=1.)
	ξ = only(cdist.params) #cdist.ξ
	m₁ = 2/√(2π)
	μx = m₁*(ξ - 1/ξ)
	sig = √((1 - m₁*m₁)*(ξ*ξ + 1/(ξ*ξ)) + 2m₁*m₁ - 1)
	z = ((q - μ)/σ)*sig + μx
	Ξ = z<0 ? 1/ξ : ξ
	g = 2/(ξ + 1/ξ)
	heaviside(z, 0) - sign(z)*g*Ξ*norm_cdf(-abs(z)/Ξ)
end
