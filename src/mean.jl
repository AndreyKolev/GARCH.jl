# Conditional Mean models

abstract type MeanModel <: Model end
abstract type ConditionalDistribution <: Model end

sim(model::MeanModel, x::Vector{Float64}) = error("sim not implemented for $(typeof(model))")
predict(model::MeanModel, x::Vector{Float64}, n::Int) = error("predict not implemented for $(typeof(model))")

mutable struct ARMA <: MeanModel
	p::Int
	q::Int
	params::Vector{Float64}
	include_mean::Bool
	ARMA(p::Int, q::Int, include_mean::Bool=true) = new(p, q, [include_mean ? 0. : Float64[]; zeros(p); zeros(q)], include_mean)
end

"Returns a vector of lower bounds for the GARCH model parameters"
lower_bounds(model::ARMA) = [model.include_mean ? -1. : Float64[]; -ones(model.p); -ones(model.q)] 

"Returns a vector of upper bounds for the GARCH model parameters"
upper_bounds(model::ARMA) = [model.include_mean ? 1. : Float64[]; ones(model.p); ones(model.q)]

"Simulates the residuals of an ARMA model"
function sim(model::ARMA, x::Vector{Float64})
	p, q = model.p, model.q
	c = model.include_mean ? first(model.params) : 0.
	ϕ = model.params[model.include_mean+1:model.include_mean+p]  # model.params[2:p+1]
	θ = last(model.params, q)
	ɛ = similar(x)
	xc = x .- c
	i0 = max(p, q)
	n = length(x)
	ɛ[1:i0] .= xc[1:i0]
	for t ∈ (i0 + 1):n
		ɛ[t] = xc[t] - sum(ϕ[i]*xc[t-i] for i ∈ 1:p; init=0.) - sum(θ[j]*ɛ[t-j] for j ∈ 1:q; init=0.)
	end
	ɛ
end

"Calculates the unconditional mean of the ARMA model."
function unc_mean(model::ARMA)
	first(model.params)
end

"Predicts the conditional mean of the ARMA model."
function predict(model::ARMA, x::Vector{Float64}, n::Int)
	p, q = model.p, model.q
	c = model.include_mean ? first(model.params) : 0.
	ϕ = model.params[model.include_mean+1:model.include_mean+p]
	θ = last(model.params, q)
	μ = zeros(n)
	ɛ = sim(model, x)
	for t ∈ 1:n
		ar = sum(ϕ[i]*((t-i < 1 ? x[end + t - i] : μ[t-i]) - c) for i ∈ 1:p)
		ma = sum(t-j < 1 ? θ[j]*ɛ[end + t - j] : 0 for j ∈ 1:q)
		μ[t] = c + ar + ma
	end
	μ
end

"Returns a tuple of symbols representing the parameter names for the ARMA model, including the mean term (if included), AR & MA coefficients"
function names(model::ARMA)
	return (Symbol.([model.include_mean ? "c" : []; ["ϕ$i" for i in 1:model.p]; ["θ$i" for i in 1:model.q]])...,)
end

