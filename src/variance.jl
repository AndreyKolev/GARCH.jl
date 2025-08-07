# Conditional variance models

sim(model::VarianceModel, ɛ::Vector{Float64}) = error("sim not implemented for $(typeof(model))")

"Calculates the long run unconditional variance of the conditional variance model"
function unc_variance(model::VarianceModel, cdist::ConditionalDistribution)
	ω = first(model.params)
	ω/(1 - persistence(model, cdist))
end

# Standard GARCH - Generalized Autoregressive Conditional Heteroskedasticity, Bollerslev 1986
mutable struct sGARCH <: VarianceModel
	p::Int
	q::Int
	params::Vector{Float64}
	sGARCH(p::Int, q::Int) = new(p, q, [1e-5; [0.09; zeros(p-1)]; [0.89; zeros(q-1)]])
end

"Returns a vector of lower bounds for the Standard GARCH model parameters"
lower_bounds(model::sGARCH) = [1e-10; zeros(model.p); zeros(model.q)]

"Returns a vector of upper bounds for the Standard GARCH model parameters"
upper_bounds(model::sGARCH) = [1; fill(0.3, model.p); fill(0.99, model.q)]

"Returns a tuple of symbols representing the parameter names of the Standard GARCH model"
function names(model::sGARCH)
	return (Symbol.(["ω"; ["α$i" for i in 1:model.p]; ["β$i" for i in 1:model.q]])...,)
end

"Standard GARCH model simulation"
function sim(model::sGARCH, ɛ::Vector{Float64})
    p, q = model.p, model.q
    ω = model.params[1]
    α = model.params[2:p+1]
    β = model.params[p+2:p+q+1]
    h = similar(ɛ)
    ɛ² = ɛ.^2
    t0 = max(p, q)
    h[1:t0] .= mean(ɛ²)
    n = length(ɛ²)
    for t = (t0 + 1):n
    	h[t] = ω + sum(α[i]*ɛ²[t-i] for i ∈ 1:p) + sum(β[j]*h[t-j] for j ∈ 1:q)
    end
    h
end

"Calulates the persistence of the GARCH model (measures how long volatility shocks affect future conditional variances)"
function persistence(model::sGARCH, cdist::ConditionalDistribution)
	p, q = model.p, model.q
	ω = first(model.params)
	α = model.params[2:p+1]
	β = model.params[p+2:p+q+1]
	sum(α; init=0.) + sum(β; init=0.)
end

"n-step volatility prediction for a Standard GARCH model"
function predict(model::sGARCH, cdist::ConditionalDistribution, ɛ::Vector{Float64}, n::Int=10)
	p, q = model.p, model.q
	ω = first(model.params)
	α = model.params[2:p+1]
	β = model.params[p+2:p+q+1]
	h = sim(model, ɛ)
	μ = zeros(n)
	σ = zeros(n)
	for t ∈ 1:n
		σ[t] = ω
		σ[t] += sum(α[i]*(t-i < 1 ? ɛ[end+(t-i)]^2 : σ[t-i]) for i ∈ 1:p)
		σ[t] += sum(β[j]*(t-j < 1 ? h[end+(t-j)] : σ[t-j]) for j ∈ 1:q)
	end
	sqrt.(σ)
end

# GJR-GARCH Glosten-Jagannathan-Runkle GARCH by Glosten, Jagannathan and Runkle (1993)
mutable struct gjrGARCH <: VarianceModel
	p::Int
	q::Int
	params::Vector{Float64}
	#1e-5, 0.09, 0.89, 0.02
	gjrGARCH(p::Int, q::Int) = new(p, q, [1e-5; [0.09; zeros(p-1)]; [0.89; zeros(q-1)]; [0.02; zeros(p-1)]])
end

"Returns a vector of lower bounds for the GJR GARCH model parameters"
lower_bounds(model::gjrGARCH) = [1e-10; zeros(model.p); zeros(model.q); zeros(model.p)]

"Returns a vector of upper bounds for the GJR GARCH model parameters"
upper_bounds(model::gjrGARCH) = [1; fill(0.3, model.p); fill(0.99, model.q); fill(1, model.p)]

"Returns a tuple of symbols representing the parameter names of the GJR GARCH model"
function names(model::gjrGARCH)
	return (Symbol.(["ω"; ["α$i" for i in 1:model.p]; ["β$i" for i in 1:model.q]; ["γ$i" for i in 1:model.p]])...,)
end

"GJR GARCH model simulation"
function sim(model::gjrGARCH, ɛ::Vector{Float64})
    p, q = model.p, model.q
    ω = model.params[1]
    α = model.params[2:p+1]
    β = model.params[p+2:p+q+1]
    γ = last(model.params, p)
    h = similar(ɛ)
    ɛ² = ɛ.^2
    t0 = max(p, q)
    h[1:t0] .= mean(ɛ²)
    n = length(ɛ²)
    for t = (t0 + 1):n
    	h[t] = ω + sum((α[i] + (ɛ[t-i]<0)*γ[i])*ɛ²[t-i] for i ∈ 1:p) + sum(β[j]*h[t-j] for j ∈ 1:q)
    end
    h
end


"Calulates the persistence of a GJR GARCH model (measures how long volatility shocks persist over time, incorporating both symmetric and asymmetric effects)"
function persistence(model::gjrGARCH, cdist::ConditionalDistribution)
	p, q = model.p, model.q
	ω = first(model.params)
	α = model.params[2:p+1]
	β = model.params[p+2:p+q+1]
	γ = last(model.params, p)
	κ = cdf(cdist, 0.)
	sum(α; init=0.) + sum(β; init=0.) + κ*sum(γ; init=0.)
end

"n-step volatility prediction for a GJR GARCH model"
function predict(model::gjrGARCH, cdist::ConditionalDistribution, ɛ::Vector{Float64}, n::Int=10)
	p, q = model.p, model.q
	ω = first(model.params)
	α = model.params[2:p+1]
	β = model.params[p+2:p+q+1]
	γ = last(model.params, p)
	h = sim(model, ɛ)
	μ = zeros(n)
	σ = zeros(n)
	κ = cdf(cdist, 0.)
	for t ∈ 1:n
		σ[t] = ω
		σ[t] += sum(t-i < 1 ? (α[i] + γ[i]*(ɛ[end+(t-i)]<0))*ɛ[end+(t-i)]^2 : (α[i] + γ[i]*κ)*σ[t-i] for i ∈ 1:p)
		σ[t] += sum(β[j]*(t-j < 1 ? h[end+(t-j)] : σ[t-j]) for j ∈ 1:q)
	end
	return sqrt.(σ)
end
