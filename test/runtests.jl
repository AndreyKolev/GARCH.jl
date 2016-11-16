using Base.Test
using GARCH

filename = Pkg.dir("GARCH", "test", "data", "SPY.csv")
close = Array{Float64}(readcsv(filename)[:,2])
ret = diff(log.(close))
ret = ret - mean(ret)
fit = garchFit(ret)

param = [2.469347e-06, 1.142268e-01, 8.691734e-01] #R fGarch garch(1,1) estimated params

@test_approx_eq_eps(fit.params, param, 1e-3)
@test_approx_eq_eps(predict(fit), 0.005617744, 1e-4)
@test_approx_eq_eps(predict(fit, 2), [0.005617744, 0.005788309], 1e-4)
