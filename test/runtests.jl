using GARCH
using Pkg, Test, DelimitedFiles, Statistics


filename = Pkg.dir("GARCH", "test", "data", "price.csv")
price = Array{Float64}(readdlm(filename, ',')[:,2])

ret = diff(log.(price))
ret = ret .- mean(ret)
fit = garchFit(ret)

param = [2.469347e-06, 1.142268e-01, 8.691734e-01] #R fGarch garch(1,1) estimated params

@test fit.params ≈ param atol=1e-3
@test predict(fit) ≈ 0.005617744 atol = 1e-4
@test predict(fit, 2) ≈ [0.005617744, 0.005788309] atol = 1e-4

