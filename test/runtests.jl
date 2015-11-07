module TestCapsule

using FactCheck
using GARCH

facts("Consistency with R's fGarch") do
    filename = Pkg.dir("GARCH", "test", "data", "SPY.csv")
    close = map(Float64, readcsv(filename)[:, 2])
    ret = diff(log(close))
    ret = ret .- mean(ret)
    fit = garchFit(ret)
    param = [2.469347e-06, 1.142268e-01, 8.691734e-01] #R fGarch garch(1,1) estimated params
    @fact fit.params --> roughly(param, atol=1e-3)
end

end #TestCapsule
