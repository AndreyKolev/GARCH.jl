"Computes the Jarque-Bera test statistic and p-value"
function jbtest(x::Vector{Float64})
  n = length(x)
  m1 = sum(x)/n
  m2 = sum((x .- m1).^2)/n
  m3 = sum((x .- m1).^3)/n
  m4 = sum((x .- m1).^4)/n
  
  b1 = (m3/m2^(3/2))^2
  b2 = (m4/m2^2)
  
  statistic = n*b1/6 + n*(b2 - 3)^2/24
  pvalue = exp(-statistic/2)
  statistic, pvalue
end
