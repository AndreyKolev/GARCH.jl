# Julia GARCH package

[![Build Status](https://travis-ci.org/AndreyKolev/GARCH.jl.svg?branch=master)](https://travis-ci.org/AndreyKolev/GARCH.jl)

The **Julia GARCH Package** provides a flexible framework for modeling time series data using Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models.

Designed for researchers and practitioners working with financial and economic time series, this package supports volatility modeling and conditional mean forecasting. It is built with extensibility in mind, enabling users to implement, customize, and extend GARCH-based models within the Julia ecosystem.

---

## Features Implemented

- **ARMA(p, q)** models for capturing the conditional mean of time series
- **GARCH(p, q)** and **gjrGARCH(p, q)** models for modeling conditional variance (volatility)
- Support for **Normal** and **Skew Normal** conditional distributions
- **n-step ahead forecasting** of both conditional mean and variance
- Built-in **error analysis** and diagnostic tools
- **Jarque-Bera test** for evaluating the normality of residuals

---

## Features Under Development

- **Enhanced model testing**
- **Simulation capabilities**

---

## Usage

The package employs an **object-oriented design**, allowing intuitive model composition, customization, and extension. Users can easily define and combine components for conditional mean, conditional variance, and distributional assumptions.

### Example: Basic Workflow

Load your data and compute log returns:

```julia
quotes = readdlm("quotes.csv", ',')
price = float.(quotes[:, 2])
rets = diff(log.(price))
```

Define a model with ARMA(1,1) for the conditional mean, gjrGARCH(1,1) for volatility, and a Skew Normal distribution:

```julia
using GARCH
model = GARCHModel(ARMA(1, 1), gjrGARCH(1, 1), SkewNormal())
```

Fit the model to the return series (stored in rets array):

```julia
fit!(model, rets)
```

Perform forecasting after fitting:

```julia
pred_mu, pred_sigma = predict(model, rets)
```

Access diagnostic information:

```julia
diagnostics_output = diagnostics(model, rets)
```

---

### Alternative Model Specification Example

You can also define a model using standard GARCH with a Normal distribution:

```julia
model = GARCHModel(ARMA(2, 2), sGARCH(2, 2), Normal())
```

---

## Extending the Package

If you wish to extend the package, you can define custom components by creating new types in the following files:
- Conditional mean: `mean.jl`
- Conditional variance: `variance.jl`
- Conditional distribution: `distribution.jl`

This modular structure allows for seamless integration of new models.

---

For more information, refer to the source code.
