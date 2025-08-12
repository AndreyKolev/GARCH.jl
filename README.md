# Julia GARCH package

[![Build Status](https://travis-ci.org/AndreyKolev/GARCH.jl.svg?branch=master)](https://travis-ci.org/AndreyKolev/GARCH.jl)

The **Julia GARCH Package** provides a flexible framework for modeling time series data using Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models.

Designed for researchers and practitioners working with financial and economic time series, this package supports volatility modeling and conditional mean forecasting. It is built with extensibility in mind, enabling users to implement, customize, and extend GARCH-based models within the Julia ecosystem.


## Features Implemented

- **ARMA(p, q)** models for capturing the conditional mean of time series
- **GARCH(p, q)** and **gjrGARCH(p, q)** models for modeling conditional variance (volatility)
- Support for **Normal** and **Skew Normal** conditional distributions
- **n-step ahead forecasting** of both conditional mean and variance
- Built-in **error analysis** and diagnostic tools
- **Jarque-Bera test** for evaluating the normality of residuals


## Features Under Development

- **Enhanced model testing**
- **Simulation capabilities**


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
diagnostics(model, rets)
```


### Alternative Model Specification Example

You can also define a model using standard GARCH with a Normal distribution:

```julia
model = GARCHModel(ARMA(2, 2), sGARCH(2, 2), Normal())
```

## ⚠️Breaking changes
The API for fitting GARCH models has been completely refactored to support greater flexibility and modularity. The previous direct garchFit(ret) interface has been replaced with a more powerful, composable architecture based on the GARCHModel type.

### Key Changes:
- **Old API**: `fit = garchFit(rets)` returned a `GARCHFit` object containing data, parameters, likelihood, diagnostics, etc.
- **New API**: 
    - You now define a `GARCHModel` with explicit components (conditional mean, conditional variance, and conditional distribution), then use `fit!` to estimate parameters, and separate methods (`predict`, `diagnostics`) for subsequent analysis. 
    - `predict` now returns forecasts for both the conditional mean and variance - unlike the previous version, which returned only the conditional variance.
    - The new API requires time series data at every stage, ensuring that fitted models can be consistently applied to new data.

### Deprecated compatibility methods:
The old `garchFit` function is still available but marked as deprecated. It wraps the new API under the hood, so it will continue to work temporarily, but you should migrate to the new syntax for future compatibility and access to advanced features.

### Migration Guide:

1. **Replace `garchFit(ret)` with `GARCHModel` and `fit!`**:
   ```julia
   # Old way (deprecated)
   # fit = garchFit(rets)

   # New way
   model = GARCHModel(ARMA(1, 1), gjrGARCH(1, 1), SkewNormal())
   fit!(model, rets)
   ```

2. **Forecasting and diagnostics**:
   ```julia
   # Forecast
   pred_mu, pred_sigma = predict(model, rets)

   # Diagnostics
   diagnostics(model, rets)
   ```
 
### Why the change?
This new design enables:
- Easy combination of different mean, variance, and distribution components.
- Better separation of model specification, estimation, and inference.
- Support for more complex models (e.g., GJR-GARCH, ARMA-GARCH, different conditional distributions).

## Types & Methods reference

### `GARCHModel`
A composite model type that combines a **conditional mean model**, a **conditional variance model (GARCH-type)**, and a **conditional distribution**. It represents a full GARCH framework for modeling time series with volatility clustering.

### `ARMA`
A concrete type representing an ARMA(p,q) model for the conditional mean.

### `sGARCH`
A GARCH model GARCH(p,q) with no leverage effect.

### `gjrGARCH`
A GJR-GARCH model that captures **asymmetric volatility** (leverage effect), where negative shocks have a larger impact on volatility.

### `Normal`
A conditional distribution type assuming normal innovations (i.e., Gaussian errors).

### `SkewNormal`
A conditional distribution type assuming skew-normal errors, allowing for asymmetry in the return distribution.

### `fit!`
Performs **maximum likelihood estimation (MLE)** of the GARCH model parameters. Returns the optimized parameters and convergence status.

### `diagnostics`
Computes and returns a comprehensive set of diagnostic statistics for the fitted GARCH model, including:
- Parameter estimates and standard errors
- t-values and p-values
- Log-likelihood
- Information criteria (AIC, BIC, etc.)
- Jarque-Bera test for normality of standardized residuals

### `llh`
Computes the **log-likelihood** of the GARCH model given historical data, based on the current parameter values.

### `params!`
Sets the parameters of a model **in-place** from a vector.

### `predict`
Predicts the **conditional mean** and **conditional standard deviation** (volatility) for future periods (`n` steps ahead) using the fitted GARCH model.

### `unc_mean`
Returns the **unconditional mean** of the GARCH model, derived from the mean model.

### `unc_variance`
Returns the **long-run unconditional variance** of the GARCH model.

### `residuals`
Returns the model **residuals** (raw or standardized). If `standardize=true`, returns standardized residuals (residuals divided by conditional standard deviation).

### `fitted`
Returns the **fitted values** of the conditional mean from the model.

### `persistence`
Measures the **persistence of volatility** — Indicates how quickly shocks to volatility decay.

### `half_life`
Calculates the **half-life of volatility shocks** — the time it takes for a shock to decay to half its initial effect.

### `sigma` (alias for `σ`)
Returns the **conditional standard deviation** (volatility) of the GARCH model over the historical sample.

### IC
Compute information criteria (AIC, BIC, SIC, HQIC)

### `garchFit`
This function is kept for backward compatibility with the previous versions (deprecated). 

## Extending the Package

If you wish to extend the package, you can define custom components by creating new types in the following files:
- Conditional mean: `mean.jl`
- Conditional variance: `variance.jl`
- Conditional distribution: `distribution.jl`
- Statistical tests: `stattests.jl`

This modular structure allows for seamless integration of new models.

---

For more information, refer to the source code.

## Author
Andrey Kolev

## References
* T. Bollerslev (1986): Generalized Autoregressive Conditional Heteroscedasticity. Journal of Econometrics 31, 307–327.
* R. F. Engle (1982): Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation. Econometrica 50, 987–1008.
* Whittle, P. (1951). Hypothesis Testing in Time Series Analysis. Almquist and Wicksell. Whittle, P. (1963). Prediction and Regulation. English Universities Press. ISBN 0-8166-1147-5.
* O'Hagan, A.; Leonard, Tom (1976). "Bayes estimation subject to uncertainty about parameter constraints". Biometrika. 63 (1): 201–203. doi:10.1093/biomet/63.1.201. ISSN 0006-3444
* Glosten, L. R., R. Jagannathan, and D. E. Runkle. "On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks." The Journal of Finance. Vol. 48, No. 5, 1993, pp. 1779–1801.
