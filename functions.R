# ------------------------------------------------------------
# Helper functions for evaluation
# FIE453 project
# Load in model scripts with: source('functions.R')
# ------------------------------------------------------------

library(data.table)
library(sandwich) # Newey-West standard errors
library(lmtest)   # coeftest()

# decile table ------------------------------------------------
# hit rates and average next-month return per predicted decile (1 = lowest score, 10 = highest)
decile_table <- function(d, pcol) {
  d <- copy(d) # don't change the original table
  d[, dec := ceiling(10 * frank(get(pcol), ties.method = 'first') / .N), by = yyyymm]
  d[, .(top10_rate    = mean(top10),
        bottom10_rate = mean(bottom10),
        avg_ret       = mean(ret_fwd)), by = dec][order(dec)]
}

# long-short portfolio ------------------------------------------
# each month: buy decile 10, short decile 1 (by predicted score)
# weight = 'equal' (every stock the same) or 'value' (by market cap)
long_short <- function(d, pcol, weight = 'equal') {
  d <- copy(d)
  if (weight == 'value') d <- d[!is.na(me)] # value weights need market cap
  d[, dec := ceiling(10 * frank(get(pcol), ties.method = 'first') / .N), by = yyyymm]
  if (weight == 'value') {
    port <- d[dec %in% c(1, 10), .(r = weighted.mean(ret_fwd, me)), by = .(yyyymm, dec)]
  } else {
    port <- d[dec %in% c(1, 10), .(r = mean(ret_fwd)), by = .(yyyymm, dec)]
  }
  ls <- dcast(port, yyyymm ~ dec, value.var = 'r') # one row per month, columns for decile 1 and 10
  setnames(ls, c('1', '10'), c('short', 'long'))
  ls[, ls := long - short][order(yyyymm)]
}

# performance summary --------------------------------------------
# input: vector of monthly returns
perf <- function(r) {
  wealth <- cumprod(1 + r)
  c(ann_return = 12 * mean(r),                       # average annual return
    ann_vol    = sqrt(12) * sd(r),                   # annual volatility
    sharpe     = sqrt(12) * mean(r) / sd(r),         # annualized Sharpe ratio
    t_stat     = mean(r) / (sd(r) / sqrt(length(r))),# is the average return different from zero?
    max_dd     = min(wealth / cummax(wealth) - 1))   # worst peak-to-trough loss
}

# Fama-French 3-factor alpha -------------------------------------
# input: output of long_short() and the ff3 table (yyyymm, mkt_rf, smb, hml, rf as decimals)
# portfolio formed in month t earns its return in t+1, so shift to align with factors of t+1
ff3_alpha <- function(ls, ff3, col = 'ls') {
  x <- copy(ls)
  x[, yyyymm := fifelse(yyyymm %% 100 == 12, yyyymm + 89L, yyyymm + 1L)]
  x <- merge(x, ff3, by = 'yyyymm')
  m <- lm(reformulate(c('mkt_rf', 'smb', 'hml'), response = col), data = x)
  ct <- coeftest(m, vcov = NeweyWest(m, lag = 6))
  c(alpha_month = ct['(Intercept)', 1],              # monthly alpha
    alpha_year  = 12 * ct['(Intercept)', 1],         # annualized alpha
    t_alpha     = ct['(Intercept)', 3],              # Newey-West t-stat
    beta_mkt    = ct['mkt_rf', 1],
    beta_smb    = ct['smb', 1],
    beta_hml    = ct['hml', 1])
}