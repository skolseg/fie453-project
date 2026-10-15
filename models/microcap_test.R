# ------------------------------------------------------------
# Test: train on investable universe only
# Model 1b (simple logit, 8 signals), top and bottom 10 %, score = difference
# A: train all stocks       -> evaluate investable
# B: train investable only  -> evaluate investable
# ------------------------------------------------------------

library(data.table)
library(pROC)
source('functions.R')

# load data ----------------------------------------------
dt <- fread('data/model_data.csv')
stopifnot('investable' %in% names(dt))

sig_small <- c('Mom12m', 'BM', 'GP', 'AssetGrowth', 'ShareIss1Y', 'MaxRet', 'Beta', 'DolVol')
keep <- c('permno', 'yyyymm', 'split', 'ret_fwd', 'top10', 'me', 'investable', sig_small)
dt <- dt[, ..keep]
gc()

# helper: rank scaled to -0.5..0.5 within a month (same as in prep script)
rank_x <- function(x) frank(x, ties.method = 'average') / length(x) - 0.5

# ============================================================
# A: train on all stocks (targets and ranks as in prep script)
# ============================================================
dt[, bottom10 := as.integer(ret_fwd <= quantile(ret_fwd, 0.1)), by = yyyymm]

train_A <- dt[split == 'train']
fA_top <- reformulate(sig_small, response = 'top10')
fA_bot <- reformulate(sig_small, response = 'bottom10')
mA_top <- glm(fA_top, data = train_A, family = binomial, model = FALSE, y = FALSE)
mA_bot <- glm(fA_bot, data = train_A, family = binomial, model = FALSE, y = FALSE)
rm(train_A) ; gc()

# ============================================================
# B: investable only - redefine targets and ranks within the universe
# ============================================================
inv <- dt[investable == 1]

# targets: top / bottom 10 % among investable stocks each month
inv[, top10_inv    := as.integer(ret_fwd >= quantile(ret_fwd, 0.9)), by = yyyymm]
inv[, bottom10_inv := as.integer(ret_fwd <= quantile(ret_fwd, 0.1)), by = yyyymm]

# signal ranks: re-rank within investable stocks each month
# (keep the all-stock ranks for model A's predictions)
sig_inv <- paste0(sig_small, '_inv')
inv[, (sig_inv) := lapply(.SD, rank_x), by = yyyymm, .SDcols = sig_small]

train_B <- inv[split == 'train']
fB_top <- reformulate(sig_inv, response = 'top10_inv')
fB_bot <- reformulate(sig_inv, response = 'bottom10_inv')
mB_top <- glm(fB_top, data = train_B, family = binomial, model = FALSE, y = FALSE)
mB_bot <- glm(fB_bot, data = train_B, family = binomial, model = FALSE, y = FALSE)
rm(train_B) ; gc()

# ============================================================
# evaluate both on the investable test sample
# ============================================================
test <- inv[split == 'test']

test[, pA_top := predict(mA_top, newdata = test, type = 'response')]
test[, pA_bot := predict(mA_bot, newdata = test, type = 'response')]
test[, scoreA := pA_top - pA_bot]

test[, pB_top := predict(mB_top, newdata = test, type = 'response')]
test[, pB_bot := predict(mB_bot, newdata = test, type = 'response')]
test[, scoreB := pB_top - pB_bot]

# use the investable targets for evaluation (what a fund cares about)
test[, top10 := top10_inv] ; test[, bottom10 := bottom10_inv]

# coefficients: does the model learn different things?
round(cbind(A_top = coef(mA_top)[-1], B_top = coef(mB_top)[-1],
            A_bot = coef(mA_bot)[-1], B_bot = coef(mB_bot)[-1]), 3)
# ^ do signs and sizes change when micro-caps are excluded?

# AUC
c(A_top = auc(test$top10, test$pA_top, quiet = TRUE),
  B_top = auc(test$top10, test$pB_top, quiet = TRUE),
  A_bot = auc(test$bottom10, test$pA_bot, quiet = TRUE),
  B_bot = auc(test$bottom10, test$pB_bot, quiet = TRUE)) |> round(4)

# decile tables
decile_table(test, 'scoreA')
decile_table(test, 'scoreB')
# ^ which one has the steadier rise in avg_ret?

# long-short
rbind(A_vw = perf(long_short(test, 'scoreA', 'value')$ls),
      B_vw = perf(long_short(test, 'scoreB', 'value')$ls),
      A_ew = perf(long_short(test, 'scoreA', 'equal')$ls),
      B_ew = perf(long_short(test, 'scoreB', 'equal')$ls)) |> round(3)
# ^ does training on investable stocks improve the tradable strategy?