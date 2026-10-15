# ------------------------------------------------------------
# Regime-aware stock selection
# FIE453 project
# Predict: will a stock be in top 10 % performers next month (0/1)
# Logit and XGBoost models
# ------------------------------------------------------------

library(data.table)

library(ggplot2)

# data -------------------------------------------------------
signals <- c("Mom12m","Mom6m","IndMom","High52","ResidualMomentum","LRreversal",
             "BM","EP","cfp","SP","NetPayoutYield","GP","OperProf","RoE","roaq",
             "AssetGrowth","Investment","ChInv","Accruals","PctAcc",
             "ShareIss1Y","CompEquIss","NetEquityFinance",
             "IdioVol3F","RealizedVol","Beta","MaxRet","Illiquidity","DolVol")
regime  <- c("vix","credit_spread","term_spread","tbill","unrate","inflation","sentiment")

crsp <- fread('data/crsp_clean.csv')
osap <- fread('data/osap_signals.csv')
macro <- fread('data/macro.csv')

panel <- merge(crsp, osap, by = c('permno', 'yyyymm'), all.x = TRUE) # merge crsp and osap data
panel <- merge(panel, macro, by = 'yyyymm', all.x = TRUE) # merge panel data with macro data

rm(crsp, osap); gc() # free memory

# data exploration and quality assurance ---------------------
dim(panel) # rows and columns
names(panel) # column names
str(panel) # types and first values
range(panel$yyyymm) # first and last month
uniqueN(panel$permno) # number of unique stocks
head(panel)

panel[, .N, by = .(year = yyyymm %/% 100)][order(year)] # stock-months per year
# ^ stort hopp fra 1972 til 1973 siden NASDAQ kommer inn da - starte treningsgrunnlag etter dette

round(colMeans(is.na(panel)) * 100, 1) # % missing per column

# signal coverage per year
cov <- panel[, lapply(.SD, function(x) mean(!is.na(x))),
             by = .(year = yyyymm %/% 100), .SDcols = signals]
cov_long <- melt(cov, id.vars = "year", variable.name = "signal", value.name = "coverage")
cov_long[coverage >= 0.5, .(first_year = min(year)), by = signal][order(first_year)] # first year for each signal with coverage > 50 %
# ^ operprof never appearing, will not include in models

# macro variables first month appearing
sapply(regime, function(v) panel[!is.na(get(v)), min(yyyymm)])
# ^ vix not appearing until 1990, everything else before 1972

# signal coverage heatmap
ggplot(cov_long, aes(year, signal, fill = coverage)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "steelblue", labels = scales::percent) +
  labs(x = NULL, y = NULL, fill = "Coverage", title = "Share of stocks with each signal")

# CRSP sanity checks
panel[ret_fwd < -1, .N]                          # should be 0 (can't lose more than 100%)
panel[me <= 0, .N]                               # should be 0 (can't have negative marketcap)
panel[, quantile(ret_fwd, c(0, .001, .01, .5, .99, .999, 1), na.rm = TRUE)]   # extreme returns
# ^ 0.1% shows highest return amongst the 0.1% worst returns (-65% ish), etc. - 100% shows the best return ever (2658% ish), 99.9% shows the highest return of the 99.9% highest (150%)
panel[, .N, by = .(permno, yyyymm)][N > 1, .N]   # duplicate stock-months, should be 0
# ^ we have harmless duplicates in crsp, which we can filter away by using unique(). the differnces are not in predictors, only in securitytype (e.g. EQTY vs. N/A)
# ^ will probably show 0 now if filtered before this line (is filtered in crsp_data.R now)
panel[is.na(ret_fwd), .N] / nrow(panel) # share without a target

# filter panel to >1977 (coverage from all signals) and has ret_fwd (our target)
dt <- panel[yyyymm >= 197701 & !is.na(ret_fwd)]
rm(panel) ; gc() # free up space, don't need panel anymore

n <- dt[, .N, by = yyyymm] # no. of stocks each month
n[, date := as.IDate(paste0(yyyymm, '01'), format = '%Y%m%d')] # turn yyyymm into proper date

# stocks we have each month in dt
ggplot(n, aes(date, N)) +
  geom_line() +
  labs(x = NULL, y = 'Number of stocks', title = 'Stocks per month after filters') +
  theme_classic()

# target ----------------------------------------------------
# we want to target the top decile for next month (top 10 %)
dt[, top10 := as.integer(ret_fwd >= quantile(ret_fwd, 0.9)), by = yyyymm] # 1 if in top 10 %

dt[, mean(top10), by = yyyymm][, summary(V1)] # check: about 0.10 every month
# ^ all around 0.1, everything is good

# investable -----------------------------------------------
# flag stocks a fund could realistically trade
# price > 5 USD, mcap above NYSE 20th percentile, measured at month t (no look-ahead)


# signal list exclusion and ranking ------------------------
coverage <- sapply(signals, function(s) dt[yyyymm <= 200912, mean(!is.na(get(s)))]) # coverage in training period
signals  <- names(coverage)[coverage >= 0.5] # keep signals with at least 50 % coverage
setdiff(names(coverage), signals) # dropped signals

rank_x <- function(x) {
  r <- frank(x, na.last = 'keep') / sum(!is.na(x)) - 0.5 # rank scaled -0.5..0.5
  fifelse(is.na(r), 0, r) # missing value -> return middle value
}
# ^ if a stock has the worst BM one month, it gets a ranking of almost -0.5 (vice versa)
# ^ this is to make outliers not have as big of an impact to the models

dt[, (signals) := lapply(.SD, rank_x), by = yyyymm, .SDcols = signals] # apply rank function to signals

# regime variables -------------------------------------
# VIX starts in 1990, sentiment starts in 1978
sapply(regime, function(v) dt[, sum(is.na(get(v)))]) # missing per variable from 1977

# what months are missing from the variables
dt[is.na(unrate), unique(yyyymm)]
# ^ missing data one month due to government shutdown - same goes for inflation
dt[is.na(sentiment), unique(yyyymm)]
# ^ used to publish quarterly before 1978
# ^ might be fixed later in fred_data.R, not shown here anymore

regime <- setdiff(regime, 'vix') # remove VIX from regime variables, starts too late

# train-test-split -------------------------------------
dt[, split := fifelse(yyyymm <= 200912, 'train', 'test')] # train 1977-2009, test 2010-

# specifications of the split
dt[, .(from   = min(yyyymm),
       to     = max(yyyymm),
       months = uniqueN(yyyymm),
       years  = round(uniqueN(yyyymm) / 12, 1),
       rows   = .N),
   by = split][, share_rows := round(rows / sum(rows), 2)][]

# time-block folds for cross-validation (training period only, test rows get NA)
dt[split == 'train', fold := cut(yyyymm %/% 100, breaks = 5, labels = FALSE)]
dt[split == 'train', .(from = min(yyyymm), to = max(yyyymm)), by = fold][order(fold)] # check the periods

# standardizing regime variables ----------------------
# turns every regime variable into "how many standard deviations away is it from normal level"
# only compute mean and sd from training data, so model doesn't "know the future"
for (v in regime) {
  mu <- dt[split == 'train', mean(get(v), na.rm = TRUE)] # mean in training period
  s <- dt[split == 'train', sd(get(v), na.rm = TRUE)] # standard dev in training period
  dt[, (v) := (get(v) - mu) / s] # apply to all rows, train and test
}

# tests
dt[split == 'train', lapply(.SD, function(x) round(c(mean = mean(x), sd = sd(x)), 2)), .SDcols = regime]
dt[split == 'test',  lapply(.SD, function(x) round(c(mean = mean(x), sd = sd(x)), 2)), .SDcols = regime]

# drop highly correlated signals --------------------
# only checking on training period (no look-ahead)
# remove anything with corr >= 0.7 (rule of thumb)
# avoid multicollinearity in models
# remove redundancy, as well as free up ram space
# if none is dropped, signals list has been changed

set.seed(1)
samp <- dt[split == 'train'][sample(.N, 200000), ..signals]
C <- abs(cor(samp))
diag(C) <- 0

drop <- character(0)
repeat {
if (max(C) <= 0.7) break
idx <- which(C == max(C), arr.ind = TRUE)[1, ]
pair <- rownames(C)[idx]
loser <- pair[which.min(coverage[pair])]        # keep the one with better coverage
drop <- c(drop, loser)
C <- C[setdiff(rownames(C), loser), setdiff(colnames(C), loser)]
}

drop                                  # which signals were removed
signals <- setdiff(signals, drop)
length(signals)

rm(samp) ; gc()

rm( cov, cov_long, n, macro); gc() # free some space

# save modelling data ----------------------------------
keep <- c('permno', 'yyyymm', 'split', 'fold', 'ret_fwd', 'top10', 'me', signals, regime)
fwrite(dt[, ..keep], 'data/model_data.csv')
saveRDS(list(signals = signals, regime = regime), 'data/model_vars.rds')
rm(list = ls()); gc() # remove all objects and free memory
