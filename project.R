# ------------------------------------------------------------
# Regime-aware stock selection
# FIE453 project
# Predict: will a stock be in top 10 % performers next month (0/1)
# Logit and XGBoost models
# ------------------------------------------------------------

library(data.table)

library(glmnet)
library(xgboost)

library(pROC)
library(sandwich)
library(lmtest)

library(ggplot2)

# data -------------------------------------------------------
signals <- c("Mom12m","Mom6m","IndMom","High52","ResidualMomentum","LRreversal",
             "BM","EP","cfp","SP","NetPayoutYield","GP","OperProf","RoE","roaq",
             "AssetGrowth","Investment","ChInv","Accruals","PctAcc",
             "ShareIss1Y","CompEquIss","NetEquityFinance",
             "IdioVol3F","RealizedVol","Beta","MaxRet","Illiquidity","DolVol")
regime  <- c("vix","credit_spread","term_spread","tbill","unrate","inflation","sentiment")

crsp <- fread('PROJECT/crsp_clean.csv')
osap <- fread('PROJECT/osap_signals.csv')
macro <- fread('PROJECT/macro.csv')

panel <- merge(crsp, osap, by = c('permno', 'yyyymm'), all.x = TRUE) # merge crsp and osap data
panel <- merge(panel, macro, by = 'yyyymm', all.x = TRUE) # merge panel data with macro data

rm(crsp, osap); gc() # free memory

# data exploration and quality assurance ---------------------
dim(panel) # rows and columns
names(panel) # column names
str(panel) # types and first values
range(panel$yyyymm) # first and last month
uniqueN(panel$permno) # number of unique stocks

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
panel[me <= 0, .N]                               # should be 0
panel[, quantile(ret_fwd, c(0, .001, .01, .5, .99, .999, 1), na.rm = TRUE)]   # extreme returns
# ^ 0.1% shows highest return amongst the 0.1% worst returns (-65% ish), etc. - 100% shows the best return ever (2658% ish), 99.9% shows the highest return of the 99.9% highest (150%)
panel[, .N, by = .(permno, yyyymm)][N > 1, .N]   # duplicate stock-months, should be 0
# ^ we have harmless duplicates in crsp, which we can filter away by using unique(). the differnces are not in predictors, only in securitytype (e.g. EQTY vs. N/A)
# ^ will probably show 0 now if filtered before this line
panel[is.na(ret_fwd), .N] / nrow(panel) # share without a target

# filter panel to >1977 (coverage from all signals)
dt <- panel[yyyymm >= 197701 &          # from 1977
            prc >= 5 &                  # price at least $5 (no penny stocks)
            !is.na(ret_fwd) &           # has next month's return
            !is.na(me)]                 # has market cap

dt <- dt[me >= quantile(me, 0.2), .SD, by = yyyymm]   # drop the smallest 20% each month (no microcaps)

n <- dt[, .N, by = yyyymm] # no. of stocks each month
n[, date := as.IDate(paste0(yyyymm, '01'), format = '%Y%m%d')] # turn yyyymm into proper date

ggplot(n, aes(date, N)) +
  geom_line() +
  labs(x = NULL, y = 'Number of stocks', title = 'Stocks per month after filters') +
  theme_classic()

# simple model
d <- copy(dt)
d[, top10 := as.integer(ret_fwd >= quantile(ret_fwd, 0.9)), by = yyyymm]

d[, (signals) := lapply(.SD, function(x) {
  r <- frank(x, na.last = "keep") / sum(!is.na(x)) - 0.5
  fifelse(is.na(r), 0, r)
}), by = yyyymm, .SDcols = signals]

# split by time
train <- d[yyyymm <  201001]
test  <- d[yyyymm >= 201001]

# logit
m <- glm(reformulate(signals, "top10"), data = train, family = binomial)

test[, p := predict(m, newdata = test, type = "response")]
auc(test$top10, test$p)
 