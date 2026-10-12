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

# load modelling data ----------------------------------
vars    <- readRDS('data/model_vars.rds')
signals <- vars$signals # ADDED
regime  <- vars$regime # ADDED
dt      <- fread('data/model_data.csv')

# model 1 --------------------------------------------
# baseline logistic regression
# only signals as predictor

train <- dt[split == 'train']
test <- dt[split == 'test']
rm(dt) ; gc() # free up space, don't need dt anymore

f1 <- reformulate(signals, response = 'top10') # returns top10 ~ *signal + signal + signal + ...*
m1 <- glm(f1, data = train, family = binomial) # logistic reg on all osap signals

summary(m1)

# out-of-sample testing
test[, p1 := predict(m1, newdata = test, type = 'response')]

roc1 <- roc(test$top10, test$p1, quiet = TRUE)
auc(roc1)
plot(roc1, legacy.axes = TRUE, main = 'ROC curve, baseline logistic regression (test 2010-)')

# lift table - share of actual top 10 % stocks in each predicted decile
test[, dec1 := ceiling(10 * frank(p1, ties.method = 'first') / .N), by = yyyymm] # dec1 is decile number (1-10)
lift <- test[, .(hit_rate = mean(top10)), by = dec1][order(dec1)] 
lift[, lift := hit_rate / mean(test$top10)] # hit rate is how many from said decile is predicted to be in decile 10 (top 10 %), lift is how many times better than random prediction is
lift
# ^ we get a very good lift table - however, we include volatility as a signal
# ^ volatile stocks can be picked by model for top decile, but are in turn also likely to perform worse due to volatility

# 1. Do decile-10 stocks also crash more often?
test[, bottom10 := as.integer(ret_fwd <= quantile(ret_fwd, 0.1)), by = yyyymm]
test[, .(top10_rate = mean(top10), bottom10_rate = mean(bottom10), avg_ret = mean(ret_fwd)), by = dec1][order(dec1)]
# ^ yes, they do

# 2. Which signals drive the model?
sort(coef(m1)[-1])
# ^ volatility drives most

# model 1b) predict bottom 10 % as well, find difference
train[, bottom10 := as.integer(ret_fwd <= quantile(ret_fwd, 0.1)), by = yyyymm]
test[,  bottom10 := as.integer(ret_fwd <= quantile(ret_fwd, 0.1)), by = yyyymm]

m1_bot <- glm(reformulate(signals, 'bottom10'), data = train, family = binomial,
              model = FALSE, y = FALSE)

test[, p1_bot := predict(m1_bot, newdata = test, type = 'response')]
test[, score1 := p1 - p1_bot]     # high = more likely to end up at the top than at the bottom

# same checks as before
test[, dec_s := ceiling(10 * frank(score1, ties.method = 'first') / .N), by = yyyymm]
test[, .(top10_rate = mean(top10), bottom10_rate = mean(bottom10), avg_ret = mean(ret_fwd)),
     by = dec_s][order(dec_s)]


# model 2 --------------------------------------------
# logit reg including regime variables
# interaction between signals and regime vars
# dropping a lot of signals to control model more (RAM etc.)
# creating a stress variable from select regime variables

sig_small <- c('Mom12m', 'BM', 'GP', 'AssetGrowth', 'ShareIss1Y', 'MaxRet', 'Beta', 'DolVol')
train[, stress := (credit_spread + unrate - sentiment) / 3]
test[,  stress := (credit_spread + unrate - sentiment) / 3]
train[, .(mean = mean(stress), sd = sd(stress))]          # check: mean 0

f2 <- as.formula(paste('top10 ~ (', paste(sig_small, collapse = ' + '), ') * stress'))
f2

m2 <- glm(f2, data = train, family = binomial, model = FALSE, y = FALSE)
summary(m2)
# ^ 
test[, p2 := predict(m2, newdata = test, type = 'response')]
auc(test$top10, test$p2)

# TODO:
# value-weigh portfolios
# include regime
# use other models than logit
# elastic net logit and xgboost