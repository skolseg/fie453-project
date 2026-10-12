# ------------------------------------------------------------
# Model 1 and 2
# Logistic regression
# Predicting deciles with OSAP signals
# Model 2 includes regime variables

# ------------------------------------------------------------
source('functions.R')

library(data.table)
library(pROC)
library(stargazer)

# load modelling data ----------------------------------------
vars    <- readRDS('data/model_vars.rds')
signals <- vars$signals
regime  <- vars$regime
dt      <- fread('data/model_data.csv')

# baseline logit regression ---------------------------------
sig_small <- c('Mom12m', 'BM', 'GP', 'AssetGrowth', 'ShareIss1Y', 'MaxRet', 'Beta', 'DolVol') # use simplified signal list in beginning for example

train <- dt[split == 'train']
test <- dt[split == 'test']

# include bottom 10 % as target for later
train[, bottom10 := as.integer(ret_fwd <= quantile(ret_fwd, 0.1)), by = yyyymm]
test[,  bottom10 := as.integer(ret_fwd <= quantile(ret_fwd, 0.1)), by = yyyymm]
rm(dt) ; gc() # free up space, don't need dt anymore

f1 <- reformulate(sig_small, response = 'top10') # returns top10 ~ signal + signal + ...
m1 <- glm(f1, data = train, family = binomial, model = FALSE, y = FALSE)

summary(m1)

# out-of-sample testing
test[, p1 := predict(m1, newdata = test, type = 'response')]
auc(test$top10, test$p1)

# lift table - share of actual top 10 % stocks in each predicted decile
test[, dec1 := ceiling(10 * frank(p1, ties.method = 'first') / .N), by = yyyymm] # dec1 is decile number (1-10)
lift <- test[, .(hit_rate = mean(top10)), by = dec1][order(dec1)]
lift[, lift := hit_rate / mean(test$top10)] # hit rate is how many from said decile is predicted to be in decile 10 (top 10 %), lift is how many times better than random prediction is
lift
# ^ we get a very good lift table - however, we include volatility as a signal
# ^ volatile stocks can be picked by model for top decile, but are in turn also likely to perform worse due to volatility

# 1. Do decile-10 stocks also crash more often?
test[, .(top10_rate = mean(top10), bottom10_rate = mean(bottom10), avg_ret = mean(ret_fwd)), by = dec1][order(dec1)]
# ^ yes, they do
# ^ 24.4 % of next month's predicted best stocks actually end up in worst decile. cannot safely use model for trading strategy

# 2. Which signals drive the model?
sort(coef(m1)[-1])
# ^ volatility drives model most - the reason a lot of the predicted winners actually turn out to be losers
# ^ how to fix?

# model 1b) -----------------------------------------------
# include bottom 10 % in predictor
# prevent predicting losers as winners

f1_bot <- reformulate(sig_small, response = 'bottom10')
m1_bot <- glm(f1_bot, data = train, family = binomial, model = FALSE, y = FALSE)

stargazer(m1, m1_bot, type = 'text')
# ^ predictors for both top and bottom 10 % have same sign (+/-)

test[, p1_bot := predict(m1_bot, newdata = test, type = 'response')]
test[, score1 := p1 - p1_bot] # high = more likely to end up at the top than at the bottom

# check scores
decile_table(test, 'score1')
# now, avg return next month climbs as deciles climb!
# ^ earlier, they fell, when not testing on score

# long-short check
ls_vw <- long_short(test, 'score1', weight = 'value')
round(perf(ls_vw$ls), 3)

ls_ew <- long_short(test, 'score1', weight = 'equal')
round(perf(ls_ew$ls), 3)

# regime-aware logit -------------------------------------
# same logit, including regime variables interaction
# creating one stress variable from select regime variables
train[, stress := (credit_spread + unrate - sentiment) / 3]
test[,  stress := (credit_spread + unrate - sentiment) / 3]
train[, .(mean = mean(stress), sd = sd(stress))]          # check: mean 0

f2 <- as.formula(paste('top10 ~ (', paste(sig_small, collapse = ' + '), ') * stress'))
f2

m2 <- glm(f2, data = train, family = binomial, model = FALSE, y = FALSE)
stargazer(m1, m1_bot, m2, type = 'text')
# ^ 
test[, p2 := predict(m2, newdata = test, type = 'response')]
auc(test$top10, test$p2)
decile_table(test, 'p2')
