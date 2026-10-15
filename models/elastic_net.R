# ------------------------------------------------------------
# Regime-aware stock selection
# FIE453 project
# Model 3: elastic net logit (alpha = 0.5)
# 24 signals x stress index, top and bottom 10 %, score = difference

# ------------------------------------------------------------

library(data.table)
library(glmnet)
library(pROC)

source('functions.R')

# load modelling data ----------------------------------
vars    <- readRDS('data/model_vars.rds')
signals <- vars$signals
regime  <- vars$regime
dt      <- fread('data/model_data.csv')

train <- dt[split == 'train']
test  <- dt[split == 'test']
rm(dt) ; gc()

# targets and stress index (same as in the logit script)
train[, bottom10 := as.integer(ret_fwd <= quantile(ret_fwd, 0.1)), by = yyyymm]
test[,  bottom10 := as.integer(ret_fwd <= quantile(ret_fwd, 0.1)), by = yyyymm]

train[, stress := (credit_spread + unrate - sentiment) / 3]
test[,  stress := (credit_spread + unrate - sentiment) / 3]

# design matrices ---------------------------------------
# all signals, stress, and every signal:stress interaction (no intercept column)
f3 <- as.formula(paste('~ (', paste(signals, collapse = ' + '), ') * stress'))
f3

# creates one column per signal, one for stress, and one for each signal * stress product (24+1+24=49)
X_train <- model.matrix(f3, data = train)[, -1] # [, -1] to drop intercept column (glm adds its own)
X_test  <- model.matrix(f3, data = test)[, -1]
dim(X_train) # rows x 49 columns

# model 3a: top 10 % --------------------------------------
# cross-validation over lambda on the 5 time-block folds
set.seed(1)
cv3_top <- cv.glmnet(X_train, train$top10, family = 'binomial', alpha = 0.5,
                     foldid = train$fold, type.measure = 'auc', nlambda = 30)
plot(cv3_top, sign.lambda = 1) # CV AUC against log(lambda); dashed lines = lambda.min and lambda.1se

# model 3b: bottom 10 % -----------------------------------
set.seed(1)
cv3_bot <- cv.glmnet(X_train, train$bottom10, family = 'binomial', alpha = 0.5,
                     foldid = train$fold, type.measure = 'auc', nlambda = 30)
plot(cv3_bot, sign.lambda = 1)

# out-of-sample predictions ---------------------------------
test[, p3_top := as.numeric(predict(cv3_top, newx = X_test, s = 'lambda.min', type = 'response'))]
test[, p3_bot := as.numeric(predict(cv3_bot, newx = X_test, s = 'lambda.min', type = 'response'))]
test[, score3 := p3_top - p3_bot] # high = more likely to end up at the top than at the bottom

rm(X_train, X_test) ; gc() # free memory

# which effects survived -------------------------------------
coef_table <- function(cv) {
  b <- coef(cv, s = 'lambda.min')
  data.table(term = rownames(b), coef = as.numeric(b))[term != '(Intercept)']
}
b_top <- coef_table(cv3_top)
b_bot <- coef_table(cv3_bot)
# ^ for both top and bottom prediction with elastic, no regime variables survive (using lambda.1se) update: changed to .min!

b <- merge(b_top, b_bot, by = 'term', suffixes = c('_top', '_bot'))
b[, top_minus_bottom := coef_top - coef_bot] # effect on the score (signed signals: + agrees with research)

b[coef_top != 0 | coef_bot != 0][order(-abs(top_minus_bottom))] # all kept effects
b[grepl(':stress', term) & top_minus_bottom != 0][order(-abs(top_minus_bottom))] # regime interactions kept
b[, .(kept_top = sum(coef_top != 0), kept_bot = sum(coef_bot != 0), total = .N)] # how many of 49 kept

# evaluation ---------------------------------------------------
auc(test$top10, test$p3_top)       # top model
auc(test$bottom10, test$p3_bot)    # bottom model

decile_table(test, 'score3')
# ^ bottom10_rate falling (good)
# ^ top10_rate falling (bad, in theory)
# ^ model bad at predicting winners - or more conservative. can check our sharpe and returns underneath

ls3_vw <- long_short(test, 'score3', weight = 'value')
ls3_ew <- long_short(test, 'score3', weight = 'equal')
rbind(value = perf(ls3_vw$ls), equal = perf(ls3_ew$ls)) |> round(3)
# ^ value-weighted long-short returns an annual return that is insignificant from zero
# ^ elastic net not good for now