# ------------------------------------------------------------
# Regime-aware stock selection
# FIE453 project
# Model 4: XGBoost
# Top and bottom 10 %, score = difference

# ------------------------------------------------------------

library(data.table)
library(xgboost)
library(pROC)

source('functions.R')

# load modelling data ----------------------------------------
vars    <- readRDS('data/model_vars.rds')
signals <- vars$signals
regime <- vars$regime
dt      <- fread('data/model_data.csv')

train <- dt[split == 'train']
test  <- dt[split == 'test']
rm(dt) ; gc()

# bottom 10 % within each month (same as in the logit and elastic net scripts)
train[, bottom10 := as.integer(ret_fwd <= quantile(ret_fwd, 0.1)), by = yyyymm]
test[,  bottom10 := as.integer(ret_fwd <= quantile(ret_fwd, 0.1)), by = yyyymm]

# time-ordered cross-validation -------------------------------
# six consecutive month blocks give five expanding-window validation folds
# each fold trains only on earlier months; all stocks in a month stay together
months <- sort(unique(train$yyyymm))
stopifnot(length(months) >= 12L)
month_id <- match(train$yyyymm, months)
block <- ceiling(6 * month_id / length(months))
folds <- lapply(2:6, function(k) which(block == k))
# leave one month between training and validation because ret_fwd is next month's return
train_folds <- lapply(folds, function(i) which(month_id < min(month_id[i]) - 1L))
stopifnot(all(lengths(train_folds) > 0L))

# design matrices --------------------------------------------
# trees learn nonlinear effects and interactions directly from the signals AND regime variables
X <- c(signals, regime)
X_train <- as.matrix(train[, ..X])
X_test  <- as.matrix(test[, ..X])
dim(X_train)

# tuning settings --------------------------------------------
# tune learning rate and depth; CV early stopping selects the number of trees
params <- list(booster = 'gbtree',
objective = 'binary:logistic',
eval_metric = 'auc',
tree_method = 'hist', # histogram splits, much faster
max_bin = 64, # signals are ranks, 64 is plenty
min_split_loss = 0,
min_child_weight = 50, # leaf needs enough data, no splits on only handful of stocks
subsample = 0.8 # each tree sees 80% of rows, gives less overfitting
)

grid <- as.data.table(expand.grid(learning_rate = c(0.1), # faster than 0.05, probably as good
                                 max_depth = c(3, 6), # how many levels of splits each tree has - cv chooses
                                colsample_bytree = c(0.5, 1))) # each tree only use random share of columns - cv decides betrween .5 and 1

max_rounds <- 300L # lower learning rates may need more trees
tol <- 0.001 # choose smallest numbeer of trees within .001 of best CV AUC

# use the same folds and candidate settings for both targets
tune4 <- function(y) {
  dtrain <- xgb.DMatrix(X_train, label = y)
  results <- copy(grid)
  cv_models <- vector('list', nrow(grid))

  for (i in seq_len(nrow(grid))) {
    settings <- modifyList(params, as.list(grid[i]))
    set.seed(453)
    cv <- xgb.cv(params = settings, data = dtrain, nrounds = max_rounds,
                  folds = folds, train_folds = train_folds,
                  early_stopping_rounds = 30, maximize = TRUE, verbose = 1, print_every_n = 50)
    log <- cv$evaluation_log
    best <- which(log$test_auc_mean >= max(log$test_auc_mean) - tol)[1]
    results[i, `:=`(nrounds = log$iter[best], cv_auc = log$test_auc_mean[best])]
    cv_models[[i]] <- cv
    if (nrow(log) == max_rounds) {
      warning('CV reached max_rounds for candidate ', i, '; consider increasing the cap.')
    }
  }

  best <- which.max(results$cv_auc)
  list(results = results[order(-cv_auc)], cv = cv_models[[best]],
       params = modifyList(params, as.list(grid[best])), nrounds = results$nrounds[best])
}

# model 4a: top 10 % ------------------------------------------
cv4_top <- tune4(train$top10)
cv4_top$results
# ^ best learning_rate = .1, max_depth = 3, nrounds = 156, CV AUC = .6724 
plot(cv4_top$cv$evaluation_log$iter, cv4_top$cv$evaluation_log$test_auc_mean,
     type = 'l', xlab = 'Number of trees', ylab = 'Mean validation AUC', main = 'Top 10 %: cross-validation')
abline(v = cv4_top$nrounds, lty = 2, col = 'red')
# ^ validation AUC peaks at .... trees for the selected learning rate and depth

# model 4b: bottom 10 % ---------------------------------------
cv4_bot <- tune4(train$bottom10)
cv4_bot$results
# ^ best learning_rate = ...., max_depth = ...., nrounds = ...., CV AUC = ....
plot(cv4_bot$cv$evaluation_log$iter, cv4_bot$cv$evaluation_log$test_auc_mean,
     type = 'l', xlab = 'Number of trees', ylab = 'Mean validation AUC', main = 'Bottom 10 %: cross-validation')
abline(v = cv4_bot$nrounds, lty = 2, col = 'red')
# ^ validation AUC peaks at .... trees for the selected learning rate and depth

# final models -----------------------------------------------
# refit on the full training sample using each target's CV-selected settings
set.seed(453)
m4_top <- xgb.train(params = cv4_top$params,
                    data = xgb.DMatrix(X_train, label = train$top10),
                    nrounds = cv4_top$nrounds, verbose = 1)

set.seed(453)
m4_bot <- xgb.train(params = cv4_bot$params,
                    data = xgb.DMatrix(X_train, label = train$bottom10),
                    nrounds = cv4_bot$nrounds, verbose = 1)

# out-of-sample predictions -----------------------------------
dtest <- xgb.DMatrix(X_test)
test[, p4_top := as.numeric(predict(m4_top, newdata = dtest))]
test[, p4_bot := as.numeric(predict(m4_bot, newdata = dtest))]
test[, score4 := p4_top - p4_bot] # high = more likely to end up at the top than at the bottom

rm(X_train, X_test, dtest) ; gc()

# which signals drive the models ------------------------------
b_top <- xgb.importance(model = m4_top)
xgb.plot.importance(b_top)
b_top
b_bot <- xgb.importance(model = m4_bot)
xgb.plot.importance(b_bot)
# ^ which signals are most important for predicting winners?
head(b_bot, 15)
# ^ do the same signals drive both models, suggesting they predict extreme returns?
# ^ gain measures contribution to tree splits; it does not show the direction of an effect

# evaluation --------------------------------------------------
auc(test$top10, test$p4_top, levels = c(0, 1), direction = '<')
# ^ top model test AUC = ....
auc(test$bottom10, test$p4_bot, levels = c(0, 1), direction = '<')
# ^ bottom model test AUC = ....
# ^ is predicting losers easier than predicting winners?

# probability accuracy (Topic 5: page 34/37) --------
# MSE and pseudo-R2: top 10 % -------------------------------
MSE_top <- mean((test$top10 - test$p4_top)^2)
MSE_baseline_top <- mean((test$top10 - mean(train$top10))^2)
PseudoR2_top <- 1 - MSE_top / MSE_baseline_top

MSE_top
PseudoR2_top
# ^ MSE = ...., pseudo-R2 = ....

# MSE and pseudo-R2: bottom 10 % ----------------------------
MSE_bot <- mean((test$bottom10 - test$p4_bot)^2)
MSE_baseline_bot <- mean((test$bottom10 - mean(train$bottom10))^2)
PseudoR2_bot <- 1 - MSE_bot / MSE_baseline_bot

MSE_bot
PseudoR2_bot
# ^ MSE = ...., pseudo-R2 = ....
# ^ positive pseudo-R2 means better predictions than the training-frequency benchmark

decile_table(test, 'p4_top')
# ^ do predicted winners also frequently end up in the bottom decile?
decile_table(test, 'score4')
# ^ does avg_ret rise and bottom10_rate fall as the score increases?

ls4_vw <- long_short(test, 'score4', weight = 'value')
ls4_ew <- long_short(test, 'score4', weight = 'equal')
rbind(value = perf(ls4_vw$ls), equal = perf(ls4_ew$ls)) |> round(3)
# ^ value-weighted: ann_return = ...., Sharpe = ...., t_stat = ....
# ^ equal-weighted: ann_return = ...., Sharpe = ...., t_stat = ....
# ^ compare returns and Sharpe ratios with elastic net
# ^ does the strategy perform well with both value and equal weights?
