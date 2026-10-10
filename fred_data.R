library(data.table)
library(fredr)

# function to pull series from FRED database
get_fred <- function(id) {
  x <- setDT(fredr(series_id = id, observation_start = as.Date("1960-01-01"),
                   frequency = "m", aggregation_method = "avg"))
  x <- x[, .(yyyymm = year(date) * 100L + month(date), value)]
  setnames(x, "value", id)
  x
}
# what we want to pull
ids <- c("VIXCLS","BAA","AAA","GS10","TB3MS","UNRATE","CPIAUCSL","UMCSENT")

# macro variables
macro <- Reduce(function(a, b) merge(a, b, by = 'yyyymm', all = TRUE), lapply(ids, get_fred)) # download each series from FRED, merge all into one table

macro[, `:=`(credit_spread = BAA - AAA, 
             term_spread   = GS10 - TB3MS, # 10yr treasury yield minus 3mnth treasury bill rate
             inflation     = 100 * (CPIAUCSL / shift(CPIAUCSL, 12) - 1))] # annual inflation growth rate in said month

macro <- macro[, .(yyyymm, vix = VIXCLS, credit_spread, term_spread,
                   tbill = TB3MS, unrate = UNRATE, inflation, sentiment = UMCSENT)]

# one month lag (IMPORTANT)
setorder(macro, yyyymm)

regime <- setdiff(names(macro), 'yyyymm') # all macro variables
macro[, (regime) := lapply(.SD, shift), .SDcols = regime] # only lagging regime macro variables, not yyyymm

# forward-fill missing values
setorder(macro, yyyymm)
macro[, c('unrate', 'inflation', 'sentiment') := lapply(.SD, nafill, type = 'locf'),
      .SDcols = c('unrate', 'inflation', 'sentiment')]

fwrite(macro, 'data/macro.csv')
