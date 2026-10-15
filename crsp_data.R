library(data.table)

crsp <- fread('data/crsp.csv')

dim(crsp) # rows and cols
sapply(crsp, class) # type cols
colSums(is.na(crsp)) # NA per col

# lowercase cols for later merge with OSAP
setnames(crsp, tolower(names(crsp)))

# include only US common stocks on NYSE, AMEX, NASDAQ
crsp <- crsp[sharetype == 'NS' & 
  securitytype == 'EQTY' & 
  securitysubtype == 'COM' & 
  usincflg == 'Y' & 
  issuertype %in% c('ACOR', 'CORP') & 
  primaryexch %in% c('N', 'A', 'Q')]

# remove duplicate stock-months from WRDS download (harmless)
crsp <- unique(crsp, by = c('permno', 'yyyymm'))
crsp[, .N, by = .(permno, yyyymm)][N > 1, .N] # check: should be 0

# market cap (market equity)
crsp[, me := abs(mthprc) * shrout]

# sort data
setorder(crsp, permno, yyyymm)

# next month's return (our target)
crsp[, next_ym := shift(yyyymm, type = 'lead'), by = permno] # next year and month (202512 -> 202601)
crsp[, ret_fwd := shift(mthret, type = 'lead'), by = permno] # next monthly return
crsp[next_ym != fifelse(yyyymm %% 100 == 12, yyyymm + 89L, yyyymm + 1L), ret_fwd := NA] # only gives next mnth ret if there is an actual next month in data

# keep only necessary columns, save as .csv
crsp <- crsp[, .(permno, yyyymm, ret = mthret, ret_fwd, me, prc = abs(mthprc), exch = primaryexch)] # price can be negative if estimate, return absolute number
fwrite(crsp, 'data/crsp_clean.csv')
