library(data.table)
library(OpenSourceAP.DownloadR)

openap <- OpenAP$new()

signals <- c("Mom12m","Mom6m","IndMom","High52","ResidualMomentum","LRreversal",
             "BM","EP","cfp","SP","NetPayoutYield","GP","OperProf","RoE","roaq",
             "AssetGrowth","Investment","ChInv","Accruals","PctAcc",
             "ShareIss1Y","CompEquIss","NetEquityFinance",
             "IdioVol3F","RealizedVol","Beta","MaxRet","Illiquidity","DolVol")

dir.create("osap_files", showWarnings = FALSE)

# a download counts as OK only if it is real data, not a Google Drive HTML page
ok <- function(x) {
  !is.null(x) && is.data.frame(x) && nrow(x) > 0 &&
    all(c("permno", "yyyymm") %in% names(x))
}

for (s in signals) {
  file <- file.path("osap_files", paste0(s, ".csv"))
  if (file.exists(file)) { cat(s, ": already saved\n"); next }

  x <- tryCatch(suppressWarnings(openap$dl_signal(s, signed = TRUE)),
                error = function(e) NULL)

  if (ok(x)) {
    x <- setDT(x)[, .SD, .SDcols = c("permno", "yyyymm", s)]
    fwrite(x, file)
    cat(s, ": OK (", nrow(x), "rows )\n")
  } else {
    cat(s, ": FAILED\n")
  }
  Sys.sleep(3)   # small pause so Google Drive is not hit too fast
}

# ---- Summary ----
saved   <- sub("\\.csv$", "", list.files("osap_files", pattern = "\\.csv$"))
missing <- setdiff(signals, saved)

cat("\nDownloaded (", length(intersect(signals, saved)), "):\n",
    paste(intersect(signals, saved), collapse = ", "), "\n")
cat("\nNOT downloaded (", length(missing), ") - get these manually:\n",
    paste(missing[order(tolower(missing))], collapse = ", "), "\n")
# ^ download missing signals manually from osap google drive, add to osap_manual

# make signed = TRUE for manual files, add to osap_files
doc  <- setDT(openap$dl_signal_doc())
sign <- setNames(doc$Sign, doc$Acronym)

for (f in list.files("osap_manual", pattern = "\\.csv$", full.names = TRUE)) {
  s <- sub("\\.csv$", "", basename(f))
  x <- fread(f)
  x[, (s) := get(s) * sign[[s]]]          # flip so higher = higher expected return
  fwrite(x[, .SD, .SDcols = c("permno", "yyyymm", s)],
         file.path("osap_files", paste0(s, ".csv")))
  cat(s, ": flipped with sign", sign[[s]], "and saved\n")
}

# merge to one file for project
files <- list.files('osap_files', pattern = '\\.csv$', full.names = TRUE)

osap <- NULL
for (f in files) {
  x <- fread(f)
  setnames(x, 1:2, c("permno", "yyyymm"))
  osap <- if (is.null(osap)) x else merge(osap, x, by = c("permno", "yyyymm"), all = TRUE)
  cat(basename(f), "merged\n")
}

setcolorder(osap, c('permno', 'yyyymm', intersect(signals, names(osap)))) # same order as signals
setorder(osap, permno, yyyymm)
fwrite(osap, 'data/osap_signals.csv')

cat("\nosap_signals.csv:", nrow(osap), "rows,", ncol(osap) - 2, "signals,",
    "months", min(osap$yyyymm), "to", max(osap$yyyymm), "\n")
