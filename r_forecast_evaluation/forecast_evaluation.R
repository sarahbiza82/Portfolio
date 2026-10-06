# Forecast evaluation on market data: returns (wheat futures) and realised volatility (KOSPI 200)
#
#  1. Returns: zero forecast and historical mean as benchmarks (a random walk in prices implies a zero return
#     forecast); persistence is kept under its own name. Diebold-Mariano tests with HAC variance,
#     Mincer-Zarnowitz as a joint Wald test (alpha = 0, beta = 1) with HAC errors, Sharpe ratio of a sign strategy.
#  2. Volatility: every model forecasts the same target (RV) so QLIKE losses are comparable; log forecasts are
#     mapped back to levels with the lognormal correction.
#
# Run from this folder:  Rscript forecast_evaluation.R

suppressPackageStartupMessages({
  library(readxl); library(dplyr); library(tidyr); library(ggplot2)
  library(sandwich); library(lmtest); library(car); library(zoo)
})
dir.create("output", showWarnings = FALSE)
stars <- function(p) ifelse(is.na(p), "", ifelse(p < .01, "***", ifelse(p < .05, "**", ifelse(p < .10, "*", ""))))

# Diebold-Mariano: loss differential regressed on a constant, HAC variance.
# Positive statistic = the candidate has a LOWER loss than the benchmark.
dm <- function(loss_bench, loss_model, lag) {
  d <- loss_bench - loss_model
  fit <- lm(d ~ 1)
  se <- sqrt(NeweyWest(fit, lag = lag, prewhite = FALSE)[1, 1])
  t <- mean(d) / se
  c(stat = t, p = 2 * pnorm(-abs(t)))
}

# ============================================================================================
# PART 1. Five-day-ahead forecasts of daily wheat futures returns, rolling windows
# ============================================================================================
w <- read_excel("data/wheat_support5_STU.xlsx") |> mutate(date = as.Date(date))
y <- w$return; n <- length(y); h <- 5
win5 <- 5 * 252; win3 <- 3 * 252

ar1_fc <- function(x, h) {                       # AR(1) with intercept, estimated by OLS on the window
  fit <- lm(x[-1] ~ x[-length(x)])
  c0 <- coef(fit)[1]; phi <- coef(fit)[2]
  mu <- c0 / (1 - phi)
  unname(mu + phi^h * (x[length(x)] - mu))
}

idx <- (win5):(n - h)
fc <- data.frame(date = w$date[idx + h], actual = y[idx + h],
                 zero = 0,
                 mean5y = sapply(idx, function(t) mean(y[(t - win5 + 1):t])),
                 persistence = y[idx],
                 AR1_5y = sapply(idx, function(t) ar1_fc(y[(t - win5 + 1):t], h)),
                 AR1_3y = sapply(idx, function(t) ar1_fc(y[(t - win3 + 1):t], h)))
models1 <- c("mean5y", "persistence", "AR1_5y", "AR1_3y")

tab1 <- bind_rows(lapply(c("zero", models1), function(m) {
  e <- fc$actual - fc[[m]]
  out <- data.frame(model = m, RMSE_bp = 1e4 * sqrt(mean(e^2)), MAE_bp = 1e4 * mean(abs(e)))
  if (m != "zero") {
    d1 <- dm((fc$actual - fc$zero)^2, e^2, lag = h - 1); d2 <- dm(abs(fc$actual - fc$zero), abs(e), lag = h - 1)
    out$DM_MSE <- sprintf("%.2f%s", d1["stat"], stars(d1["p"])); out$DM_MAE <- sprintf("%.2f%s", d2["stat"], stars(d2["p"]))
    mz <- lm(fc$actual ~ fc[[m]])
    lh <- linearHypothesis(mz, c("(Intercept) = 0", "fc[[m]] = 1"), vcov. = NeweyWest(mz, lag = h - 1, prewhite = FALSE), test = "Chisq")
    out$MZ_beta <- round(coef(mz)[2], 3); out$MZ_joint_p <- signif(lh$`Pr(>Chisq)`[2], 2)
  } else { out$DM_MSE <- "benchmark"; out$DM_MAE <- "benchmark"; out$MZ_beta <- NA; out$MZ_joint_p <- NA }
  # economic value: long/short one unit according to the sign of the forecast
  pos <- sign(fc[[m]]); r <- pos * fc$actual
  out$Sharpe_ann <- if (sd(r) > 0) round(sqrt(252) * mean(r) / sd(r), 2) else NA
  out
}))
bh <- round(sqrt(252) * mean(fc$actual) / sd(fc$actual), 2)
tab1$RMSE_bp <- round(tab1$RMSE_bp, 1); tab1$MAE_bp <- round(tab1$MAE_bp, 1)
write.csv(tab1, "output/returns_forecast_table.csv", row.names = FALSE)
cat("PART 1. Wheat futures, 5-day-ahead daily return.", nrow(fc), "forecasts,", format(min(fc$date)), "to", format(max(fc$date)), "\n")
cat("DM statistics are against the zero forecast (positive = better than zero). Buy-and-hold annualised Sharpe:", bh, "\n")
print(tab1, row.names = FALSE)

# ============================================================================================
# PART 2. One-day-ahead forecasts of KOSPI 200 realised variance, expanding window
# ============================================================================================
k <- read_excel("data/Kospi_STU.xls") |> mutate(Date = as.Date(Date))
har_x <- function(x) data.frame(d = dplyr::lag(x, 1), w = dplyr::lag(rollmeanr(x, 5, fill = NA), 1), m = dplyr::lag(rollmeanr(x, 22, fill = NA), 1))
RV <- k$RV5
X_rv  <- har_x(RV);            names(X_rv)  <- c("d", "w", "m")
X_lrv <- har_x(log(RV));       names(X_lrv) <- c("d", "w", "m")
X_lmed <- har_x(log(k$MedRV)); names(X_lmed) <- c("d", "w", "m")

cut <- as.Date("2006-12-31"); test <- which(k$Date > cut)
f <- matrix(NA, length(test), 4, dimnames = list(NULL, c("RW", "HAR_level", "HAR_log", "HAR_log_MedRV")))
for (j in seq_along(test)) {
  t <- test[j]; tr <- 23:(t - 1)                # only past observations
  f[j, "RW"] <- RV[t - 1]
  m1 <- lm(RV[tr] ~ d + w + m, data = X_rv[tr, ]);          f[j, "HAR_level"] <- max(predict(m1, X_rv[t, ]), min(RV[tr]))
  m2 <- lm(log(RV[tr]) ~ d + w + m, data = X_lrv[tr, ]);    f[j, "HAR_log"] <- exp(predict(m2, X_lrv[t, ]) + summary(m2)$sigma^2 / 2)
  m3 <- lm(log(RV[tr]) ~ d + w + m, data = X_lmed[tr, ]);   f[j, "HAR_log_MedRV"] <- exp(predict(m3, X_lmed[t, ]) + summary(m3)$sigma^2 / 2)
}
act <- RV[test]
qlike <- function(a, p) a / p - log(a / p) - 1          # zero when the forecast is perfect
tab2 <- bind_rows(lapply(colnames(f), function(m) {
  out <- data.frame(model = m, QLIKE = round(mean(qlike(act, f[, m])), 4), RMSE_x1e4 = round(1e4 * sqrt(mean((act - f[, m])^2)), 3))
  if (m != "HAR_log") {
    d1 <- dm(qlike(act, f[, "HAR_log"]), qlike(act, f[, m]), lag = 5)
    out$DM_QLIKE_vs_HAR_log <- sprintf("%.2f%s", d1["stat"], stars(d1["p"]))
  } else out$DM_QLIKE_vs_HAR_log <- "benchmark"
  out
}))
write.csv(tab2, "output/volatility_forecast_table.csv", row.names = FALSE)
cat("\nPART 2. KOSPI 200 realised variance, 1-day-ahead.", length(test), "forecasts,", format(min(k$Date[test])), "to", format(max(k$Date[test])), "\n")
cat("All models forecast the same target (RV). DM against HAR_log (positive = better than HAR_log).\n")
print(tab2, row.names = FALSE)

# ---- figures ---------------------------------------------------------------------------------
p1 <- data.frame(date = k$Date[test], Realised = sqrt(252 * act) * 100, HAR_log = sqrt(252 * f[, "HAR_log"]) * 100) |>
  pivot_longer(-date) |>
  ggplot(aes(date, value, colour = name)) + geom_line(linewidth = .4) +
  scale_colour_manual(values = c(HAR_log = "#c0392b", Realised = "grey30")) +
  labs(title = "KOSPI 200: realised volatility and one-day-ahead HAR forecast", y = "annualised volatility (%)", x = NULL, colour = NULL) + theme_minimal()
ggsave("output/har_forecast.png", p1, width = 9, height = 3.8, dpi = 120)

cum <- data.frame(date = fc$date, AR1_5y = cumsum((fc$actual - fc$zero)^2 - (fc$actual - fc$AR1_5y)^2),
                  mean5y = cumsum((fc$actual - fc$zero)^2 - (fc$actual - fc$mean5y)^2)) |> pivot_longer(-date)
p2 <- ggplot(cum, aes(date, value * 1e4, colour = name)) + geom_line() + geom_hline(yintercept = 0) +
  labs(title = "Wheat: cumulative squared-error gain over the zero forecast (above 0 = better)", y = "x 1e-4", x = NULL, colour = NULL) + theme_minimal()
ggsave("output/returns_cumulative_gain.png", p2, width = 9, height = 3.6, dpi = 120)
cat("\nDone. See output/.\n")
