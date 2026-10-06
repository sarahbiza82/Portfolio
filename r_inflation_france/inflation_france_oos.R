# French inflation forecasting: out-of-sample evaluation
#
# Direct h-step forecasts (h = 1, 3, 6, 12) built only from information available at the forecast origin,
# expanding window from 2012-01 (133 to 144 forecast origins per horizon), Diebold-Mariano tests with the
# Harvey-Leybourne-Newbold correction. ARX_oracle uses the realised predictors at the target date: it is
# not a feasible forecast, it is a reference showing how much known future predictors would help.
#
# Run:  Rscript inflation_france_oos.R   (from this folder)

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(ggplot2); library(forecast); library(glmnet); library(tidyr)
})
set.seed(1)

# ---- data --------------------------------------------------------------------------------
raw <- read_csv("france_inflation_data.csv", col_types = cols(.default = "c"))   # read as text, then convert
names(raw) <- trimws(names(raw))
raw <- raw |> mutate(across(-date, ~ as.numeric(trimws(.x))))
stopifnot(!anyNA(raw))                                    # stop loudly if any value fails to parse
d <- raw |>
  mutate(date = as.Date(date),
         d_oil    = 100 * (log(prix_petrole) - lag(log(prix_petrole), 3)),
         d_energy = 100 * (log(indice_prix)  - lag(log(indice_prix), 3)),
         d_usd    = 100 * (log(usd_eur)      - lag(log(usd_eur), 3)),
         d_bond   = obligations_10y - lag(obligations_10y, 3),
         infl_l1  = lag(inflation, 1)) |>
  arrange(date)
X_cols <- c("inflation", "infl_l1", "d_oil", "d_energy", "d_usd", "d_bond", "obligations_10y")

H <- c(1, 3, 6, 12)
START <- as.Date("2012-01-31")
n <- nrow(d)

# ---- forecasts ---------------------------------------------------------------------------
one_origin <- function(i, h) {
  # information set: rows 1..i. Training pairs (x_s, y_{s+h}) need s + h <= i.
  s_max <- i - h
  tr <- d[4:s_max, ]                                  # first 3 rows lose lags
  y  <- d$inflation[(4:s_max) + h]
  xn <- d[i, ]
  f <- c(naive = d$inflation[i])
  ar <- lm(y ~ inflation + infl_l1, data = cbind(tr, y = y))
  f["AR"] <- predict(ar, xn)
  arx <- lm(y ~ ., data = cbind(tr[, X_cols], y = y))
  f["ARX"] <- predict(arx, xn[, X_cols])
  Xm <- as.matrix(tr[, X_cols]); sc <- list(m = colMeans(Xm), s = apply(Xm, 2, sd))
  Xs <- scale(Xm, sc$m, sc$s)
  cv <- cv.glmnet(Xs, y, alpha = 1, nfolds = 5)
  f["Lasso"] <- as.numeric(predict(cv, scale(as.matrix(xn[, X_cols]), sc$m, sc$s), s = "lambda.min"))
  f["Combo"] <- mean(c(f["AR"], f["ARX"], f["Lasso"]))
  # NOT a feasible forecast: uses the realised predictor values at the target date (a conditional forecast).
  # Shown only as a reference, to measure how much knowing the future predictors would help.
  x_t <- c("d_oil", "d_energy", "d_usd", "d_bond")
  trg <- d[(4:s_max) + h, x_t]; names(trg) <- paste0(x_t, "_target")
  orc <- lm(y ~ ., data = cbind(y = y, inflation = tr$inflation, trg))
  nt <- d[i + h, x_t]; names(nt) <- paste0(x_t, "_target")
  f["ARX_oracle"] <- predict(orc, cbind(inflation = xn$inflation, nt))
  f
}

res <- list()
for (h in H) {
  origins <- which(d$date >= START & seq_len(n) + h <= n)
  fc <- t(sapply(origins, one_origin, h = h))
  res[[as.character(h)]] <- data.frame(date = d$date[origins], h = h, actual = d$inflation[origins + h], fc)
}
all_fc <- bind_rows(res)
write_csv(all_fc, "forecasts.csv")

# ---- evaluation --------------------------------------------------------------------------
models <- c("naive", "ARX", "Lasso", "Combo", "ARX_oracle")
eval_period <- function(df, label) {
  bind_rows(lapply(H, function(h) {
    z <- filter(df, h == !!h)
    e_ar <- z$AR - z$actual
    out <- data.frame(period = label, h = h, n = nrow(z), AR_RMSE = round(sqrt(mean(e_ar^2)), 2))
    for (m in models) {
      e <- z[[m]] - z$actual
      p <- tryCatch(dm.test(e_ar, e, alternative = "greater", h = h, power = 2, varestimator = "bartlett")$p.value,
                    error = function(err) NA_real_)
      stars <- ifelse(is.na(p), "", ifelse(p < .01, "***", ifelse(p < .05, "**", ifelse(p < .10, "*", ""))))
      out[[m]] <- paste0(sprintf("%.2f", sqrt(mean(e^2)) / sqrt(mean(e_ar^2))), stars)
    }
    out
  }))
}
tab <- bind_rows(eval_period(all_fc, "2012-2024 (all)"),
                 eval_period(filter(all_fc, date + 0 < as.Date("2020-01-01")), "2012-2019"),
                 eval_period(filter(all_fc, date >= as.Date("2021-01-01")), "2021 onward (energy shock)"))
write_csv(tab, "results_table.csv")
print(as.data.frame(tab), row.names = FALSE)

# ---- figure ------------------------------------------------------------------------------
p <- all_fc |> filter(h == 12) |>
  select(date, actual, AR, ARX, Lasso, naive) |>
  pivot_longer(-c(date, actual), names_to = "model", values_to = "forecast") |>
  mutate(target_date = date) |>
  ggplot(aes(date)) +
  geom_line(aes(y = actual), colour = "black", linewidth = 1) +
  geom_line(aes(y = forecast, colour = model), alpha = .8) +
  labs(title = "French inflation: 12-month-ahead forecasts vs realised value 12 months later",
       subtitle = "Black line: realised inflation 12 months after the forecast origin",
       x = "forecast origin", y = "inflation (%)", colour = NULL) +
  theme_minimal()
ggsave("forecasts_h12.png", p, width = 9, height = 4, dpi = 120)
cat("\nDone. Files: forecasts.csv, results_table.csv, forecasts_h12.png\n")
