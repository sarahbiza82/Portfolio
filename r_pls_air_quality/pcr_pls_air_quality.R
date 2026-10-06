# PCR vs PLS vs OLS for PM2.5 in Beijing, validated chronologically
#
# Hourly data from the Aotizhongxin station, 2013-03 to 2017-02. Chronological split (train 2013-2015, test
# 2016-2017) with blocked cross-validation; a random split is run alongside to measure its optimism. Results are
# shown with and without PM10 (PM2.5 is part of PM10), against OLS and a persistence forecast.
#
# Run from this folder:  Rscript pcr_pls_air_quality.R

suppressPackageStartupMessages({ library(readr); library(dplyr); library(tidyr); library(ggplot2); library(pls) })
set.seed(1)
dir.create("output", showWarnings = FALSE)

raw <- read_csv("beijing_aotizhongxin.csv", show_col_types = FALSE)
d <- raw |>
  mutate(time = as.POSIXct(sprintf("%d-%02d-%02d %02d:00", year, month, day, hour), tz = "UTC")) |>
  select(time, PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, WSPM) |>
  arrange(time) |>
  mutate(lPM25_prev = log(dplyr::lag(PM2.5))) |>
  na.omit() |>
  mutate(lPM25 = log(PM2.5), lPM10 = log(PM10), lSO2 = log(SO2), lNO2 = log(NO2), lCO = log(CO), lO3 = log(O3))
cat(nrow(d), "complete hourly observations,", format(min(d$time)), "to", format(max(d$time)), "\n")

BASE <- c("lSO2", "lNO2", "lCO", "lO3", "TEMP", "PRES", "DEWP", "WSPM")
SETS <- list(`without PM10` = BASE, `with PM10` = c("lPM10", BASE))
cat("Variance inflation factors (with PM10):\n")
print(round(sapply(SETS[[2]], function(v) 1 / (1 - summary(lm(reformulate(setdiff(SETS[[2]], v), v), d))$r.squared)), 2))

rmse <- function(a, p) sqrt(mean((a - p)^2)); r2 <- function(a, p) 1 - sum((a - p)^2) / sum((a - mean(a))^2)

one_sigma <- function(fit) selectNcomp(fit, method = "onesigma", plot = FALSE)

evaluate <- function(train, test, vars, label_split, label_set) {
  f <- reformulate(vars, "lPM25")
  ols <- lm(f, train)
  # blocked cross-validation: 10 consecutive segments, so validation folds are not interleaved with training hours
  pc <- pcr(f, data = train, scale = TRUE, validation = "CV", segments = 10, segment.type = "consecutive")
  pl <- plsr(f, data = train, scale = TRUE, validation = "CV", segments = 10, segment.type = "consecutive")
  k_pc <- max(1, one_sigma(pc)); k_pl <- max(1, one_sigma(pl))
  p <- list(OLS = predict(ols, test), PCR = drop(predict(pc, test, ncomp = k_pc)), PLS = drop(predict(pl, test, ncomp = k_pl)),
            Persistence = test$lPM25_prev)
  data.frame(split = label_split, predictors = label_set, model = names(p), components = c(length(vars), k_pc, k_pl, NA),
             test_RMSE = round(sapply(p, rmse, a = test$lPM25), 4), test_R2 = round(sapply(p, r2, a = test$lPM25), 4), row.names = NULL)
}

chrono_train <- d$time < as.POSIXct("2016-01-01", tz = "UTC")
rand_train <- sample(nrow(d)) <= sum(chrono_train)                  # same training size, hours drawn at random
res <- bind_rows(lapply(names(SETS), function(s) bind_rows(
  evaluate(d[chrono_train, ], d[!chrono_train, ], SETS[[s]], "chronological (train 2013-2015, test 2016-2017)", s),
  evaluate(d[rand_train, ], d[!rand_train, ], SETS[[s]], "random hours", s))))
write_csv(res, "output/results.csv")
cat("\nTrain:", sum(chrono_train), "hours | test:", sum(!chrono_train), "hours\n\n"); print(res, row.names = FALSE)

# test error by number of components, chronological split, without PM10
tr <- d[chrono_train, ]; te <- d[!chrono_train, ]; f <- reformulate(BASE, "lPM25")
pc <- pcr(f, data = tr, scale = TRUE); pl <- plsr(f, data = tr, scale = TRUE)
curve <- data.frame(components = 1:length(BASE),
                    PCR = sapply(1:length(BASE), function(k) rmse(te$lPM25, drop(predict(pc, te, ncomp = k)))),
                    PLS = sapply(1:length(BASE), function(k) rmse(te$lPM25, drop(predict(pl, te, ncomp = k)))))
write_csv(curve, "output/test_error_by_components.csv")
g <- curve |> pivot_longer(-components, names_to = "model", values_to = "rmse") |>
  ggplot(aes(components, rmse, colour = model)) + geom_line() + geom_point() +
  geom_hline(yintercept = rmse(te$lPM25, predict(lm(f, tr), te)), linetype = 2) +
  annotate("text", x = 1.2, y = rmse(te$lPM25, predict(lm(f, tr), te)), label = "OLS", vjust = -0.6, size = 3) +
  labs(title = "Out-of-sample error by number of components (2016-2017, predictors without PM10)", y = "test RMSE of log PM2.5", colour = NULL) + theme_minimal()
ggsave("output/test_error_by_components.png", g, width = 8, height = 3.8, dpi = 120)
cat("\nDone. See output/.\n")
