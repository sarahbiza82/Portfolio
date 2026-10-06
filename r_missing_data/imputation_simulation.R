# Which imputation method works on a small financial cross-section? A repeated-deletion experiment.
#
# CAC 40 firms, 40 rows, 7 quantitative variables. 15% of the cells are deleted at random, 300 times, under two
# mechanisms (MCAR and MAR); five imputation methods are compared on the imputed cells only, in standard-deviation
# units. Multiple imputation is then used for inference, pooling estimates with Rubin's rules.
#
# Run from this folder:  Rscript imputation_simulation.R     (about 20 minutes)

suppressPackageStartupMessages({ library(readr); library(dplyr); library(tidyr); library(ggplot2); library(mice); library(missMDA) })
set.seed(2026)
dir.create("output", showWarnings = FALSE)

full <- read_csv("cac40_complete.csv", show_col_types = FALSE)
Q <- c("RENDEMENT", "CA", "VOLUME", "PER", "BETA", "ESG", "DETTE")
X <- as.data.frame(full[, Q]); n <- nrow(X); p <- ncol(X)
sds <- sapply(X, sd)
cat(n, "firms,", p, "quantitative variables. Skewness:\n"); print(round(sapply(X, function(v) mean((v - mean(v))^3) / sd(v)^3), 2))

# ---- missingness mechanisms -------------------------------------------------------------------
make_na <- function(X, mech, rate = 0.15) {
  M <- matrix(FALSE, n, p, dimnames = list(NULL, Q))
  if (mech == "MCAR") {
    M[sample(n * p, floor(rate * n * p))] <- TRUE
  } else {                                   # MAR: CA is always observed; large firms (by CA) are more often missing elsewhere
    w <- rank(X$CA) / n                      # 0..1
    for (j in setdiff(Q, "CA")) M[, j] <- runif(n) < rate * (p / (p - 1)) * 2 * w
  }
  M[rowSums(M) == p, 1] <- FALSE             # never delete a whole row
  Xna <- X; Xna[M] <- NA
  list(Xna = Xna, M = M)
}

# ---- imputation methods -----------------------------------------------------------------------
imp_mean   <- function(Z) { for (j in Q) Z[is.na(Z[[j]]), j] <- mean(Z[[j]], na.rm = TRUE); Z }
imp_median <- function(Z) { for (j in Q) Z[is.na(Z[[j]]), j] <- median(Z[[j]], na.rm = TRUE); Z }
imp_pca    <- function(Z) as.data.frame(imputePCA(Z, ncp = 2, scale = TRUE)$completeObs)
imp_mice   <- function(Z, method) {          # point imputation = average of the m completed datasets
  mi <- mice(Z, m = 5, method = method, maxit = 5, printFlag = FALSE)
  as.data.frame(Reduce(`+`, lapply(1:5, function(i) as.matrix(complete(mi, i)))) / 5)
}
METHODS <- list(Mean = imp_mean, Median = imp_median, PCA = imp_pca,
                `MICE (pmm)` = function(Z) imp_mice(Z, "pmm"), `MICE (norm)` = function(Z) imp_mice(Z, "norm"))

nrmse <- function(imp, M) {                  # error on imputed cells only, each variable scaled by its standard deviation
  E <- sweep(as.matrix(imp)[, Q] - as.matrix(X), 2, sds, "/")
  sqrt(mean(E[M]^2))
}

# ---- experiment -------------------------------------------------------------------------------
R <- 300
res <- list()
for (mech in c("MCAR", "MAR")) {
  for (r in 1:R) {
    d <- make_na(X, mech)
    for (m in names(METHODS)) {
      v <- tryCatch(nrmse(METHODS[[m]](d$Xna), d$M), error = function(e) NA_real_)
      res[[length(res) + 1]] <- data.frame(mechanism = mech, rep = r, method = m, nrmse = v)
    }
  }
  cat(mech, "done\n")
}
res <- bind_rows(res)
write_csv(res, "output/simulation_draws.csv")

tab <- res |> group_by(mechanism, method) |>
  summarise(mean = mean(nrmse, na.rm = TRUE), sd = sd(nrmse, na.rm = TRUE),
            q05 = quantile(nrmse, .05, na.rm = TRUE), q95 = quantile(nrmse, .95, na.rm = TRUE), failed = sum(is.na(nrmse)), .groups = "drop") |>
  arrange(mechanism, mean) |> mutate(across(mean:q95, ~ round(.x, 3)))
write_csv(tab, "output/simulation_table.csv")
cat("\nError on imputed cells, in standard deviations (lower is better), over", R, "random deletions\n"); print(as.data.frame(tab), row.names = FALSE)

# how often does each method win? and how often does PCA beat MICE?
wide <- res |> pivot_wider(names_from = method, values_from = nrmse)
win <- res |> group_by(mechanism, rep) |> slice_min(nrmse, n = 1, with_ties = FALSE) |> ungroup() |> count(mechanism, method) |>
  group_by(mechanism) |> mutate(share = round(n / sum(n), 3)) |> select(-n) |> arrange(mechanism, desc(share))
cat("\nShare of replications in which each method has the lowest error\n"); print(as.data.frame(win), row.names = FALSE)
pm <- wide |> group_by(mechanism) |> summarise(`PCA better than MICE (pmm)` = round(mean(PCA < `MICE (pmm)`, na.rm = TRUE), 3),
                                               `Mean better than MICE (pmm)` = round(mean(Mean < `MICE (pmm)`, na.rm = TRUE), 3),
                                               `Mean better than PCA` = round(mean(Mean < PCA, na.rm = TRUE), 3))
cat("\nPairwise comparisons (share of replications)\n"); print(as.data.frame(pm), row.names = FALSE)
write_csv(win, "output/win_shares.csv"); write_csv(pm, "output/pairwise.csv")

g <- ggplot(res, aes(reorder(method, nrmse, median, na.rm = TRUE), nrmse)) + geom_boxplot(outlier.size = .4, fill = "#dbe7f3") +
  facet_wrap(~ mechanism) + coord_flip() +
  labs(title = "Imputation error over 300 random deletions (15% of cells)", x = NULL, y = "RMSE on imputed cells, in standard deviations") + theme_minimal()
ggsave("output/imputation_error.png", g, width = 9, height = 3.8, dpi = 120)

# ---- multiple imputation used as intended: pooling with Rubin's rules -----------------------------
d <- make_na(X, "MCAR")
f <- RENDEMENT ~ ESG + BETA
complete_fit <- summary(lm(f, X))$coefficients
listwise_fit <- summary(lm(f, d$Xna))$coefficients
mi <- mice(d$Xna, m = 20, method = "pmm", maxit = 10, printFlag = FALSE)
pooled <- summary(pool(with(mi, lm(RENDEMENT ~ ESG + BETA))))
cmp <- data.frame(term = rownames(complete_fit),
                  complete_est = round(complete_fit[, 1], 3), complete_se = round(complete_fit[, 2], 3),
                  listwise_est = round(listwise_fit[, 1], 3), listwise_se = round(listwise_fit[, 2], 3),
                  pooled_est = round(pooled$estimate, 3), pooled_se = round(pooled$std.error, 3))
write_csv(cmp, "output/rubin_pooling.csv")
cat("\nRegression RENDEMENT ~ ESG + BETA: complete data, listwise deletion (", nobs(lm(f, d$Xna)), "rows left), MICE pooled over 20 imputations\n")
print(cmp, row.names = FALSE)
cat("\nDone. See output/.\n")
