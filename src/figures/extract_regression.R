library(dplyr)
library(tidyr)
library(ggplot2)
library(tidymodels)
library(glmnet)
library(randomForest)
library(corrplot)
library(vip)

PROJ <- "c:/Users/aland/Documents/Knowledge/Data Science & AI/Second Year/Data Science in R/02. Project"
OUT  <- file.path(PROJ, "presentation/figures")

# Nord dark palette
nord_bg    <- "#0F1117"
nord_surf  <- "#161C27"
nord_bord  <- "#252E3F"
nord_tx    <- "#C9D4E8"
nord_tx2   <- "#7E8FA6"
nord_acc   <- "#88C0D0"
nord_yel   <- "#EBCB8B"
nord_grn   <- "#A3BE8C"
nord_red   <- "#BF616A"

theme_nord <- function() {
  theme_minimal() +
  theme(
    plot.background    = element_rect(fill = nord_surf, color = NA),
    panel.background   = element_rect(fill = nord_surf, color = NA),
    panel.grid.major   = element_line(color = nord_bord, linewidth = 0.4),
    panel.grid.minor   = element_blank(),
    strip.background   = element_rect(fill = nord_bord, color = NA),
    strip.text         = element_text(color = nord_tx,  size = 9),
    axis.text          = element_text(color = nord_tx2, size = 8),
    axis.title         = element_text(color = nord_tx,  size = 10),
    plot.title         = element_text(color = nord_tx,  size = 12, face = "bold"),
    plot.subtitle      = element_text(color = nord_tx2, size = 9),
    legend.background  = element_rect(fill = nord_surf, color = NA),
    legend.text        = element_text(color = nord_tx2, size = 9),
    legend.title       = element_text(color = nord_tx,  size = 9)
  )
}

rmse_fn <- function(a, p) sqrt(mean((a - p)^2, na.rm = TRUE))
r2_fn   <- function(a, p) cor(a, p, use = "complete.obs")^2

# ── Load data ─────────────────────────────────────────────────────────────────
cat("Loading subject_features.csv...\n")
subject_features <- read.csv(file.path(PROJ, "data/processed/subject_features.csv"))

reg_data <- subject_features |>
  select(subject_id, ends_with("_mean"), stability_score) |>
  drop_na()

cat("Subjects:", nrow(reg_data), "| Predictors:", ncol(reg_data) - 2, "\n")

# ── 1. Correlation heatmap ────────────────────────────────────────────────────
cat("Plotting correlation heatmap...\n")
numeric_cols <- reg_data |> select(where(is.numeric), -subject_id)
cor_matrix   <- cor(numeric_cols, use = "complete.obs")

png(file.path(OUT, "fig_corr_heatmap.png"),
    width = 900, height = 820, bg = nord_surf)
par(bg = nord_surf, col.main = nord_tx, col.axis = nord_tx2,
    col.lab = nord_tx, fg = nord_bord)
corrplot(cor_matrix,
         method    = "color",
         type      = "upper",
         tl.cex    = 0.55,
         tl.col    = nord_tx2,
         col       = colorRampPalette(c("#5E81AC", nord_surf, "#BF616A"))(200),
         title     = "Feature Correlation Matrix (mean EEG features + stability score)",
         mar       = c(0, 0, 2, 0),
         cl.cex    = 0.7,
         cl.col    = nord_tx2)
dev.off()
cat("Saved fig_corr_heatmap.png\n")

# ── Train/test split ──────────────────────────────────────────────────────────
set.seed(42)
reg_data_split  <- initial_split(reg_data, prop = 0.7)
train_data      <- training(reg_data_split)
test_data       <- testing(reg_data_split)
train_model     <- train_data |> select(-subject_id)
test_model      <- test_data  |> select(-subject_id)

x_train_raw <- train_data |> select(-stability_score, -subject_id)
y_train_raw <- train_data |> pull(stability_score)
x_test_raw  <- test_data  |> select(-stability_score, -subject_id)
y_test_raw  <- test_data  |> pull(stability_score)

cat("Train:", nrow(train_data), "| Test:", nrow(test_data), "\n")

# ── OLS raw (overfit demo) ────────────────────────────────────────────────────
cat("Fitting OLS (raw)...\n")
model_raw        <- lm(y_train_raw ~ ., data = x_train_raw)
pred_raw_train   <- predict(model_raw, newdata = x_train_raw)
pred_raw_test    <- predict(model_raw, newdata = x_test_raw)

metrics_raw <- data.frame(
  Model   = "OLS (Raw)",
  Dataset = c("Train", "Test"),
  RMSE    = round(c(rmse_fn(y_train_raw, pred_raw_train), rmse_fn(y_test_raw, pred_raw_test)), 4),
  R2      = round(c(r2_fn(y_train_raw, pred_raw_train),   r2_fn(y_test_raw, pred_raw_test)), 4)
)

# ── OLS normalized ────────────────────────────────────────────────────────────
cat("Fitting OLS (normalized)...\n")
train_means <- x_train_raw |> summarise(across(everything(), ~mean(.x, na.rm = TRUE)))
train_sds   <- x_train_raw |> summarise(across(everything(), ~sd(.x,   na.rm = TRUE)))
normalize   <- function(d, m, s) d |> mutate(across(everything(), ~(.x - m[[cur_column()]]) / s[[cur_column()]]))

x_train_norm <- normalize(x_train_raw, train_means, train_sds)
x_test_norm  <- normalize(x_test_raw,  train_means, train_sds)

zero_var <- names(x_train_norm)[sapply(x_train_norm, function(x) sd(x, na.rm = TRUE) == 0)]
if (length(zero_var) > 0) {
  x_train_norm <- x_train_norm |> select(-all_of(zero_var))
  x_test_norm  <- x_test_norm  |> select(-all_of(zero_var))
}

model_norm       <- lm(y_train_raw ~ ., data = x_train_norm)
pred_norm_train  <- predict(model_norm, newdata = x_train_norm)
pred_norm_test   <- predict(model_norm, newdata = x_test_norm)

metrics_norm <- data.frame(
  Model   = "OLS (Normalized)",
  Dataset = c("Train", "Test"),
  RMSE    = round(c(rmse_fn(y_train_raw, pred_norm_train), rmse_fn(y_test_raw, pred_norm_test)), 4),
  R2      = round(c(r2_fn(y_train_raw, pred_norm_train),   r2_fn(y_test_raw, pred_norm_test)), 4)
)

# ── Lasso ─────────────────────────────────────────────────────────────────────
cat("Fitting Lasso (10-fold CV)...\n")
set.seed(42)
cv_folds    <- vfold_cv(train_model, v = 10)
lasso_recipe <- recipe(stability_score ~ ., data = train_model) |> step_normalize(all_predictors())
lasso_spec   <- linear_reg(penalty = tune(), mixture = 1) |> set_engine("glmnet")
lasso_wf     <- workflow() |> add_recipe(lasso_recipe) |> add_model(lasso_spec)
lambda_grid  <- grid_regular(penalty(), levels = 50)

lasso_tune  <- tune_grid(lasso_wf, resamples = cv_folds, grid = lambda_grid,
                         metrics = metric_set(rmse, rsq))
best_lambda <- select_best(lasso_tune, metric = "rmse")
cat("Best lambda:", round(best_lambda$penalty, 6), "\n")

lasso_final      <- finalize_workflow(lasso_wf, best_lambda)
lasso_fit        <- fit(lasso_final, data = train_model)
pred_lasso_train <- predict(lasso_fit, new_data = train_model)$.pred
pred_lasso_test  <- predict(lasso_fit, new_data = test_model)$.pred

metrics_lasso <- data.frame(
  Model   = "Lasso",
  Dataset = c("Train", "Test"),
  RMSE    = round(c(rmse_fn(y_train_raw, pred_lasso_train), rmse_fn(y_test_raw, pred_lasso_test)), 4),
  R2      = round(c(r2_fn(y_train_raw, pred_lasso_train),   r2_fn(y_test_raw, pred_lasso_test)), 4)
)

lasso_coef_df <- lasso_fit |>
  extract_fit_parsnip() |>
  tidy() |>
  filter(term != "(Intercept)", estimate != 0) |>
  arrange(desc(abs(estimate)))
cat("Lasso selected:", nrow(lasso_coef_df), "features\n")

# ── Random Forest ─────────────────────────────────────────────────────────────
cat("Fitting Random Forest...\n")
set.seed(42)
rf_spec  <- rand_forest(mode = "regression") |> set_engine("randomForest")
rf_wf    <- workflow() |>
  add_recipe(recipe(stability_score ~ ., data = train_model)) |>
  add_model(rf_spec)
rf_fit   <- fit(rf_wf, data = train_model)

pred_rf_train <- predict(rf_fit, new_data = train_model)$.pred
pred_rf_test  <- predict(rf_fit, new_data = test_model)$.pred

metrics_rf <- data.frame(
  Model   = "Random Forest",
  Dataset = c("Train", "Test"),
  RMSE    = round(c(rmse_fn(y_train_raw, pred_rf_train), rmse_fn(y_test_raw, pred_rf_test)), 4),
  R2      = round(c(r2_fn(y_train_raw, pred_rf_train),   r2_fn(y_test_raw, pred_rf_test)), 4)
)

# ── Print & save metrics ──────────────────────────────────────────────────────
all_metrics <- rbind(metrics_raw, metrics_norm, metrics_lasso, metrics_rf)
cat("\n=== METRICS ===\n")
print(all_metrics)
write.csv(all_metrics, file.path(OUT, "metrics_regression.csv"), row.names = FALSE)

# ── 2. Predicted vs Actual ────────────────────────────────────────────────────
cat("Plotting predicted vs actual...\n")
pred_df <- bind_rows(
  data.frame(actual = y_test_raw, predicted = pred_lasso_test, Model = "Lasso"),
  data.frame(actual = y_test_raw, predicted = pred_rf_test,    Model = "Random Forest")
)

p_pred <- ggplot(pred_df, aes(x = actual, y = predicted, color = Model)) +
  geom_point(alpha = 0.8, size = 2) +
  geom_abline(slope = 1, intercept = 0, color = nord_tx2, linetype = "dashed") +
  facet_wrap(~Model) +
  scale_color_manual(values = c("Lasso" = nord_acc, "Random Forest" = nord_yel)) +
  theme_nord() +
  theme(legend.position = "none") +
  labs(title = "Predicted vs. Actual Stability Score (test set)",
       x = "Actual", y = "Predicted")

ggsave(file.path(OUT, "fig_pred_vs_actual.png"), p_pred,
       width = 8, height = 4, dpi = 150, bg = nord_surf)
cat("Saved fig_pred_vs_actual.png\n")

# ── 3. Residuals vs Fitted (Lasso) ────────────────────────────────────────────
cat("Plotting residuals...\n")
resid_df <- data.frame(
  fitted    = pred_lasso_train,
  residuals = y_train_raw - pred_lasso_train
)

p_resid <- ggplot(resid_df, aes(x = fitted, y = residuals)) +
  geom_point(alpha = 0.7, color = nord_acc, size = 2) +
  geom_hline(yintercept = 0, color = nord_red, linetype = "dashed") +
  geom_smooth(method = "loess", se = FALSE, color = nord_yel, linewidth = 0.8) +
  theme_nord() +
  labs(title = "Residuals vs. Fitted (Lasso — train set)",
       x = "Fitted Values", y = "Residuals")

ggsave(file.path(OUT, "fig_residuals.png"), p_resid,
       width = 6, height = 4, dpi = 150, bg = nord_surf)
cat("Saved fig_residuals.png\n")

# ── 4. Q-Q plot (Lasso) ───────────────────────────────────────────────────────
p_qq <- ggplot(resid_df, aes(sample = residuals)) +
  stat_qq(color = nord_acc, alpha = 0.8, size = 2) +
  stat_qq_line(color = nord_red, linetype = "dashed") +
  theme_nord() +
  labs(title = "Q-Q Plot of Residuals (Lasso — train set)",
       x = "Theoretical Quantiles", y = "Sample Quantiles")

ggsave(file.path(OUT, "fig_qqplot.png"), p_qq,
       width = 5, height = 4, dpi = 150, bg = nord_surf)
cat("Saved fig_qqplot.png\n")

# ── 5. Lasso coefficients ─────────────────────────────────────────────────────
cat("Plotting Lasso coefficients...\n")
p_coef <- ggplot(lasso_coef_df,
                 aes(x = reorder(term, estimate), y = estimate,
                     fill = estimate > 0)) +
  geom_col() +
  scale_fill_manual(values = c("TRUE" = nord_acc, "FALSE" = nord_red),
                    labels = c("TRUE" = "positive", "FALSE" = "negative")) +
  coord_flip() +
  theme_nord() +
  theme(legend.position = "bottom") +
  labs(title = paste("Lasso selected features (n =", nrow(lasso_coef_df), ")"),
       x = NULL, y = "Coefficient", fill = NULL)

h <- max(3.5, nrow(lasso_coef_df) * 0.22)
ggsave(file.path(OUT, "fig_lasso_coefs.png"), p_coef,
       width = 7, height = h, dpi = 150, bg = nord_surf)
cat("Saved fig_lasso_coefs.png\n")

# ── 6. RF feature importances ─────────────────────────────────────────────────
cat("Plotting RF importances...\n")
p_rf <- rf_fit |>
  extract_fit_engine() |>
  vip(num_features = 20,
      aesthetics = list(fill = nord_yel, color = "transparent")) +
  theme_nord() +
  labs(title = "Random Forest — top 20 features (node impurity)",
       x = NULL, y = "Importance")

ggsave(file.path(OUT, "fig_rf_importance.png"), p_rf,
       width = 7, height = 5, dpi = 150, bg = nord_surf)
cat("Saved fig_rf_importance.png\n")

# ── 7. Feature agreement Lasso vs RF ─────────────────────────────────────────
rf_vi <- rf_fit |> extract_fit_engine() |> vip::vi() |>
  transmute(Feature = Variable, RF_Rank = row_number())

n_lasso <- nrow(lasso_coef_df)
comparison_df <- lasso_coef_df |>
  transmute(Feature = term, Lasso_Coef = round(estimate, 4)) |>
  left_join(rf_vi, by = "Feature") |>
  mutate(RF_Agrees = ifelse(!is.na(RF_Rank) & RF_Rank <= n_lasso, "YES", "NO")) |>
  arrange(desc(abs(Lasso_Coef)))

cat(sprintf("\nLasso kept %d features | RF agrees on %d (top-%d overlap)\n",
            n_lasso,
            sum(comparison_df$RF_Agrees == "YES"),
            n_lasso))

write.csv(comparison_df, file.path(OUT, "feature_agreement.csv"), row.names = FALSE)
cat("Saved feature_agreement.csv\n")
cat("\nDone.\n")
