# Regenerates all figures with a clean white/light theme that works
# equally well in dark and light mode presentations.

library(dplyr)
library(tidyr)
library(ggplot2)
library(tidymodels)
library(glmnet)
library(randomForest)
library(corrplot)
library(vip)
library(edfReader)
library(signal)
library(tidyverse)
library(discrim)

PROJ <- "c:/Users/aland/Documents/Knowledge/Data Science & AI/Second Year/Data Science in R/02. Project"
OUT  <- file.path(PROJ, "presentation/figures")

# ── Neutral light theme ───────────────────────────────────────────────────────
theme_clean <- function(base = 9) {
  theme_minimal(base_size = base) +
  theme(
    plot.background   = element_rect(fill = "white", color = NA),
    panel.background  = element_rect(fill = "white", color = NA),
    panel.grid.major  = element_line(color = "#E5E7EB", linewidth = 0.4),
    panel.grid.minor  = element_blank(),
    strip.background  = element_rect(fill = "#F3F4F6", color = NA),
    strip.text        = element_text(color = "#374151", size = base - 1, face = "bold"),
    axis.text         = element_text(color = "#6B7280", size = base - 1),
    axis.title        = element_text(color = "#374151", size = base),
    plot.title        = element_text(color = "#111827", size = base + 2, face = "bold"),
    plot.subtitle     = element_text(color = "#6B7280", size = base - 1),
    legend.background = element_rect(fill = "white", color = NA),
    legend.text       = element_text(color = "#6B7280", size = base - 1),
    legend.title      = element_text(color = "#374151", size = base - 1)
  )
}

rmse_fn <- function(a, p) sqrt(mean((a - p)^2, na.rm = TRUE))
r2_fn   <- function(a, p) cor(a, p, use = "complete.obs")^2

# ─────────────────────────────────────────────────────────────────────────────
# 1. Band filter figure (from exploration)
# ─────────────────────────────────────────────────────────────────────────────
cat("--- Band filter figure ---\n")
header  <- readEdfHeader(file.path(PROJ, "data/raw/S001/S001R04.edf"))
sig_c3  <- readEdfSignals(header, signals = "C3..")
annot   <- readEdfSignals(header, signals = "EDF Annotations")
events  <- as.data.frame(annot$annotations)
evt     <- events[events$annotation %in% c("T1", "T2"), ][1, ]
sr      <- 160
s_idx   <- round(evt$onset * sr) + 1
e_idx   <- round(evt$end   * sr)
c3_raw  <- sig_c3$signal[s_idx:e_idx]
t_vec   <- seq(0, length.out = length(c3_raw), by = 1 / sr)
nyq     <- sr / 2
c3_mu   <- filtfilt(butter(4, c(8,  13) / nyq, type = "pass"), c3_raw)
c3_beta <- filtfilt(butter(4, c(13, 30) / nyq, type = "pass"), c3_raw)

df_band <- bind_rows(
  data.frame(time = t_vec, value = c3_raw,  band = "broadband"),
  data.frame(time = t_vec, value = c3_mu,   band = "mu  (8-13 Hz)"),
  data.frame(time = t_vec, value = c3_beta, band = "beta  (13-30 Hz)")
)
df_band$band <- factor(df_band$band, levels = c("broadband", "mu  (8-13 Hz)", "beta  (13-30 Hz)"))

p_band <- ggplot(df_band, aes(x = time, y = value)) +
  geom_line(color = "#2563EB", linewidth = 0.45) +
  facet_wrap(~band, ncol = 1, scales = "free_y") +
  theme_clean() +
  labs(title    = "C3 - trial 1 (left imagery): broadband vs filtered bands",
       subtitle = "Band power = log(var()) of the filtered signal",
       x = "Time (s)", y = "Amplitude (uV)")

ggsave(file.path(OUT, "fig_bandfilter.png"), p_band,
       width = 7, height = 5, dpi = 150, bg = "white")
cat("Saved fig_bandfilter.png\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Classification figures
# ─────────────────────────────────────────────────────────────────────────────
cat("\n--- Classification figures ---\n")
filter <- dplyr::filter
features <- read_csv(file.path(PROJ, "data/processed/trial_features.csv"),
                     show_col_types = FALSE) |>
  mutate(label = as.factor(label)) |>
  mutate(
    asym_mu_C   = C3_mu_power   - C4_mu_power,
    asym_beta_C = C3_beta_power - C4_beta_power,
    asym_mu_F   = FC3_mu_power  - FC4_mu_power,
    asym_beta_F = FC3_beta_power - FC4_beta_power,
    asym_mu_P   = CP3_mu_power  - CP4_mu_power,
    asym_beta_P = CP3_beta_power - CP4_beta_power
  )

feature_cols <- features |> select(-subject_id, -trial_id, -run, -label) |> colnames()
features <- features |>
  group_by(subject_id) |>
  mutate(across(all_of(feature_cols),
                ~ (. - mean(., na.rm = TRUE)) / (sd(., na.rm = TRUE) + 1e-8))) |>
  ungroup()

# C3 density distributions
p_density <- features |>
  select(starts_with("C3_"), label) |>
  select(-contains("mu_power"), -contains("beta_power"), -contains("erd")) |>
  pivot_longer(starts_with("C3_"), names_to = "feature", values_to = "value") |>
  mutate(feature = gsub("C3_", "", feature)) |>
  ggplot(aes(x = value, fill = label, color = label)) +
  geom_density(alpha = 0.35, linewidth = 0.6) +
  facet_wrap(~feature, scales = "free", ncol = 4) +
  scale_fill_manual(values  = c("left" = "#2563EB", "right" = "#D97706")) +
  scale_color_manual(values = c("left" = "#2563EB", "right" = "#D97706")) +
  theme_clean(8) +
  labs(title    = "C3 feature distributions by class (left vs right)",
       subtitle = "All features overlap heavily after per-subject normalization",
       x = NULL, y = "Density", fill = NULL, color = NULL)

ggsave(file.path(OUT, "fig_c3_density.png"), p_density,
       width = 11, height = 7, dpi = 150, bg = "white")
cat("Saved fig_c3_density.png\n")

# Models
set.seed(42)
data_split  <- initial_split(features, prop = 0.8, strata = subject_id)
train_data  <- training(data_split)
test_data   <- testing(data_split)

base_recipe <- recipe(label ~ ., data = train_data) |>
  update_role(subject_id, trial_id, run, new_role = "ID") |>
  step_impute_mean(all_numeric_predictors()) |>
  step_zv(all_numeric_predictors()) |>
  step_corr(all_numeric_predictors(), threshold = 0.9)
eval_metrics <- metric_set(accuracy, sensitivity, specificity, roc_auc)

fit_eval <- function(spec, nm) {
  cat("  Fitting", nm, "...\n")
  wf <- workflow() |> add_recipe(base_recipe) |> add_model(spec)
  augment(fit(wf, train_data), new_data = test_data) |> mutate(model = nm)
}

lr_p  <- fit_eval(logistic_reg() |> set_engine("glm") |> set_mode("classification"), "Logistic Regression")
nb_p  <- fit_eval(naive_Bayes()  |> set_engine("naivebayes") |> set_mode("classification"), "Naive Bayes")
knn_p <- fit_eval(nearest_neighbor(neighbors = 10) |> set_engine("kknn") |> set_mode("classification"), "KNN")
rf_p  <- fit_eval(rand_forest(trees = 500) |> set_engine("ranger") |> set_mode("classification"), "Random Forest")

model_colors <- c(
  "Logistic Regression" = "#2563EB",
  "Naive Bayes"         = "#D97706",
  "KNN"                 = "#DC2626",
  "Random Forest"       = "#059669"
)

roc_df <- bind_rows(lr_p, nb_p, knn_p, rf_p) |>
  group_by(model) |>
  roc_curve(truth = label, .pred_left)

p_roc <- ggplot(roc_df, aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_line(linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0, color = "#9CA3AF", linetype = "dashed") +
  scale_color_manual(values = model_colors) +
  theme_clean() +
  labs(title = "ROC curves - all models (test set)",
       x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)", color = NULL)

ggsave(file.path(OUT, "fig_roc_curves.png"), p_roc,
       width = 6, height = 5, dpi = 150, bg = "white")
cat("Saved fig_roc_curves.png\n")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Regression figures
# ─────────────────────────────────────────────────────────────────────────────
cat("\n--- Regression figures ---\n")
subject_features <- read.csv(file.path(PROJ, "data/processed/subject_features.csv"))
reg_data <- subject_features |>
  select(subject_id, ends_with("_mean"), stability_score) |>
  drop_na()

# Correlation heatmap
cor_matrix <- reg_data |> select(where(is.numeric), -subject_id) |> cor(use = "complete.obs")
png(file.path(OUT, "fig_corr_heatmap.png"), width = 900, height = 820, bg = "white")
corrplot(cor_matrix,
         method = "color", type = "upper", tl.cex = 0.55, tl.col = "#374151",
         col    = colorRampPalette(c("#2563EB", "white", "#DC2626"))(200),
         title  = "Feature Correlation Matrix",
         mar    = c(0, 0, 1.5, 0), cl.cex = 0.7)
dev.off()
cat("Saved fig_corr_heatmap.png\n")

# Models
set.seed(42)
reg_split    <- initial_split(reg_data, prop = 0.7)
train_data_r <- training(reg_split)
test_data_r  <- testing(reg_split)
train_m <- train_data_r |> select(-subject_id)
test_m  <- test_data_r  |> select(-subject_id)
y_train <- train_data_r$stability_score
y_test  <- test_data_r$stability_score

set.seed(42)
cv_folds <- vfold_cv(train_m, v = 10)

lasso_wf <- workflow() |>
  add_recipe(recipe(stability_score ~ ., data = train_m) |> step_normalize(all_predictors())) |>
  add_model(linear_reg(penalty = tune(), mixture = 1) |> set_engine("glmnet"))
lasso_tune <- tune_grid(lasso_wf, resamples = cv_folds,
                        grid = grid_regular(penalty(), levels = 50),
                        metrics = metric_set(rmse, rsq))
best_lambda <- select_best(lasso_tune, metric = "rmse")
lasso_fit   <- fit(finalize_workflow(lasso_wf, best_lambda), data = train_m)
pred_lasso_train <- predict(lasso_fit, new_data = train_m)$.pred
pred_lasso_test  <- predict(lasso_fit, new_data = test_m)$.pred

set.seed(42)
rf_wf   <- workflow() |>
  add_recipe(recipe(stability_score ~ ., data = train_m)) |>
  add_model(rand_forest(mode = "regression") |> set_engine("randomForest"))
rf_fit  <- fit(rf_wf, data = train_m)
pred_rf_train <- predict(rf_fit, new_data = train_m)$.pred
pred_rf_test  <- predict(rf_fit, new_data = test_m)$.pred

# Predicted vs actual
pred_df <- bind_rows(
  data.frame(actual = y_test, predicted = pred_lasso_test, Model = "Lasso"),
  data.frame(actual = y_test, predicted = pred_rf_test,    Model = "Random Forest")
)
p_pred <- ggplot(pred_df, aes(x = actual, y = predicted, color = Model)) +
  geom_point(alpha = 0.8, size = 2) +
  geom_abline(slope = 1, intercept = 0, color = "#9CA3AF", linetype = "dashed") +
  facet_wrap(~Model) +
  scale_color_manual(values = c("Lasso" = "#2563EB", "Random Forest" = "#D97706")) +
  theme_clean() +
  theme(legend.position = "none") +
  labs(title = "Predicted vs. Actual Stability Score (test set)",
       x = "Actual", y = "Predicted")
ggsave(file.path(OUT, "fig_pred_vs_actual.png"), p_pred,
       width = 8, height = 4, dpi = 150, bg = "white")
cat("Saved fig_pred_vs_actual.png\n")

# Lasso coefficients
lasso_coef_df <- lasso_fit |> extract_fit_parsnip() |> tidy() |>
  filter(term != "(Intercept)", estimate != 0) |>
  arrange(desc(abs(estimate)))
p_coef <- ggplot(lasso_coef_df,
                 aes(x = reorder(term, estimate), y = estimate, fill = estimate > 0)) +
  geom_col() +
  scale_fill_manual(values = c("TRUE" = "#2563EB", "FALSE" = "#DC2626"),
                    labels  = c("TRUE" = "positive", "FALSE" = "negative")) +
  coord_flip() +
  theme_clean() +
  theme(legend.position = "bottom") +
  labs(title = paste("Lasso selected features (n =", nrow(lasso_coef_df), ")"),
       x = NULL, y = "Coefficient", fill = NULL)
ggsave(file.path(OUT, "fig_lasso_coefs.png"), p_coef,
       width = 7, height = max(3.5, nrow(lasso_coef_df) * 0.22), dpi = 150, bg = "white")
cat("Saved fig_lasso_coefs.png\n")

# RF feature importances
p_rf <- rf_fit |> extract_fit_engine() |>
  vip(num_features = 20, aesthetics = list(fill = "#2563EB", color = "white")) +
  theme_clean() +
  labs(title = "Random Forest - top 20 features (node impurity)",
       x = NULL, y = "Importance")
ggsave(file.path(OUT, "fig_rf_importance.png"), p_rf,
       width = 7, height = 5, dpi = 150, bg = "white")
cat("Saved fig_rf_importance.png\n")

cat("\nAll figures saved.\n")
