# Reproduction of 02_classification.Rmd for figure extraction.
# Faithful to AB's methodology; only the output paths and visual theme differ.
# Runtime: ~20-40 min (per-subject fitting for 109 subjects × 4 models).

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(discrim)
  library(glmnet)
})

PROJ <- normalizePath(file.path(
  tryCatch(dirname(normalizePath(sub("--file=", "",
    grep("--file=", commandArgs(FALSE), value = TRUE)))),
    error = function(e) getwd()),
  "../.."))
OUT  <- file.path(PROJ, "presentation/figures")
set.seed(42)

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

# ── 1. Load and build feature set (AB's exact feature selection) ──────────────
cat("Loading data...\n")
features_raw <- read_csv(file.path(PROJ, "data/processed/trial_features.csv"),
                         show_col_types = FALSE)

features <- features_raw |>
  mutate(
    C_asym_mu    = C3_mu_power    - C4_mu_power,
    C_asym_beta  = C3_beta_power  - C4_beta_power,
    FC_asym_mu   = FC3_mu_power   - FC4_mu_power,
    FC_asym_beta = FC3_beta_power - FC4_beta_power,
    CP_asym_mu   = CP3_mu_power   - CP4_mu_power,
    CP_asym_beta = CP3_beta_power - CP4_beta_power
  )

mi_feature_cols <- features |>
  select(matches("^(C3|C4|FC3|FC4|CP3|CP4)_(mu|beta)_(power|erd)$")) |>
  colnames()

predictor_cols <- c(
  mi_feature_cols,
  "C_asym_mu", "C_asym_beta",
  "FC_asym_mu", "FC_asym_beta",
  "CP_asym_mu", "CP_asym_beta"
)

model_df <- features |>
  select(subject_id, trial_id, run, label, all_of(predictor_cols)) |>
  mutate(label = factor(label, levels = c("left", "right")))

# Drop subjects with NaN values
bad_subjects <- model_df |>
  filter(if_any(all_of(predictor_cols), is.na)) |>
  distinct(subject_id)

model_df <- model_df |> anti_join(bad_subjects, by = "subject_id")

cat("Trials:", nrow(model_df), " Subjects:", n_distinct(model_df$subject_id), "\n")
cat("Predictors:", length(predictor_cols), "\n")

# ── 2. Density figure (C3, C4, asymmetry features) ───────────────────────────
cat("Plotting density distributions...\n")
p_density <- model_df |>
  select(starts_with(c("C3_", "C4_", "C_")), label) |>
  pivot_longer(cols = -label, names_to = "feature", values_to = "value") |>
  mutate(feature = factor(feature, levels = sort(unique(feature), decreasing = TRUE))) |>
  ggplot(aes(x = value, fill = label, color = label)) +
  geom_density(alpha = 0.4, linewidth = 0.6) +
  facet_wrap(~feature, scales = "free") +
  scale_fill_manual(values  = c("left" = "#2563EB", "right" = "#D97706")) +
  scale_color_manual(values = c("left" = "#2563EB", "right" = "#D97706")) +
  theme_clean(8) +
  labs(title    = "Feature distributions by class — C3, C4, and asymmetry features",
       subtitle = "Band power, ERD, and lateral asymmetry; distributions overlap across classes",
       x = NULL, y = "Density", fill = NULL, color = NULL)

ggsave(file.path(OUT, "fig_c3_density.png"), p_density,
       width = 12, height = 7, dpi = 150, bg = "white")
cat("Saved fig_c3_density.png\n")

# ── 3. Train / test split (AB's approach: per subject per class) ──────────────
test_keys <- model_df |>
  group_by(subject_id, label) |>
  slice_sample(prop = 0.2) |>
  ungroup() |>
  select(subject_id, trial_id)

train_data <- model_df |> anti_join(test_keys, by = c("subject_id", "trial_id"))
test_data  <- model_df |> semi_join(test_keys,  by = c("subject_id", "trial_id"))
cat("Train:", nrow(train_data), "| Test:", nrow(test_data), "\n")

# ── 4. Within-subject scaler (fit on train, apply to test) ────────────────────
fit_subject_scaler <- function(train_df, feature_cols) {
  train_df |>
    group_by(subject_id) |>
    summarise(
      across(all_of(feature_cols),
             list(mean = ~mean(.x, na.rm = TRUE), sd = ~sd(.x, na.rm = TRUE)),
             .names = "{.col}__{.fn}"),
      .groups = "drop"
    )
}

transform_subject_scaler <- function(df, scaler, feature_cols, eps = 1e-8) {
  out <- df |> left_join(scaler, by = "subject_id")
  for (feature in feature_cols) {
    m <- out[[paste0(feature, "__mean")]]
    s <- out[[paste0(feature, "__sd")]]
    s[is.na(s) | s < eps] <- 1
    out[[feature]] <- (out[[feature]] - m) / (s + eps)
  }
  out |> select(-ends_with("__mean"), -ends_with("__sd"))
}

scaler       <- fit_subject_scaler(train_data, predictor_cols)
train_scaled <- transform_subject_scaler(train_data, scaler, predictor_cols)
test_scaled  <- transform_subject_scaler(test_data,  scaler, predictor_cols)

# ── 5. Model specs (faithful to AB) ──────────────────────────────────────────
p <- length(predictor_cols)

logreg_spec <- logistic_reg(penalty = tune(), mixture = tune()) |>
  set_engine("glmnet") |> set_mode("classification")
logreg_grid <- dials::grid_space_filling(
  dials::penalty(range = c(-3, 0)),
  dials::mixture(range = c(0, 1)),
  size = 10
)

nb_spec  <- naive_Bayes() |> set_engine("naivebayes") |> set_mode("classification")
knn_spec <- nearest_neighbor(neighbors = 15) |> set_engine("kknn") |> set_mode("classification")

rf_spec <- rand_forest(trees = 500, mtry = tune(), min_n = tune()) |>
  set_engine("ranger") |> set_mode("classification")
rf_grid <- dials::grid_space_filling(
  dials::mtry(range = c(1L, floor(p / 2L))),
  dials::min_n(range = c(2L, 5L)),
  size = 10
)

eval_metrics <- metric_set(accuracy, sensitivity, specificity, roc_auc)

base_recipe <- function(train_df) {
  recipe(label ~ ., data = train_df) |>
    update_role(subject_id, trial_id, run, new_role = "ID") |>
    step_impute_mean(all_numeric_predictors()) |>
    step_zv(all_numeric_predictors())
}

make_subject_folds <- function(train_df) {
  rsample::group_vfold_cv(train_df, v = n_distinct(train_df$run), group = run)
}

fit_and_predict <- function(model_spec, rec, train_df, test_df,
                             resamples = NULL, grid = NULL) {
  wf     <- workflow() |> add_recipe(rec) |> add_model(model_spec)
  params <- workflows::extract_parameter_set_dials(wf)

  if (nrow(params) == 0) {
    fit <- fit(wf, data = train_df)
    return(augment(fit, new_data = test_df))
  }
  tuned  <- tune::tune_grid(wf, resamples = resamples, grid = grid,
                             metrics = eval_metrics)
  best   <- tune::select_best(tuned, metric = "roc_auc")
  fit    <- tune::finalize_workflow(wf, best) |> fit(data = train_df)
  augment(fit, new_data = test_df)
}

# ── 6. Per-subject fitting ────────────────────────────────────────────────────
subjects <- sort(unique(train_scaled$subject_id))
cat("Fitting", length(subjects), "subjects...\n")

all_preds <- purrr::map_dfr(subjects, function(sid) {
  s_train <- train_scaled |> filter(subject_id == sid)
  s_test  <- test_scaled  |> filter(subject_id == sid)
  rec     <- base_recipe(s_train)
  folds   <- make_subject_folds(s_train)

  bind_rows(
    fit_and_predict(logreg_spec, rec, s_train, s_test, folds, logreg_grid) |> mutate(model = "Logistic Regression"),
    fit_and_predict(nb_spec,     rec, s_train, s_test)                       |> mutate(model = "Naive Bayes"),
    fit_and_predict(knn_spec,    rec, s_train, s_test)                       |> mutate(model = "KNN"),
    fit_and_predict(rf_spec,     rec, s_train, s_test, folds, rf_grid)       |> mutate(model = "Random Forest")
  )
}, .progress = "Fitting subjects")

write_csv(all_preds, file.path(OUT, "all_preds_classification.csv"))
cat("Saved all_preds_classification.csv\n")

# ── 7. Pooled metrics table ───────────────────────────────────────────────────
pooled_metrics <- all_preds |>
  group_by(model) |>
  group_modify(~eval_metrics(.x, truth = label, estimate = .pred_class, .pred_left)) |>
  ungroup() |>
  select(model, .metric, .estimate) |>
  pivot_wider(names_from = .metric, values_from = .estimate) |>
  mutate(across(where(is.numeric), ~round(., 4)))

cat("\n=== POOLED METRICS ===\n")
print(pooled_metrics)
write_csv(pooled_metrics, file.path(OUT, "metrics_classification.csv"))

# ── 8. ROC curves ─────────────────────────────────────────────────────────────
cat("Plotting ROC curves...\n")
model_colors <- c(
  "Logistic Regression" = "#2563EB",
  "Naive Bayes"         = "#D97706",
  "KNN"                 = "#DC2626",
  "Random Forest"       = "#059669"
)

p_roc <- all_preds |>
  group_by(model) |>
  roc_curve(truth = label, .pred_left) |>
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_line(linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0, color = "#9CA3AF", linetype = "dashed") +
  scale_color_manual(values = model_colors) +
  theme_clean() +
  labs(title = "ROC curves - all models (test set, pooled across subjects)",
       x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)", color = NULL)

ggsave(file.path(OUT, "fig_roc_curves.png"), p_roc,
       width = 6, height = 5, dpi = 150, bg = "white")
cat("Saved fig_roc_curves.png\n")

# ── 9. Per-subject ROC AUC ────────────────────────────────────────────────────
per_subject_roc <- all_preds |>
  group_by(model, subject_id) |>
  summarise(
    roc_auc  = yardstick::roc_auc_vec(truth = label, estimate = .pred_left),
    n_trials = n(),
    .groups  = "drop"
  )

roc_summary <- per_subject_roc |>
  group_by(model) |>
  summarise(
    mean_auc       = mean(roc_auc),
    median_auc     = median(roc_auc),
    above_70       = sum(roc_auc > 0.70),
    at_or_below_50 = sum(roc_auc <= 0.50),
    n_subjects     = n(),
    .groups = "drop"
  ) |>
  arrange(desc(median_auc), desc(mean_auc))

cat("\n=== PER-SUBJECT ROC AUC SUMMARY ===\n")
print(roc_summary)

# Boxplot per model
p_box <- ggplot(per_subject_roc, aes(x = reorder(model, roc_auc, median), y = roc_auc, fill = model)) +
  geom_boxplot(show.legend = FALSE, width = 0.5, outlier.size = 1) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "#9CA3AF") +
  scale_fill_manual(values = model_colors) +
  scale_y_continuous(labels = scales::percent_format()) +
  theme_clean() +
  labs(title    = "Per-subject ROC AUC by model",
       subtitle = "Gray dashed: chance level (AUC = 0.50)",
       x = NULL, y = "ROC AUC")

ggsave(file.path(OUT, "fig_per_subject_roc_boxplot.png"), p_box,
       width = 7, height = 4.5, dpi = 150, bg = "white")
cat("Saved fig_per_subject_roc_boxplot.png\n")

# Bar chart for best model
best_model_name <- roc_summary |> slice(1) |> pull(model)
cat("Best model (median ROC AUC):", best_model_name, "\n")

best_per_subject <- per_subject_roc |>
  filter(model == best_model_name) |>
  arrange(roc_auc)

cat("Mean ROC AUC:", round(mean(best_per_subject$roc_auc), 3), "\n")
cat("Above 0.70: ", sum(best_per_subject$roc_auc > 0.70), "/", nrow(best_per_subject), "\n")
cat("At/below 0.50:", sum(best_per_subject$roc_auc <= 0.50), "/", nrow(best_per_subject), "\n")

p_bar <- ggplot(best_per_subject,
                aes(x = reorder(subject_id, roc_auc), y = roc_auc, fill = roc_auc)) +
  geom_col(width = 0.85, show.legend = FALSE) +
  geom_hline(yintercept = 0.5, color = "#9CA3AF", linetype = "dashed", linewidth = 0.7) +
  geom_hline(yintercept = mean(best_per_subject$roc_auc),
             color = "#2563EB", linetype = "dashed", linewidth = 0.7) +
  scale_fill_gradient(low = "#F59E0B", high = "#2563EB") +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1), expand = c(0, 0.01)) +
  scale_x_discrete(expand = c(0.01, 0)) +
  annotate("text", x = 3, y = mean(best_per_subject$roc_auc) + 0.04,
           label = paste0("mean = ", round(mean(best_per_subject$roc_auc) * 100, 1), "%"),
           color = "#2563EB", size = 2.8, hjust = 0) +
  annotate("text", x = 3, y = 0.5 - 0.04,
           label = "chance (50%)", color = "#9CA3AF", size = 2.8, hjust = 0) +
  theme_clean() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
        panel.grid.major.x = element_blank()) +
  labs(title    = paste0("Per-subject ROC AUC (", best_model_name, ")"),
       subtitle = "Blue dashed: global mean  |  Gray dashed: chance level",
       x = "Subjects (sorted by ROC AUC)", y = "ROC AUC")

ggsave(file.path(OUT, "fig_per_subject_accuracy.png"), p_bar,
       width = 8, height = 4, dpi = 150, bg = "white")
cat("Saved fig_per_subject_accuracy.png\n")

cat("\nDone.\n")
