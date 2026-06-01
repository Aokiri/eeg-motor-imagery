library(tidyverse)
library(tidymodels)
library(discrim)

PROJ <- "c:/Users/aland/Documents/Knowledge/Data Science & AI/Second Year/Data Science in R/02. Project"
OUT  <- file.path(PROJ, "presentation/figures")

# Nord dark palette
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
    strip.text         = element_text(color = nord_tx, size = 9),
    axis.text          = element_text(color = nord_tx2, size = 8),
    axis.title         = element_text(color = nord_tx,  size = 10),
    plot.title         = element_text(color = nord_tx,  size = 12, face = "bold"),
    plot.subtitle      = element_text(color = nord_tx2, size = 9),
    legend.background  = element_rect(fill = nord_surf, color = NA),
    legend.text        = element_text(color = nord_tx2, size = 9),
    legend.title       = element_text(color = nord_tx,  size = 9)
  )
}

cat("Loading trial_features.csv...\n")
features <- read_csv(file.path(PROJ, "data/processed/trial_features.csv"),
                     show_col_types = FALSE)
cat("Loaded:", nrow(features), "trials,", ncol(features), "cols\n")

# ── Feature engineering ───────────────────────────────────────────────────────
features <- features |>
  mutate(label = as.factor(label)) |>
  mutate(
    asym_mu_C   = C3_mu_power   - C4_mu_power,
    asym_beta_C = C3_beta_power - C4_beta_power,
    asym_mu_F   = FC3_mu_power  - FC4_mu_power,
    asym_beta_F = FC3_beta_power - FC4_beta_power,
    asym_mu_P   = CP3_mu_power  - CP4_mu_power,
    asym_beta_P = CP3_beta_power - CP4_beta_power
  )

# Per-subject z-score normalization
feature_cols <- features |> select(-subject_id, -trial_id, -run, -label) |> colnames()
features <- features |>
  group_by(subject_id) |>
  mutate(across(all_of(feature_cols),
                ~ (. - mean(., na.rm = TRUE)) / (sd(., na.rm = TRUE) + 1e-8))) |>
  ungroup()

# ── 1. Density distributions C3 features (left vs right) ────────────────────
cat("Plotting C3 density distributions...\n")
p_density <- features |>
  select(starts_with("C3_"), label) |>
  select(-contains("mu_power"), -contains("beta_power")) |>  # keep tsfeatures only
  pivot_longer(cols = starts_with("C3_"), names_to = "feature", values_to = "value") |>
  mutate(feature = gsub("C3_", "", feature)) |>
  ggplot(aes(x = value, fill = label, color = label)) +
  geom_density(alpha = 0.4, linewidth = 0.6) +
  facet_wrap(~feature, scales = "free", ncol = 4) +
  scale_fill_manual(values  = c("left" = nord_acc, "right" = nord_yel)) +
  scale_color_manual(values = c("left" = nord_acc, "right" = nord_yel)) +
  theme_nord() +
  labs(
    title    = "C3 feature distributions by class (left vs right)",
    subtitle = "Distributions overlap heavily; no single feature separates classes",
    x = NULL, y = "Density", fill = NULL, color = NULL
  )

ggsave(file.path(OUT, "fig_c3_density.png"), p_density,
       width = 11, height = 7, dpi = 150, bg = nord_surf)
cat("Saved fig_c3_density.png\n")

# ── Train/test split ──────────────────────────────────────────────────────────
set.seed(42)
data_split <- initial_split(features, prop = 0.8, strata = subject_id)
train_data <- training(data_split)
test_data  <- testing(data_split)

base_recipe <- recipe(label ~ ., data = train_data) |>
  update_role(subject_id, trial_id, run, new_role = "ID") |>
  step_impute_mean(all_numeric_predictors()) |>
  step_zv(all_numeric_predictors()) |>
  step_corr(all_numeric_predictors(), threshold = 0.9)

eval_metrics <- metric_set(accuracy, sensitivity, specificity, roc_auc)

fit_and_eval <- function(spec, name) {
  cat("Fitting", name, "...\n")
  wf   <- workflow() |> add_recipe(base_recipe) |> add_model(spec)
  fit  <- wf |> fit(data = train_data)
  pred <- augment(fit, new_data = test_data)
  list(preds = pred, name = name)
}

# ── 2. Models ─────────────────────────────────────────────────────────────────
logreg_res <- fit_and_eval(
  logistic_reg() |> set_engine("glm") |> set_mode("classification"),
  "Logistic Regression"
)

nb_res <- fit_and_eval(
  naive_Bayes() |> set_engine("naivebayes") |> set_mode("classification"),
  "Naive Bayes"
)

knn_res <- fit_and_eval(
  nearest_neighbor(neighbors = 10) |> set_engine("kknn") |> set_mode("classification"),
  "KNN"
)

rf_res <- fit_and_eval(
  rand_forest(trees = 500) |> set_engine("ranger") |> set_mode("classification"),
  "Random Forest"
)

# ── 3. ROC curves ─────────────────────────────────────────────────────────────
cat("Plotting ROC curves...\n")
model_colors <- c(
  "Logistic Regression" = nord_grn,
  "Naive Bayes"         = nord_yel,
  "KNN"                 = nord_red,
  "Random Forest"       = nord_acc
)

roc_data <- bind_rows(
  logreg_res$preds |> mutate(model = "Logistic Regression"),
  nb_res$preds     |> mutate(model = "Naive Bayes"),
  knn_res$preds    |> mutate(model = "KNN"),
  rf_res$preds     |> mutate(model = "Random Forest")
) |>
  group_by(model) |>
  roc_curve(truth = label, .pred_left)

p_roc <- roc_data |>
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_line(linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0, color = nord_tx2, linetype = "dashed") +
  scale_color_manual(values = model_colors) +
  theme_nord() +
  labs(title = "ROC curves — all models (test set)",
       x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)", color = NULL)

ggsave(file.path(OUT, "fig_roc_curves.png"), p_roc,
       width = 6, height = 5, dpi = 150, bg = nord_surf)
cat("Saved fig_roc_curves.png\n")

# ── 4. Metrics table ──────────────────────────────────────────────────────────
all_preds <- list(
  "Logistic Regression" = logreg_res$preds,
  "Naive Bayes"         = nb_res$preds,
  "KNN"                 = knn_res$preds,
  "Random Forest"       = rf_res$preds
)

metrics_tbl <- purrr::imap_dfr(all_preds, ~ {
  res <- eval_metrics(.x, truth = label, estimate = .pred_class, .pred_left)
  res$model <- .y
  res
}) |>
  select(model, .metric, .estimate) |>
  tidyr::pivot_wider(names_from = .metric, values_from = .estimate) |>
  mutate(across(where(is.numeric), ~round(., 4)))

cat("\n=== CLASSIFICATION METRICS ===\n")
print(metrics_tbl)
write.csv(metrics_tbl, file.path(OUT, "metrics_classification.csv"), row.names = FALSE)
cat("Saved metrics_classification.csv\n")

# ── 5. Per-subject accuracy (RF) ──────────────────────────────────────────────
per_subject_acc <- rf_res$preds |>
  group_by(subject_id) |>
  summarise(
    accuracy = mean(.pred_class == label),
    n_trials = n(),
    .groups  = "drop"
  ) |>
  arrange(accuracy)

cat("\n=== PER-SUBJECT ACCURACY (RF) ===\n")
cat("Global mean:", round(mean(per_subject_acc$accuracy), 4), "\n")
cat("Above 70%:  ", sum(per_subject_acc$accuracy > 0.70), "/", nrow(per_subject_acc), "\n")
cat("At/below 50%:", sum(per_subject_acc$accuracy <= 0.50), "/", nrow(per_subject_acc), "\n")

write.csv(per_subject_acc, file.path(OUT, "per_subject_accuracy.csv"), row.names = FALSE)
cat("Saved per_subject_accuracy.csv (for future bar chart)\n")
cat("\nDone.\n")
