# ==============================================================================
# EEG Preprocessing Pipeline - PhysioNet Dataset
# Purpose: Extract trials from EDF files and compute tsfeatures.
# Usage: Rscript src/preprocess.R
# ==============================================================================

library(edfReader)
library(tidyverse)
library(tsfeatures)

# Configuration #
input_dir      <- "data/raw/"
output_dir     <- "data/processed/"
imagery_runs   <- c("R04", "R08", "R12")
# Channels over the motor cortex
# (the only ones relevant for left vs right imagery)
motor_channels <- c("C3..", "C4..", "Fc3.", "Fc4.", "Cp3.", "Cp4.")
sampling_rate  <- 160

feature_list <- c(
  "entropy", "stability", "lumpiness", "crossing_points",
  "flat_spots", "hurst", "acf_features", "nonlinearity"
)

# Ensure output directory exists
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# Helper Functions #

# Extracts features for all trials and channels
# Also handles bad trials (flat signals, corrupted data)
extract_trial_features <- function(trial_data, feature_list) {
  result <- tryCatch({
    trial_data |>
      group_by(channel) |>
      summarise(
        features = list(tsfeatures(list(ts(value, frequency = sampling_rate)), 
                                   features = feature_list)),
        .groups = "drop"
      ) |>
      unnest(features) |>
      pivot_wider(names_from = channel, values_from = -channel,
                  names_glue = "{channel}_{.value}")
  }, error = function(e) {
    # Return NULL for trials that can't be processed (zero-variance, etc.)
    return(NULL)
  })
  result
}

# Main Processing #

message("Starting processing for 109 subjects...")

all_subjects_features <- tibble()

for (subj_num in 1:109) {
  subject <- sprintf("S%03d", subj_num)
  message("Processing ", subject, "...")

  subj_trials <- tibble()
  trial_counter <- 1

  for (run in imagery_runs) {
    filepath <- paste0(input_dir, subject, "/", subject, run, ".edf")

    if (!file.exists(filepath)) {
      message("  Skipping ", run, " (file not found)")
      next
    }

    # Read data and annotations
    hdr   <- readEdfHeader(filepath)
    sigs  <- readEdfSignals(hdr, signals = motor_channels)
    annot <- readEdfSignals(hdr, signals = "EDF Annotations")

    # The discriminative information lives
    # in the statistical properties of the signal.
    # Filter task trials (T1 = left hand, T2 = right hand)
    evts <- annot$annotations |>
      as_tibble() |>
      filter(annotation %in% c("T1", "T2"))

    for (i in seq_len(nrow(evts))) {
      evt <- evts[i, ]

      # Convert event times (seconds) to sample indices
      # At 160 Hz, +1 because R is 1-indexed
      s_idx <- round(evt$onset * sampling_rate) + 1
      e_idx <- round(evt$end * sampling_rate)

      # Build the trial data frame: one row per sample per channel
      trial_data <- tibble(
        channel = rep(c("C3", "C4", "FC3", "FC4", "CP3", "CP4"),
                      each = e_idx - s_idx + 1),
        value = c(
          sigs$`C3..`$signal[s_idx:e_idx], sigs$`C4..`$signal[s_idx:e_idx],
          sigs$`Fc3.`$signal[s_idx:e_idx], sigs$`Fc4.`$signal[s_idx:e_idx],
          sigs$`Cp3.`$signal[s_idx:e_idx], sigs$`Cp4.`$signal[s_idx:e_idx]
        )
      )

      # Each trial is a time series of ~656 samples per channel.
      # We compute summary statistics to
      # compress it into a reasonable dimensionality.
      features <- extract_trial_features(trial_data, feature_list)

      if (!is.null(features)) {
        features$subject_id <- subject
        features$trial_id   <- paste0(subject, "_", trial_counter)
        features$run        <- run
        features$label      <- ifelse(evt$annotation == "T1", "left", "right")

        subj_trials <- bind_rows(subj_trials, features)
      }
      trial_counter <- trial_counter + 1
    }
  }
  all_subjects_features <- bind_rows(all_subjects_features, subj_trials)
}

# Aggregation and Saving #

# Save trial-level features (one row per trial) for classification tasks
message("Saving trial_features.csv...")
write_csv(all_subjects_features, paste0(output_dir, "trial_features.csv"))

# Aggregate per subject: mean and std of each feature across all their trials
message("Aggregating subject-level features for regression...")
feature_cols <- all_subjects_features |>
  select(-subject_id, -trial_id, -run, -label) |>
  colnames()

subject_features <- all_subjects_features |>
  group_by(subject_id) |>
  summarise(
    across(all_of(feature_cols), list(mean = mean, std = sd),
           .names = "{.col}_{.fn}"),
    n_trials = n(),
    .groups = "drop"
  )

write_csv(subject_features, paste0(output_dir, "subject_features.csv"))

message("Done. Final dataset dimensions: ", nrow(all_subjects_features),
        " trials across ", n_distinct(all_subjects_features$subject_id),
        " subjects.")