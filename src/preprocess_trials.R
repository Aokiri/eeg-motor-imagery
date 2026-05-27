# ==============================================================================
# EEG Preprocessing Pipeline - Trial Features
# Purpose: Extract trials from EDF files and compute tsfeatures + band power
#          features (mu 8-13 Hz, beta 13-30 Hz) per trial.
# Output:  data/processed/trial_features.csv
# Usage:   Rscript src/preprocess_trials.R
# ==============================================================================

library(edfReader)
library(tidyverse)
library(tsfeatures)
library(signal)
filter <- dplyr::filter

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

# Returns log(var()) of the signal filtered to [low, high] Hz.
# Returns NA if the signal is flat or the filter fails.
compute_band_power <- function(x, low, high, sr = 160) {
  nyq      <- sr / 2
  bf       <- butter(4, c(low, high) / nyq, type = "pass")
  filtered <- tryCatch(filtfilt(bf, x),
                       error = function(e) rep(NA_real_, length(x)))
  v <- var(filtered, na.rm = TRUE)
  if (is.na(v) || v <= 0) return(NA_real_)
  log(v)
}

# Extracts mu (8-13 Hz) and beta (13-30 Hz) band power for all channels.
extract_band_power_features <- function(trial_data, sr = 160) {
  trial_data |>
    group_by(channel) |>  # nolint
    summarise(  # nolint
      mu_power   = compute_band_power(value, 8,  13, sr),
      beta_power = compute_band_power(value, 13, 30, sr),
      .groups = "drop"
    ) |>
    pivot_wider(names_from  = channel,  # nolint
                values_from = c(mu_power, beta_power),
                names_glue  = "{channel}_{.value}")
}

# Extracts features for all trials and channels
# Also handles bad trials (flat signals, corrupted data)
extract_trial_features <- function(trial_data, feature_list, sr = 160) {
  result <- tryCatch({
    trial_data |>
      group_by(channel) |>  # nolint
      summarise(  # nolint
        features = list(tsfeatures(list(ts(value, frequency = sr)), # nolint
                                   features = feature_list)),
        .groups = "drop"
      ) |>
      unnest(features) |> # nolint
      pivot_wider(names_from = channel, values_from = -channel, # nolint
                  names_glue = "{channel}_{.value}")
  }, error = function(e) {
    # Return NULL for trials that can't be processed (zero-variance, etc.)
    return(NULL)  # nolint
  })
  result
}

# Main Processing #

n_subjects <- 109
message("Starting processing for ", n_subjects, " subjects...")
t_start <- proc.time()[["elapsed"]]

all_subjects_features <- tibble()

for (subj_num in 1:n_subjects) {
  subject <- sprintf("S%03d", subj_num)
  message(sprintf("[%3d/%d] %s ...", subj_num, n_subjects, subject))
  t_subj <- proc.time()[["elapsed"]]

  subj_trials <- tibble()
  trial_counter <- 1

  for (run in imagery_runs) {
    filepath <- paste0(input_dir, subject, "/", subject, run, ".edf")

    if (!file.exists(filepath)) {
      message("         ", run, ": file not found, skipping")
      next
    }

    # Read data and annotations
    hdr   <- readEdfHeader(filepath)
    sigs  <- readEdfSignals(hdr, signals = motor_channels)
    annot <- readEdfSignals(hdr, signals = "EDF Annotations")

    # Filter for imagery task events only (T1 = left hand, T2 = right hand)
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
      features <- extract_trial_features(trial_data, feature_list,
                                         sr = sampling_rate)

      if (!is.null(features)) {
        band_features <- extract_band_power_features(trial_data,
                                                     sr = sampling_rate)
        features <- bind_cols(features, band_features)

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

  elapsed_subj  <- round(proc.time()[["elapsed"]] - t_subj, 1)
  elapsed_total <- round(proc.time()[["elapsed"]] - t_start)
  n_subj_trials <- nrow(subj_trials)
  n_cumulative  <- nrow(all_subjects_features)
  pct           <- round(subj_num / n_subjects * 100)
  message(sprintf("         %d trials | cumulative: %d | %.1fs | total elapsed: %ds (%d%%)",
                  n_subj_trials, n_cumulative, elapsed_subj, elapsed_total, pct))
}

# Save #

message("Saving trial_features.csv...")
write_csv(all_subjects_features, paste0(output_dir, "trial_features.csv"))

message("Done. ", nrow(all_subjects_features), " trials across ",
        n_distinct(all_subjects_features$subject_id), " subjects.")
