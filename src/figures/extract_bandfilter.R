library(edfReader)
library(signal)
library(ggplot2)
library(dplyr)
library(tidyr)

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

theme_nord <- function() {
  theme_minimal() +
  theme(
    plot.background    = element_rect(fill = nord_surf, color = NA),
    panel.background   = element_rect(fill = nord_surf, color = NA),
    panel.grid.major   = element_line(color = nord_bord, linewidth = 0.4),
    panel.grid.minor   = element_blank(),
    strip.background   = element_rect(fill = nord_bord, color = NA),
    strip.text         = element_text(color = nord_tx,  size = 10, face = "bold"),
    axis.text          = element_text(color = nord_tx2, size = 9),
    axis.title         = element_text(color = nord_tx,  size = 10),
    plot.title         = element_text(color = nord_tx,  size = 12, face = "bold"),
    plot.subtitle      = element_text(color = nord_tx2, size = 9),
    legend.background  = element_rect(fill = nord_surf, color = NA),
    legend.text        = element_text(color = nord_tx2, size = 9),
    legend.title       = element_text(color = nord_tx,  size = 9)
  )
}

# ── Read one EDF, extract C3 from trial 1 ────────────────────────────────────
# readEdfSignals with a single channel returns the signal object directly (not a list)
cat("Reading EDF...\n")
header  <- readEdfHeader(file.path(PROJ, "data/raw/S001/S001R04.edf"))
sig_c3  <- readEdfSignals(header, signals = "C3..")   # single signal → direct object
annot   <- readEdfSignals(header, signals = "EDF Annotations")

events    <- as.data.frame(annot$annotations)
task_evts <- events[events$annotation %in% c("T1", "T2"), ]
evt       <- task_evts[1, ]

sr    <- 160
s_idx <- round(evt$onset * sr) + 1
e_idx <- round(evt$end   * sr)

c3_raw  <- sig_c3$signal[s_idx:e_idx]
t_vec   <- seq(0, length.out = length(c3_raw), by = 1 / sr)

# ── Butterworth band filters ──────────────────────────────────────────────────
nyq     <- sr / 2
bf_mu   <- butter(4, c(8,  13) / nyq, type = "pass")
bf_beta <- butter(4, c(13, 30) / nyq, type = "pass")

c3_mu   <- filtfilt(bf_mu,   c3_raw)
c3_beta <- filtfilt(bf_beta, c3_raw)

# ── Build tidy dataframe ──────────────────────────────────────────────────────
df <- bind_rows(
  data.frame(time = t_vec, value = c3_raw,  band = "broadband"),
  data.frame(time = t_vec, value = c3_mu,   band = "mu  (8–13 Hz)"),
  data.frame(time = t_vec, value = c3_beta, band = "beta  (13–30 Hz)")
)
df$band <- factor(df$band, levels = c("broadband", "mu  (8–13 Hz)", "beta  (13–30 Hz)"))

# ── Plot ──────────────────────────────────────────────────────────────────────
p <- ggplot(df, aes(x = time, y = value)) +
  geom_line(color = nord_acc, linewidth = 0.5) +
  facet_wrap(~band, ncol = 1, scales = "free_y") +
  theme_nord() +
  labs(
    title    = "C3 — trial 1 (left imagery): broadband vs filtered bands",
    subtitle = "Band power (log-variance) is computed on each filtered signal",
    x = "Time (s)", y = "Amplitude (µV)"
  )

ggsave(file.path(OUT, "fig_bandfilter.png"), p,
       width = 7, height = 5, dpi = 150, bg = nord_surf)
cat("Saved fig_bandfilter.png\n")
