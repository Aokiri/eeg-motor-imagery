library(ggplot2)
library(dplyr)

PROJ <- "c:/Users/aland/Documents/Knowledge/Data Science & AI/Second Year/Data Science in R/02. Project"
OUT  <- file.path(PROJ, "presentation/figures")

df <- read.csv(file.path(OUT, "per_subject_accuracy.csv")) |>
  arrange(accuracy) |>
  mutate(rank = row_number())

global_mean <- mean(df$accuracy)

p <- ggplot(df, aes(x = rank, y = accuracy, fill = accuracy)) +
  geom_col(width = 0.85, show.legend = FALSE) +
  geom_hline(yintercept = 0.5,         color = "#9CA3AF", linetype = "dashed", linewidth = 0.7) +
  geom_hline(yintercept = global_mean, color = "#2563EB", linetype = "dashed", linewidth = 0.7) +
  scale_fill_gradient(low = "#F59E0B", high = "#2563EB") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                     limits = c(0, 1), expand = c(0, 0.01)) +
  scale_x_continuous(expand = c(0.01, 0)) +
  annotate("text", x = nrow(df) * 0.02, y = global_mean + 0.04,
           label = paste0("mean = ", round(global_mean * 100, 1), "%"),
           color = "#2563EB", size = 2.8, hjust = 0,
           family = "sans") +
  annotate("text", x = nrow(df) * 0.02, y = 0.5 - 0.04,
           label = "chance (50%)",
           color = "#9CA3AF", size = 2.8, hjust = 0,
           family = "sans") +
  theme_minimal() +
  theme(
    plot.background   = element_rect(fill = "white", color = NA),
    panel.background  = element_rect(fill = "white", color = NA),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.4),
    axis.text.x       = element_blank(),
    axis.ticks.x      = element_blank(),
    axis.text.y       = element_text(color = "#6B7280", size = 8),
    axis.title        = element_text(color = "#374151", size = 9),
    plot.title        = element_text(color = "#111827", size = 11, face = "bold"),
    plot.subtitle     = element_text(color = "#6B7280", size = 8)
  ) +
  labs(
    title    = "Per-subject classification accuracy (Random Forest)",
    subtitle = "Blue dashed: global mean  |  Gray dashed: chance level (50%)",
    x        = "Subjects (sorted by accuracy)",
    y        = "Accuracy"
  )

ggsave(file.path(OUT, "fig_per_subject_accuracy.png"), p,
       width = 8, height = 4, dpi = 150, bg = "white")
cat("Saved fig_per_subject_accuracy.png\n")
cat("Global mean:", round(global_mean, 4), "\n")
cat("Above 70%:  ", sum(df$accuracy > 0.70), "/", nrow(df), "\n")
cat("At/below 50%:", sum(df$accuracy <= 0.50), "/", nrow(df), "\n")
