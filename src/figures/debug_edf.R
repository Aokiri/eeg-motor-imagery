suppressPackageStartupMessages({
  library(edfReader)
  library(signal)
  library(dplyr)
})

PROJ <- "c:/Users/aland/Documents/Knowledge/Data Science & AI/Second Year/Data Science in R/02. Project"
f <- file.path(PROJ, "data/raw/S001/S001R04.edf")
cat("File exists:", file.exists(f), "\n")

h    <- readEdfHeader(f)
cat("Labels:", paste(h$sHeaders$label, collapse=", "), "\n")

sigs <- readEdfSignals(h, signals = c("C3.."))
cat("Names in sigs:", paste(names(sigs), collapse=", "), "\n")

s1 <- sigs[[1]]
cat("Class:", class(s1), "\n")
cat("Names:", paste(names(s1), collapse=", "), "\n")
cat("Signal length:", length(s1$signal), "\n")
cat("First 5 values:", head(s1$signal, 5), "\n")

# Also check annotations
annot <- readEdfSignals(h, signals = "EDF Annotations")
cat("Annot class:", class(annot), "\n")
cat("Annot names:", paste(names(annot), collapse=", "), "\n")
events <- as.data.frame(annot$annotations)
cat("Events head:\n")
print(head(events))
