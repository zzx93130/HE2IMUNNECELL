# Run the installed, pinned tlsR package without modifying its detection code.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
jobs <- read.csv(args[1], stringsAsFactors = FALSE)
output <- args[2]
dir.create(output, recursive = TRUE, showWarnings = FALSE)
suppressPackageStartupMessages(library(tlsR))
stopifnot(as.character(packageVersion("tlsR")) == "0.3.0")
defaults <- formals(tlsR::detect_TLS)
stopifnot(defaults$k == 30L, defaults$bcell_density_threshold == 10,
          defaults$min_B_cells == 50L, defaults$min_T_cells_nearby == 10L,
          defaults$max_distance_T == 50, defaults$expand_distance == 80)
writeLines(capture.output(sessionInfo(), print(defaults)), file.path(output, "R_session.txt"))
save_csv <- function(df, path) {
  connection <- gzfile(path, "wt")
  on.exit(close(connection))
  write.csv(df, connection, row.names = FALSE, na = "")
}
metrics <- list()
for (job_number in seq_len(nrow(jobs))) {
  job <- jobs[job_number, ]
  df <- read.csv(job$input_csv, stringsAsFactors = FALSE)
  stopifnot(!anyDuplicated(df$global_instance_id), all(is.finite(df$x)), all(is.finite(df$y)))
  stopifnot(identical(grepl("B", df$phenotype, ignore.case = TRUE), df$phenotype == "B cell"),
            identical(grepl("T", df$phenotype, ignore.case = TRUE), df$phenotype == "T cell"))
  b <- as.matrix(df[df$phenotype == "B cell", c("x", "y")])
  dense_n <- if (nrow(b) >= 31L) sum(rowMeans(FNN::get.knn(b, k = 30L)$nn.dist) <= job$threshold) else 0L
  parameters <- list(LSP = "HEslide", ldata = list(HEslide = df), bcell_density_threshold = job$threshold)
  start <- proc.time()[["elapsed"]]
  native <- do.call(tlsR::detect_TLS, parameters)
  detection_seconds <- proc.time()[["elapsed"]] - start
  native_df <- native$HEslide
  save_csv(native_df, file.path(output, paste0(job$run_id, "_cells.csv.gz")))
  ids <- sort(unique(native_df$tls_id_knn[native_df$tls_id_knn > 0]))
  regions <- data.frame(native_id = integer(), x_um = numeric(), y_um = numeric(),
                        core_B_count = integer(), native_total_cells = integer(),
                        icat = numeric(), membership_max_distance_um = numeric())
  if (length(ids)) {
    # A second official call without expansion recovers exact core centers and
    # detects any overwrite of core labels during native expansion.
    parameters$expand_distance <- NULL
    core <- do.call(tlsR::detect_TLS, c(parameters, list(expand_distance = NULL)))$HEslide
    save_csv(core[core$tls_id_knn > 0, ], file.path(output, paste0(job$run_id, "_cores.csv.gz")))
    rows <- lapply(ids, function(cid) {
      cdf <- core[core$tls_id_knn == cid, ]
      stopifnot(nrow(cdf) >= 50L)
      cx <- cdf$tls_center_x[1]; cy <- cdf$tls_center_y[1]
      members <- native_df[native_df$tls_id_knn == cid, ]
      set.seed(569L + as.integer(cid))
      data.frame(native_id = cid, x_um = cx, y_um = cy, core_B_count = nrow(cdf),
                 native_total_cells = nrow(members), icat = tlsR::calc_icat("HEslide", cid, native),
                 membership_max_distance_um = max(sqrt((members$x - cx)^2 + (members$y - cy)^2)))
    })
    regions <- do.call(rbind, rows)
  }
  write.csv(regions, file.path(output, paste0(job$run_id, "_regions.csv")), row.names = FALSE, na = "")
  write.csv(tlsR::summarize_TLS(native), file.path(output, paste0(job$run_id, "_summary.csv")), row.names = FALSE, na = "")
  metrics[[job_number]] <- data.frame(run_id = job$run_id, threshold = job$threshold,
    eligible_B = sum(df$phenotype == "B cell"), eligible_T = sum(df$phenotype == "T cell"),
    dense_B = dense_n, detections = length(ids), detection_seconds = detection_seconds,
    total_seconds = proc.time()[["elapsed"]] - start)
  write.csv(do.call(rbind, metrics), file.path(output, "tlsr_metrics.csv"), row.names = FALSE)
  cat(job$run_id, ":", length(ids), "detections;", dense_n, "dense B cells;", round(detection_seconds, 3), "seconds\n")
  flush.console()
}
