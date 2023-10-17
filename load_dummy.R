
set.seed(2023-10-15)
library(dplyr)

as_ssl_data <- function(df, u = 0.99) {
  df <- df |>
    select(x1=X1, x2=X2, class=Class) |> 
    mutate(across(c(x1, x2), ~. - mean(.))) |>
    mutate(across(c(x1, x2), ~. / sd(.)))
  
  df$class <- as.integer(factor(df$class))-1
  
  
  df$labeled <- sample(0:1, nrow(df), replace=TRUE, prob = c(u, 1-u))
  df$labeled <- as.logical(df$labeled)
  
  df <- split(df, df$labeled)
  df[[2]] <- dplyr::sample_n(df[[2]], nrow(df[[1]]), replace = TRUE)
  df <- do.call(rbind, unname(df))
  
  with(df, plot(x1, x2, col=labeled+1, pch = ifelse(labeled, 16, 20)))
  
  df
}

# Moons ------------------------------------------------------------------------
moons <- RSSL::generateCrescentMoon(n=1000, sigma = 0.5)
moons <- moons |> as_ssl_data()
table(moons$labeled)

write.csv(moons, "data/moons.csv", row.names = FALSE)

# Blobs ------------------------------------------------------------------------
blobs <- RSSL::generateFourClusters(1000, 10, expected = TRUE)
blobs <- blobs |> as_ssl_data()
write.csv(blobs, "data/blobs.csv", row.names = FALSE)

# Circles -----------------------------------------------------------------------
circles <- RSSL::generateTwoCircles(1000, noise_var = 0.1)
circles <-  circles |> as_ssl_data(0.98)
table(circles$labeled)

write.csv(circles, "data/circles.csv", row.names = FALSE)

# Grid -------------------------------------------------------------------------

grid <- expand.grid(
  x1 = seq(from=-2.5, to=2.5, length.out=100),
  x2 = seq(from=-2.5, to=2.5, length.out=100)
)

write.csv(grid, "data/grid.csv", row.names = FALSE)





