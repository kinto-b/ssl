
plot_ssl <- function(data = "moons", method = "pi", ...) {
  pred <- read.csv(
    sprintf("data/predictions/%s-toy_nn-%s.keras.csv", data, method),
    header = FALSE
  )
  # true <- read.csv("data/tmpy.csv", header=FALSE)
  
  df <- read.csv(sprintf("data/%s.csv", data))
  grid <- read.csv("data/grid.csv")
  
  plot(
    grid$x1, 
    grid$x2, 
    col = pred[[1]] + 2,
    pch=3, cex=0.5, xlab="", ylab="", axes = F, ...
  )
  with(
    df,
    points(
      x1, x2, 
      col = "darkgrey",
      pch=20
    )
  )
  with(
    unique(df[df$labeled, ]),
    points(x1, x2, col="black", pch=16)
  )
}


par(mfrow=c(1,3))
plot_ssl("moons", "sl", main = expression("Benchmark"))
plot_ssl("moons", "pi", main = expression(Pi-model))
plot_ssl("moons", "te", main = expression("Temporal Ensemble"))

par(mfrow=c(1,3))
plot_ssl("blobs", "sl", main = expression("Benchmark"))
plot_ssl("blobs", "pi", main = expression(Pi-model))
plot_ssl("blobs", "te", main = expression("Temporal Ensemble"))

par(mfrow=c(1,3))
plot_ssl("circles", "sl", main = expression("Benchmark"))
plot_ssl("circles", "pi", main = expression(Pi-model))
plot_ssl("circles", "te", main = expression("Temporal Ensemble"))
