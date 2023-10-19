set.seed(2023-10-19)

n <- 1000
class_loc <- c(0, 1, 2)
planes <- lapply(class_loc, \(x) cbind(class = x, x1 = runif(n), x2 = rnorm(n, x, 0.25)))
planes <- do.call(rbind, planes) |> as.data.frame()

# Entropy selection
m <- nnet::multinom(factor(class)~., planes)
p <- predict(m, type = "probs")
p <- p + 1e-15 # p=0 leads to NaNs
e <- apply(p, 1, \(p) - sum(p * log(p)))

selection_probs <- plogis(7*e-4)
hist(selection_probs)

idx_ent <- runif(nrow(planes)) < selection_probs

# Plot
plot_sampling <- function(sample_idx, ...) {
  plot(
    planes$x1,
    planes$x2,
    pch = 20,
    col = c("grey90", "grey60", "grey20")[planes$class + 1],
    xlab="", ylab="", axes = F, ...
  )
  with(planes[sample_idx, ], points(x1, x2, pch=20, col=16))
  abline(h=0.5*c(1, 3), lty=2)
}

par(mfrow=c(1,2))
idx_rand <- runif(nrow(planes)) < mean(i)
plot_sampling(idx_rand, main = expression("Completely random labeling"))
plot_sampling(idx_ent, main = expression("Entropy-based labeling"))

