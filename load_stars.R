#' Prepare star data 
#'
#' Select the useful predictors, center them, scale them so that they are mean
#' zero and standard devation one. Then drop extreme outliers by filtering
#' out records with any predictor greater than 10. 
#' 
#' Save the prepared data.
#'
#'

library(dplyr)
set.seed(2023-10-04)


# Prepare ----------------------------------------------------------------------

stars <- read.csv("data/stars/raw.csv")

# Discard metadata variables
stars <- stars |>
  as_tibble() |> 
  select(u:z, redshift, class)

# Remove extreme outliers based on z-score
stars <- stars |> 
  filter(if_all(u:redshift, ~abs(. - mean(.))/sd(.) < 10))

# Standardize and centre features
stars <- stars |> 
  mutate(across(u:redshift, ~. - mean(.))) |> 
  mutate(across(u:redshift, ~. / sd(.))) 

# TODO: Remove this
# While models are still in development, we'll work with a 1/10th of the data
# to speed things along
# stars <- sample_n(stars, 10000)

# Splits -----------------------------------------------------------------------

split_proportion <- function(tbl, prop) {
  if (length(prop)==1) prop <- c(prop, 1-prop)
  sp <- sample(seq_along(prop), size = nrow(tbl), replace = TRUE, prob = prop)
  
  split(tbl, sp)
}

# First do a train/validate/test split
dat <- split_proportion(stars, c(0.8, 0.1, 0.1))
names(dat) <- c("train", "test", "validate")

# Now assign make some of the training data unlabeled completely at random (CAR)
unlabeled_proportions <- c(0.001, 0.01, 0.1, 0.25)
train_car <- lapply(
  unlabeled_proportions,
  function(p) {
    tr <- split_proportion(dat$train, p)
    names(tr) <- c("labeled", "unlabeled")
    tr$unlabeled$class <- NA
    bind_rows(!!!tr)
  }
)
names(train_car) <- unlabeled_proportions

# We will up-sample the labeled data so we get equal sized labeled/unlabeled.
train_car <- purrr::map(train_car, function(df) {
  u <- df |> filter(is.na(class))
  
  l <- df |>
    filter(!is.na(class)) |> 
    sample_n(nrow(.env$u), replace = TRUE) 
  
  rbind(u, l)
})

# Write it out for modelling
write.csv(dat$test, "data/stars/prepared-test.csv", row.names = FALSE)
write.csv(dat$validate, "data/stars/prepared-validate.csv", row.names = FALSE)
purrr::iwalk(train_car, \(x, p) write.csv(x, sprintf("data/stars/prepared-train-car-%s.csv", p), row.names = FALSE))


# Entropy ----------------------------------------------------------------------
# Ziyang has suggested sampling labeled data using entropy.

# First we model the class probabilities
m <- nnet::multinom(class~., dat$train)
p <- predict(m, type = "probs")

# Next we compute the case-level entropy
p <- p + 1e-15 # Deal with predicted prob of zero
e <- apply(p, 1, \(p) - sum(p * log(p)))
boxplot(e)
p[which.max(e), ]
p[which.min(e), ]

# Finally we sample with probability Pr(i in I) = logistic(e)
entropy_sampling_prob <- list(
  `0.25` = plogis(1.25*e-1.25),
  `0.1`  = plogis(1.25*e-2.25),
  `0.01` = plogis(1.25*e-4.75),
  `0.001` = plogis(1.25*e-7)
)

train_ent <- lapply(
  entropy_sampling_prob,
  function(p) {
    idx <- runif(length(e)) < p
    df <- dat$train
    df$class[!idx] <- NA
    df
  }
)

lapply(train_ent, \(df) df |> count(is.na(class)) |> mutate(p=100*n/sum(n))) |> 
  print()

# We will up-sample the labeled data so we get equal sized labeled/unlabeled.
train_ent <- purrr::map(train_ent, function(df) {
  u <- df |> filter(is.na(class))
  
  l <- df |>
    filter(!is.na(class)) |> 
    sample_n(nrow(.env$u), replace = TRUE) 
  
  rbind(u, l)
})


purrr::iwalk(train_ent, \(x, p) write.csv(x, sprintf("data/stars/prepared-train-ent-%s.csv", p), row.names = FALSE))

