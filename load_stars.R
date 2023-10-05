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

# Standardize and centre features
stars <- stars |> 
  mutate(across(u:redshift, ~. - mean(.))) |> 
  mutate(across(u:redshift, ~. / sd(.))) 

# Remove extreme outliers
stars <- stars |> filter(if_all(u:redshift, ~abs(.) < 10))


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
    tr <- split_proportion(stars, p)
    names(tr) <- c("labeled", "unlabeled")
    tr$unlabeled$class <- NA
    bind_rows(!!!tr)
  }
)
names(train) <- unlabeled_proportions

# Write it out for modelling
write.csv(dat$test, "data/stars/prepared-test.csv", row.names = FALSE)
write.csv(dat$validate, "data/stars/prepared-validate.csv", row.names = FALSE)
purrr::iwalk(train, \(x, p) write.csv(x, sprintf("data/stars/prepared-train-car-%s.csv", p), row.names = FALSE))


