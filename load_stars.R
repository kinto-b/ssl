#' Prepare star data 
#'
#' Select the useful predictors, center them, scale them so that they are mean
#' zero and standard devation one. Then drop extreme outliers by filtering
#' out records with any predictor greater than 10. 
#' 
#' Save the prepared data.
#'
#'
#' TODO: Create labeled/unlabeled cuts here instead of in the python scripts.

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

# Save prepared data
stars |> 
  sample_n(10000) |>
  mutate(class = as.integer(factor(class))-1) |> 
  write.csv("data/stars/prepared.csv", row.names = FALSE)

