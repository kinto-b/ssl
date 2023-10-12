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


# Functions --------------------------------------------------------------------

split_proportion <- function(tbl, prop) {
  if (length(prop)==1) prop <- c(prop, 1-prop)
  sp <- sample(seq_along(prop), size = nrow(tbl), replace = TRUE, prob = prop)
  
  split(tbl, sp)
}

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

# Remove labels ----------------------------------------------------------------
# We remove labels first completely at random, and then using the unit entropy.
# To do this we produce selection indicators to tell whether each record is 
# labeled or not.

# Completely at random
proportions <- c(0.001, 0.01, 0.1, 0.25)
lbl_mat_car <- sapply(proportions, \(p) runif(nrow(stars)) < p)
colnames(lbl_mat_car) <- paste0("car-", proportions)

# Entropy
m <- nnet::multinom(class~., stars)
p <- predict(m, type = "probs")
p <- p + 1e-15 # p=0 leads to NaNs
e <- apply(p, 1, \(p) - sum(p * log(p)))

proportions <- list(
  `0.001` = plogis(1.25*e-7),
  `0.01` = plogis(1.25*e-4.75),
  `0.1`  = plogis(1.25*e-2.25),
  `0.25` = plogis(1.25*e-1.25)
)

lbl_mat_ent <- sapply(proportions, \(p) runif(nrow(stars)) < p)
colnames(lbl_mat_ent) <- paste0("ent-", names(proportions))

stars <- bind_cols(stars, lbl_mat_car, lbl_mat_ent)

# write.csv(stars, "data/stars/prepared.csv", row.names = FALSE)

# Folds ------------------------------------------------------------------------
# Now we'll create five cuts of the data for cross-validation.

folds <- stars |> split_proportion(rep(0.2, 5))

purrr::iwalk(folds, \(x, i) write.csv(x, sprintf("data/stars/prepared-fold%s.csv", i), row.names = FALSE))


# README ------------------------------------------------------------------

sink("data/stars/README.md")
cat(
  "# Stars", 
  "",
  "The raw data has been prepared by excluding one extreme outlier, dropping useless columns, and mean-centering and scaling the predictors so that they have sd=1.0.",
  "",
  "Indicator columns have been added to flag whether each observation should be treated as labeled or unlabeled.",
  "For example, `car-0.1` contains approx. 10% TRUE selected completely at random while `ent-0.01` contains approx. 1% TRUE selected based on unit-entropy.",
  "",
  "Exact proportions are shown below.",
  "", "",
  sep = "\n"
)

stars |> 
  select(`car-0.001`:`ent-0.25`) |> 
  sapply(\(x) 100*table(x)/length(x)) |> 
  print()

cat("\n\n")
cat(
  "To use the data, do something like", "",
  "```",
  "df <- read.csv('prepared-fold1.csv') |>",
  "  mutate(class = ifelse(`car-0.1`, class, NA) # Unlabel",
  "  select(u:class) # Drop selection indicators",
  "```", "",
  "We can do CV by training on 4 of the folds and testing on the remaining held-out fold.",
  sep = "\n"
)

sink()




