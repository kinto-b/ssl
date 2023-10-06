
#' Conduct self-training
#' 
#' Fit a model on the labeled data. Use the fitted model to predict classes
#' for the unlabelled data. Add the predictions to the labeled data if they
#' have high confidence. Repeat.
#'
#' @param formula A model formula
#' @param model A function taking formula and data as its first two args and 
#' returning a fitted model.
#' @param predict_probs A function taking a fitted model and a new data.frame
#'   and returning a matrix of fitted class probabilities
#' @param predict_class A function taking a fitted model and a new data.frame
#'   and returning a vector of fitted classes
#' @param labeled A data.frame to use as the labeled data
#' @param unlabeled A data.frame to use as the unlabeled data
#' @param test A data.frame to compute accuracy on
#' @param alpha The psuedo-labelling probability cutoff. Unlabeled data will be
#'   assigned to whatever class has probability greater than this value.
#' @param epochs The maximum number of epochs to pseudo-label for.
#'
#' @return Invisibly, a list containing the original and final fitted models, the
#'   pseudolabelled data and the remainders
selftrain <- function(formula, model, predict_probs, predict_class, labeled, unlabeled, test, alpha = 0.9, epochs = 10) {
  lhs <- as.character(formula)[2]
  
  test[[lhs]] <- factor(test[[lhs]])
  labeled[[lhs]] <- factor(labeled[[lhs]])
  unlabeled[[lhs]] <- NA
  lvls <- levels(labeled[[lhs]])
  
  cat(sprintf(
    "Training on %d labeled and %d unlabeled...\n",
    nrow(labeled),
    nrow(unlabeled)
  ))
  
  epoch <- 1
  while (nrow(unlabeled)>1 && epoch <= epochs) {
    m <- model(formula, data=labeled)
    
    if (epoch == 1) {
      m0 <- m
      acc0 <- mean(predict_class(m, test) == test[[lhs]])
      cat(sprintf("Starting accuracy %0.1f.\n", 100*acc0))
    }
    
    # Psuedo-label
    pred <- predict_probs(m, unlabeled) > alpha
    for (i in seq_along(lvls)) { 
      unlabeled[[lhs]][pred[,i]] <- lvls[i]
    }
    
    # Prepare for next epoch
    still_unlabeled <- is.na(unlabeled[[lhs]])
    labeled <- rbind(labeled, unlabeled[!still_unlabeled, ])
    unlabeled <- unlabeled[still_unlabeled, ]
    
    cat(sprintf("After epoch %d, %d unlabeled points remain.\n", epoch, nrow(unlabeled)))
    epoch <- epoch + 1
  }
  
  acc <- mean(predict_class(m, test) == test[[lhs]])
  cat(sprintf("Final accuracy %0.1f (%+0.1f).\n", 100*acc, 100*(acc-acc0)))
  
  invisible(list(
    model = m,
    model0 = m0,
    pseudolabeled = labeled,
    remainder = unlabeled
  ))
}


# Main -------------------------------------------------------------------------
set.seed(2023-10-06)
stars <- read.csv("data/stars/prepared-train-car-0.001.csv")

stars <- split(stars, is.na(stars$class))
names(stars) <- c("labeled", "unlabeled")

stars$validate <- read.csv("data/stars/prepared-validate.csv")

# Try a self-training with a few different classifiers
cat("\n\n# Multinomial classifier ------- \n")
selftrain(
  class~., 
  model = \(...) nnet::multinom(..., trace=FALSE),
  predict_probs = \(m, df) predict(m, type = "probs", newdata = df),
  predict_class = \(m, df) predict(m, newdata = df),
  stars$labeled, stars$unlabeled, stars$validate
)

cat("\n\n# LDA classifier ------- \n")
selftrain(
  class~., 
  model = \(...) MASS::lda(...),
  predict_probs = \(m, df) predict(m, newdata = df)$posterior,
  predict_class = \(m, df) predict(m, newdata = df)$class,
  stars$labeled, stars$unlabeled, stars$validate
)

cat("\n\n# QDA classifier ------- \n")
selftrain(
  class~., 
  model = \(...) MASS::qda(...),
  predict_probs = \(m, df) predict(m, newdata = df)$posterior,
  predict_class = \(m, df) predict(m, newdata = df)$class,
  stars$labeled, stars$unlabeled, stars$validate
)

cat("\n\n# SVM classifier ------- \n")
selftrain(
  class~., 
  model = \(...) kernlab::ksvm(..., prob.model=TRUE),
  predict_probs = \(m, df) kernlab::predict(m, df, type="probabilities"),
  predict_class = \(m, df) kernlab::predict(m, newdata = df),
  stars$labeled, stars$unlabeled, stars$validate
)






