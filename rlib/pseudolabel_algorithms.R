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
#' @param labeled A data.frame to use as the labeled data
#' @param unlabeled A data.frame to use as the unlabeled data
#' @param test A data.frame to compute accuracy on
#' @param alpha The psuedo-labelling probability cutoff. Unlabeled data will be
#'   assigned to whatever class has probability greater than this value.
#' @param n The psuedo-labelling probability cutoff. The most confident `n`
#'   predictions for each class will be assigned to that class.
#' @param epochs The maximum number of epochs to pseudo-label for.
#'
#' @return Invisibly, a list containing the final fitted model, the
#'   pseudolabelled data and the remainders
selftrain_alpha <- function(formula, model, predict_probs, labeled, unlabeled, test, alpha = 0.9, epochs = 10) {
  psuedolabeler <- function(...) .pseudolabel_alpha(..., alpha = alpha)
  .selftrain(formula, model, predict_probs, labeled, unlabeled, test, psuedolabeler, epochs)
}

selftrain_n <- function(formula, model, predict_probs, labeled, unlabeled, test, n, epochs = 10) {
  psuedolabeler <- function(...) .pseudolabel_n(..., n = n)
  .selftrain(formula, model, predict_probs, labeled, unlabeled, test, psuedolabeler, epochs)
}

.selftrain <- function(formula, model, predict_probs, labeled, unlabeled, test, .pseudolabeler, epochs = 10) {
  lhs <- as.character(formula)[2]
  data <- .prepare_data(formula, labeled, unlabeled, test)
  cat(sprintf(
    "Training on %d labeled and %d unlabeled...\n",
    nrow(labeled),
    nrow(unlabeled)
  ))
  
  epoch <- 1
  while (nrow(data$unlabeled)>1 && epoch <= epochs) {
    m <- model(formula, data=data$labeled)
    data <- .pseudolabeler(formula, m, data, predict_probs)
    
    if (epoch == 1) {
      acc0 <- .accuracy_selftrain(formula, m, predict_probs, data$test)
      cat(sprintf("Starting accuracy %0.1f.\n", 100*acc0))
    }
    cat(sprintf("After epoch %d, %d unlabeled points remain.\n", epoch, nrow(data$unlabeled)))
    epoch <- epoch + 1
  }
  
  acc <- .accuracy_selftrain(formula, m, predict_probs, data$test)
  cat(sprintf("Final accuracy %0.1f (%+0.1f).\n", 100*acc, 100*(acc-acc0)))
  
  invisible(list(
    model = m,
    pseudolabeled = labeled,
    remainder = unlabeled,
    acc_diff = acc-acc0
  ))
}


#' Conduct co-training
#' 
#' Create two copies of the data. Fit one model on each copy of the labeled 
#' data. Use model 1 to predict classes for unlabeled data 2. Add the predictions
#' to labeled data 2 if they have high confidence. Do the same for model 2 and
#' data 1. Repeat.
#' 
#' @param model1 A function taking formula and data as its first two args and 
#' returning a fitted model.
#' @param predict_probs1 A function taking a fitted model and a new data.frame
#'   and returning a matrix of fitted class probabilities
#' @param model2 A function taking formula and data as its first two args and 
#' returning a fitted model.
#' @param predict_probs2 A function taking a fitted model and a new data.frame
#'   and returning a matrix of fitted class probabilities
#' @inheritParams selftrain
#'
#' @return Invisibly, a list containing the final fitted models, the
#'   pseudolabelled data and the remainders
cotrain_alpha <- function(
    formula, 
    model1, predict_probs1, 
    model2, predict_probs2,
    labeled, unlabeled, test,
    alpha = 0.9,
    epochs = 10
) {
  psuedolabeler <- function(...) .pseudolabel_alpha(..., alpha = alpha)
  .cotrain(
    formula, 
    model1, predict_probs1, 
    model2, predict_probs2,
    labeled, unlabeled, test,
    psuedolabeler,
    epochs
  )
}

cotrain_n <- function(
    formula, 
    model1, predict_probs1, 
    model2, predict_probs2,
    labeled, unlabeled, test,
    n,
    epochs = 10
) {
  psuedolabeler <- function(...) .pseudolabel_n(..., n = n)
  .cotrain(
    formula, 
    model1, predict_probs1, 
    model2, predict_probs2,
    labeled, unlabeled, test,
    psuedolabeler,
    epochs
  )
}

.cotrain <- function(
    formula, 
    model1, predict_probs1, 
    model2, predict_probs2,
    labeled, unlabeled, test,
    .pseudolabeler,
    epochs
) {
  lhs <- as.character(formula)[2]

  data1 <- .prepare_data(formula, labeled, unlabeled, test)
  data2 <- .prepare_data(formula, labeled, unlabeled, test)
  cat(sprintf(
    "Training on %d labeled and %d unlabeled...\n",
    nrow(labeled),
    nrow(unlabeled)
  ))
  
  epoch <- 1
  while (nrow(data1$unlabeled)>1 && nrow(data2$unlabeled)>1 && epoch <= epochs) {
    m1 <- model1(formula, data=data1$labeled)
    m2 <- model2(formula, data=data2$labeled)
    
    data1 <- .pseudolabeler(formula, m2, data1, predict_probs2)
    data2 <- .pseudolabeler(formula, m1, data2, predict_probs1)
    
    if (epoch == 1) {
      acc0 <- .accuracy_cotrain(formula, m1, predict_probs1, m2, predict_probs2, data1$test)
      cat(sprintf("Starting accuracy %0.1f.\n", 100*acc0))
    }
    cat(sprintf(
      "After epoch %d, (%d, %d) unlabeled points remain.\n",
      epoch,
      nrow(data1$unlabeled),
      nrow(data2$unlabeled)
    ))
    epoch <- epoch + 1
  }
  
  acc <- .accuracy_cotrain(formula, m1, predict_probs1, m2, predict_probs2, data1$test)
  cat(sprintf("Final accuracy %0.1f (%+0.1f).\n", 100*acc, 100*(acc-acc0)))
  
  invisible(list(
    model1 = m1,
    model2 = m2,
    pseudolabeled1 = data1$labeled,
    pseudolabeled2 = data2$labeled,
    remainder1 = data1$unlabeled,
    remainder2 = data2$unlabeled,
    acc_diff = acc-acc0
  ))
}



# Internal ----------------------------------------------------------------

.accuracy_selftrain <- function(formula, model, predict_probs, df) {
  lhs <- as.character(formula)[2]
  lvls <- levels(df[[lhs]])
  
  pred <- predict_probs(model, df)
  pred <- lvls[apply(pred, 1, which.max)] # Class that maximises avg pred prob
  pred <- factor(pred)
  
  mean(pred == df[[lhs]])
}

.accuracy_cotrain <- function(
    formula, 
    model1, predict_probs1, 
    model2, predict_probs2,
    df
) {
  lhs <- as.character(formula)[2]
  lvls <- levels(df[[lhs]])
  
  pred1 <- predict_probs1(model1, df)
  pred2 <- predict_probs2(model2, df)
  pred <- (pred1 + pred2) / 2
  pred <- lvls[apply(pred, 1, which.max)] # Class that maximises avg pred prob
  pred <- factor(pred)
  
  mean(pred == df[[lhs]])
}

.prepare_data <- function(formula, labeled, unlabeled, test) {
  lhs <- as.character(formula)[2]
  
  test[[lhs]] <- factor(test[[lhs]])
  labeled[[lhs]] <- factor(labeled[[lhs]])
  unlabeled[[lhs]] <- NA
  
  list(
    labeled = labeled,
    unlabeled = unlabeled,
    test = test
  )
}

.pseudolabel_alpha <- function(formula, fit, data, predict_probs, alpha) {
  lhs <- as.character(formula)[2]
  lvls <- levels(data$labeled[[lhs]])
  
  pred <- predict_probs(fit, data$unlabeled) > alpha
  for (i in seq_along(lvls)) { 
    data$unlabeled[[lhs]][pred[,i]] <- lvls[i]
  }
  
  still_unlabeled <- is.na(data$unlabeled[[lhs]])
  data$labeled <- rbind(data$labeled, data$unlabeled[!still_unlabeled, ])
  data$unlabeled <- data$unlabeled[still_unlabeled, ]
  
  data
}

.pseudolabel_n <- function(formula, fit, data, predict_probs, n) {
  lhs <- as.character(formula)[2]
  lvls <- levels(data$labeled[[lhs]])
  
  pred <- predict_probs(fit, data$unlabeled)
  for (i in seq_along(lvls)) {
    m <- min(n, nrow(pred))
    idx <- order(pred[,i], decreasing = TRUE)[1:m]
    data$unlabeled[[lhs]][idx] <- lvls[i]
  }
  
  still_unlabeled <- is.na(data$unlabeled[[lhs]])
  data$labeled <- rbind(data$labeled, data$unlabeled[!still_unlabeled, ])
  data$unlabeled <- data$unlabeled[still_unlabeled, ]
  
  data
}

