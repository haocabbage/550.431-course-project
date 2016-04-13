trans <- function(m) {
  # transform values {to 0, 1, 2, 3}
  m_new <- as.vector(m)
  label_value <- sort(unique(m_new))
  for (i in 0:3) {
    m_new[m_new == label_value[i+1]] <- i
  }
  return(matrix(m_new, dim(m)[1], dim(m)[2]))  
}

to_zero <- function(m) {
  # transform black pixel value to zero
  m_new <- m
  m_new[m_new < 1] <- 0
  return(m_new)
}

error_rate <- function (m1, m2) {
  # compute the percentage of pixels that are mislabeled
  missed <- length(m2[m2 != m1])
  return(missed/length(m2[m2 != 0]))
}


