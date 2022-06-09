make_stimuli <- function(l_info) {
  #' create stimuli from 2D feature space
  #' 
  #' @param l_info list with parameters
  #' @return a list with a tbl containing the stimulus set and 
  #' the parameter list with the added infos
  #' 
  l_info$space_edges <- c(0, sqrt(l_info$n_stimuli) - 1)
  x1 <- seq(l_info$space_edges[1], l_info$space_edges[2], by = 1)
  x2 <- seq(l_info$space_edges[1], l_info$space_edges[2], by = 1)
  features <- crossing(x1, x2)
  tbl <- tibble(stim_id = seq(1, nrow(features)), features)
  if (l_info$category_shape == "squares") {
    tbl <- create_categories(tbl, sqrt(l_info$n_categories)) %>% 
      select(-c(x1_cat, x2_cat))
  } else if (l_info$category_shape == "ellipses") {
    l <- create_ellipse_categories(tbl, l_info$n_categories)
    tbl <- l[[1]]
    l_info$tbl_ellipses <- l[[2]]
  }
  
  l_info$feature_names <- c("x1", "x2")
  l_info$label <- "category"
  tbl$category <- as.factor(tbl$category)
  l_info$categories <- levels(tbl$category)
  tbl$cat_type <- l_info$cat_type
  tbl <- tbl %>% mutate(observed = 1) %>%
    pivot_wider(
      names_from = category, values_from = observed,
      names_sort = TRUE, names_prefix = "cat"
    ) %>%
    mutate(category = tbl$category)
  tbl[is.na(tbl)] <- 0
  if (l_info$is_reward) {
    tbl$prior_sd <- l_info$prior_sd
  }
  
  return(list(tbl, l_info))
}


create_ellipse_categories <- function(tbl, n_categories) {
  #' create ellipse categories from feature space
  #' 
  #' @description create one ellipse or three ellipses within feature space
  #' creating different categories; assign all points to those one/three
  #' categories or to a baseline category (i.e., points not within any
  #' of the ellipses)
  #' @param tbl \code{tibble} containing each of the two features in a column
  #' @param n_categories \code{integer} stating how many categories to create
  #' @return a list with the \code{tibble} with an added column stating 
  #' the category and another \code{tibble} with the ellipse contours
  #' 
  thxs <- c(0, apply(tbl[, c("x1")], 2, function(x) (min(x) + max(x))/2))
  theta_deg <- 45
  fctr_mid <- list(
    "squash_all" = .9, "squash_y" = 1, "squash_x" = .45, 
    "move_x" = 0, "move_y" = 0, "category" = 2
  )
  fctr_hi <- list(
    "squash_all" = .85, "squash_y" = .75, "squash_x" = .225,
    "move_x" = 3, "move_y" = -3, "category" = 3
  )
  fctr_lo <- list(
    "squash_all" = .85, "squash_y" = .75, "squash_x" = .225,
    "move_x" = -3, "move_y" = 3, "category" = 4
  )
  fctr_mid_hi <- list(
    "squash_all" = .9, "squash_y" = .75, "squash_x" = .45, 
    "move_x" = -1.5, "move_y" = 1.5, "category" = 2
  )
  fctr_mid_lo <- list(
    "squash_all" = .9, "squash_y" = .75, "squash_x" = .45, 
    "move_x" = 1.5, "move_y" = -1.5, "category" = 3 
  )
  if (n_categories == 4) {
    l_map <- list(fctr_mid, fctr_hi, fctr_lo)
  } else if (n_categories == 3) {
    l_map <- list(fctr_mid_hi, fctr_mid_lo)
  } else if (n_categories == 2) {
    l_map <- list(fctr_mid)
  }
  
  
  l <- map(l_map, assign_grid_points, tbl = tbl, thxs = thxs, theta_deg = theta_deg)
  tbl_all_cats <- reduce(
    map(l, 2), function(x, y) inner_join(x, y, by = c("stim_id", "x1", "x2"))
  )
  tbl_ellipses <- reduce(map(l, 1), rbind)
  cat_tmp <- apply(tbl_all_cats[, 4: ncol(tbl_all_cats)], 1, max)
  tbl_all_cats <- tbl_all_cats[, 1:3]
  tbl_all_cats$category <- as.factor(cat_tmp)
  
  return(list(tbl_all_cats, tbl_ellipses))
}

assign_grid_points <- function(fctrs, tbl, thxs, theta_deg) {
  #' define whether 2D points are inside or outside an ellipse
  #' 
  #' @description take all integer pairs from a 2D space and decide whether
  #' each pair is inside or outside a given ellipse
  #' @param fctrs squashing and moving of ellipse
  #' @param tbl tibble with 2D points (x1 and x2)
  #' @param thxs min and max vals on x and y axis
  #' @param theta_deg rotation angle in degrees
  #' @return list with tbl defining ellipse and tbl with all 2D points
  #' 
  tbl_ellipse <- ellipse(thxs, fctrs, theta_deg)
  # https://stackoverflow.com/questions/9508518/why-are-these-numbers-not-equal
  elementwise.all.equal <- Vectorize(function(x, y) {isTRUE(all.equal(x, y))})
  is_within_ellipse <- function(x1, x2, tbl_ellipse) {
    tryCatch(
      warning = function(cnd) FALSE,
      {
        y_range <- tbl_ellipse %>% filter(elementwise.all.equal(x_rotated, x1)) %>%
          summarize(y_min = min(y_rotated), y_max = max(y_rotated))
        return(between(x2, y_range$y_min, y_range$y_max))
      }
    )
  }
  in_ellipse <- pmap_lgl(tbl[, c("x1", "x2")], is_within_ellipse, tbl_ellipse = tbl_ellipse)
  tbl$category <- 1
  tbl$category[in_ellipse] <- fctrs[["category"]]
  tbl_ellipse$category <- fctrs[["category"]]
  return(list(tbl_ellipse, tbl))
}

sample_ellipse_space <- function(fctrs, n, thxs, theta_deg) {
  #' uniformly sample from ellipse
  #' 
  #' @description randomly sample from a space defined by an ellipse
  #' @param fctrs squashing and moving of ellipse
  #' @param n number of samples
  #' @param thxs min and max vals on x and y axis
  #' @param theta_deg rotation angle in degrees
  #' @return x and y values of samples
  #' 
  tbl_ellipse <- ellipse(thxs, fctrs, theta_deg)
  min_max <- apply(tbl_ellipse[, c("x_rotated", "y_rotated")], 2, function(x) c(min(x), max(x)))
  x1 <- round(runif(n, min_max[1, 1], min_max[2, 1]), 1)
  # https://stackoverflow.com/questions/9508518/why-are-these-numbers-not-equal
  elementwise.all.equal <- Vectorize(function(x, y) {isTRUE(all.equal(x, y))})
  
  sample_y_uniform <- function(x_val) {
    y_bounds <- tbl_ellipse %>% filter(elementwise.all.equal(x_rotated, x_val)) %>% 
      summarize(min_y = min(y_rotated), max_y = max(y_rotated)) %>% 
      as_vector() %>% sort() %>% unname()
    return(runif(1, y_bounds[1], y_bounds[2]))
  }
  x2 <- map_dbl(x1, sample_y_uniform)
  tbl <- tibble(x1, x2)
  return(tbl_ellipse)
}

ellipse <- function(thxs, fctrs, theta_deg) {
  #' create an ellipse and rotate it
  #' 
  #' @description create an ellipse within 2D range
  #' @param thxs min and max vals on x and y axis
  #' @param fctrs squashing and moving of ellipse
  #' @param theta_deg rotation angle in degrees
  #' @return fine grained 2D values of a rotated ellipse
  #' 
  x_prep <- seq(0, 2*pi, by = .01)
  tbl_circle <- tibble(
    x = thxs[2] * fctrs[["squash_all"]] * fctrs[["squash_x"]] * sin(x_prep),
    y = thxs[2] * fctrs[["squash_all"]] * fctrs[["squash_y"]] * cos(x_prep)
  )
  tbl_circle <- cbind(
    tbl_circle, 
    t(tbl_circle %>% as.matrix()) %>% rotate_points(theta_deg) %>% t() %>%
      round(1)
  ) %>% rename(x_rotated = `1`, y_rotated = `2`) %>% 
    mutate(
      x_rotated = x_rotated + thxs[2],
      y_rotated = y_rotated + thxs[2]
    )
  tbl_circle$x_rotated <- tbl_circle$x_rotated + fctrs[["move_x"]]
  tbl_circle$y_rotated <- tbl_circle$y_rotated + fctrs[["move_y"]]
  return(tbl_circle)
}



rotate_points <- function(x, theta_deg) {
  #' rotate 2D points in clockwise direction
  #' according to theta_deg (rotation angle in degrees)
  #' 
  #' @description rotate 2D points
  #' @param x a matrix of 2D points
  #' @param theta_deg rotation angle in degrees
  #' @return the rotate x matrix
  #' 
  theta_rad <- (theta_deg * pi) / 180
  theta_sin <- sin(theta_rad)
  theta_cos <- cos(theta_rad)
  m_rotate <- matrix(c(theta_cos, -theta_sin, theta_sin, theta_cos), ncol = 2, byrow = FALSE)
  x_rotated <- apply(x, 2, function(a) m_rotate %*% a)
  x_rotated
  return(x_rotated)
}
