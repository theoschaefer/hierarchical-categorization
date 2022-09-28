library(tidyverse)
library(cmdstanr)
library(rutils)
library(ggrepel)
library(grid)
library(gridExtra)
library(furrr)
library(loo)

utils_loc <- c("R/utils/plotting-utils.R", "R/utils/utils.R")
walk(utils_loc, source)

# Load Data and Preprocess Them -------------------------------------------

tbl_both <- readRDS(file = "data/infpro_task-cat_beh/tbl_both.RDS")
tbl_train <- readRDS(file = "data/infpro_task-cat_beh/tbl_train.RDS")
tbl_transfer <- readRDS(file = "data/infpro_task-cat_beh/tbl_transfer.RDS")

# params to revert back to untransformed space
mean_d1i <- mean(tbl_both$d1i)
sd_d1i <- sd(tbl_both$d1i)
mean_d2i <- mean(tbl_both$d2i)
sd_d2i <- sd(tbl_both$d2i)


p_id <- 101

file_loc_gcm <- str_c("data/infpro_task-cat_beh/models/gcm-model-", p_id, ".RDS")
file_loc_gaussian <- str_c(
  "data/infpro_task-cat_beh/models/gaussian-model-", p_id, ".RDS"
)
m_gcm <- readRDS(file_loc_gcm)
m_pt <- readRDS(file_loc_gaussian)
post_c <- m_gcm$draws(variables = "c", format = "df") %>% as_tibble()
post_pts <- m_pt$draws(variables = c("mu1", "mu2"), format = "df")

# all the exemplars observed during training that can be referred to in memory
l_tbl_exemplars <- tbl_train %>% filter(participant == p_id) %>%
  group_by(category, d1i_z, d2i_z) %>%
  count() %>% select(-n) %>%
  split(.$category)

# this has to be replaced once completion data are available
# as exemplars will contain the cue from one dimension
# plus a fine grid over plausible category values from the second dimension

l_tbl_lookup <- map2(
  l_tbl_exemplars, names(l_tbl_exemplars),
  ~ crossing(
    category = ..2,
    d1i_z = c(unique(..1$d1i_z), seq(min(..1$d1i_z), max(..1$d1i_z), by = .1)),
    d2i_z = c(unique(..1$d2i_z), seq(min(..1$d2i_z), max(..1$d2i_z), by = .1))
  )
)

# iterate over A & C categories aka target categories
ids_cat <- c("A", "C")
# iterate over cue dimensions
ids_dim <- c("d1i_z", "d2i_z")
tbl_cat_dim <- crossing(i_cat = ids_cat, i_dim = ids_dim)

l_closest <- pmap(
  tbl_cat_dim, 
  max_sim_responses, 
  l_tbl_lookup = l_tbl_lookup,
  l_tbl_exemplars = l_tbl_exemplars
)
tbl_closest <- reduce(l_closest, rbind)

max_sim_responses <- function(i_cat, i_dim, l_tbl_lookup, l_tbl_exemplars) {
  # iterate over all cues from the given dimension
  i_cue <- seq(1, nrow(l_tbl_exemplars[[i_cat]]), by = 1)
  map(
    i_cue, 
    max_sim_response,
    l_tbl_lookup = l_tbl_lookup,
    l_tbl_exemplars = l_tbl_exemplars,
    i_cat = i_cat,
    i_dim = i_dim
  ) %>% reduce(rbind)
}



max_sim_response <- function(
    i_cue, i_cat, i_dim, l_tbl_lookup, l_tbl_exemplars 
) {
  
  cue <- l_tbl_exemplars[[i_cat]][i_cue, i_dim] %>% as_vector()
  # then compute the similarities to all exemplars from all available values on the grid
  rows_selected <- as.logical(l_tbl_lookup[[i_cat]][, i_dim] == cue)
  grid_vals <- l_tbl_lookup[[i_cat]][rows_selected, ]
  
  sum_within_category_similarity <- function(x1, x2, c, tbl_df) {
    sum(map_dbl(c, ~ sum(exp(-.x^2*((x1 - tbl_df$d1i_z)^2 + (x2 - tbl_df$d2i_z)^2)))))
  }
  
  idx_min <- which.max(map2_dbl(
    grid_vals$d1i_z, grid_vals$d2i_z, 
    sum_within_category_similarity, 
    c = as.list(post_c$c),
    tbl_df = l_tbl_exemplars[[i_cat]]
  ))
  
  return(tibble(grid_vals[idx_min, ], cue_dim = i_dim))
}




