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

# all the exemplars observed during training
l_tbl_exemplars <- tbl_train %>% filter(participant == p_id) %>%
  group_by(category, d1i_z, d2i_z) %>%
  count() %>% select(-n) %>%
  split(.$category)

# this has to be replaced once completion data are available
l_tbl_lookup <- map2(
  l_tbl_exemplars, names(l_tbl_exemplars),
  ~ crossing(
    category = ..2,
    d1i_z = seq(min(..1$d1i_z), max(..1$d1i_z), by = .1),
    d2i_z = seq(min(..1$d2i_z), max(..1$d2i_z), by = .1)
  )
)

# categories to be looked at: A & C aka target categories

# iterate over cue dimensions
cue_dimension <- c("d1i_z", "d2i_z")[1]
# iterate over all cues from that dimension
cue <- l_tbl_exemplars$A[1, cue_dimension] %>% as_vector()
# then compute the similarities to all exemplars from all available values on the grid
rows_selected <- as.logical(l_tbl_lookup$A[, cue_dimension] == cue)
grid_vals <- l_tbl_lookup$A[rows_selected, ]

sum_within_category_similarity <- function(x1, x2, c, tbl_df) {
  sum(map_dbl(c, ~ sum(exp(-.x^2*((x1 - tbl_df$d1i_z)^2 + (x2 - tbl_df$d2i_z)^2)))))
}

idx_min <- which.max(map2_dbl(
  grid_vals$d1i_z, grid_vals$d2i_z, 
  sum_within_category_similarity, 
  c = as.list(post_c$c),
  tbl_df = l_tbl_exemplars$A
))

grid_vals[idx_min, ]



