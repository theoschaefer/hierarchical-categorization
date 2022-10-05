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


tbl_train %>% group_by(participant, d1i, d2i, category) %>%
  count()
# all category A and category B stimuli were seen 17 times during training
# we can therefore use all distinct category exemplars only once
# when computing similarities towards within-category exemplars as 17 cancels out

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
tbl_closest$d1i_z * sd_d1i + mean_d1i


max_sim_responses <- function(i_cat, i_dim, post_c, l_tbl_lookup, l_tbl_exemplars, l_tbl_cues = NULL) {
  #' iterates over max_sim_response for all possible cue values
  #' on the given dimension
  #' 
  #' @description compute maximally similar responses for given
  #' category and dimension
  #' 
  #' @param i_cat category value
  #' @param i_dim dimension value
  #' @param post_c posterior samples of gcm c parameter
  #' @param l_tbl_lookup tbl df containing lookup table with
  #' grid of possible response values on both dimensions
  #' @param l_tbl_exemplars tbl df with all exemplars encountered
  #' during category learning
  #' @param l_tbl_cues list containing tbl dfs with inference cues;
  #' optional, if nothing is provided, just uses all possible cues 
  #' from the category learning training data (all available x1s and x2s)
  #' @return tbl df with maximally similar responses for all cues
  #'   
  # iterate over all cues from the given dimension
  if (!is.null(l_tbl_cues)) l_tbl_range <- l_tbl_cues
  if (is.null(l_tbl_cues)) l_tbl_range <- l_tbl_exemplars
  i_cue <- seq(1, nrow(l_tbl_range[[i_cat]]), by = 1)
  map(
    i_cue, 
    max_sim_response,
    post_c = post_c,
    l_tbl_lookup = l_tbl_lookup,
    l_tbl_exemplars = l_tbl_exemplars,
    i_cat = i_cat,
    i_dim = i_dim,
    l_tbl_cues = l_tbl_cues
  ) %>% reduce(rbind)
}

max_sim_response <- function(
    i_cue, i_cat, i_dim, post_c, l_tbl_lookup, l_tbl_exemplars, l_tbl_cues = NULL
) {
  #' @description compute maximally similar response 
  #' for given category, dimension, and cue
  #' 
  #' @param i_cue cue value
  #' @param i_cat category value
  #' @param i_dim dimension value
  #' @param post_c posterior samples of gcm c parameter
  #' @param l_tbl_lookup tbl df containing lookup table with
  #' grid of possible response values on both dimensions
  #' @param l_tbl_exemplars tbl df with all exemplars encountered
  #' during category learning
  #' @param l_tbl_cues list containing tbl dfs with inference cues;
  #' optional, if nothing is provided, just uses all possible cues 
  #' from the category learning training data (all available x1s and x2s)
  #' @return one row tbl df with maximally similar response
  #' 
  if (!is.null(l_tbl_cues)) {
    cue = l_tbl_cues[[i_cat]][i_cue, i_dim] %>% as_vector()
  } else {
    cue <- l_tbl_exemplars[[i_cat]][i_cue, i_dim] %>% as_vector()
  }
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


varied_cues <- function(tbl_df) {
  #' @description cross cue-values from one dimension with fine grid
  #' of values from other dimension
  #' 
  #' @param tbl_df tbl df with unique cue values from 1D and NAs on other dimension
  #' @return tbl df with seen cues crossed with fine grid from other dimension
  #' 
  mask <- which(!is.na(tbl_df[1,])) %>% as_vector()
  tbl_df_filtered <- tbl_df[, mask]
  dim_cued <- names(tbl_df_filtered)[2]
  if (dim_cued == "d1i_z") dim_response = "d2i_z"
  if (dim_cued == "d2i_z") dim_response = "d1i_z"
  
  tbl_cross <- crossing(
    dim_cued = c(unique(tbl_df_filtered[, dim_cued]) %>% as_vector(), seq(min(tbl_df_filtered[, dim_cued]), max(tbl_df_filtered[, dim_cued]), by = .1)),
    dim_response = seq(-1.5, 1.5, by = .1)
  )
  names(tbl_cross) <- c(dim_cued, dim_response)
  return(tbl_cross)
}


tbl_completion_prep <- read_csv(file = "data/infpro_task-cat_beh/sub-all_task-inf_beh-distances.csv")
cols_required <- c("participant", "category", "rep", "cuedim", "cue_val", "respdim", "resp_i")
tbl_completion <- tbl_completion_prep[, cols_required] %>%
  group_by(participant, category, cuedim, respdim, cue_val, rep) %>%
  mutate(rwn = row_number(resp_i)) %>%
  filter(rwn == 1) %>% ungroup()



load_parameter_posteriors <- function(p_id) {
  #' @description load posterior samples from gcm and gaussian nb for one participant
  #'
  file_loc_gcm <- str_c("data/infpro_task-cat_beh/models/gcm-model-", p_id, ".RDS")
  file_loc_gaussian <- str_c(
    "data/infpro_task-cat_beh/models/gaussian-model-", p_id, ".RDS"
  )
  m_gcm <- readRDS(file_loc_gcm)
  m_pt <- readRDS(file_loc_gaussian)
  post_c <- m_gcm$draws(variables = "c", format = "df") %>% as_tibble()
  post_pts <- m_pt$draws(variables = c("mu1", "mu2"), format = "df")
  
  return(list(gcm = post_c, gaussian = post_pts))
}


model_based_inference_responses <- function(tbl_completion, tbl_train, p_id) {
  #' model-based inferences given gcm model posteriors
  #' 
  #' @description computes inference responses that maximize within-category
  #' similarity given posterior samples from c parameter of gcm
  #' 
  #' @param tbl_completion empirical completion data
  #' @param tbl_train empirical category learning data from training
  #' @param p_id participant id

  #' @return list containing tbl dfs with maximally similar responses
  #' for all categories and cues
  #'  
  
  # all the exemplars observed during training that can be referred to in memory
  l_tbl_exemplars <- tbl_train %>% filter(participant == p_id) %>%
    group_by(category, d1i_z, d2i_z) %>%
    count() %>% select(-n) %>%
    split(.$category)

  # load posterior samples
  l_posteriors <- load_parameter_posteriors(p_id)
  post_c <- l_posteriors$gcm
  
  # get unique inference cues and bring them in format to compute model-based responses
  l_tbl_cues <- tbl_completion %>%
    mutate(cue_val_dupl = cue_val, cuedim_dupl = cuedim) %>%
    pivot_wider(
      id_cols = c(participant, category, cuedim_dupl, cue_val_dupl, respdim, rep, resp_i),
      names_from = cuedim, values_from = cue_val, names_glue = "{cuedim}_{.value}"
    ) %>%
    mutate(
      d1i_z = (`1_cue_val` - mean_d1i) / sd_d1i, 
      d2i_z = (`2_cue_val` - mean_d2i) / sd_d2i,
    ) %>% 
    filter(participant == p_id) %>%
    select(-c(
      participant, cuedim_dupl, cue_val_dupl, respdim, 
      rep, resp_i, `1_cue_val`, `2_cue_val`
    )) %>% 
    group_by(category, d1i_z, d2i_z) %>%
    count() %>% select(-n) %>%
    split(.$category)
  
  # expand the not-varied dimension as a grid for a given participant
  l_tbl_lookup <- map(l_tbl_cues, varied_cues)

  tbl_cat_dim <- crossing(i_cat = c("A", "B"), i_dim = names(l_tbl_lookup$A)[1])
  
  l_closest <- pmap(
    tbl_cat_dim, 
    max_sim_responses, 
    post_c = post_c,
    l_tbl_lookup = l_tbl_lookup,
    l_tbl_exemplars = l_tbl_exemplars,
    l_tbl_cues = l_tbl_cues
  )
  
  return(l_closest)
  
}

