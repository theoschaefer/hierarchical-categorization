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



tbl_completion_prep <- read_csv(file = "data/infpro_task-cat_beh/sub-all_task-inf_beh-distances.csv")
cols_required <- c("participant", "category", "rep", "cuedim", "cue_val", "respdim", "resp_i")
tbl_completion <- tbl_completion_prep[, cols_required] %>%
  group_by(participant, category, cuedim, respdim, cue_val, rep) %>%
  mutate(rwn = row_number(resp_i)) %>%
  filter(rwn == 1) %>% ungroup()
