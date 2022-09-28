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
c <- m_gcm$draws(variables = "c", format = "df") %>% as_tibble()
pts <- m_pt$draws(variables = c("mu1", "mu2"), format = "df")

# all the exemplars observed during training
l_tbl_exemplars <- tbl_train %>% filter(participant == p_id) %>%
  group_by(category, d1i_z, d2i_z) %>%
  count() %>% select(-n) %>%
  split(.$category)

# this has to be replaced once completion data are available
tbl_cues <- crossing(x1 = unique(l_tbl_e))

l_tbl_cues <- map2(
  l_tbl_exemplars, names(l_tbl_exemplars),
  ~ crossing(
    category = ..2,
    d1i_z = unique(..1$d1i_z),
    d2i_z = unique(..1$d2i_z)
  )
)

