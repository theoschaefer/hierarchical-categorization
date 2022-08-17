library(tidyverse)
library(cmdstanr)
library(rutils)
library(ggrepel)
library(grid)
library(gridExtra)
library(furrr)

utils_loc <- c("R/utils/plotting-utils.R", "R/utils/utils.R")
walk(utils_loc, source)


# Load Data and Preprocess Them -------------------------------------------


file_loc_train <- "data/infpro_task-cat_beh/infpro_task-cat_beh.csv"
file_loc_transfer <- "data/infpro_task-cat_beh/infpro_task-cat2_beh.csv"
tbl_train <- read_csv(file_loc_train, show_col_types = FALSE)
tbl_transfer <- read_csv(file_loc_transfer, show_col_types = FALSE)

tbl_train <- tbl_train  %>% 
  mutate(
    d1i_z = scale(d1i)[, 1],
    d2i_z = scale(d2i)[, 1]
  )

tbl_stim_id <- tbl_train %>% count(d1i, d2i, d1i_z, d2i_z, category) %>%
  arrange(d1i, d2i) %>% mutate(stim_id = seq_along(d1i + d2i)) %>%
  select(-n)
tbl_train <- tbl_train %>% 
  left_join(tbl_stim_id, by = c("d1i", "d2i", "d1i_z", "d2i_z", "category")) %>%
  relocate(stim_id, .before = d1i)

# define how many trials starting from the last trial should be analyzed
n_last_trials <- 500

tbl_train_last <- tbl_train %>% group_by(participant) %>%
  mutate(
    rwn_fwd = row_number(block),
    rwn_bkwd = row_number(desc(rwn_fwd))
  ) %>% ungroup() %>%
  filter(rwn_bkwd <= n_last_trials)



# Plot Overall Proportion Responses By Stimulus and Category --------------

# only correct responses
pl_train <- plot_average_categorization_accuracy(tbl_train_last, "Train")
pl_tf <- plot_average_categorization_accuracy(tbl_transfer, "Transfer")
marrangeGrob(list(pl_train, pl_tf), ncol = 2, nrow = 1)

tbl_train_agg <- aggregate_by_stimulus_and_response(tbl_stim_id, tbl_train)
tbl_train_agg_overall <- tbl_train_agg %>%
  group_by(d1i, d2i, d1i_z, d2i_z, stim_id, category, response) %>%
  summarize(
    n_responses = sum(n_responses),
    n_trials = sum(n_trials)
  ) %>%
  mutate(prop_responses = n_responses / n_trials)

# all responses
participant_sample <- "Average of All"
plot_proportion_responses(
  tbl_train_agg_overall %>% mutate(response = str_c("Response = ", response)) %>%
    filter(prop_responses > .025),
  facet_by_response = TRUE
)


# todos
# three functions fitting gcm, naive gaussian classifier, and multivariate gaussian classifiers
# all of them should take the same arguments: by-participant tbl_df, l_stan_params (i.e., n samples etc.), 
# safely map over participant_ids
# functions should throw an error if rhat values are above 1.05
# functions should save a png of the main parameters

l_tbl_train_agg <- split(tbl_train_agg, tbl_train_agg$participant)
l_stan_params <- list(
  n_samples = 500,
  n_warmup = 500,
  n_chains = 1
)



# GCM ---------------------------------------------------------------------


stan_gcm <- write_gcm_stan_file()
mod_gcm <- cmdstan_model(stan_gcm)
safe_gcm <- safely(bayesian_gcm)

n_workers_available <- parallel::detectCores()
plan(multisession, workers = n_workers_available - 2)
l_loo_gcm <- furrr::future_map(l_tbl_train_agg, safe_gcm, l_stan_params = l_stan_params, mod_gcm = mod_gcm, .progress = TRUE)
# ok
l_gcm_results <- map(l_loo_gcm, "result")
# not ok
map(l_loo_gcm, "error") %>% reduce(c)

tbl_participant <- l_tbl_train_agg[[1]]



# Gaussian ----------------------------------------------------------------

stan_gaussian <- write_gaussian_naive_bayes_stan()
mod_gaussian <- cmdstan_model(stan_gaussian)

# todo
# write functions for gaussian naive bayes and for multivariate gaussian as for gcm
safe_gaussian <- safely()






