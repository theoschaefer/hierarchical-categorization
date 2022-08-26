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


file_loc_train <- "data/infpro_task-cat_beh/infpro_task-cat_beh.csv"
file_loc_transfer <- "data/infpro_task-cat_beh/infpro_task-cat2_beh.csv"
tbl_train <- read_csv(file_loc_train, show_col_types = FALSE)
tbl_transfer <- read_csv(file_loc_transfer, show_col_types = FALSE)
colnames(tbl_transfer) <- str_replace(colnames(tbl_transfer), "cat2", "cat")
tbl_train$session <- "train"
tbl_transfer$session <- "transfer"

tbl_both <- tbl_train  %>% 
  rbind(tbl_transfer) %>%
  mutate(
    d1i_z = scale(d1i)[, 1],
    d2i_z = scale(d2i)[, 1],
    response_int = as.numeric(factor(response))
    
  )
tbl_train <- tbl_both %>% filter(session == "train")
tbl_transfer <- tbl_both %>% filter(session == "transfer")

tbl_stim_id <- tbl_train %>% count(d1i, d2i, d1i_z, d2i_z, category) %>%
  arrange(d1i, d2i) %>% mutate(stim_id = seq_along(d1i + d2i)) %>%
  select(-n)
tbl_stim_id_transfer <- tbl_transfer %>% count(d1i, d2i, d1i_z, d2i_z, category) %>%
  arrange(d1i, d2i) %>% mutate(stim_id = seq_along(d1i + d2i)) %>%
  select(-n)
tbl_train <- tbl_train %>% 
  left_join(tbl_stim_id, by = c("d1i", "d2i", "d1i_z", "d2i_z", "category")) %>%
  relocate(stim_id, .before = d1i)
tbl_transfer <- tbl_transfer %>%
  left_join(tbl_stim_id_transfer, by = c("d1i", "d2i", "d1i_z", "d2i_z", "category")) %>%
  relocate(stim_id, .before = d1i)

# define how many trials starting from the last trial should be analyzed
n_last_trials <- 500

tbl_train_last <- tbl_train %>% group_by(participant) %>%
  mutate(
    rwn_fwd = row_number(block),
    rwn_bkwd = row_number(desc(rwn_fwd))
  ) %>% ungroup() %>%
  filter(rwn_bkwd <= n_last_trials) %>%
  dplyr::select(-c(rwn_fwd, rwn_bkwd))

tbl_both <- rbind(tbl_train_last, tbl_transfer)

# Plot Overall Proportion Responses By Stimulus and Category --------------

# only correct responses
pl_train <- plot_average_categorization_accuracy(tbl_train_last, "Train")
pl_tf <- plot_average_categorization_accuracy(tbl_transfer, "Transfer")
marrangeGrob(list(pl_train, pl_tf), ncol = 2, nrow = 1)

tbl_train_agg <- aggregate_by_stimulus_and_response(tbl_stim_id, tbl_train_last)
tbl_transfer_agg <- aggregate_by_stimulus_and_response(tbl_stim_id_transfer, tbl_transfer)
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
  tbl_train_agg_overall %>% 
    mutate(response = str_c("Response = ", response)) %>%
    filter(prop_responses > .025),
  facet_by_response = TRUE
)

tbl_train_agg$response_int <- as.numeric(factor(tbl_train_agg$response))
tbl_transfer_agg$response_int <- as.numeric(factor(tbl_transfer_agg$response))


l_stan_params <- list(
  n_samples = 2000,
  n_warmup = 1000,
  n_chains = 3
)


n_workers_available <- parallel::detectCores()
plan(multisession, workers = n_workers_available / 2)


# GCM ---------------------------------------------------------------------

tbl_both_agg <- rbind(tbl_train_agg, tbl_transfer_agg)

l_tbl_both_agg <- split(tbl_both_agg, tbl_both_agg$participant)
tbl_participant <- l_tbl_both_agg[["101"]]

stan_gcm <- write_gcm_stan_file_predict()
mod_gcm <- cmdstan_model(stan_gcm)
safe_gcm <- safely(bayesian_gcm)

options(warn = -1)
l_loo_gcm <- furrr::future_map(
  l_tbl_both_agg, safe_gcm, 
  l_stan_params = l_stan_params, 
  mod_gcm = mod_gcm, 
  .progress = TRUE
)
options(warn = 0)
saveRDS(l_loo_gcm, file = "data/infpro_task-cat_beh/gcm-loos.RDS")

# ok
l_gcm_results <- map(l_loo_gcm, "result")
# not ok
map(l_loo_gcm, "error") %>% reduce(c)


# Gaussian ----------------------------------------------------------------


l_tbl_both <- split(tbl_both, tbl_both$participant)
tbl_participant <- l_tbl_both[["101"]]
tbl_participant_agg <- l_tbl_both_agg[["101"]]

stan_gaussian <- write_gaussian_naive_bayes_stan()
mod_gaussian <- cmdstan_model(stan_gaussian)
safe_gaussian <- safely(bayesian_gaussian_naive_bayes)

l_loo_gaussian <- furrr::future_map2(
  l_tbl_both, l_tbl_both_agg, safe_gaussian, 
  l_stan_params = l_stan_params,
  mod_gaussian = mod_gaussian, 
  .progress = TRUE
)
saveRDS(l_loo_gaussian, file = "data/infpro_task-cat_beh/gaussian-loos.RDS")

# ok
l_gaussian_results <- map(l_loo_gaussian, "result")
# not ok
map(l_loo_gaussian, "error") %>% reduce(c)


# Multivariate Gaussian ---------------------------------------------------


stan_multi <- write_gaussian_multi_bayes_stan()
mod_multi <- cmdstan_model(stan_multi)
safe_multi <- safely(bayesian_gaussian_multi_bayes)

l_loo_multi <- furrr::future_map2(
  l_tbl_both, l_tbl_both_agg, safe_multi, 
  l_stan_params = l_stan_params,
  mod_multi = mod_multi, 
  .progress = TRUE
)
saveRDS(l_loo_multi, file = "data/infpro_task-cat_beh/multi-loos.RDS")

# ok
l_multi_results <- map(l_loo_multi, "result")
# not ok
map(l_loo_multi, "error") %>% reduce(c)



# Model Weights -----------------------------------------------------------

safe_weights <- safely(loo_model_weights)

l_loo_weights <- pmap(
  list(l_gcm_results, l_gaussian_results), #, l_multi_results 
  ~ safe_weights(list(..1, ..2)), #, , ..3
  method = "stacking"
)
l_loo_weights_results <- map(l_loo_weights, ~ .x$"result"[2])
v_weights <- l_loo_weights_results[map_lgl(l_loo_weights_results, ~ !is.null(.x))] %>% 
  unlist()
participants <- str_match(names(v_weights), "^([0-9]*).model2")[,2]
tbl_weights <- tibble(
  participant = participants,
  weight_prototype = v_weights
)
ggplot(tbl_weights, aes(weight_prototype)) + 
  geom_histogram(fill = "#66CCFF", color = "white") +
  theme_bw() +
  labs(x = "Model Weight Prototype Model", y = "Nr. Participants")



