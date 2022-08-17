library(tidyverse)
library(cmdstanr)
library(rutils)
library(ggrepel)
library(grid)
library(gridExtra)

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

stan_gcm <- write_gcm_stan_file()
mod_gcm <- cmdstan_model(stan_gcm)

tbl_participant <- l_tbl_train_agg[[1]]

bayesian_gcm <- function(tbl_participant, l_stan_params, mod_gcm) {
  
  participant_sample <- tbl_participant$participant[1]
  tbl_gcm <- tbl_participant %>% 
    filter(category == response) %>% 
    mutate(prop_correct = prop_responses)
  
  # compute pairwise distances
  m_distances_x1 <- map(1:nrow(tbl_gcm), distance_1d, tbl_gcm, "d1i_z") %>% unlist() %>%
    matrix(byrow = TRUE, nrow = nrow(tbl_gcm), ncol = nrow(tbl_gcm))
  m_distances_x2 <- map(1:nrow(tbl_gcm), distance_1d, tbl_gcm, "d2i_z") %>% unlist() %>%
    matrix(byrow = TRUE, nrow = nrow(tbl_gcm), ncol = nrow(tbl_gcm))
  
  l_data <- list(
    n_stim = nrow(tbl_gcm), n_trials = tbl_gcm$n_trials, 
    n_correct = tbl_gcm$n_responses, n_cat = length(unique(tbl_gcm$category)),
    cat = as.numeric(factor(tbl_gcm$category, labels = c(1, 2, 3))),
    d1 = m_distances_x1, d2 = m_distances_x2,
    n_correct_predict = tbl_gcm$n_responses,
    n_trials_per_item = tbl_gcm$n_trials
  )
  
  fit_gcm <- mod_gcm$sample(
    data = l_data, chains = l_stan_params$n_chains, 
    iter_sampling = l_stan_params$n_samples, iter_warmup = l_stan_params$n_warmup
  )
  
  file_loc <- str_c("data/infpro_task-cat_beh/gcm-model-", participant_sample, ".RDS")
  fit_gcm$save_object(file = file_loc)
  
  pars_interest <- c("theta", "bs", "c", "w")
  pars_interest_no_theta <- c("bs", "c", "w")
  tbl_draws <- fit_gcm$draws(variables = pars_interest, format = "df")
  
  names_thetas <- names(tbl_draws)[startsWith(names(tbl_draws), "theta")]
  tbl_gcm$pred_theta <- colMeans(tbl_draws[, names_thetas])
  tbl_gcm$pred_difference <- tbl_gcm$pred_theta - tbl_gcm$prop_responses
  
  
  tbl_summary <- fit_gcm$summary(variables = pars_interest)
  tbl_summary_nok <- tbl_summary %>% filter(rhat > 1.02 | rhat < 0.98)
  if (nrow(tbl_summary_nok) > 0) {
    stop(str_c(
      "participant = ", participant_sample, "; Rhat for some parameters not ok",
      "model can be found under: ", file_loc
    ))
  }
  
  loo_gcm <- fit_gcm$loo(variables = "log_lik_pred")
  
  idx_no_theta <- map(pars_interest_no_theta, ~ str_detect(tbl_summary$variable, .x)) %>%
    reduce(rbind) %>% colSums()
  tbl_label <- tbl_summary[as.logical(idx_no_theta), ]
  
  tbl_posterior <- tbl_draws %>% 
    select(starts_with(pars_interest_no_theta), .chain) %>%
    rename(chain = .chain) %>%
    pivot_longer(starts_with(pars_interest_no_theta), names_to = "parameter", values_to = "value") %>%
    filter(parameter != "chain")
  tbl_posterior$parameter <- fct_inorder(tbl_posterior$parameter)
  
  pl_thetas <- plot_item_thetas(tbl_gcm, str_c("GCM; Participant = ", participant_sample))
  pl_posteriors <- plot_posteriors(tbl_posterior)
  pl_pred_uncertainty <- plot_proportion_responses(tbl_gcm, color_pred_difference = TRUE)
  
  # save plots
  c_names <- function(x, y) str_c("data/infpro_task-cat_beh/model-plots/", x, y, ".png")
  l_pl_names <- map(c("gcm-thetas-", "gcm-posteriors-", "gcm-uncertainty-"), c_names, y = participant_sample)
  l_pl <- list(pl_thetas, pl_posteriors, pl_pred_uncertainty)
  l_vals_size <- list(c(3, 3), c(5, 5), c(5.5, 5.5))
  pwalk(list(l_pl, l_pl_names, l_vals_size), save_my_png)
  
  return(loo_gcm)
}

save_my_png <- function(pl, f_name, vals_size) {
  png(filename = f_name, vals_size[1], vals_size[2], "in", res = 200)
  grid.draw(pl)
  dev.off()
}
