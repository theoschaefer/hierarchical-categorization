library(tidyverse)
library(cmdstanr)
library(rutils)
library(ggrepel)
library(gridExtra)

utils_loc <- c("R/utils/plotting-utils.R", "R/utils/utils.R")
walk(utils_loc, source)

file_loc_train <- "data/infpro_task-cat_beh/infpro_task-cat_beh.csv"
file_loc_transfer <- "data/infpro_task-cat_beh/infpro_task-cat2_beh.csv"
tbl_train <- read_csv(file_loc_train, show_col_types = FALSE)
tbl_transfer <- read_csv(file_loc_transfer, show_col_types = FALSE)

# define how many trials starting from the last trial should be analyzed
tbl_train %>% count(participant) %>% arrange(n)
n_last_trials <- 500

tbl_train_last <- tbl_train %>% group_by(participant) %>%
  mutate(
    rwn_fwd = row_number(block),
    rwn_bkwd = row_number(desc(rwn_fwd))
  ) %>% ungroup() %>%
  filter(rwn_bkwd <= n_last_trials)


# plot average accuracy across participants in train and transfer tests
pl_train <- plot_average_categorization_accuracy(tbl_train_last, "Train")
pl_tf <- plot_average_categorization_accuracy(tbl_transfer, "Transfer")
marrangeGrob(list(pl_train, pl_tf), ncol = 2, nrow = 1)


tbl_train_agg <- tbl_train %>% 
  group_by(participant, d1i, d2i, category) %>%
  summarize(
    n_trials = n(), 
    n_correct = sum(accuracy),
    prop_correct = n_correct / n_trials
  ) %>% ungroup()

participant_sample <- sample(unique(tbl_train_agg$participant), 1)
tbl_sample <- tbl_train_agg %>% filter(participant == participant_sample)



# GCM ---------------------------------------------------------------------


# compute pairwise distances
m_distances_x1 <- map(1:nrow(tbl_sample), distance_1d, tbl_sample, "d1i") %>% unlist() %>%
  matrix(byrow = TRUE, nrow = nrow(tbl_sample), ncol = nrow(tbl_sample))
m_distances_x2 <- map(1:nrow(tbl_sample), distance_1d, tbl_sample, "d2i") %>% unlist() %>%
  matrix(byrow = TRUE, nrow = nrow(tbl_sample), ncol = nrow(tbl_sample))


stan_gcm <- write_gcm_stan_file()
mod <- cmdstan_model(stan_gcm)

l_data <- list(
  n_stim = nrow(tbl_sample), n_trials = tbl_sample$n_trials, 
  n_correct = tbl_sample$n_correct, n_cat = length(unique(tbl_sample$category)),
  cat = as.numeric(factor(tbl_sample$category, labels = c(1, 2, 3))),
  d1 = m_distances_x1, d2 = m_distances_x2
)


fit_gcm <- mod$sample(
  data = l_data, iter_sampling = 500, iter_warmup = 1000, chains = 1
  )

pars_interest <- c("theta", "b", "c", "w")
tbl_draws <- fit_gcm$draws(variables = pars_interest, format = "df")
idx_params <- map(pars_interest, ~ str_starts(names(tbl_draws), .x)) %>%
  reduce(rbind) %>% colSums()
names_params <- names(tbl_draws)[as.logical(idx_params)]
tbl_posterior <- tbl_draws[, c(all_of(names_params), ".chain")] %>% 
  rename(chain = .chain) %>%
  pivot_longer(all_of(names_params), names_to = "parameter", values_to = "value")
kd <- rutils::estimate_kd(tbl_posterior, names_params)
l <- sd_bfs(tbl_posterior, names_params, sqrt(2)/4)
map(names_params, plot_posterior, tbl = tbl_posterior, tbl_thx = l[[2]])


tbl_summary <- fit_gcm$summary(variables = pars_interest)
names_thetas <- names(tbl_draws)[startsWith(names(tbl_draws), "theta")]
tbl_sample$pred_theta <- colMeans(tbl_draws[, names_thetas])


ggplot(tbl_sample, aes(pred_theta, prop_correct)) +
  geom_point() +
  geom_abline() +
  theme_bw() +
  labs(x = "Predicted Theta", y = "Proportion Correct")
