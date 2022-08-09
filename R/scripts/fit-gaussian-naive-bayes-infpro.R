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


tbl_train_agg <- tbl_train %>% 
  group_by(participant, d1i, d2i, category, response) %>%
  summarize(
    n_responses = n(),
    n_correct = sum(accuracy)
  ) %>% group_by(participant, d1i, d2i) %>%
  mutate(
    n_trials = sum(n_responses), 
    prop_responses = n_responses / n_trials) %>%
  ungroup()

tbl_sample <- tbl_train_agg %>% filter(participant == participant_sample)
tbl_sample <- tbl_sample %>% mutate(
  d1i_z = scale(d1i)[, 1],
  d2i_z = scale(d2i)[, 1]
)

ggplot(tbl_sample %>% mutate(category = str_c("True Cat. = ", category)), aes(d1i_z, d2i_z)) +
  geom_point(aes(size = prop_responses, color = category), show.legend = FALSE) +
  geom_label_repel(aes(label = round(prop_responses, 2)), size = 2.5) +
  #ggtitle(str_c("Participant = ", participant_sample)) +
  facet_wrap(~ response) +
  theme_bw() +
  labs(x = expr(x[1]), y = expr(x[2]))


# Gaussian Naive Bayes ----------------------------------------------------


write_gaussian_naive_bayes_stan <- function() {
  write_stan_file("
data {
 int D; //number of dimensions
 int K; //number of categories
 int N; //number of data
 array[N, K] int n_responses; //data
 matrix[N, D] y;
}

parameters {
 array[K] row_vector[D] mu; //category means
 array[K,D] real <lower=0> sigma; //variance
}

transformed parameters {
  array[N] vector[K] theta;
  matrix[N,K] lps1;
  matrix[N,K] lps2;
  
  for (n in 1:N){
     for (k in 1:K){
        lps1[n, k] = normal_lpdf(y[n,1] | mu[k][1], sigma[k][1]); 
        lps2[n, k] = normal_lpdf(y[n,2] | mu[k][2], sigma[k][2]);
     }
     for (k in 1:K){
        theta[n][k] = (exp(lps1[n,k]) + exp(lps2[n,k])) / (sum(exp(lps1[n,])) + sum(exp(lps2[n,])));
     }
  }
}

model {
  
 for(k in 1:K){
   for (d in 1:D) {
     sigma[k][d] ~ uniform(0.1, 5);
   }
 }
 mu[1, 1] ~ normal(-.5, .5);
 mu[1, 2] ~ normal(.5, .5);
 mu[2, 1] ~ normal(0, .5);
 mu[2, 2] ~ normal(0, .5);
 mu[3, 1] ~ normal(.5, .5);
 mu[3, 2] ~ normal(-.5, .5);
 
 for (n in 1:N){
  n_responses[n] ~ multinomial(theta[n]);
 }
}
")
}

tbl_sample_naive <- tbl_sample %>% 
  select(-c(n_correct, n_trials, prop_responses)) %>%
  pivot_wider(names_from = response, values_from = n_responses, values_fill = 0)

tbl_sample_naive_1 <- tbl_sample_naive[1, ]

l_data <- list(
  D = 2, K = length(unique(tbl_sample_naive$category)),
  N = nrow(tbl_sample_naive),
  n_responses = tbl_sample_naive[, c("A", "B", "C")] %>% 
    as.matrix(),
  y = tbl_sample_naive[, c("d1i_z", "d2i_z")] %>% as.matrix()
)

stan_naive <- write_gaussian_naive_bayes_stan()
mod_naive <- cmdstan_model(stan_naive)

fit_naive <- mod_naive$sample(
  data = l_data, iter_sampling = 5000, iter_warmup = 5000, chains = 1
)

file_loc <- str_c("data/infpro_task-cat_beh/gcm-model-", participant_sample, ".RDS")
fit_naive$save_object(file = file_loc)
pars_interest <- c("theta", "mu", "sigma")
tbl_draws <- fit_naive$draws(variables = pars_interest, format = "df")
tbl_summary <- fit_naive$summary(variables = pars_interest)
names_thetas <- names(tbl_draws)[startsWith(names(tbl_draws), "theta")]
tbl_sample_naive$pred_theta <- colMeans(tbl_draws[, names_thetas])

plot_item_thetas(tbl_sample_naive, "GCM")

tbl_posterior <- tbl_draws %>% 
  select(starts_with("theta"), .chain) %>%
  rename(chain = .chain) %>%
  pivot_longer(starts_with("theta"), names_to = "parameter", values_to = "value")

ggplot(tbl_posterior, aes(value)) +
  geom_density(aes(color = parameter)) +
  facet_wrap(~ parameter, scales = "free_y")





# Gaussian Naive Bayes V2 -------------------------------------------------


write_gaussian_naive_bayes_stan_v2 <- function() {
  write_stan_file("
  
data {
 int D; //number of dimensions
 int K; //number of categories
 int N; //number of data
 array[N] int cat; //category response for a stimulus
 matrix[N, D] y;
}

parameters {
 array[K] row_vector[D] mu; //category means
 array[K,D] real <lower=0> sigma; //variance
}

model {
  
 for(k in 1:K){
   for (d in 1:D) {
     sigma[k, d] ~ uniform(0.1, 5);
     mu[k, d] ~ normal(0, 1);
   }
 }


 for (n in 1:N){
 vector[K] LL1;
 vector[K] LL2;
   for (k in 1:K) {
     LL1[k] = normal_lpdf(y[n, 1] | mu[k, 1], sigma[k, 1]);
     LL2[k] = normal_lpdf(y[n, 2] | mu[k, 2], sigma[k, 2]);
   }
  target += (LL1[cat[n]] - log_sum_exp(LL1)) + (LL2[cat[n]] - log_sum_exp(LL2));
  //theta[n] = (exp(LL1[cat[n]] - log_sum_exp(LL1)) + exp(LL2[cat[n]] - log_sum_exp(LL2))) / 2;
 }
}

")
}




participant_sample <- sample(unique(tbl_train_agg$participant), 1)
tbl_train_agg <- tbl_train %>% 
  group_by(participant, d1i, d2i, category, response) %>%
  summarize(
    n_responses = n(),
    n_correct = sum(accuracy)
  ) %>% group_by(participant, d1i, d2i) %>%
  mutate(
    n_trials = sum(n_responses), 
    prop_responses = n_responses / n_trials) %>%
  ungroup()
tbl_sample <- tbl_train_agg %>% filter(participant == participant_sample)
tbl_sample <- tbl_sample %>% mutate(
  d1i_z = scale(d1i)[, 1],
  d2i_z = scale(d2i)[, 1]
)

tbl_naive2 <- tbl_train %>% filter(participant == participant_sample) %>%
  mutate(
    d1i_z = scale(d1i), d2i_z = scale(d2i),
    response_int = as.numeric(factor(response, labels = seq(1, 3, by = 1)))
  ) %>% group_by(d1i, d2i) %>% mutate(stim_id = )

l_data <- list(
  D = 2, K = length(unique(tbl_naive2$category)),
  N = nrow(tbl_naive2),
  y = tbl_naive2[, c("d1i_z", "d2i_z")] %>% as.matrix(),
  cat = tbl_naive2$response_int
)

stan_naive_v2 <- write_gaussian_naive_bayes_stan_v2()
mod_naive_v2 <- cmdstan_model(stan_naive_v2)

fit_naive_v2 <- mod_naive_v2$sample(
  data = l_data, iter_sampling = 2000, iter_warmup = 2000, chains = 1
)

file_loc <- str_c("data/infpro_task-cat_beh/gcm-model-", participant_sample, ".RDS")
fit_naive_v2$save_object(file = file_loc)
pars_interest <- c("mu", "sigma", "theta")
tbl_draws <- fit_naive_v2$draws(variables = pars_interest, format = "df")
tbl_summary <- fit_naive_v2$summary(variables = pars_interest)
names_thetas <- names(tbl_draws)[startsWith(names(tbl_draws), "theta")]
tbl_sample_naive$pred_theta <- colMeans(tbl_draws[, names_thetas])

plot_item_thetas(tbl_sample_naive, "GCM")

tbl_posterior <- tbl_draws %>% 
  select(starts_with(c("mu", "sigma")), .chain) %>%
  rename(chain = .chain) %>%
  pivot_longer(starts_with(c("mu", "sigma")), names_to = "parameter", values_to = "value")

ggplot(tbl_posterior, aes(value)) +
  geom_density(aes(color = parameter)) +
  facet_wrap(~ parameter, scales = "free_y")




ggplot(tbl_sample %>% mutate(category = str_c("True Cat. = ", category)), aes(d1i_z, d2i_z)) +
  geom_point(aes(size = prop_responses, color = category), show.legend = FALSE) +
  geom_label_repel(aes(label = round(prop_responses, 2)), size = 2.5) +
  #ggtitle(str_c("Participant = ", participant_sample)) +
  facet_wrap(~ response) +
  theme_bw() +
  labs(x = expr(x[1]), y = expr(x[2]))

