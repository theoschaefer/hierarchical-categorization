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
    n_trials = n(), 
    n_correct = sum(accuracy),
    prop_correct = n_correct / n_trials
  ) %>% ungroup()

participant_sample <- sample(unique(tbl_train_agg$participant), 1)
tbl_sample <- tbl_train_agg %>% filter(participant == participant_sample)
tbl_sample <- tbl_sample %>% mutate(
  d1i_z = scale(d1i)[, 1],
  d2i_z = scale(d2i)[, 1]
)

ggplot(tbl_sample, aes(d1i_z, d2i_z)) +
  geom_point(aes(size = prop_correct, color = category), show.legend = FALSE) +
  geom_label_repel(aes(label = round(prop_correct, 2)), size = 2.5) +
  ggtitle(str_c("Participant = ", participant_sample)) +
  theme_bw() +
  labs(x = expr(x[1]), y = expr(x[2]))


# Gaussian Naive Bayes ----------------------------------------------------


write_gaussian_naive_bayes_stan <- function() {
  write_stan_file("
data {
 int D; //number of dimensions
 int K; //number of gaussians
 int N; //number of data
 array[N] int n_trials; // number of trials per item
 array[N] int n_correct; // number of true categorization responses per item
 array[N] int cat; // true category labels
 matrix[N, D] y; //data
}

parameters {
 array[K] row_vector[D] mu; //category means
 array[K] row_vector[D] sigma; //variance
}

transformed parameters {
  matrix[N,K] theta;
  matrix[N,K] lps1;
  matrix[N,K] lps2;
  
  for (n in 1:N){
     for (k in 1:K){
        //increment log probability of the gaussian
        // could this be expressed as an array of length N containing matrices of KxD?
        lps1[n, k] = normal_lpdf(y[n,1] | mu[k,1], sigma[k,1]); 
        lps2[n, k] = normal_lpdf(y[n,2] | mu[k,2], sigma[k,2]);
     }
     for (k in 1:K){
     theta[n,k] = (lps1[n,k] + lps2[n,k]) / (sum(lps1[n,]) + sum(lps2[n,]));
     }
  }
}

model {

 mu[1,1] ~ normal(-1, 1);
 mu[1,2] ~ normal(.5, 1);
 mu[2,1] ~ normal(0, 1);
 mu[2,2] ~ normal(0, 1);
 mu[3,1] ~ normal(1, 1);
 mu[3,2] ~ normal(-.5, 1);
  
 for(k in 1:K){
   mu[k] ~ normal(0, 2);
   sigma[k] ~ exponential(1);
 }

 n_correct ~ multinomial(theta);
}

generated quantities {
 array[K] corr_matrix[D] Sigma;
 array[N] int n_correct_predict;
 
 for (k in 1:K){
 Sigma[k] = multiply_lower_tri_self_transpose(L[k]);
 }
  n_correct_predict = binomial_rng(n_trials, theta);
}
")
}

