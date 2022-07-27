#==============================================================================#
#-------------------- INFPRO - Categorization Analysis ------------------------#
#==============================================================================#

# Author: Theo Schaefer
# Contact: tschaefer@cbs.mpg.de


#================================ Setup ========================================

# Load libraries
library(tidyverse)  # for most data wrangling functions (filter, ggplot,...)
library(ggforce)  # to draw ellipses in feature space
library(viridisLite)  # color theme
library(rutils)

# Specify column select function
select = dplyr::select

# Set working directory to script path
dir_data = 'data/infpro_task-cat_beh/'

# Define category colors
cat_colors = c('#C88D0D','#6A0DAD','#808080') 
# Set axis labels for feature space plots
dim_labs = c('Dim 1 (head)', 'Dim 2 (stomach)')

# Feature space resolution (important for drawing ellipses)
res = 10
# Prototype locations (dim1, dim2)
pt = list(A= c(4,7), B = c(7,4), C = NA)


#----------------------------- Costum functions --------------------------------

getmode <- function(v) {
  # Calculates the modal value of a vector
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

#--------------------------------- Load data -----------------------------------

ds_cat = read_csv(paste0(dir_data, 'infpro_task-cat_beh.csv'))  # training
ds_cat2 = read_csv(paste0(dir_data, 'infpro_task-cat2_beh.csv'))  # transfer

# Select categorization blocks
ds_cat %>% 
  # group_by(participant) %>% mutate(maxblock = max(block, na.rm=T)) %>%
  filter(block >= maxblock-3) %>%  # select last three blocks
  filter(block > 1) %>%  # remove practice block 1
  filter(!response=='NaN')


#============================= Plotting ========================================

#------------------------ Categorization training -----------------------------#

ds_plot = ds_cat

# Blockwise accuracy (lineplot)
ds_plot %>% 
  group_by(participant, block) %>% summarise(accuracy = mean(accuracy)) %>% 
  ggplot(aes(block, accuracy, color=factor(participant))) +
  geom_line(size=2, alpha=0.6) +
  geom_line(data=ds_plot %>% group_by(block) %>% summarise(accuracy = mean(accuracy)), 
            aes(block, accuracy), size=3, inherit.aes=F) +
  theme_classic() #+
# theme(legend.position = 'None')


# Modal participant responses (single plot, summarised)
ds_plot %>% 
  group_by(d1i, d2i, category) %>% 
  summarise(response = getmode(response)) %>%
  ggplot(aes(d1i,d2i, color=response, shape=category)) +
  geom_ellipse(aes(x0=pt$A[1], y0=pt$A[2], a=res*(.3), b=res*(.15), angle=pi/4)) +
  geom_ellipse(aes(x0=pt$B[1], y0=pt$B[2], a=res*(.3), b=res*(.15), angle=pi/4)) +
  geom_point(size=3, alpha=1) +
  scale_color_manual(values=cat_colors) +
  labs(x=dim_labs[1], y=dim_labs[2]) +
  theme_classic() 

# Overlayed participant responses (single plot)
ds_plot %>% 
  group_by(participant, d1i, d2i, category) %>% 
  summarise(response = getmode(response)) %>%
  ggplot(aes(d1i,d2i, color=response, shape=category)) +
  geom_ellipse(aes(x0=pt$A[1], y0=pt$A[2], a=res*(.3), b=res*(.15), angle=pi/4)) +
  geom_ellipse(aes(x0=pt$B[1], y0=pt$B[2], a=res*(.3), b=res*(.15), angle=pi/4)) +
  geom_point(size=3, alpha=0.05) +
  scale_color_manual(values=cat_colors) +
  labs(x=dim_labs[1], y=dim_labs[2]) +
  theme_classic() 

# Modal participant-trial responses (facet grid)
ds_plot %>% 
  group_by(participant, d1i, d2i, category) %>% 
  summarise(response = getmode(response)) %>% 
  ggplot(aes(d1i,d2i, color=response, shape=category)) +
  geom_ellipse(aes(x0=pt$A[1], y0=pt$A[2], a=res*(.3), b=res*(.15), angle=pi/4)) +
  geom_ellipse(aes(x0=pt$B[1], y0=pt$B[2], a=res*(.3), b=res*(.15), angle=pi/4)) +
  geom_point(size=3, alpha=1) +
  facet_wrap(~participant, ncol=10, scales='free') +
  ggtitle('Categorization training: Modal participant responses') +
  scale_color_manual(values=cat_colors) +
  labs(x=dim_labs[1], y=dim_labs[2]) +
  theme_classic() +
  theme(legend.position='None',
        axis.title=element_blank(),
        axis.text=element_blank(),
        axis.ticks=element_blank())

# Overlayed participant-trial responses (facet grid)
ds_plot %>% 
  ggplot(aes(d1i,d2i, color=response, shape=category)) +
  geom_ellipse(aes(x0=pt$A[1], y0=pt$A[2], a=res*(.3), b=res*(.15), angle=pi/4)) +
  geom_ellipse(aes(x0=pt$B[1], y0=pt$B[2], a=res*(.3), b=res*(.15), angle=pi/4)) +
  geom_point(size=3, alpha=0.2) +
  facet_wrap(~participant, ncol=10, scales='free') +
  ggtitle('Categorization training: Overlayed participant responses') +
  scale_color_manual(values=cat_colors) +
  labs(x=dim_labs[1], y=dim_labs[2]) +
  theme_classic() +
  theme(legend.position='None',
        axis.title=element_blank(),
        axis.text=element_blank(),
        axis.ticks=element_blank())

# Categorization training: participant RT (single plot)
ds_plot %>% 
  group_by(participant) %>% 
  mutate(z_rt = scale(rt)) %>% 
  group_by(d1i, d2i, category) %>% 
  summarise(rt = mean(rt), z_rt = mean(z_rt)) %>%
  ggplot(aes(d1i,d2i, color=rt, shape=category)) +
  geom_ellipse(aes(x0=pt$A[1], y0=pt$A[2], a=res*(.3), b=res*(.15), angle=pi/4), color='black') +
  geom_ellipse(aes(x0=pt$B[1], y0=pt$B[2], a=res*(.3), b=res*(.15), angle=pi/4), color='black') +
  geom_point(size=3, alpha=1) +
  scale_color_viridis_c(option = 'magma') +
  labs(x=dim_labs[1], y=dim_labs[2]) +
  theme_classic() 

# Categorization training: participant RT, z-scored (single plot)
ds_plot %>% 
  group_by(participant) %>% 
  mutate(z_rt = scale(rt)) %>% 
  group_by(d1i, d2i, category) %>% 
  summarise(rt = mean(rt), z_rt = mean(z_rt)) %>%
  ggplot(aes(d1i,d2i, color=z_rt, shape=category)) +
  geom_ellipse(aes(x0=pt$A[1], y0=pt$A[2], a=res*(.3), b=res*(.15), angle=pi/4), color='black') +
  geom_ellipse(aes(x0=pt$B[1], y0=pt$B[2], a=res*(.3), b=res*(.15), angle=pi/4), color='black') +
  geom_point(size=3, alpha=1) +
  scale_color_viridis_c(option = 'magma') +
  labs(x=dim_labs[1], y=dim_labs[2]) +
  theme_classic() 

# Categorization training: participant Accuracy (single plot)
ds_plot %>% 
  group_by(d1i, d2i, category) %>% 
  summarise(error = mean(1-accuracy)) %>%
  ggplot(aes(d1i,d2i, color=error, shape=category)) +
  geom_ellipse(aes(x0=pt$A[1], y0=pt$A[2], a=res*(.3), b=res*(.15), angle=pi/4), color='black') +
  geom_ellipse(aes(x0=pt$B[1], y0=pt$B[2], a=res*(.3), b=res*(.15), angle=pi/4), color='black') +
  geom_point(size=3, alpha=1) +
  scale_color_viridis_c(option = 'magma') +
  labs(x=dim_labs[1], y=dim_labs[2]) +
  theme_classic() 

# Response error between categories (1 plot)
ds_plot %>% 
  group_by(category) %>% 
  summarise(error = mean(1-accuracy),
            error_sd = sd(1-accuracy),
            error_sem = error_sd / sqrt(.$participant %>% n_distinct())) %>% 
  ggplot(aes(category, error, fill=category)) +
  geom_bar(stat='identity') + 
  geom_errorbar(aes(ymin=error-error_sem, ymax=error+error_sem), width=.2,
                position=position_dodge(.9)) +
  # ylim(c(0,0.5)) +
  scale_fill_manual(values=cat_colors) +
  theme_classic() +
  ggtitle('Categorization error between \ncategories (mean & sem)')

# Response error time course
ds_plot %>% 
  group_by(participant, block) %>% summarise(accuracy = mean(accuracy)) %>% 
  ggplot(aes(block, accuracy, color=factor(participant))) +
  geom_line(size=1.5, alpha=0.4) +
  geom_line(data=ds_plot %>% group_by(block) %>% summarise(accuracy = mean(accuracy)), 
            aes(block, accuracy), size=3, inherit.aes=F) +
  # scale_color_viridis_d() +
  theme_classic() +
  theme(legend.position = 'None') + 
  ggtitle('Categorization error accross blocks \n(single participants)')




#------------------------ Categorization transfer -----------------------------#

ds_plot = ds_cat2

# Categorization transfer: Modal participant responses (single plot)
ds_plot %>% 
  group_by(d1i, d2i, category) %>% 
  summarise(response = getmode(response)) %>%
  ggplot(aes(d1i,d2i, color=response, shape=category)) +
  geom_ellipse(aes(x0=pt$A[1], y0=pt$A[2], a=res*(.3), b=res*(.15), angle=pi/4)) +
  geom_ellipse(aes(x0=pt$B[1], y0=pt$B[2], a=res*(.3), b=res*(.15), angle=pi/4)) +
  geom_point(size=3, alpha=1) +
  scale_color_manual(values=cat_colors) +
  labs(x=dim_labs[1], y=dim_labs[2]) +
  theme_classic() 

# Overlayed participant responses (single plot)
ds_plot %>% 
  group_by(participant, d1i, d2i, category) %>% 
  summarise(response = getmode(response)) %>%
  ggplot(aes(d1i,d2i, color=response, shape=category)) +
  geom_ellipse(aes(x0=pt$A[1], y0=pt$A[2], a=res*(.3), b=res*(.15), angle=pi/4)) +
  geom_ellipse(aes(x0=pt$B[1], y0=pt$B[2], a=res*(.3), b=res*(.15), angle=pi/4)) +
  geom_point(size=3, alpha=0.05) +
  scale_color_manual(values=cat_colors) +
  labs(x=dim_labs[1], y=dim_labs[2]) +
  theme_classic() 

# Modal participant responses (facet grid)
ds_plot %>% 
  group_by(participant, d1i, d2i, category) %>% 
  summarise(response = getmode(response)) %>% 
  ggplot(aes(d1i,d2i, color=response, shape=category)) +
  geom_ellipse(aes(x0=pt$A[1], y0=pt$A[2], a=res*(.3), b=res*(.15), angle=pi/4)) +
  geom_ellipse(aes(x0=pt$B[1], y0=pt$B[2], a=res*(.3), b=res*(.15), angle=pi/4)) +
  geom_point(size=3, alpha=1) +
  facet_wrap(~participant, ncol=10, scales='free') +
  ggtitle('Categorization transfer: participant responses') +
  scale_color_manual(values=cat_colors) +
  labs(x=dim_labs[1], y=dim_labs[2]) +
  theme_classic() +
  theme(legend.position='None',
        axis.title=element_blank(),
        axis.text=element_blank(),
        axis.ticks=element_blank())



# participant RT (single plot, feature space)
ds_plot %>% 
  group_by(participant) %>% 
  mutate(z_rt = scale(rt)) %>%
  group_by(d1i, d2i, category, block) %>% 
  summarise(rt = mean(rt), z_rt = mean(z_rt)) %>%
  ggplot(aes(d1i,d2i, color=rt, shape=category)) +
  geom_ellipse(aes(x0=pt$A[1], y0=pt$A[2], a=res*(.3), b=res*(.15), angle=pi/4), color='black') +
  geom_ellipse(aes(x0=pt$B[1], y0=pt$B[2], a=res*(.3), b=res*(.15), angle=pi/4), color='black') +
  geom_point(size=3, alpha=1) +
  facet_wrap(~block, labeller=label_both) + 
  scale_color_viridis_c(option = 'magma') +
  labs(x=dim_labs[1], y=dim_labs[2]) +
  theme_light() 

# participant RT (single plot, heatmap)
ds_plot %>% 
  group_by(participant) %>% 
  mutate(z_rt = scale(rt)) %>%
  group_by(d1i, d2i, category, block) %>% 
  summarise(rt = mean(rt), z_rt = mean(z_rt)) %>%
  ggplot(aes(d1i,d2i)) +
  geom_tile(aes(fill=rt)) +      
  geom_ellipse(aes(x0=pt$A[1], y0=pt$A[2], a=res*(.3), b=res*(.15), angle=pi/4), color='black') +
  geom_ellipse(aes(x0=pt$B[1], y0=pt$B[2], a=res*(.3), b=res*(.15), angle=pi/4), color='black') +
  facet_wrap(~block, labeller=label_both) + 
  scale_fill_viridis_c(option = 'magma') +
  labs(x=dim_labs[1], y=dim_labs[2]) +
  theme_minimal() 


ds_plot %>% 
  group_by(participant) %>%
  mutate(z_rt = scale(rt)) %>%
  group_by(participant, d1i, d2i, category) %>% 
  summarise(rt = mean(z_rt))  %>%
  ggplot(aes(d1i,d2i)) +
  geom_tile(aes(fill=rt)) +      
  geom_ellipse(aes(x0=pt$A[1], y0=pt$A[2], a=res*(.3), b=res*(.15), angle=pi/4), color='black') +
  geom_ellipse(aes(x0=pt$B[1], y0=pt$B[2], a=res*(.3), b=res*(.15), angle=pi/4), color='black') +
  facet_wrap(~participant, ncol=10, labeller=label_both) + 
  scale_fill_viridis_c(option = 'magma') +
  labs(x=dim_labs[1], y=dim_labs[2]) +
  theme_minimal() 

# Error between categories
ds_plot %>% 
  group_by(category) %>% 
  summarise(error = mean(1-accuracy),
            error_sd = sd(1-accuracy),
            error_sem = error_sd / sqrt(.$participant %>% n_distinct())) %>% 
  ggplot(aes(category, error, fill=category)) +
  geom_bar(stat='identity') + 
  geom_errorbar(aes(ymin=error-error_sem, ymax=error+error_sem), width=.2,
                position=position_dodge(.9)) +
  # ylim(c(0,0.5)) +
  scale_fill_manual(values=cat_colors) +
  theme_classic() +
  ggtitle('Categorization error between \ncategories (mean & sem)')


ds_plot %>% mutate(block = factor(block)) %>%
  group_by(participant, block) %>% 
  summarise(accuracy=mean(accuracy)) %>% 
  group_by(block) %>% grouped_agg(block, accuracy) %>%
  ggplot(aes(block, mean_accuracy, group = 1)) +
  geom_errorbar(aes(
    x = block, 
    ymin = mean_accuracy - 1.96 * se_accuracy,
    ymax = mean_accuracy + 1.96 * se_accuracy
    ), width = .2) + 
  geom_line() +
  geom_point(color = "white", size = 3) +
  geom_point() +
  coord_cartesian(ylim = c(.45, 1)) +
  scale_color_brewer(palette = "Set1") +
  theme_bw() +
  labs(
    x = "Transfer Block",
    y = "Mean Accuracy"
  )

ds_plot$accuracy %>% mean()
