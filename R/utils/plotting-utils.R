library(ggrepel)

plot_average_categorization_accuracy <- function(tbl_df, title) {
  #' scatter plot with proportion correct by x1 and x2
  #' 
  #' @description size of points is plotted according to categorization accuracy
  #' 
  #' @param tbl_df tbl df with n trials and n correct per x1-x2 combination
  #' @param title title of the plot
  #' @return ggplot object
  #' 
  
  tbl_df %>% group_by(d1i, d2i, category) %>% 
    summarize(acc_mn = mean(accuracy)) %>%
    ungroup() %>%
    ggplot(aes(d1i, d2i, group = category)) +
    geom_point(aes(color = category, size = acc_mn), show.legend = FALSE) +
    geom_label_repel(aes(label = round(acc_mn, 2)), size = 2.5, show.legend = FALSE) +
    scale_color_brewer(palette = "Set1") +
    scale_x_continuous(breaks = seq(2, 10, by = 2)) +
    scale_y_continuous(breaks = seq(2, 10, by = 2)) +
    theme_bw() +
    labs(x = expr(x[1]), y = expr(x[2]), title = title)
}


plot_item_thetas <- function(tbl_df, title) {
  #' scatter plot of posterior theta means against proportion correct per item
  #' 
  #' @description adds a label with the correlation between posterior means 
  #' and empirical data (i.e., proportion correct)
  #' 
  #' @param tbl_df tbl df with n trials and n correct per x1-x2 combination
  #' @param title title of the plot
  #' @return ggplot object
  #' 
  
  tbl_cor <- tibble(
    cor = cor(tbl_df$prop_correct, tbl_df$pred_theta)
  )
  
  ggplot(tbl_df, aes(pred_theta, prop_correct)) +
    geom_point() +
    geom_abline() +
    geom_label(data = tbl_cor, aes(label = str_c("r = ", round(cor, 2)), x = .75, y = .2)) +
    theme_bw() +
    labs(x = "Predicted Theta", y = "Proportion Correct", title = title)
  
}


plot_proportion_responses <- function(
    tbl_df, participant_sample, 
    facet_by_response = FALSE, color_pred_difference = FALSE
) {
  #' scatter plot of stimuli in feature space
  #' 
  #' @description filled by true category, faceted by response, 
  #' and sized by proportion responses
  #' 
  #' @param tbl_df tbl df with x vals in z space, stim_id, category, response,
  #' and proportion responses
  #' @param facet_by_response facets by given responses; default to FALSE
  #' @return ggplot object
  #' 
  
  pl <- ggplot(tbl_df, aes(d1i_z, d2i_z)) +
    geom_label_repel(aes(label = str_c(stim_id, "=", round(prop_responses, 2))), size = 2.5) +
    ggtitle(str_c("Participant = ", participant_sample)) +
    theme_bw() +
    scale_color_brewer(palette = "Set1", name = "True Category") +
    scale_size_continuous(guide = "none", range = c(2, 6)) +
    labs(x = expr(x[1]), y = expr(x[2]))
  
  if(color_pred_difference) {
    pl <- pl + geom_point(aes(color = pred_difference), size = 6) +
      scale_color_gradient2(
        name = "Prediction Difference",
        low = "#0099FF", mid = "white", high = "#FF9999"
      )
  } else {
    pl <- pl + geom_point(aes(size = prop_responses, color = category)) 
  }
  
  if(facet_by_response) {
    pl + facet_wrap(~ response)
  } else {pl}
  
}


plot_posteriors <- function(tbl_posterior, tbl_label, n_cols = 4) {
  #' histograms of posterior distributions
  #' 
  #' @description faceted by parameter
  #' 
  #' @param tbl_posterior tbl_df with samples from the posterior in long format
  #' @param tbl_label tbl_df with labels and their positions
  #' @param n_cols nr of columns to plot in the arrangement
  #' @return ggplot object
  #' 
  
  pl <- ggplot(tbl_posterior, aes(value)) +
    geom_histogram(bins = 30, fill = "#66CCFF", color = "white") +
    facet_wrap(~ parameter, scales = "free", ncol = n_cols) +
    theme_bw()
  gg_y_scales <- ggplot_build(pl)$layout$panel_scales_y
  maxy <- map(1:length(gg_y_scales), ~ gg_y_scales[[.x]]$range$range) %>% 
    unlist() %>% max()
  
  pl + geom_label(
    data = tbl_label %>% 
      mutate(parameter = variable), 
    aes(x = mean, y = maxy, label = str_c("Mean = ", round(mean, 2)))
  ) + labs(x = "Parameter Value", y = "Nr. Samples")
}
