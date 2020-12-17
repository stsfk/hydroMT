if (!require("pacman"))
  install.packages("pacman")
pacman::p_load(
  tidyverse,
  tidymodels,
  caret,
  lubridate,
  zeallot,
  xgboost,
  cowplot,
  doParallel,
  Cubist,
  pls,
  earth,
  elasticnet,
  ipred,
  plyr,
  e1071,
  kernlab,
  randomForest,
  ParBayesianOptimization,
  xgboost,
  lemon,
  ggrepel,
  twosamples,
  scales
)


# Rain MT -----------------------------------------------------------------

load("./mt_results/rain_mt.Rda")

# model's GOF and consistent rate

get_consistent_rate <- function(x) {
  (sum(x == 4)) / length(x)
}

eval_grid <- eval_grid %>%
  mutate(
    consistent_rate_tes = vector("list", 1),
    consistent_rate_trs = vector("list", 1)
  )

# iterate over region, season, data splits
for (i in 1:nrow(eval_grid)) {
  mt_trs <- eval_grid$mt_trs[[i]]
  mt_tes <- eval_grid$mt_tes[[i]]
  
  # iteration over machine learning methods
  consistent_rate_tr <- tibble(model = names(mt_trs),
                               consistent_rate = vector("list", 1))
  consistent_rate_te <- consistent_rate_tr
  
  for (j in 1:length(mt_trs)) {
    mt_te <- mt_tes[[j]]
    mt_tr <- mt_trs[[j]]
    
    consistent_rate_te$consistent_rate[[j]] <-
      sapply(mt_te, get_consistent_rate)
    consistent_rate_tr$consistent_rate[[j]] <-
      sapply(mt_tr, get_consistent_rate)
  }
  
  eval_grid$consistent_rate_tes[[i]] <- consistent_rate_te
  eval_grid$consistent_rate_trs[[i]] <- consistent_rate_tr
}


data_gof <- eval_grid %>%
  select(region, season, iter, gof_result) %>%
  unnest(gof_result)

model_order <- data_gof %>%
  group_by(model) %>%
  dplyr::summarise(mean_gof = mean(r2)) %>%
  arrange(desc(mean_gof)) %>%
  pull(model)

data_plot <- eval_grid %>%
  select(region, season, iter, consistent_rate_tes) %>%
  unnest(cols = consistent_rate_tes) %>%
  mutate(mean_consistent_rate = map_dbl(consistent_rate, function(x)
    unlist(mean(x)))) %>%
  left_join(data_gof, by = c("region", "season", "iter", "model")) %>%
  dplyr::select(-consistent_rate) %>%
  dplyr::rename(gof = r2,
                consistent_rate = mean_consistent_rate) %>%
  mutate(
    region = factor(
      region,
      levels = c(1:4),
      labels = str_c("Region ", 1:4)
    ),
    season = factor(
      season,
      levels = c("S", "W"),
      labels = c("Summer", "Winter")
    ),
    model = factor(model, levels = model_order)
  )

data_plot2 <- data_plot %>%
  group_by(region, season, model) %>%
  dplyr::summarise(
    mean_consistent_rate = mean(consistent_rate),
    min_consistent_rate = min(consistent_rate),
    max_consistent_rate = max(consistent_rate),
    mean_gof = mean(gof),
    max_gof = max(gof),
    min_gof = min(gof)
  )

data_plot3 <- data_plot2 #%>%
#dplyr::filter(region == "Region 3", season == "Summer")

ggplot() +
  geom_point(
    data = data_plot,
    aes(x = gof, y = consistent_rate, color = model),
    size = 1,
    alpha = 0.5
  ) +
  geom_errorbar(
    data = data_plot2,
    aes(
      x = mean_gof,
      y = mean_consistent_rate,
      ymin = min_consistent_rate,
      ymax = max_consistent_rate,
      color = model
    ),
    size = 0.5
  ) +
  geom_errorbar(
    data = data_plot2,
    aes(
      x = mean_gof,
      y = mean_consistent_rate,
      xmin = min_gof,
      xmax = max_gof,
      color = model
    ),
    size = 0.5
  ) +
  geom_text_repel(
    data = data_plot3,
    aes(
      label = data_plot3$model %>% as.character(),
      x = mean_gof,
      y = mean_consistent_rate,
    ),
    max.iter = 100000,
    force = 0.5,
    xlim = c(0.2, 1),
    size = 2,
    segment.size = 0.2
  )+
  scale_x_continuous(limits = c(0.2, 1),
                     breaks = c(0.2, 0.4, 0.6, 0.8, 1))+
  scale_color_discrete() +
  facet_grid(season ~ region)+
  labs(y = "Consistent rate",
       x = "R²",
       color = "Model") +
  theme_bw(base_size = 10)+
  theme(legend.position = "right",
        strip.background = element_rect(fill = "grey80", size = 0))

ggsave(
  filename = "./mt_results/GoF_vs_consistent_rate.png",
  width = 7,
  height = 5.5,
  units = "in",
  dpi = 600
)


# Change ratio vs. evaluation results --------------------------------------

load("./mt_results/rain_mt.Rda")

get_consistent_rate <- function(x) {
  (sum(x == 4)) / length(x)
}

get_inconsistent_rate <- function(x) {
  (sum(x == 3)) / length(x)
}

get_invalid_rate <- function(x) {
  (sum(x == 2)) / length(x)
}

get_inconclusive_rate <- function(x) {
  (sum(x == 1)) / length(x)
}

get_evaluation_df <- function(xs, change_ratio = (50:150) * 0.01) {
  tibble(
    change_ratio = change_ratio,
    Consistent = sapply(xs, get_consistent_rate),
    Inconsistent = sapply(xs, get_inconsistent_rate),
    Invalid = sapply(xs, get_invalid_rate),
    Inconclusive = sapply(xs, get_inconclusive_rate)
  ) %>%
    gather(outcome, value, -change_ratio)
}

eval_grid <- eval_grid %>%
  mutate(evaluation_tes = vector("list", 1),
         evaluation_trs = vector("list", 1))

# iterate over region, season, data splits
for (i in 1:nrow(eval_grid)) {
  mt_trs <- eval_grid$mt_trs[[i]]
  mt_tes <- eval_grid$mt_tes[[i]]
  
  # iteration over machine learning methods
  evaluation_tr <- tibble(model = names(mt_trs),
                          evaluation = vector("list", 1))
  evaluation_te <- evaluation_tr
  
  for (j in 1:length(mt_trs)) {
    mt_te <- mt_tes[[j]]
    mt_tr <- mt_trs[[j]]
    
    evaluation_te$evaluation[[j]] <- get_evaluation_df(mt_te)
    evaluation_tr$evaluation[[j]] <- get_evaluation_df(mt_tr)
  }
  
  eval_grid$evaluation_tes[[i]] <- evaluation_te %>%
    unnest(evaluation)
  eval_grid$evaluation_trs[[i]] <- evaluation_tr %>%
    unnest(evaluation)
}

data_gof <- eval_grid %>%
  select(region, season, iter, gof_result) %>%
  unnest(gof_result)

model_order <- data_gof %>%
  group_by(model) %>%
  dplyr::summarise(mean_gof = mean(r2)) %>%
  arrange(desc(mean_gof)) %>%
  pull(model)

data_plot <- eval_grid %>%
  select(region, season, iter, evaluation_tes) %>%
  unnest(cols = evaluation_tes)

data_plot <- data_plot %>%
  mutate(outcome = factor(
    outcome,
    levels = c("Consistent", "Inconsistent", "Invalid", "Inconclusive")
  )) %>%
  mutate(model = factor(model, levels = model_order))

data_plot2 <- data_plot %>%
  dplyr::filter(region %in% c(1:4),
                season %in% c("S"))%>%
  group_by(region, season, model, change_ratio, outcome) %>%
  dplyr::summarise(value = mean(value)) %>%
  mutate(region = paste0("Region ", region))

ggplot(data_plot2, aes(change_ratio, value, fill = outcome)) +
  geom_area(alpha = 0.8) +
  scale_fill_manual(values = c("mediumseagreen", "tan2", "tomato2", "grey")) +
  scale_x_continuous(breaks = c(0.5, 1, 1.5), expand = c(0, 0))+
  scale_y_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1),
                     expand = c(0, 0)) +
  facet_grid(region ~ model) +
  labs(fill = "Assessment\nresult",
       x = "Magnitude of precipitation of TCs compared to samples from test sets",
       y = "Proportion of assessment result")+
  theme_bw(base_size = 8)+
  theme(
    legend.position = "top",
    strip.background = element_rect(fill = "grey80", size = 0),
    strip.text = element_text(size = 6),
    panel.spacing.x = unit(0.5, "lines"),
    panel.spacing.y = unit(0.4, "lines"),
    panel.border = element_blank(),
    axis.text.x = element_text(angle = 90)
  )

ggsave(
  filename = "./mt_results/change_ratio_vs_evaluation_S.png",
  width = 7,
  height = 4,
  units = "in",
  dpi = 600
)

# Distribution of consistent rate between train and test data --------------
get_consistent_rate <- function(x) {
  (sum(x == 4)) / length(x)
}

get_consistent_rate_per_event <- function(xs) {
  m <-
    xs %>% unlist() %>% matrix(nrow = length(xs)) # num of mt * num of flood events
  
  apply(m, 2, get_consistent_rate)
}

eval_grid <- eval_grid %>%
  mutate(consistent_dis = vector("list", 1))

# iterate over region, season, data splits
for (i in 1:nrow(eval_grid)) {
  mt_trs <- eval_grid$mt_trs[[i]]
  mt_tes <- eval_grid$mt_tes[[i]]
  
  # iteration over machine learning methods
  consistent_dis_tr <- tibble(model = names(mt_trs),
                              consistent_dis = vector("list", 1))
  consistent_dis_te <- consistent_dis_tr
  
  for (j in 1:length(mt_trs)) {
    mt_te <- mt_tes[[j]]
    mt_tr <- mt_trs[[j]]
    
    consistent_dis_te$consistent_dis[[j]] <-
      get_consistent_rate_per_event(mt_te)
    consistent_dis_tr$consistent_dis[[j]] <-
      get_consistent_rate_per_event(mt_tr)
  }
  
  consistent_dis_te <- consistent_dis_te %>%
    unnest(consistent_dis) %>%
    mutate(case = "Test set")
  
  consistent_dis_tr <- consistent_dis_tr %>%
    unnest(consistent_dis) %>%
    mutate(case = "Training set")
  
  eval_grid$consistent_dis[[i]] <- consistent_dis_te %>%
    rbind(consistent_dis_tr)
}

data_gof <- eval_grid %>%
  select(region, season, iter, gof_result) %>%
  unnest(gof_result)

model_order <- data_gof %>%
  group_by(model) %>%
  dplyr::summarise(mean_gof = mean(r2)) %>%
  arrange(desc(mean_gof)) %>%
  pull(model)

data_plot <- eval_grid %>%
  select(region, season, iter, consistent_dis) %>%
  unnest(consistent_dis)

data_plot <- data_plot %>%
  mutate(model = factor(model, levels = model_order))

data_plot2 <- data_plot %>%
  dplyr::filter(region %in% c(1:4),
                season %in% c("S"))%>%
  group_by(region, season, model) %>%
  mutate(region = paste0("Region ", region))

# K-S test

dist_test <- function(data_test1,
                      data_test2,
                      confidence_level = 0.05) {
  test_result <- cvm_test(data_test1, data_test2)
  
  if (test_result[2] <= confidence_level) {
    "reject H[0]"
  } else {
    "fail to reject H[0]"
  }
}

dist_test_df <- function(df) {
  data_test1 <- df %>%
    dplyr::filter(case == "Test set") %>%
    pull(consistent_dis)
  
  data_test2 <- df %>%
    dplyr::filter(case == "Training set") %>%
    pull(consistent_dis)
  
  dist_test(data_test1, data_test2)
}

data_plot_dist_test <- data_plot2 %>%
  group_by(region, season, model) %>%
  group_split()
data_plot_dist_test_summary <-  data_plot_dist_test %>%
  lapply(function(x)
    x[1, ]) %>%
  bind_rows() %>%
  mutate(test_result = "")

for (i in seq_along(data_plot_dist_test)) {
  data_plot_dist_test_summary$test_result[i] <-
    data_plot_dist_test[[i]] %>%
    dist_test_df()
}

data_plot_dist_test_summary2 <- data_plot_dist_test_summary %>%
  mutate(test_result = replace(test_result, test_result == "reject H[0]", "reject H0")) %>%
  mutate(test_result = replace(
    test_result,
    test_result == "fail to reject H[0]",
    "fail to reject H0"
  ))

# plot
ggplot(data_plot2, aes(consistent_dis, fill = case)) +
  geom_histogram(
    aes(y = 0.05 * ..density..),
    alpha = 0.5,
    position = 'identity',
    binwidth = 0.05,
    color = "grey30",
    size = 0.05
  )+
  geom_text(
    data = data_plot_dist_test_summary2,
    aes(
      x = 0,
      y = 0.8,
      label = test_result,
      color = factor(test_result)
    ),
    size = 1.5,
    hjust = 0,
    vjust = 0.5,
    parse = F
  )+
  scale_color_manual(values = c("midnightblue", "indianred4"),
                     guide = F) +
  facet_grid(region ~ model)+
  labs(fill = "Associated\ndataset",
       x = "Consistent rate of the TCs associated with a sample",
       y = "Proportion",
       tag = "H0: distributions of the consistent rate of TCs associated with training and test sets are the same.") +
  theme_bw(base_size = 8)+
  theme(
    legend.position = 'top',
    legend.justification = 'left',
    legend.direction = 'horizontal',
    strip.background = element_rect(fill = "grey80", size = 0),
    strip.text = element_text(size = 6),
    panel.spacing.x = unit(0.3, "lines"),
    panel.spacing.y = unit(0.4, "lines"),
    panel.border = element_blank(),
    axis.text.x = element_text(angle = 90, size = 5),
    axis.text.y = element_text(size = 5),
    panel.grid.minor = element_blank(),
    plot.tag.position = c(0.68, 0.955),
    plot.tag = element_text(size = 7)
  )

ggsave(
  filename = "./mt_results/compare_distribution_S.png",
  width = 7,
  height = 4,
  units = "in",
  dpi = 600
)

# Multiple MT -------------------------------------------------------------
load("./mt_results/rain_mt.Rda")
rain_eval_grid <- eval_grid

load("./mt_results/pet_mt.Rda")
pet_eval_grid <- eval_grid

data_gof <- eval_grid %>%
  select(region, season, iter, gof_result) %>%
  unnest(gof_result)

model_order <- data_gof %>%
  group_by(model) %>%
  dplyr::summarise(mean_gof = mean(r2)) %>%
  arrange(desc(mean_gof)) %>%
  pull(model)

# get the test result of pet for change ratio == 1
get_consistent_rate <- function(x) {
  (sum(x == 4)) / length(x)
}

pet_eval_grid <- pet_eval_grid %>%
  mutate(test_results = vector("list", 1))

# iterate over region, season, data splits
for (i in 1:nrow(pet_eval_grid)) {
  mt_tes <- pet_eval_grid$mt_tes[[i]]
  test_results <- tibble(model = names(mt_tes),
                         test_result = vector("list", 1))
  
  # iteration over machine learning methods
  for (j in seq_along(mt_tes)) {
    test_results$test_result[[j]] <- mt_tes[[j]] %>% unlist()
  }
  
  pet_eval_grid$test_results[[i]] <- test_results %>%
    unnest(test_result)
}

pet_eval_grid <- pet_eval_grid %>%
  select(region, season, iter, test_results)

# process rain_eval_grid
rain_eval_grid <- rain_eval_grid %>%
  select(region, season, iter, mt_tes)

# iterate over region, season, data splits
rain_eval_grid <- rain_eval_grid %>%
  mutate(consistent_rates = vector("list", 1))

for (i in 1:nrow(rain_eval_grid)) {
  mt_tes <- rain_eval_grid$mt_tes[[i]]
  pet_tes <- pet_eval_grid$test_results[[i]]
  
  consistent_rate <- tibble(
    model = names(mt_tes),
    rain_consistent_test = vector("list", 1),
    pet_consistent_test = vector("list", 1),
    rain_and_pet_consistent_test = vector("list", 1)
  )
  # iteration over machine learning methods
  for (j in seq_along(mt_tes)) {
    mt_te <- mt_tes[[j]][[51]]
    
    model_name <- names(mt_tes)[[j]]
    pet_mt_te <- pet_tes %>%
      dplyr::filter(model == model_name) %>%
      pull(test_result)
    
    consistent_rate$rain_consistent_test[[j]] <- mt_te
    consistent_rate$pet_consistent_test[[j]] <- pet_mt_te
    consistent_rate$rain_and_pet_consistent_test[[j]] <-
      pmin(mt_te, pet_mt_te)
  }
  
  rain_eval_grid$consistent_rates[[i]] <- consistent_rate
}


rain_eval_grid <- rain_eval_grid %>%
  unnest(consistent_rates)

data_plot <- rain_eval_grid %>%
  select(-mt_tes) %>%
  mutate(
    rain_consistent_test = map_dbl(rain_consistent_test, get_consistent_rate),
    pet_consistent_test = map_dbl(pet_consistent_test, get_consistent_rate),
    rain_and_pet_consistent_test = map_dbl(rain_and_pet_consistent_test, get_consistent_rate)
  ) %>%
  gather(
    item,
    value,
    rain_consistent_test,
    pet_consistent_test,
    rain_and_pet_consistent_test
  ) %>%
  mutate(item = factor(
    item,
    levels = c(
      "rain_consistent_test",
      "pet_consistent_test",
      "rain_and_pet_consistent_test"
    ),
    labels = c("MR[1]", "MR[2]", "MR[1]*' & M'*R[2]")
  )) %>%
  mutate(
    region = factor(
      region,
      levels = c(1:4),
      labels = str_c("Region ", 1:4)
    ),
    season = factor(
      season,
      levels = c("S", "W"),
      labels = c("Summer", "Winter")
    ),
    model = factor(model, levels = model_order)
  )

lab <- c(expression(MR[1]),
         expression(MR[2]),
         expression(MR[1] * ' & M' * R[2]))

ggplot(data_plot %>% filter(season == "Winter"), aes(item, value)) +
  geom_point(aes(color = item, shape = item),
             size = 1,
             stroke = 0.3) +
  geom_line(aes(group = interaction(iter, season)), size = 0.2, color =
              "grey50") +
  scale_color_manual(
    name = "MR considered",
    values = c("#00AFBB", "#E7B800", "#FC4E07"),
    labels = lab
  ) +
  scale_shape_discrete(name = "MR considered",
                       solid = F,
                       labels = lab) +
  scale_x_discrete(labels = parse(text = levels(data_plot$item))) +
  facet_grid(region ~ model) +
  labs(x = "MR considered",
       y = "Consistent rate") +
  theme_bw(base_size = 8) +
  theme(legend.position = "top") +  theme_bw(base_size = 8) +
  theme(
    legend.position = "top",
    strip.background = element_rect(fill = "grey80", size = 0),
    strip.text = element_text(size = 6),
    panel.border = element_blank(),
    axis.text.y =  element_text(size = 5),
    axis.text.x =  element_text(size = 5, angle = 90),
    panel.background = element_rect(fill = "grey95")
  )

ggsave(
  filename = "./mt_results/multiple_MRs_W.png",
  width = 7,
  height = 4,
  units = "in",
  dpi = 600
)
