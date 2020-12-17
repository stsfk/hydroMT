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
  lemon
)


# Constant ----------------------------------------------------------------

old_model_names <-
  c(
    "lmFit",
    "MARSFit"  ,
    "PLSFit"   ,
    "RidgeFit"    ,
    "LassoFit",
    "CARTFit"  ,
    "KNNFit",
    "CubistFit"  ,
    "SVMFit"    ,
    "svmRadialFit" ,
    "RFFit"       ,
    "XGBoost"
  )

new_model_names <-
  c(
    "LM",
    "MARSBag",
    "PLS",
    "Ridge",
    "Lasso",
    "CART",
    "KNN",
    "Cubist",
    "SVMPoly",
    "SVMRadial",
    "RF",
    "XGBoost"
  )


# Data --------------------------------------------------------------------

eval_grid <- expand.grid(
  region = c(1:4),
  season = c("S", "W"),
  iter = c(1:5)
) %>%
  as_tibble() %>%
  mutate(
    file_name = paste0(
      "./modeling_results/Reg",
      region,
      "_",
      season,
      "_iter",
      iter,
      ".Rda"
    ),
    xgb_file_name = paste0(
      "./modeling_results/xgb_region",
      region,
      "_",
      season,
      "_iter",
      iter,
      ".model"
    )
  ) %>%
  mutate(gof_result = vector("list", 1))


for (i in 1:nrow(eval_grid)) {
  # gof of all models except xgboost
  load(eval_grid$file_name[i])
  
  df1 <-
    sapply(models, function(x)
      hydroGOF::gof(predict(x, dtest), dtest$Qmax, digits = 8)[17]) %>%
    as_tibble() %>%
    mutate(model = names(models))
  
  # gof of xgboost model
  xgb_dtest <-
    xgb.DMatrix(data = data.matrix(dtest %>% select(-Qmax)),
                label = dtest$Qmax)
  
  xgb_model <- xgboost::xgb.load(eval_grid$xgb_file_name[i])
  
  df2 <- tibble(model = "XGBoost",
                value = hydroGOF::gof(
                  sim = predict(xgb_model, xgb_dtest),
                  ob = dtest$Qmax,
                  digits = 8
                )[17])
  
  # return
  eval_grid$gof_result[[i]] <- bind_rows(df1, df2) %>%
    dplyr::rename(r2 = value) %>%
    dplyr::select(model, r2) %>%
    mutate(model = plyr::mapvalues(model, old_model_names, new_model_names))
}


# Plot --------------------------------------------------------------------

data_plot <- eval_grid %>%
  unnest(gof_result) %>%
  select(region, season, iter, model, r2) %>%
  mutate(
    region = factor(
      region,
      levels = c(1:4),
      labels = str_c("Region ", 1:4)
    ),
    season = factor(
      season,
      levels = c("W", "S"),
      labels = c("Winter", "Summer")
    )
  )

model_order <- data_plot %>%
  group_by(model) %>%
  dplyr::summarise(mean = mean(r2)) %>%
  arrange(mean) %>%
  mutate(mean = sprintf("%.2f", round(mean, 2)),
         label = paste0(model, "\n", "mean=", mean))

data_plot <- data_plot %>%
  mutate(model = factor(model, levels = model_order$model, labels = model_order$label))

ggplot(data_plot, aes(model, r2, color = season)) +
  geom_boxplot(size = 0.4,
               outlier.size = 1,
               fatten = 1.5) +
  facet_wrap( ~ region, nrow = 1) +
  scale_color_manual(values = c("steelblue", "chocolate1")) +
  scale_y_continuous(limits = c(0.2, 1)) +
  coord_flex_flip(left = brackets_vertical(direction = "right", length = unit(0.03, "npc"))) +
  labs(x = "Model",
       y = "R²",
       color = "Season") +
  theme_bw(base_size = 10) +
  theme(
    legend.position = "top",
    strip.background = element_rect(fill = "grey75", size = 0),
    panel.border = element_blank(),
    panel.background = element_rect(fill = "grey95"),
    panel.spacing = unit(1, "lines")
  )

ggsave(
  filename = "./mt_results/GoF.png",
  width = 7,
  height = 5.5,
  units = "in",
  dpi = 600
)
