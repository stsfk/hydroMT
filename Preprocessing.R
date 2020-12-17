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
  xgboost
)

read_data_single_catchment <- function(file_names) {
  # Read data ---------------------------------------------------------------
  
  # Schmidt, Lennart, Heße, Falk, Kumar, Rohini, & Attinger, Sabine. (2019).
  # Dataset of Floods in Germany 1950-2010 (Version 1.0) [Data set]. Zenodo.
  # http://doi.org/10.5281/zenodo.3538207
  
  read_flood_data <- function(file_name) {
    read_csv(
      file_name,
      col_types = cols(
        .default = col_double(),
        EventID = col_character(),
        YYYY = col_character(),
        Startdate = col_date(format = "%Y-%m-%d"),
        Enddate = col_date(format = "%Y-%m-%d"),
        Group_ID = col_character(),
        Keep = col_logical()
      )
    )
  }
  
  data_raw <- lapply(file_names, read_flood_data) %>%
    bind_rows()
  
  # Get predictors, reponse, and context variable names ---------------------
  
  # Predictor names
  #   dynamic predictors
  p_predictor <-
    str_subset(names(data_raw), "^P([0-9]|min|max|maxT|minT)$") # precipitation [mm/day]
  sm_predictor <-
    str_subset(names(data_raw), "^SM([0-9]|min|max|maxT|minT)$") # daily mean soil moisture [%]
  t_predictor <-
    str_subset(names(data_raw), "^T([0-9]|min|max|maxT|minT)$") # daily mean temperature [degree C]
  PET_predictor <-
    str_subset(names(data_raw), "^PET([0-9]|min|max|maxT|minT)$") # PET
  AET_predictor <-
    str_subset(names(data_raw), "^AET([0-9]|min|max|maxT|minT)$") # AET
  
  #   static predictors
  average_climatic <-
    c("AI_Ann", "P_Ann") # aridity index and mean annual precipitation
  topography <-
    c("Area", "Altitude", "Slope") # area, slope, and elevation
  geomorphology <-
    c("ChSlopeM",
      "DD",
      "FLMax",
      "FLCV",
      "FLSD",
      "StrahlerMax",
      "LengthAll") # channel slope, drainage density, and flow path lengths
  land_cover <-
    c("Forest", "Impervious") # Permeable is linear combination of Forest and Impervious, and it is excluded
  other_static <- c("Duration")
  
  #   combine predictors
  predictors <- c(
    p_predictor,
    PET_predictor,
    t_predictor,
    average_climatic,
    topography,
    geomorphology,
    land_cover
  )
  
  # Response names
  response <- c("Qmax")
  other_response <- c("Duration", "Qmean", "Qvol")
  
  # Context variables
  context <- c("Startdate", "Region")
  
  # Output ------------------------------------------------------------------
  
  data_process <- data_raw %>%
    select(all_of(c(response, context, predictors))) %>%
    mutate(Season = month(Startdate)) %>%
    mutate(Season = map_dbl(Season, function(x)
      ifelse(x >= 3 && x <= 9, 1, 0))) %>%
    select(-Startdate) %>%
    select(Qmax, Region, Season, everything())
  
  list(data_process = data_process,
       data_raw = data_raw)
}
