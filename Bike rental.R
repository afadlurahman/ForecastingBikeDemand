# LOADING PACKAGES
library(plyr)
library(dplyr)
library(ggplot2)
library(caret)
install.packages("caret")
library(lubridate)
library(reshape2)
library(tidyr)
library(MASS)
library(car)
library(xgboost)
library(randomForest)
library(ranger)
library(neuralnet)
library(superheat)
#===================================================================

# LOADING DATA
df_train <- read.csv("train.csv")
View(df_train)
summary(df_train)

str(df_train)

# pisah dulu variabel datetime jadi year,month,day,hour,minute,second
df_train$datetime <- ymd_hms(df_train$datetime)
df_train$year <- year(df_train$datetime)
df_train$month <- month(df_train$datetime)
df_train$day <- day(df_train$datetime)
df_train$hour <- hour(df_train$datetime)
#===============================================================================

# EXPLORATORY DATA ANALYSIS
# bike rentals each hour
ggplot(df_train, aes(x = hour, y = count))+
  geom_bar(stat="identity",fill="aquamarine3")+
  theme(legend.position="none")+ 
  labs(x = "Hour", y="Number of Rentals") + 
  scale_x_continuous(breaks = 0:23) 
ggsave("RentperHour.png", plot = last_plot(), units="in",  device = "png", width=10, height=7, 
       dpi=500)

# bike rentals each season
ggplot(df_train,
       aes(x = factor(season),
           y = count,fill=factor(season))) +
  geom_boxplot() + xlab("Season")+
  theme(legend.position="none")+
  scale_x_discrete(labels=c("Spring","Summer","Fall","Winter"))
ggsave("RentperSeason.png", plot = last_plot(), units="in",  device = "png", width=10, height=7, 
       dpi=500)

# bike rentals per weather
ggplot(df_train, aes(x=factor(weather), y=count,fill=factor(weather)))+
  geom_boxplot()+
  scale_x_discrete(labels=c('Clear', "Cloudy", 'Light Rain and Snow', 'Thunderstorm'))+
  labs(x="Weather",
       y="Count"
  )+ theme(legend.position="none")
ggsave("WeatherCount.png", plot = last_plot(), units="in",  device = "png", width=10, height=7, 
       dpi=500)

# summary bike rentals per month each season
monthly_seasonal_data <- df_train %>%
  group_by(month, season) %>%
  summarise(total_rentals = sum(count),.groups = 'drop') %>%
  ungroup()

ggplot(monthly_seasonal_data, aes(x = factor(month), y = total_rentals, fill = as.factor(season))) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Month",
       y = "Total Number of Rentals",
       fill = "Season") + 
  scale_fill_discrete(labels=c('Spring', 'Summer', 'Fall', 'Winter'))+
  theme(
    legend.position = c(0.999, 0.999), # x and y coordinates in [0,1] range
    legend.justification = c("right", "top") # Adjust legend to be justified from the right and top
  ) 
ggsave("RentperSeasonMonth.png", plot = last_plot(), units="in",  device = "png", width=10, height=7, 
       dpi=500)


month_seasonal_data <- df_train %>%
  group_by(month,weather) %>%
  summarise(mean_rentals = mean(count),.groups = 'drop') %>%
  ungroup()

ggplot(month_seasonal_data, aes(x = month, y = mean_rentals, group = weather)) +
  geom_line(aes(color=weather)) +
  geom_point(aes(color=weather)) +
  labs(color = "Weather")+
  guides(color = guide_legend(override.aes = list(shape = NA)))+
  labs(title = "Average Bike Rentals per Hour Across Season",
       x = "Hour",
       y = "Average Number of Rentals")

work_hour_data <- df_train %>%
  group_by(hour,workingday) %>%
  summarise(rentals = sum(count),.groups = 'drop') %>%
  ungroup()

ggplot(work_hour_data, aes(x = hour, y = rentals, group = factor(workingday))) +
  geom_bar(stat = "identity", position=position_dodge(preserve = "single"),
           aes(fill=factor(workingday))) +
  labs(x = "Hour",
       y = " Number of Rentals",
       fill = "Working Day" )+
  theme(
    legend.position = c(0.999, 0.999), # x and y coordinates in [0,1] range
    legend.justification = c("right", "top") # Adjust legend to be justified from the right and top
  ) +
  scale_x_continuous(breaks = 0:23) 
ggsave("RentperWorkingdayH.png", plot = last_plot(), units="in",  device = "png", width=10, height=7, 
       dpi=500)

ggplot(df_train, aes(x=temp,y=atemp))+
  geom_point(color="purple")+
  labs(x = "temperature",
       y = " Feels like temperature")
ggsave("TempAtemp.png", plot = last_plot(), units="in",  device = "png", width=10, height=7, 
       dpi=500)
cor(df_train$temp,df_train$atemp)

# the distribution of Count
ggplot(df_train, aes(x=count))+
  geom_histogram(bins=30, fill="brown4",color="white")+
  labs(x = "Bike Rentals",
       y = "Frekuensi")+
  theme(legend.position="none")
ggsave("BikeRent.png", plot = last_plot(), units="in",  device = "png", width=10, height=7, 
       dpi=500)

ggplot(df_train, aes(x=log(count))) +
  geom_histogram(fill="brown4",color="white")+
  labs(x = "Log Bike Rentals",
       y = "Frekuensi")+
  theme(legend.position="none")
ggsave("LogBikeRent.png", plot = last_plot(), units="in",  device = "png", width=10, height=7, 
       dpi=500)

ggplot(df_train, aes(x=factor(season), y=temp,fill=factor(season)))+
  geom_boxplot()+
  scale_x_discrete(labels=c('Spring', 'Summer', 'Fall', 'Winter'))+
  labs(x="Season",
       y="Temperature"
  )+ theme(legend.position="none")
ggsave("TempSeason.png", plot = last_plot(), units="in",  device = "png", width=10, height=7, 
       dpi=500)

data <- df_train %>%
  mutate(temp_bins = cut(
    temp, 
    breaks = seq(floor(min(temp)), ceiling(max(temp)), by = 2))
  )

# Summarize data to get count of rents in each temperature bin
binned_data <- data %>%
  group_by(temp_bins) %>%
  summarise(count_of_rents = sum(count),.groups = 'drop')

# Plot using ggplot2
ggplot(binned_data, aes(x = temp_bins, y = count_of_rents)) +
  geom_bar(stat = "identity", fill = "lightblue", color = "black") +
  labs(title = "Count of Rents by Temperature Bins", x = "Temperature (bins)", y = "Count of Rents") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#=================================================================================

# MODEL BUILDING USING TRAIN
# konversi kategori ke factor
df_train$season <- as.factor(df_train$season)
df_train$holiday <- as.factor(df_train$holiday)
df_train$workingday <- as.factor(df_train$workingday)
df_train$weather <- as.factor(df_train$weather)
df_train$year <- as.factor(df_train$year)

# Generalized Linear Model
# tambaha temp buat memastikan
glm1 <- lm(log(count) ~ hour + month + year + weather + season + holiday + 
             workingday + temp + humidity + windspeed + workingday*hour + 
             humidity*temp, data = df_train
)
summary(glm1)
step_glm1 <- step(glm1, direction="both")
summary(step_glm1)
vif(step_glm1)

# Interaksi kuat temp dengan atemp, jadi hilangkan salah satu

# dari seleksi glm, days cukup ga penting

# Random Forest
#days ilangin
set.seed(8)
rf_bikedata <- randomForest(log(count) ~ hour + month  + year + weather + season + 
                              workingday + temp + humidity + windspeed + humidity*temp + 
                              workingday*hour, data = df_train, mtry=6, 
                            importance = TRUE, ntree=1000, nodesize = 20)
rf_bikedata
varImpPlot(rf_bikedata, sort=T, main= "Variable Importance", pch=16)



# XGBOOST model
set.seed(888)
xgbGrid <- expand.grid(max_depth = 7, nrounds = 100, eta = c(0.01, 0.1), 
                       colsample_bytree = 0.6, gamma =c(0,1), min_child_weight = 1, 
                       subsample = 0.6
)

# days ilangin?
model.xgb.tuned <- train(log(count) ~ hour + month + year + weather + season + 
                           workingday + temp + humidity + windspeed + humidity*temp +
                           workingday*hour, data = df_train, 
                         method = "xgbTree", eval_metric ="error", tuneGrid=xgbGrid,
                         objective = "count:poisson"
)
model.xgb.tuned
ggplot(model.xgb.tuned)
imp <- varImp(model.xgb.tuned)
imp
plot(imp, top = 5)


#==============================================================================

# TESTING 
df_test <- read.csv("test.csv")

df_test$datetime_sep <- ymd_hms(df_test$datetime)
df_test$year <- year(df_test$datetime_sep)
df_test$month <- month(df_test$datetime_sep)
df_test$day <- day(df_test$datetime_sep)
df_test$hour <- hour(df_test$datetime_sep)

df_test$season <- as.factor(df_test$season)
df_test$holiday <- as.factor(df_test$holiday)
df_test$workingday <- as.factor(df_test$workingday)
df_test$weather <- as.factor(df_test$weather)
df_test$year <- as.factor(df_test$year)

# GLM test
predict_glm1 <- exp(predict(step_glm1, newdata=df_test, type = "response"))
output <- data.frame(datetime = df_test$datetime, count = predict_glm1)
output
write.csv(output, file = "sample_submission_glm1.csv", row.names = FALSE)

# Random Forest Test
predict_rf <- exp(predict(rf_bikedata, newdata = df_test))
output <- data.frame(datetime = df_test$datetime, count = predict_rf)
output
write.csv(output, file = "sample_submissionrf.csv", row.names = FALSE)

# XGBOOST Test
predict.xgb <- exp(predict(model.xgb.tuned, df_test, type="raw"))
output <- data.frame(datetime = df_test$datetime, count = predict.xgb)
output
write.csv(output, file = "sample_submission_xgb.csv", row.names = FALSE)