#Setting work directory
setwd("C:\\Users\\rishabh.parihar\\Desktop\\Amex Hack")
getwd()

#Loading libraries
library(dplyr)
library(missForest)
library(ROCR)
library(data.table)
library(mlr)
library(caTools)
library(h2o)
library(lubridate)
library(xgboost)
library(rfUtilities)
library(dummies)
library(ranger)
library(ROSE)
library(reshape)

#Reading train and test data
train <- fread("train.csv" , stringsAsFactors = FALSE , na.strings = c(" " , "" , "NA" , "NAN" , "NULL"))
history <- fread("historical_user_logs.csv" , stringsAsFactors = FALSE ,na.strings = c(" " , "" , "NA" , "NAN" , "NULL"))
test <- fread("test.csv" ,stringsAsFactors = FALSE , na.strings = c(" " , "" , "NA" , "NAN" , "NULL"))

test$is_click <- sample(0:1 , size = nrow(test) , replace = T)

#combining train and test data
alldata <- bind_rows(list(train, test))

#Imputing missing values
alldata$product_category_2 <- NULL

#Imputing with mode
alldata$user_group_id_miss_flag <- ifelse(is.na(alldata$user_group_id) , 1 , 0)
alldata$user_group_id[is.na(alldata$user_group_id)] <- 3

alldata$gender_miss_flag <- ifelse(is.na(alldata$gender) , 1 , 0)
alldata$gender[is.na(alldata$gender)] <- "Unknown"

alldata$age_level_miss_flag <- ifelse(is.na(alldata$age_level) , 1 , 0)
alldata$age_level[is.na(alldata$age_level)] <- 3

alldata$user_depth_miss_flag <- ifelse(is.na(alldata$user_depth) , 1 , 0)
alldata$user_depth[is.na(alldata$user_depth)] <- 3

alldata$city_development_index_miss_flag <- ifelse(is.na(alldata$city_development_index) , 1 , 0)
alldata$city_development_index[is.na(alldata$city_development_index)] <- 2

#Creating features from datetime variables

alldata$hour_of_day <- as.numeric(substr(alldata$DateTime , 12,13))
alldata$minute <- as.numeric(substr(alldata$DateTime , 15,16))

#de-duplicate the history data

history <- history[!duplicated(history[,1:4])]
head(history)

#Getting user-product level features from historical log data
#Number of times the user has viewed a particular product
#Number of distinct products that user has viewed


user_product_views <- mutate(history , viewed = ifelse(action %in% c("view") , 1 , 0) , 
                                       interested = ifelse(action %in% c("interest") , 1 , 0),
                                       month_day = substr(history$DateTime , 6,10) )%>%
                                        group_by(user_id , product)%>%summarise(view_counts = sum(viewed) ,
                                        interested_counts = sum(interested) ,
                                        days_viewed = n_distinct(month_day) , 
                                        last_viewed_date = max(DateTime))

user_tot_products_viewed <- user_product_views%>%group_by(user_id)%>%summarise(number_of_products = n_distinct(product))

# user_product_view_pivot_table <- cast(user_product_views[1:3] , user_id ~ product ,fun.aggregate = sum ,  fill = 0) 
# user_product_view_pivot_table <- data.frame(user_product_view_pivot_table)
# head(user_product_view_pivot_table)
# 
# #Applying k means clustering on this data
# 
# set.seed(1)
# 
# clusters <- kmeans(user_product_view_pivot_table[,-1] , centers = 10)
# user_product_view_pivot_table$cluster <- clusters$cluster


# user_surfing_hour_day <- mutate(history , month_day = substr(DateTime , 6,10))%>%
#                      group_by(user_id , month_day)%>%summarise(in_time = min(DateTime) , 
#                                                                out_time = max(DateTime))
# 
# user_surfing_hour_day <- mutate(user_surfing_hour_day , time_spent = difftime(out_time , in_time , units = "mins"))

#Number of maximum views and interests of a particular product in a day
user_product_frequent_views <- mutate(history , viewed = ifelse(action %in% c("view") , 1 , 0) , 
                                      interested = ifelse(action %in% c("interest") , 1 , 0),
                                      month_day = substr(history$DateTime , 6,10))%>%
                                      group_by(user_id , product , month_day)%>%summarise(frequent_views = sum(viewed) , 
                                      frequent_interests = sum(interested))%>%group_by(user_id , product)%>%
                                      summarise(max_frequent_views = max(frequent_views) , 
                                                max_frequent_interests = max(frequent_interests))

#Number of total views and interests of a user in historical logs
user_tot_views_interests <- mutate(history , viewed = ifelse(action %in% c("view") , 1 , 0) , 
                                   interested = ifelse(action %in% c("interest") , 1 , 0))%>%
                            group_by(user_id)%>%summarise(tot_views = sum(viewed) , 
                                                          tot_interests = sum(interested))

#Number of days spent online by user
user_online_days <- mutate(history , month_day = substr(history$DateTime , 6,10))%>%
                    group_by(user_id)%>%summarise(days_spent_online = n_distinct(month_day))

#Merging with data
alldata_new <- merge(alldata , user_product_views , by = c("user_id" , "product") , 
                 all.x = T , all.y = F, sort = F)

alldata_new <- merge(alldata_new , user_product_frequent_views , by = c("user_id" , "product") , 
                     all.x = T , all.y = F, sort = F)

alldata_new <- merge(alldata_new , user_online_days , by = c("user_id") , 
                     all.x = T , all.y = F, sort = F)

alldata_new <- merge(alldata_new , user_tot_products_viewed , by = "user_id" , all.x = T , all.y = F , sort = F)

alldata_new <- merge(alldata_new , user_product_view_pivot_table[,c(1,12)] , by = c("user_id") , all.x = T , all.y = F , sort = F)

alldata_new <- merge(alldata_new , user_tot_views_interests , by = c("user_id") , all.x = T , all.y = F , sort = F)

#max_frequent_views , max_frequent_interests , days_spent_online

alldata_new$view_counts[is.na(alldata_new$view_counts)] = 0

alldata_new$interested_counts[is.na(alldata_new$interested_counts)] = 0

alldata_new$interested_fraction <- ifelse(alldata_new$view_counts == 0 , 0 ,
                                          alldata_new$interested_counts/alldata_new$view_counts)

alldata_new$days_viewed[is.na(alldata_new$days_viewed)] <- 0

alldata_new$last_viewed_date <- ifelse(is.na(alldata_new$last_viewed_date) , alldata_new$DateTime ,
                                     alldata_new$last_viewed_date)

alldata_new$user_interest_fraction <- ifelse(alldata_new$tot_views == 0 , 0 ,
                                             alldata_new$tot_interests/alldata_new$tot_views)
  
alldata_new$days_since_last_viewed <- as.Date(alldata_new$DateTime) - as.Date(alldata_new$last_viewed_date)

alldata_new$max_frequent_views[is.na(alldata_new$max_frequent_views)] =  0

alldata_new$max_frequent_interests[is.na(alldata_new$max_frequent_interests)] =  0

alldata_new$days_spent_online[is.na(alldata_new$days_spent_online)] = 0

#viewed before flag
alldata_new$viewed_before_flag <- ifelse(alldata_new$view_counts==0 , 0 , 1)

#interested flag
alldata_new$interested_flag <- ifelse(alldata_new$interested_counts > 0 , 1 , 0)

#views per day
alldata_new$views_per_day <- ifelse(alldata_new$days_viewed == 0 , 0 , alldata_new$view_counts/alldata_new$days_viewed)

train_new <- alldata_new[1:nrow(train) , ]
test_new <- alldata_new[-(1:nrow(train)) , ]

#target encoding various categorical variables combinations

user_level_products <- train_new%>%group_by(user_id)%>%summarise(products_viewed = n_distinct(product))
user_level_campaigns <- train_new%>%group_by(user_id)%>%summarise(number_of_campaigns = n_distinct(campaign_id))
user_level_webpages <- train_new%>%group_by(user_id)%>%summarise(number_of_webpages = n_distinct(webpage_id))
campaign_id_event_rate <- train_new%>%group_by(campaign_id)%>%summarise(campaign_id_event_rate = mean(is_click)*100)
webpage_id_event_rate <- train_new%>%group_by(webpage_id)%>%summarise(webpage_id_event_rate = mean(is_click)*100)
campaign_webpage_event_rate <- train_new%>%group_by(campaign_id , webpage_id)%>%summarise(campaign_webpage_id_event_rate = mean(is_click)*100)
campaign_webpage_product_event_rate <- train_new%>%group_by(campaign_id , webpage_id , product)%>%summarise(campaign_webpage_product_event_rate = mean(is_click)*100)
campaign_product_event_rate <- train_new%>%group_by(campaign_id , product)%>%summarise(campaign_product_event_rate = mean(is_click)*100)
product_cats_event_rate <-train_new%>%group_by(product,product_category_1)%>%summarise(product_cats_event_rate = mean(is_click)*100)
user_group_event_rate <- train_new%>%group_by(user_group_id)%>%summarise(user_group_event_rate = mean(is_click)*100)
age_level_event_rate <- train_new%>%group_by(age_level)%>%summarise(age_level_event_rate = mean(is_click)*100)
user_depth_event_rate <- train_new%>%group_by(user_depth)%>%summarise(user_depth_event_rate = mean(is_click)*100)
user_group_depth_event_rate <- train_new%>%group_by(user_group_id , user_depth)%>%summarise(user_group_depth_event_rate = mean(is_click)*100)
development_index_event_rate <-train_new%>%group_by(city_development_index)%>%summarise(development_index_event_rate = mean(is_click)*100)
product_category_1_event_rate <- train_new%>%group_by(product_category_1)%>%summarise(product_category_1_event_rate = mean(is_click)*100)
product_event_rate <- train_new%>%group_by(product)%>%summarise(product_event_rate = mean(is_click)*100)
gender_event_rate <- train_new%>%group_by(gender)%>%summarise(gender_event_rate = mean(is_click)*100)

#combining train and test sets
alldata_new <- bind_rows(list(train_new , test_new))

#Merging
alldata_new <- merge(alldata_new , user_level_products, by = "user_id" , all.x = T , all.y = F, sort = F)
alldata_new <- merge(alldata_new , user_level_campaigns, by = "user_id" , all.x = T , all.y = F, sort = F)
alldata_new <- merge(alldata_new , user_level_webpages , by = "user_id" , all.x = T , all.y = F, sort = F)
alldata_new <- merge(alldata_new , campaign_id_event_rate , by = "campaign_id" , all.x = T , all.y = F, sort = F)
alldata_new <- merge(alldata_new , webpage_id_event_rate , by = "webpage_id" , all.x = T , all.y = F, sort = F)
alldata_new <- merge(alldata_new , campaign_webpage_event_rate , by = c("campaign_id" , "webpage_id") , all.x = T , all.y = F, sort = F)
alldata_new <- merge(alldata_new , campaign_webpage_product_event_rate , by = c("campaign_id" , "webpage_id" , "product") , all.x = T , all.y = F, sort = F)
alldata_new <- merge(alldata_new , campaign_product_event_rate , by = c("campaign_id" , "product") , all.x = T , all.y = F, sort = F)
alldata_new <- merge(alldata_new , product_cats_event_rate , by = c("product" , "product_category_1") , all.x = T , all.y = F, sort = F)
alldata_new <- merge(alldata_new , user_group_event_rate , by = "user_group_id" , all.x = T , all.y = F, sort = F)
alldata_new <- merge(alldata_new , age_level_event_rate , by = "age_level" , all.x = T , all.y = F, sort = F)
alldata_new <- merge(alldata_new , user_depth_event_rate , by = "user_depth" , all.x = T , all.y = F, sort = F)
alldata_new <- merge(alldata_new , user_group_depth_event_rate , by = c("user_group_id" , "user_depth") , all.x = T , all.y = F, sort = F)
alldata_new <- merge(alldata_new , development_index_event_rate , by = "city_development_index" , all.x = T , all.y = F, sort = F)
alldata_new <- merge(alldata_new , product_category_1_event_rate , by = "product_category_1" , all.x = T , all.y = F, sort = F)
alldata_new <- merge(alldata_new , product_event_rate , by = "product" , all.x = T , all.y = F, sort = F)
alldata_new <- merge(alldata_new , gender_event_rate , by = "gender" , all.x = T , all.y = F, sort = F)

#Imputing missing values
alldata_new[is.na(alldata_new)] <- 0
alldata_new$day <- as.numeric(substr(alldata_new$DateTime , 10 , 11))
names(alldata_new)

#Variables to dummy encode
ohe_feats <- c("product" , "gender" , "webpage_id" , "campaign_id" , "cluster" , "user_group_id" ,
               "city_development_index")

# 
alldata_new <- dummy.data.frame(alldata_new , names = ohe_feats)
#dropping variables

drop_vars <- c("user_id","session_id" , "DateTime" , "product_category_1",
               "last_viewed_date" )

for (i in drop_vars){
  
  alldata_new[[i]] = NULL
}

View(alldata_new)

for (i in names(alldata_new)){
  
  alldata_new[[i]] = as.numeric(alldata_new[[i]])
}

#
dim(train);dim(test)

dim(alldata_new)

names(alldata_new) <- make.names(names(alldata_new))

train_final <- alldata_new[(1:nrow(train)) , ]
test_final <- alldata_new[-(1:nrow(train)) , ]
test_final <- as.data.frame(test_final)
#train validation split
train_set <- as.data.frame(train_final[train_final$day < 6,])
prop.table(table(train_set$is_click))

indep_vars <- names(train_set)[!names(train_set) %in% c("is_click" , "day")]
dep_var <- "is_click"

validation_set <- as.data.frame(train_final[train_final$day >=6,])
prop.table(table(validation_set$is_click))

get_auc <- function(label  , probs){
  library(ROCR)
  AUC <- ROCR::performance(ROCR::prediction(probs,label) , "auc")@y.values[[1]]
  return (AUC)
}

##Fitting xgboost
set.seed(1)

xgb_model <- xgboost(data = data.matrix(train_set[,indep_vars]) , 
                     label = data.matrix(train_set[,dep_var]) ,
                    # watchlist = list(data.matrix(validation_set)) ,
                     nrounds = 100,objective = "binary:logistic",
                     params = list(booster = "gbtree" , eta = 0.1 , max_depth = 5 , eval_metric = "auc" , 
                                   colsample_by_tree = 0.5) ,
                     print_every_n = 100,
                     early_stopping_rounds = 5)
head(train_set[,top_20_vars])


#Predicting on validation data
xgb_valid <- predict(xgb_model , data.matrix(validation_set[,indep_vars]))

xgb_auc <- get_auc(label = validation_set$is_click , probs = xgb_valid)
xgb_auc

#0.5730

#predicting on test set
xgb_test <- predict(xgb_model ,data.matrix(test_final[,indep_vars]))
head(xgb_test)

#Reading submission file
submit <- read.csv("sample_submission_2s8l9nF.csv")

xgb_sub_1 <- data.frame(session_id = submit$session_id , is_click = xgb_test)
write.csv(xgb_sub_1 , "xgb_submission_valid_0.5935.csv" , row.names = F)

##Fitting glm model

glm_model <- glm(is_click ~ . , data = mutate(train_set , is_click = as.factor(is_click)) , family = binomial)

summary(glm_model)

glm_train_auc <- get_auc(train_set$is_click , glm_model$fitted.values)
glm_train_auc

glm_valid <- predict(glm_model , validation_set , type = "response")

glm_valid_auc <- get_auc(validation_set$is_click , glm_valid)
glm_valid_auc

glm_test <- predict(glm_model , newdata = test_final , type = "response")
cor(glm_valid , xgb_valid)

#Reading submission file
submit <- read.csv("sample_submission_2s8l9nF.csv")

glm_sub_1 <- data.frame(session_id = submit$session_id , is_click = glm_test)
write.csv(glm_sub_1 , "glm_submission_valid_0.5928.csv" , row.names = F)

cor(xgb_valid , glm_valid)

#ensemble glm and xgboost
glm_xgb_valid<- (0.65*glm_valid + 0.35*xgb_valid)
get_auc(label = validation_set$is_click , probs =  glm_xgb_valid)

#Reading submission file
submit <- read.csv("sample_submission_2s8l9nF.csv")

glm_xgb_sub_1 <- data.frame(session_id = submit$session_id , is_click = (0.4*xgb_test + 0.6*glm_test))
write.csv(glm_xgb_sub_1 , "glm_xgb_submission_valid_0.5952.csv" , row.names = F)





