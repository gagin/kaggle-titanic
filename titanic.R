# setwd("kaggle-titanic/")
#train<-read.csv("../input/train.csv")
#test<-read.csv("../input/test.csv")
train<-read.csv("train.csv")
test<-read.csv("test.csv")

library(dplyr)
library(tidyr)
library(neuralnet)

### Plan
### 1. Glue the train and the test for processing
### 2. Deal with NAs and empty values
### 3. Fix types
### 4. Decide which strings can be numerized creatively
### 5. Normalize numbers
### 6. Numerize flat factors
### 7. Split
### 8. Create training, predicting, rounding and assessment functions
### 9. Train a bunch of nnets
kTries     <- 500
kSeedShift <- 1
## 9.1 Split train to train/test subsets
## 9.2 Train a net on train subset
## 9.3 Test the net on test subset, assess result
### 10. Select more productive nnets and do a bunch of predictions on them
### 11. Merge predictions
### Profit

## 1. Glue

# Reorder columns in train set to move result column to the last position
# and align column order for train and test
train<-train[, c(1, 3:ncol(train), 2)]

# Extend testset with an extra column
test$Survived <- NA

# Glue

all <- rbind(train, test)

# Are PassengerIDs consistent with row numbers?
# identical(1:length(all$PassengerId), all$PassengerId)
# They are.

## 2. Deal with NA

# Count NAs
# all %>% sapply(function(x){sum(is.na(x))})

# Mark ages to be approximated
all$Age.Known <- !is.na(all$Age)

# Will set NA ages after Titles are assigned - see step 4

# Also single NA for Fare - take mean for same class and port
# all[is.na(all$Fare), ]
all[is.na(all$Fare), ]$Fare <- mean(all$Fare[all$Pclass==3 &
                                                     all$Embarked == "S"], 
                                    na.rm=TRUE)
# Count NULLs
# all %>% sapply(function(x){sum(is.null(x))})

# Count empties
# all %>% sapply(function(x){sum(x=="", na.rm=TRUE)})

# Too many Cabins missing, let's drop it. Pity. Placement could have been an
# indicator.
all1 <- dplyr::select(all, -Cabin)

# What are passengers with the port missing?
all1[all1$Embarked=="", ]

# Suppose tickets with similar number sold for the same port?
# all[grepl(pattern = "1135", x = all$Ticket), ]
# No, but suppose we should take cabin placement and similar price
# Either that, or drop them, but these two lines could be more useful
# for other features than the error introduced by setting it to C
all1[all1$Embarked=="", ]$Embarked <-  "C"

### 3. Fix types

all1$Survived <- as.logical(all1$Survived)
all1$Pclass.factor <- factor(all1$Pclass)

### 4. Decide which strings can be numerized creatively

# Extract Ms/Miss/Mrs/Dr/Prof
all1$Title[grepl(pattern = "Miss", x = all$Name)] <- "Ms"
all1$Title[grepl(pattern = "Ms.", x = all$Name)] <- "Ms"
all1$Title[grepl(pattern = "Mlle.", x = all$Name)] <- "Ms"
all1$Title[grepl(pattern = "Mrs.", x = all$Name)] <- "Mrs"
all1$Title[grepl(pattern = "Mme.", x = all$Name)] <- "Mrs"
all1$Title[grepl(pattern = "Dr.", x = all$Name)] <- "Dr"
all1$Title[grepl(pattern = "Mr.", x = all$Name)] <- "Mr"
all1$Title[grepl(pattern = "Don.", x = all$Name)] <- "Mr"
all1$Title[grepl(pattern = "Master.", x = all$Name)] <- "Master"
all1$Title[grepl(pattern = "Rev.", x = all$Name)] <- "Rev"
all1$Title[grepl(pattern = "Col.", x = all$Name)] <- "Mil"
all1$Title[grepl(pattern = "Major.", x = all$Name)] <- "Mil"
all1$Title[grepl(pattern = "Capt.", x = all$Name)] <- "Mil"
all1$Title[grepl(pattern = "Jonkheer.", x = all$Name)] <- "Noble"
all1$Title[grepl(pattern = "Don.", x = all$Name)] <- "Noble"
all1$Title[grepl(pattern = "Dona.", x = all$Name)] <- "Noble"
all1$Title[grepl(pattern = "Countess.", x = all$Name)] <- "Noble"

# Anyone left?
# all$Name[is.na(all1$Title)]

# Set NA ages to mean for the class and sex and title
all1$Age <- mapply(function(age,class,sex,title) {
        if(is.na(age)) {
                res <- mean(all1[all1$Sex == sex &
                                 all1$Pclass == class &
                                 all1$Title == title, ]$Age,
                     na.rm=TRUE)
                if (is.na(res)) { # In case there's no mean for this group
                        res <- mean(all1[all1$Sex == sex &
                                                 all1$Pclass == class, ]$Age,
                                    na.rm=TRUE)}
                return(res)
        }
        else age},
        all1$Age,
        all1$Pclass,
        all1$Sex,
        all1$Title)

# Take name length, the nnet will drop it if irrelevant
all1 <- mutate(all1, Name.Length=sapply(as.character(Name),nchar))

# Drop names and tickets too
all2 <- dplyr::select(all1, -Ticket,-Name)

### 5. Normalize numbers

classes2 <- sapply(all2, class)
for(i in which(classes2 == "numeric" | classes2 == "integer"))
        if(i != 1) # Except for ID column
                all2[, i] <- (all2[, i] - min(all2[, i]))/
                             (max(all2[, i]) - min(all2[, i]))


### 6. Numerize flat factors


if (FALSE) { # Preparation code
        # Print a list of int/num/logic columns to exclude from numerizaton
        # Pay attention to 1/0 factors if they weren't converted to logical
        sapply(names(classes2[classes2 == "integer" | classes2 == "numeric" |
                                      classes2 == "logical"]),
               function(x) cat(paste0("-", x, ", \n")))
}

bins <- all2 %>% # piped functions follow
        
        # make it narrow, don't touch numeric variables and IDs
        gather(catnames, catvalues,
               -PassengerId, 
               -Pclass, 
               -Age, 
               -SibSp, 
               -Parch, 
               -Fare, 
               -Age.Known, 
               -Name.Length,
               -Survived) %>%
        
        # make single column out of them
        unite(newfactor, catnames, catvalues, sep=".") %>%
        
        # add a new column - it's "1" for every record
        mutate( is = 1) %>%
        
        # create a column from each factor, and where there's no record, add "0"
        spread(newfactor, is, fill=0)

### 7. Split

bins.train <- bins[train$PassengerId, ] %>% dplyr::select(-PassengerId)
bins.test <- bins[test$PassengerId, ]

### 8. Create training, predicting, rounding and assessment functions

if (FALSE) {  # Preparation code     
        # Prepare list for neuralnet() call
        cat(paste0(names(bins)[-1], sep=" +\n"))
        # Remove target one
}

CalculateNet <- function(df, rep=1, hidden=c(1), threshold=0.1) {
        neuralnet(Survived ~ Pclass +
                                  Age +
                                  SibSp +
                                  Parch +
                                  Fare +
                                  Age.Known +
                                  Name.Length +
                                  Embarked.C +
                                  Embarked.Q +
                                  Embarked.S +
                                  Pclass.factor.1 +
                                  Pclass.factor.2 +
                                  Pclass.factor.3 +
                                  Sex.female +
                                  Sex.male +
                                  Title.Dr +
                                  Title.Master +
                                  Title.Mil +
                                  Title.Mr +
                                  Title.Mrs +
                                  Title.Ms +
                                  Title.Noble +
                                  Title.Rev,
                          data=df,
                          hidden=hidden,
                          lifesign="full",
                          lifesign.step=2000,
                          threshold=threshold,
                          rep=rep)
}

# If algorithm did not converge in all repetitions, return 0s
TryPredict <- function(nnet,test) {
        if (is.null(nnet$weights)) matrix(0,nrow=nrow(test))
        else neuralnet::compute(nnet, test)$net.result
}

Chop <- function(res) {
        # Takes result of neuralnet's compute
        # Returns vector of 0/1 predictons
        # Uses ifelse() instead of if() b/c former is vectorized
        ifelse(res < 0,
               0,
               ifelse(res > 1,
                      1,
                      round(res)))
}

Qualify <- function(real, guess) {
        # round(100 * cor(real, guess)) # this was for numeric predictions
        check <- table(real, guess)
        # Some bad models will give all 1s or all 0s, then table won't be 2x2
        if(!all(dim(check) == c(2, 2))) percentage <- 0
        else {
                good.ones  <- check[1, 1] + check[2, 2]
                bad.ones   <- check[1, 2] + check[2, 1]
                percentage <- round(100 * good.ones / (good.ones + bad.ones))
        }
        return(percentage)
}


### 9. Train a bunch of nnets


nnets      <- list()
qualities  <- vector()

for(i in 1:kTries){
        # Print entry number
        cat("Training neural net #", i, "/", kTries, "\n", sep="")
        
        ## 9.1 Split train to train/test subsets
        nr <- nrow(bins.train)
        set.seed(i+kSeedShift)
        train.numbers <- sample.int(nr, round(0.9 * nr))
        trainers <- bins.train[train.numbers, ]
        testers  <- bins.train[-train.numbers, ]
        
        ## 9.2 Train a net on train subset
        set.seed(i+kSeedShift)
        r <- as.integer(runif(1,2,5))
        h <- as.integer(runif(1,1,30))
        t <- runif(1,0.001,1)
        nnets[[i]] <- CalculateNet(trainers, rep=r, hidden=h, threshold=t)

        ## 9.3 Test the net on test subset, assess result
        res <- TryPredict(nnets[[i]], dplyr::select(testers,-Survived))
        qualities[i] <- Qualify(Chop(res), testers$Survived)
        print(qualities[i])
}

### 10. Select more productive nnets and do a bunch of predictions on them
predictions <- matrix(NA, ncol=0, nrow=nrow(bins.test))
all.tries <- 1:kTries
productive.quality <- quantile(qualities,c(0.85))[1]##########################
good.tries <- all.tries[qualities >= productive.quality]
for(i in good.tries){
        res <- TryPredict(nnets[[i]],
                                  dplyr::select(bins.test,
                                                -PassengerId,
                                                -Survived))
        predictions <- cbind(predictions, Chop(res))
}

### 11. Merge predictions

predictions.sums <- rowSums(predictions)
cut.off <- mean(predictions.sums[predictions.sums!=0]) ###############
if (is.na(cut.off)) stop("Only zero results are present")
upload <- sapply(predictions.sums, function(x) if (x >= cut.off) 1 else 0)

### Profit

upload <- cbind(bins.test$PassengerId,upload)
colnames(upload) <- c("PassengerId","Survived")
write.csv(upload,file="res.csv",row.names=FALSE,quote=FALSE)
