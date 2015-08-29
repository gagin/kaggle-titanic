# setwd("libertymutual/")
#train<-read.csv("../input/train.csv")
#test<-read.csv("../input/test.csv")
train<-read.csv("train.csv")
test<-read.csv("test.csv")

library(dplyr)
library(tidyr)
library(neuralnet)

#### Prep

### Plan
### 1. Glue the train and the test for processing
### 2. Deal with NAs and empty values
### 3. Decide which strings can be numerized creatively
### 4. Normalize numbers
### 5. Numerize flat factors
### 6. Split

### 1. Glue

## Reorder columns in train set to move result column to the last position
## and equalize column order for train and test
train<-train[, c(1, 3:ncol(train), 2)]

## Extend testset with extra column
test$Survived <- NA

## Glue

all <- rbind(train, test)

# Are PassengerIDs consistent with row numbers?



### 2. Deal with NA

## Count NAs
all %>% sapply(function(x){sum(is.na(x))})

# set NA ages to mean for the class and sex
newage <- function(age,class,sex){if(is.na(age)){
        mean(all[all$Sex==sex &
                         all$Pclass==class, ]$Age, na.rm=TRUE)}
        else age
}

train$newage <- mapply(newage, train$Age, train$Pclass, train$Sex)

## Count empties
all %>% sapply(function(x){sum(x=="", na.rm=TRUE)})


## Normalize non-factor columns
train0 <- train[, 2:(ncol(train)-1)]
all <- rbind(train0,test[, 2:ncol(test)])
classes <- sapply(all, class)
for(i in which(classes=="integer" | classes== "numeric")){
        from<-min(all[, i])
        print(from)
}
        
mins <- sapply(all, min)

if (FALSE) { # Preparation code
        # Check for NAs
        train %>% sapply(function(x){sum(is.na(x))})
        
        # Print a list of integer columns to exclude from numerizaton
        classes <- sapply(train, class)
        sapply(names(classes[classes=="integer"]),
               function(x) cat(paste0("-", x, ",")))
}

bins <- train %>% # piped functions follow
        
        # make it narrow, don't touch numeric variables and IDs
        gather(catnames, catvalues, -Id,-Hazard,-T1_V1,-T1_V2,-T1_V3,-T1_V10,-T1_V13,-T1_V14,-T2_V1,-T2_V2,-T2_V4,-T2_V6,-T2_V7,-T2_V8,-T2_V9,-T2_V10,-T2_V14,-T2_V15) %>%
        
        # make single column out of them
        unite(newfactor, catnames, catvalues, sep=".") %>%
        
        # add a new column - it's "1" for every record
        mutate( is = 1) %>%
        
        # create a column from each factor, and where there's no record, add "0"
        spread(newfactor, is, fill=0)

bins.test <- test %>% # piped functions follow
        
        # make it narrow, don't touch numeric variables and IDs
        gather(catnames, catvalues, -Id,-T1_V1,-T1_V2,-T1_V3,-T1_V10,-T1_V13,-T1_V14,-T2_V1,-T2_V2,-T2_V4,-T2_V6,-T2_V7,-T2_V8,-T2_V9,-T2_V10,-T2_V14,-T2_V15) %>%
        
        # make single column out of them
        unite(newfactor, catnames, catvalues, sep=".") %>%
        
        # add a new column - it's "1" for every record
        mutate(is=1) %>%
        
        # create a column from each factor, and where there's no record, add "0"
        spread(newfactor, is, fill=0)


kSeed<-2
##prepare list for neuralnet() call
#cat(paste0(names(bins),sep="+"))
CalculateNet <- function(df, rep=1, hidden=c(1), threshold=0.1) {
        set.seed(kSeed)
        nn.obj<-neuralnet(Hazard ~ T1_V1+ T1_V2+ T1_V3+ T1_V10+ T1_V13+ T1_V14+ T2_V1+ T2_V2+ T2_V4+ T2_V6+ T2_V7+ T2_V8+ T2_V9+ T2_V10+ T2_V14+ T2_V15+ T1_V11.A+ T1_V11.B+ T1_V11.D+ T1_V11.E+ T1_V11.F+ T1_V11.H+ T1_V11.I+ T1_V11.J+ T1_V11.K+ T1_V11.L+ T1_V11.M+ T1_V11.N+ T1_V12.A+ T1_V12.B+ T1_V12.C+ T1_V12.D+ T1_V15.A+ T1_V15.C+ T1_V15.D+ T1_V15.F+ T1_V15.H+ T1_V15.N+ T1_V15.S+ T1_V15.W+ T1_V16.A+ T1_V16.B+ T1_V16.C+ T1_V16.D+ T1_V16.E+ T1_V16.F+ T1_V16.G+ T1_V16.H+ T1_V16.I+ T1_V16.J+ T1_V16.K+ T1_V16.L+ T1_V16.M+ T1_V16.N+ T1_V16.O+ T1_V16.P+ T1_V16.Q+ T1_V16.R+ T1_V17.N+ T1_V17.Y+ T1_V4.B+ T1_V4.C+ T1_V4.E+ T1_V4.G+ T1_V4.H+ T1_V4.N+ T1_V4.S+ T1_V4.W+ T1_V5.A+ T1_V5.B+ T1_V5.C+ T1_V5.D+ T1_V5.E+ T1_V5.H+ T1_V5.I+ T1_V5.J+ T1_V5.K+ T1_V5.L+ T1_V6.N+ T1_V6.Y+ T1_V7.A+ T1_V7.B+ T1_V7.C+ T1_V7.D+ T1_V8.A+ T1_V8.B+ T1_V8.C+ T1_V8.D+ T1_V9.B+ T1_V9.C+ T1_V9.D+ T1_V9.E+ T1_V9.F+ T1_V9.G+ T2_V11.N+ T2_V11.Y+ T2_V12.N+ T2_V12.Y+ T2_V13.A+ T2_V13.B+ T2_V13.C+ T2_V13.D+ T2_V13.E+ T2_V3.N+ T2_V3.Y+ T2_V5.A+ T2_V5.B+ T2_V5.C+ T2_V5.D+ T2_V5.E+ T2_V5.F,
                          data=df,
                          hidden=hidden,
                          lifesign="full",
                          lifesign.step=2000,
                          threshold=threshold,
                          rep=rep)
        return(nn.obj)
}

Qualify <- function(real, guess) {
        round(100 * cor(real, guess))
}


nfeat <- ncol(bins)
nfeat.test <- ncol(bins.test)

mult <- list()
eff  <- vector()

kTries=20
for(i in 1:tries){
        cat("Iteration #", i, "/", kTries, "\n", sep="")
        set.seed(i)
        r <- 2#as.integer(runif(1,5,10))
        h <- 20#as.integer(runif(1,5,10))
        t <- 55#as.integer(runif(1,5,10))
        nr1 <- nrow(bins)
        trainset <- sample.int(nr1, round(0.9 * nr1))
        trainers <- bins[trainset, ]
        testers  <- bins[-trainset, ]
        mult[[i]] <- CalculateNet(trainers, rep=r, hidden=h, threshold=t)
        
        res <- neuralnet::compute(mult[[i]], testers[, 3:nfeat])
        eff[i] <- Qualify(res$net.result, testers$Hazard)
        print(eff[i])
}

pult <- matrix(NA, nrow=nrow(bins.test))
all.tries <- 1:kTries
min.eff <- 0 #mean(eff)##########################
good.tries <- all.tries[eff>min.eff]
for(i  in good.tries){
        res <- neuralnet::compute(mult[[i]], bins.test[,2:nfeat.test])
        pult <- cbind(pult, res$net.result)                           
}
pult <- dplyr::select(as.data.frame(pult), -V1) # drop NA column
#predi<-rowSums(pult)
#cu<-mean(predi[predi!=0]) ###############
#cu<-0.5*max(predi)
#upload<-sapply(predi,function(x)if(x>cu) 1 else 0)
upload <- round(rowMeans(pult))

upload <- sapply(upload, function(x) if(x<0) 0 else x)

names(upload) <- c("Hazard")
upload1 <- data.frame(cbind(bins.test$Id, upload))
names(upload1) <- c("Id", "Hazard")
write.csv(upload1, file="res.csv", row.names=FALSE, quote=FALSE)