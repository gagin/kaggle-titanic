# setwd("libertymutual/")
#train<-read.csv("../input/train.csv")
#test<-read.csv("../input/test.csv")
train<-read.csv("train.csv")
test<-read.csv("test.csv")

library(dplyr)
library(tidyr)
library(neuralnet)

## Prep

# check for NAs
# train %>% sapply(function(x){sum(is.na(x))})

# classes <- sapply(train, class)
# sapply(names(classes[classes=="integer"]),function(x) cat(paste0("-",x,",")))


bins<-train %>% # piped functions follow
        
        # make it narrow, don't touch numeric variables and IDs
        gather(catnames,catvalues,-Id,-Hazard,-T1_V1,-T1_V2,-T1_V3,-T1_V10,-T1_V13,-T1_V14,-T2_V1,-T2_V2,-T2_V4,-T2_V6,-T2_V7,-T2_V8,-T2_V9,-T2_V10,-T2_V14,-T2_V15) %>%
        
        # make single column out of them
        unite(newfactor,catnames,catvalues,sep=".") %>%
        
        # add a new column - it's "1" for every record
        mutate( is = 1) %>%
        
        # create a column from each factor, and where there's no record, add "0"
        spread(newfactor, is, fill = 0)

bins.test<-test %>% # piped functions follow
        
        # make it narrow, don't touch numeric variables and IDs
        gather(catnames,catvalues,-Id,-T1_V1,-T1_V2,-T1_V3,-T1_V10,-T1_V13,-T1_V14,-T2_V1,-T2_V2,-T2_V4,-T2_V6,-T2_V7,-T2_V8,-T2_V9,-T2_V10,-T2_V14,-T2_V15) %>%
        
        # make single column out of them
        unite(newfactor,catnames,catvalues,sep=".") %>%
        
        # add a new column - it's "1" for every record
        mutate( is = 1) %>%
        
        # create a column from each factor, and where there's no record, add "0"
        spread(newfactor, is, fill = 0)


seed<-2
##prepare list for neuralnet() call
#cat(paste0(names(bins),sep="+"))
bins.nn<-function(df,rep=1,hidden=c(1),threshold=0.1) {
        set.seed(seed)
        nn.obj<-neuralnet(Hazard ~ T1_V1+ T1_V2+ T1_V3+ T1_V10+ T1_V13+ T1_V14+ T2_V1+ T2_V2+ T2_V4+ T2_V6+ T2_V7+ T2_V8+ T2_V9+ T2_V10+ T2_V14+ T2_V15+ T1_V11.A+ T1_V11.B+ T1_V11.D+ T1_V11.E+ T1_V11.F+ T1_V11.H+ T1_V11.I+ T1_V11.J+ T1_V11.K+ T1_V11.L+ T1_V11.M+ T1_V11.N+ T1_V12.A+ T1_V12.B+ T1_V12.C+ T1_V12.D+ T1_V15.A+ T1_V15.C+ T1_V15.D+ T1_V15.F+ T1_V15.H+ T1_V15.N+ T1_V15.S+ T1_V15.W+ T1_V16.A+ T1_V16.B+ T1_V16.C+ T1_V16.D+ T1_V16.E+ T1_V16.F+ T1_V16.G+ T1_V16.H+ T1_V16.I+ T1_V16.J+ T1_V16.K+ T1_V16.L+ T1_V16.M+ T1_V16.N+ T1_V16.O+ T1_V16.P+ T1_V16.Q+ T1_V16.R+ T1_V17.N+ T1_V17.Y+ T1_V4.B+ T1_V4.C+ T1_V4.E+ T1_V4.G+ T1_V4.H+ T1_V4.N+ T1_V4.S+ T1_V4.W+ T1_V5.A+ T1_V5.B+ T1_V5.C+ T1_V5.D+ T1_V5.E+ T1_V5.H+ T1_V5.I+ T1_V5.J+ T1_V5.K+ T1_V5.L+ T1_V6.N+ T1_V6.Y+ T1_V7.A+ T1_V7.B+ T1_V7.C+ T1_V7.D+ T1_V8.A+ T1_V8.B+ T1_V8.C+ T1_V8.D+ T1_V9.B+ T1_V9.C+ T1_V9.D+ T1_V9.E+ T1_V9.F+ T1_V9.G+ T2_V11.N+ T2_V11.Y+ T2_V12.N+ T2_V12.Y+ T2_V13.A+ T2_V13.B+ T2_V13.C+ T2_V13.D+ T2_V13.E+ T2_V3.N+ T2_V3.Y+ T2_V5.A+ T2_V5.B+ T2_V5.C+ T2_V5.D+ T2_V5.E+ T2_V5.F,
                          data=df,
                          hidden=hidden,
                          lifesign="full",
                          lifesign.step=2000,
                          threshold=threshold,
                          rep=rep)
        return(nn.obj)}

qualify<-function(real,guess){
        round(100*cor(real,guess))
}


nfeat<-dim(bins)[2] 
nfeat.test<-dim(bins.test)[2] 

mult<-list()
eff<-vector()
tries=20
for(i in 1:tries){
        cat("Iteration #",i,"/",tries,"\n", sep="")
        set.seed(i)
        r<- 2#as.integer(runif(1,5,10))
        h<- 20#as.integer(runif(1,5,10))
        t<- 55#as.integer(runif(1,5,10))
        nr1<-dim(bins)[1]
        trainset<-sample.int(nr1,round(0.9*nr1))
        trainers<-bins[trainset,]
        testers<-bins[-trainset,]
        mult[[i]]<-bins.nn(trainers,rep=r,hidden=h,threshold=t)
        
        res<-neuralnet::compute(mult[[i]],testers[,3:nfeat])
        eff[i]<-qualify(res$net.result,
                          testers$Hazard)
        print(eff[i])
}

pult<-matrix(NA, nrow=dim(bins.test)[1])
alltries<-1:tries
mineff<-0#mean(eff)##########################
goodtries<-alltries[eff>mineff]
for(i  in goodtries){
        res<-neuralnet::compute(mult[[i]],bins.test[,2:nfeat.test])#testers[,3:nfeat])
        pult<-cbind(pult,res$net.result)                           
}
pult<-dplyr::select(as.data.frame(pult),-V1) # drop NA column
#predi<-rowSums(pult)
#cu<-mean(predi[predi!=0]) ###############
#cu<-0.5*max(predi)
#upload<-sapply(predi,function(x)if(x>cu) 1 else 0)
upload<-round(rowMeans(pult))

upload<-sapply(upload,function(x)if(x<0) 0 else x)

names(upload)<-c("Hazard")
upload1<-data.frame(cbind(bins.test$Id,upload))
names(upload1)<-c("Id","Hazard")
write.csv(upload1,file="res.csv",row.names=FALSE,quote=FALSE)