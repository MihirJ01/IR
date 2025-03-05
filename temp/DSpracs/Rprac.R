attach(mtcars)
plot(wt,mpg,main="Scatterplot Example",xlab="Car Weight",ylab="Miels Per Gallon",pch=19)




slices<-c(20,20,30,30)
label<-c("INDIA","US","DUBAI","FRANCE")
pct<-round(slices/sum(slices)*100)
label<-paste(label,pct)
label<-paste(label,"%",sep="")
pie(slices,labels=label,col=rainbow(length(label)),main="Pie Chart of Countries")




boxplot(mpg~cyl,data=mtcars,main="Car Milege Data",
        xlab="Number of Cylinders",ylab="Miles per Gallon")





slices<-c(20,20,30,30)
label<-c("INDIA","US","DUBAI","FRANCE")
pie(slices,labels=label,col=rainbow(length(label)),main="Pie Chart of Countries")





counts<-table(mtcars$vs,mtcars$gear)
barplot(counts,main="Car Distribution by Gears and VS",
        xlab="Numbers of Gears" ,col=c("darkblue","red"),
        legend=row.names(counts),beside=TRUE)





counts <- table(mtcars$vs, mtcars$gear)
barplot(counts, main = "Car Distribution", horiz = TRUE,
        names.arg = c("3 Gears", "4 Gears", "5 Gears"))
