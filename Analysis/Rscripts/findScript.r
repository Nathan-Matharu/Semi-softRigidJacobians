# source('findScript.r')

library(dplyr)
library(data.table)
setwd("/media/djtface/Chonky/Classes of milk/BME504/finalProject/computerVision/FinalVideos") #this should be changed based on the file system

inputFile <- "4cmMonteCarloDLC_resnet50_4cmMonteCarloDec1shuffle1_5000.csv" #this should be changed for each video
outputFile <- "4cmMonteCarloOut.csv" # this should be changed for each video
postureSeconds <- list(1.1,
4.7,
8.6,
12,
15.6,
19.3,
22.7,
26.3,
30,
33.4,
36.9,
40.5,
43.95,
47.4,
51,
54.8,
58,
61.2,
65,
68.4,
72,
75.2,
79.2,
82.7,
86.6,
89.9,
94,
96.8,
101,
103.9,
107.8,
111,
114.7,
118,
122.3,
126,
128.6,
132,
135.9,
139,
143,
146,
149.9,
153.4,
3) # this should be changed for each video
fRate <- 30

data <- read.csv(inputFile)
ground <- select(data, 2:4)
eps <- select(data, 5:7)
hMax <- select(data, 8:10)
vMax <- select(data, 11:13)

print("Beginning Analysis")

oFlen <- length(postureSeconds)
outFrame <- data.frame(matrix(0, oFlen, 3))

outFrame[1, ] <- data.frame("Ground X", "Ground Y", "Likelihood")
outFrame[3, ] <- data.frame("hMax X", "hMax Y", "Likelihood")
outFrame[5, ] <- data.frame("vMax X", "vMax Y", "Likelihood")
outFrame[7, ] <- data.frame("Endpoint X", "Endpoint Y", "Likelihood")

outFrame[2,] <- data.frame(ground %>% top_n(1))
outFrame[4,] <- data.frame(hMax %>% top_n(1))
outFrame[6,] <- data.frame(vMax %>% top_n(1))

j <- 8
for (i in postureSeconds) {
    hBound <- (i * fRate) + (fRate / 3)
    lBound <- (i * fRate) - (fRate / 3)
    if (lBound < 0) {
        lBound <- 0
    }
    if (hBound > nrow(eps)) {
        hBound <- nrow(eps)
    }
    tuv <- eps[lBound:hBound, c(1:3)]
    epMax <- data.frame(tuv %>% top_n(1))
    outFrame[j,] <- epMax
    j <- j + 1
}
print("Analysis Done")
fwrite(outFrame, outputFile, col.names = FALSE)
print("output file written")