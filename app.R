categoriseFeatures<-function(names)
{
  featurecategory = vector(length = length(names))
  for(i in c(1:length(names)))
  {
    if(grepl("Vol",names[i]) == TRUE || grepl("Rad",names[i]) == TRUE || grepl("Len",names[i]) == TRUE
       || grepl("Wid",names[i]) == TRUE || grepl("Area",names[i]) == TRUE)
    {
      featurecategory[i] = "Size"
    }
    
    if(grepl("Dis",names[i]) == TRUE || grepl("Trac",names[i]) == TRUE || grepl("trajArea",names[i]) == TRUE
       || grepl("Vel",names[i]) == TRUE || grepl("D2T",names[i]) == TRUE)
    {
      featurecategory[i] = "Movement"
    }
    
    if(grepl("Sph",names[i]) == TRUE || grepl("VfC",names[i]) == TRUE || grepl("Curv",names[i]) == TRUE
       || grepl("A2B",names[i]) == TRUE || grepl("Rect",names[i]) == TRUE 
       || grepl("poly",names[i]) == TRUE  || grepl("Box",names[i]) == TRUE)
    {
      featurecategory[i] = "Shape"
    }
    
    if(grepl("FO",names[i]) == TRUE || grepl("IQ",names[i]) == TRUE || grepl("Cooc",names[i]) == TRUE)
    {
      featurecategory[i] = "Texture"
    }
    
    if(grepl("den",names[i]) == TRUE && grepl("Cooc",names[i]) == FALSE)
    {
      featurecategory[i] = "Density"
    }
  }
  return(featurecategory)
}

categoriseDataFeatures<-function(dataset)
{
  featurecategory = vector(length = dim(dataset)[2])
  for(i in c(1:dim(dataset)[2]))
  {
    if(grepl("Vol",colnames(dataset)[i]) == TRUE || grepl("Rad",colnames(dataset)[i]) == TRUE || grepl("Len",colnames(dataset)[i]) == TRUE
       || grepl("Wid",colnames(dataset)[i]) == TRUE || grepl("Area",colnames(dataset)[i]) == TRUE || grepl("Box",colnames(dataset)[i]) == TRUE)
    {
      featurecategory[i] = "Size"
    }
    
    if(grepl("Dis",colnames(dataset)[i]) == TRUE || grepl("Trac",colnames(dataset)[i]) == TRUE || grepl("trajArea",colnames(dataset)[i]) == TRUE
       || grepl("Vel",colnames(dataset)[i]) == TRUE || grepl("D2T",colnames(dataset)[i]) == TRUE)
    {
      featurecategory[i] = "Movement"
    }
    
    if(grepl("Sph",colnames(dataset)[i]) == TRUE || grepl("VfC",colnames(dataset)[i]) == TRUE || grepl("Curv",colnames(dataset)[i]) == TRUE
       || grepl("A2B",colnames(dataset)[i]) == TRUE || grepl("Rect",colnames(dataset)[i]) == TRUE 
       || grepl("poly",colnames(dataset)[i]) == TRUE)
    {
      featurecategory[i] = "Shape"
    }
    
    if(grepl("FO",colnames(dataset)[i]) == TRUE || grepl("IQ",colnames(dataset)[i]) == TRUE || grepl("Cooc",colnames(dataset)[i]) == TRUE)
    {
      featurecategory[i] = "Texture"
    }
    
    if(grepl("den",colnames(dataset)[i]) == TRUE && grepl("Cooc",colnames(dataset)[i]) == FALSE)
    {
      featurecategory[i] = "Density"
    }
  }
  return(featurecategory)
}


categoriseSummaryStat<-function(dataset)
{
  featurecategory = vector(length = dim(dataset)[2])
  for(i in c(1:dim(dataset)[2]))
  {
    if(grepl("mean",colnames(dataset)[i]) == TRUE)
    {
      featurecategory[i]="mean"
    }
    
    if(grepl("std",colnames(dataset)[i]) == TRUE)
    {
      featurecategory[i]="std"
    }
    
    if(grepl("skew",colnames(dataset)[i]) == TRUE)
    {
      featurecategory[i]="skew"
    }
    
    if(grepl("asc",colnames(dataset)[i]) == TRUE)
    {
      featurecategory[i]="asc"
    }
    
    if(grepl("des",colnames(dataset)[i]) == TRUE)
    {
      featurecategory[i]="des"
    }
    
    if(grepl("max",colnames(dataset)[i]) == TRUE)
    {
      featurecategory[i]="max"
    }
  }
  return(featurecategory)
}

subsetBySeparationThreshold<-function(outputfile,separationscores,t)
{
  subbedOutputFile<-outputfile[ ,separationscores[[t]][,2]]
  return(subbedOutputFile)
}

prepareSegmentationTrainingSet<-function(segerrors,correctsegs)
{
  Segerrortraining <- segerrors
  Correctsegtraining <- correctsegs
  
  ## collate correct segmentation and segmentation error data into one data frame
  data = rbind(Segerrortraining, Correctsegtraining) 
  segerrordata<-Segerrortraining[,-1] 
  correctsegdata<-Correctsegtraining[,-1]
  alldata = rbind(segerrordata, correctsegdata)
  
  ## first column lists ground truth data labels
  class1 = rep("segerror", dim(segerrordata)[1])
  class2 = rep("correct", dim(correctsegdata)[1]) 
  class = c(class1, class2)
  all = data.frame(class, alldata)
  
  ## SMOTE() to over-sample segmentation errors 
  smoteseg<-smotefamily::SMOTE(all[,-1], all$class, K = 3, dup_size = 1)
  
  ## add smote segmentation errors to the training sets and update segmentation error data table
  smotesegsyn_data<-smoteseg$syn_data[,c((dim(segerrors)[2]), 1:(dim(segerrors)[2]-1))] 
  all<-rbind(all, smoteseg$syn_data) 
  segdata<-rbind(segerrordata, smotesegsyn_data[,-1]) 
  newclass<-c(class, smotesegsyn_data[,1]) 
  class1 = rep("segerror", dim(segerrordata)[1])
  seginfo<-list(segerrordata, correctsegdata, class1, class2)
  return(seginfo)
}


predictSegErrors<-function(segerrors, correctsegs,
                           num, testset, dataID, proportion) 
{ 
  seginfo<-prepareSegmentationTrainingSet(segerrors, correctsegs)
  smalldata = seginfo[[1]]
  bigdata = seginfo[[2]]
  smallclass = seginfo[[3]]
  bigclass = seginfo[[4]]
  n1 = length(smallclass)
  n2 = length(bigclass)
  treelist = list()
  for (i in 1:num)
  { 
    inds = sample.int(n2, n1)
    data = rbind(bigdata[inds,],smalldata)
    class = c(bigclass[1:n1], smallclass)
    data = data.frame(class, data)
    mytree = tree::tree(as.factor(class)~., data=data)
    treelist[[i]] <- mytree 
  } 
  
  trees<-treelist
  
  pred = matrix(" ", nrow = num, ncol = nrow(testset)) 
  for (i in 1:num)
  { 
    x = predict(trees[[i]], testset, type = "class")
    pred[i,] <- x 
  } 
  predictions<-pred
  
  testnumseg = vector(mode = "integer", length = ncol(predictions)) 
  for (i in 1:ncol(predictions))
  { 
    testnumseg[i] = length(which(predictions[,i] == 2)) 
  } 
  testvote = rep("nonseg", length = ncol(predictions))
  for (i in 1:ncol(predictions))
  { 
    if (testnumseg[i] >  proportion * nrow(predictions)) testvote[i] = "seg" 
  } 
  ind = which(testvote == "seg")
  segtest = dataID[ind]
  return(segtest)
}


predictSegErrors_Ensemble<-function(segerrors, correctsegs, num, K, testset, dataID, proportion) 
{ 
  votes<-list()
  for(i in c(1:K))
  {
    incProgress(amount = 1/K)
    segtest<-predictSegErrors(segerrors, correctsegs, num, testset, dataID, proportion)
    votes[[i]]<-segtest
  }
  
  list<-NULL
  for(i in c(1:K)) 
  { 
    list<-c(list, as.character(votes[[i]])) 
  } 
  
  segtest<-vector()
  j=1 
  freqdata<-as.data.frame(table(list)) 
  for(i in c(1:dim(freqdata)[1]))
  { 
    if(freqdata$Freq[i] >= K/2) 
    { 
      segtest[j] = as.character(freqdata$list[i]) 
      j=j+1 
    }
  }
  return(segtest)
}


removePredictedSegErrors<-function(testset, k, predictedSegErrors)
{
  testset<-subset(testset, testset[,k] %in% predictedSegErrors[,1] == FALSE)
  return(testset)
}

cellPopulationClassification<-function(TrainingSet, TestSet, TrainingLabels)
{
  dataforscaling<-rbind(TrainingSet, TestSet)
  dataforscaling<-scale(dataforscaling)
  TrainingSet<-dataforscaling[c(1:nrow(TrainingSet)),]
  TestSet<-dataforscaling[-c(1:nrow(TrainingSet)),]
  
  ## classifier training
  ldamodel<-MASS::lda(TrainingSet, TrainingLabels)
  rfmodel <- randomForest::randomForest(TrainingLabels~., data = TrainingSet, ntree=200, mtry=5, importance=TRUE, norm.votes = TRUE)
  svmmodel<-e1071::svm(TrainingSet, TrainingLabels, kernel = 'radial', probability = TRUE)
  
  ## classifier testing
  ldapred = predict(ldamodel, TestSet)
  rfpred = predict(rfmodel, TestSet)
  svmpred = predict(svmmodel, TestSet)
  
  ## ensemble classification, final predicted label based on majority vote
  classificationvotes<-cbind(as.character(ldapred$class), as.character(rfpred), as.character(svmpred))
  
  classificationvotes<-as.data.frame(classificationvotes)
  
  for(i in c(1:dim(TestSet)[1])) 
  {
    if((classificationvotes[i,1]==unique(TrainingLabels)[1] && classificationvotes[i,2]==unique(TrainingLabels)[1]) ||(classificationvotes[i,1]==unique(TrainingLabels)[1] && classificationvotes[i,3]==unique(TrainingLabels)[1]) 
       ||(classificationvotes[i,2]==unique(TrainingLabels)[1] && classificationvotes[i,3]==unique(TrainingLabels)[1])) 
    {
      classificationvotes[i,4] = unique(TrainingLabels)[1] 
    }
    else
    {
      classificationvotes[i,4] = unique(TrainingLabels)[2] 
    }
  }
  
  colnames(classificationvotes)<-c("LDA", "RF", "SVM", "Ensemble")
  return(classificationvotes)
}

remotes::::install_github('uoy-research/CellPhe')
library(CellPhe)
library(shiny)
library(shinythemes)
library(umap)
library(caret)
library(Rtsne)
library(shinycssloaders)
library(gridExtra)
library(DT)
library(pls)
library(plotly)
library(lubridate)
library(timetk)
library(caret)
library(PerformanceAnalytics)

options(shiny.maxRequestSize = 50*1024^2)
ui<-fluidPage(theme = shinytheme("spacelab"),
              navbarPage("The interactive CellPhe toolkit",
                         tabPanel("Cell time series",
                                  
                                  sidebarPanel(width = 3, 
                                               h3("Data preparation"),
                                               p("To use the CellPhe app, you must have already obtained a .rds file (R Data file) of time series
                                   information for your data set using the CellPhe R package."), br(),
                                               p("Documentation on how to do this is provided in the CellPhe user manual"),
                                               fileInput("Timeseries1", 
                                                         label = "Time series .rds file:", accept = "rds"),
                                               textInput("cellType1", "Enter a name for your cells:", value = "Control"),
                                               checkboxInput("showsumstats", "Display summary statistics only",value=FALSE)
                                  ),
                                  
                                  mainPanel(
                                    
                                    tabsetPanel(type = "tabs",
                                                
                                                tabPanel("Single cell time series",
                                                         uiOutput("cellID"),
                                                         uiOutput("variableNames"),
                                                         plotlyOutput("plot")),
                                                
                                                tabPanel("Extract time series variables",
                                                         br(),
                                                         helpText("Click on the download button to download a .csv file of extracted time series variables for all cells"),
                                                         helpText("Note: a file of this format will be required for the remaining CellPhe analysis tabs"),
                                                         downloadButton("downloadData", "Download extracted time series variables"),
                                                         br(),
                                                         br(),
                                                         helpText("Select a feature name from the dropdown list to display a beeswarm plot"),
                                                         uiOutput("variableNameforSumStats"),
                                                         tabsetPanel(type = "tabs",
                                                                     tabPanel("Original variables",
                                                                              conditionalPanel(fluidRow(column(plotlyOutput("variablePlot1", height = 500, width = 400), plotlyOutput("variablePlot4", height = 500, width = 400), width = 4),
                                                                                                        column(plotlyOutput("variablePlot2", height = 500, width = 400), plotlyOutput("variablePlot5", height = 500, width = 400), width = 4),
                                                                                                        column(plotlyOutput("variablePlot3", height = 500, width = 400), plotlyOutput("variablePlot6", height = 500, width = 400), width = 4)), condition = "input.showsumstats == 0"),
                                                                              conditionalPanel(fluidRow(column(plotlyOutput("variablePlot1_summary", height = 500, width = 400), plotlyOutput("variablePlot4_summary", height = 500, width = 400), width = 4),
                                                                                                        column(plotlyOutput("variablePlot2_summary", height = 500, width = 400), plotlyOutput("variablePlot5_summary", height = 500, width = 400), width = 4),
                                                                                                        column(plotlyOutput("variablePlot3_summary", height = 500, width = 400), plotlyOutput("variablePlot6_summary", height = 500, width = 400), width = 4)), condition = "input.showsumstats == 1")),
                                                                     tabPanel("Wavelet transforms",
                                                                              sliderInput("wavelevel",
                                                                                          "Wavelet level:",
                                                                                          min = 1,
                                                                                          max = 3,
                                                                                          value = 1,
                                                                                          step = 1,
                                                                                          width = "25%"),
                                                                              conditionalPanel(fluidRow(
                                                                                column(plotlyOutput("variablePlot4_l", height = 500, width = 400), width = 4),
                                                                                column(plotlyOutput("variablePlot5_l", height = 500, width = 400), width = 4),
                                                                                column(plotlyOutput("variablePlot6_l", height = 500, width = 400), width = 4)), condition = "input.showsumstats == 0"),
                                                                              conditionalPanel(fluidRow(
                                                                                column(plotlyOutput("variablePlot4_l_summary", height = 500, width = 400), width = 4),
                                                                                column(plotlyOutput("variablePlot5_l_summary", height = 500, width = 400), width = 4),
                                                                                column(plotlyOutput("variablePlot6_l_summary", height = 500, width = 400), width = 4)), condition = "input.showsumstats == 1")
                                                                              
                                                                              
                                                                     ))
                                                )))),
                         tabPanel("Identify discriminatory variables", 
                                  sidebarPanel(width = 3,
                                               p("Use this page to identify discriminatory variables for your two cell populations."),
                                               fileInput("extracted1", 
                                                         label = "Upload file for cell type 1:", accept = "csv"),
                                               textInput("cell1name", "Enter a name for cell type 1:", value = "Control"), br(), br(),
                                               fileInput("extracted2", 
                                                         label = "Upload file for cell type 2:", accept = "csv"),
                                               textInput("cell2name", "Enter a name for cell type 2:", value = "Treated"), br(),
                                               checkboxInput("showsumstats1", "Display summary statistics only",value=FALSE), br(), br(),
                                               helpText("Please enter a separation threshold to use for feature selection.
                                              Use the button below to determine the optimal separation threshold as described in CellPhe
                                              or define your own threshold."),
                                               helpText("Note: the threshold set here will be used for PCA, UMAP and t-SNE on the following tabs."),
                                               actionButton("errorRates1", "Determine optimal separation threshold"),
                                               textOutput("errorRateNote1", h5),
                                               numericInput("septhreshold", "Enter a separation score threshold:", value = "0", step = 0.05, min = 0)),
                                  mainPanel(
                                    tabsetPanel(type = "tabs",
                                                tabPanel("Beeswarm plots",
                                                         uiOutput("variableNameforPlots"),
                                                         tabsetPanel(type = "tabs",
                                                                     tabPanel("Original variables",
                                                                              
                                                                              conditionalPanel(br(),fluidRow(column(plotlyOutput("comparisonPlot1", height = 500, width = 400), plotlyOutput("comparisonPlot4", height = 500, width = 400), width = 4),
                                                                                                             column(plotlyOutput("comparisonPlot2", height = 500, width = 400), plotlyOutput("comparisonPlot5", height = 500, width = 400), width = 4),
                                                                                                             column(plotlyOutput("comparisonPlot3", height = 500, width = 400), plotlyOutput("comparisonPlot6", height = 500, width = 400), width = 4)), condition = "input.showsumstats1 == 0"),
                                                                              conditionalPanel(br(),fluidRow(column(plotlyOutput("comparisonPlot1_summary", height = 500, width = 400), plotlyOutput("comparisonPlot4_summary", height = 500, width = 400), width = 4),
                                                                                                             column(plotlyOutput("comparisonPlot2_summary", height = 500, width = 400), plotlyOutput("comparisonPlot5_summary", height = 500, width = 400), width = 4),
                                                                                                             column(plotlyOutput("comparisonPlot3_summary", height = 500, width = 400), plotlyOutput("comparisonPlot6_summary", height = 500, width = 400), width = 4)), condition = "input.showsumstats1 == 1")),
                                                                     tabPanel("Wavelet transforms",
                                                                              sliderInput("wavelevel1",
                                                                                          "Wavelet level:",
                                                                                          min = 1,
                                                                                          max = 3,
                                                                                          value = 1,
                                                                                          step = 1,
                                                                                          width = "25%"),
                                                                              
                                                                              conditionalPanel(br(),
                                                                                               fluidRow(column(plotlyOutput("comparisonPlot4_l", height = 500, width = 400), width = 4),
                                                                                                        column(plotlyOutput("comparisonPlot5_l", height = 500, width = 400), width = 4),
                                                                                                        column(plotlyOutput("comparisonPlot6_l", height = 500, width = 400), width = 4)), condition = "input.showsumstats1 == 0"),
                                                                              
                                                                              conditionalPanel(br(),
                                                                                               fluidRow(column(plotlyOutput("comparisonPlot4_l_summary", height = 500, width = 400), width = 4),
                                                                                                        column(plotlyOutput("comparisonPlot5_l_summary", height = 500, width = 400), width = 4),
                                                                                                        column(plotlyOutput("comparisonPlot6_l_summary", height = 500, width = 400), width = 4)), condition = "input.showsumstats1 == 1")
                                                                     ))),
                                                tabPanel("Display separation scores",
                                                         fluidRow(column(
                                                           helpText("Click on the download button to download the current table of separation scores"),
                                                           downloadButton("downloadSepScores", "Download table of separation scores"), withSpinner(tableOutput("sepscores"), type = 6), width = 4),
                                                           column(withSpinner(plotlyOutput("sepscoreBarPlot"), type = 6), width = 8))),
                                                tabPanel("Principal Component Analysis",
                                                         br(),
                                                         fluidRow(column(withSpinner(plotlyOutput("PCAscores", width = 500, height = 500), type = 6), width = 6),
                                                                  column(withSpinner(plotOutput("PCAbiplot", width = 500, height = 500), type = 6), width = 6))),
                                                tabPanel("UMAP/t-SNE", br(), fluidRow(column(withSpinner(plotlyOutput("UMAP", width = 500, height = 500), type = 6), width = 6),
                                                                                      column(withSpinner(plotlyOutput("tSNE", width = 500, height = 500), type = 6), width = 6)))))),
                         tabPanel("Population classification",
                                  sidebarPanel(width = 3,
                                               helpText("Note that for unbiased classifier training, training sets will be balanced prior to classification."),
                                               fileInput("group1", 
                                                         label = "Upload training file for cell type 1:", accept = "csv"),
                                               textInput("group1name", "Enter a name for cell type 1:", value = "Untreated"), br(),
                                               fileInput("group2", 
                                                         label = "Upload training file for cell type 2:", accept = "csv"),
                                               textInput("group2name", "Enter a name for cell type 2:", value = "Treated"),
                                               br(),
                                               helpText("Please enter a separation threshold to use for feature selection prior to classification.
                                            Use the button below to determine the optimal separation threshold as described in CellPhe
                                            or define your own threshold."),
                                               actionButton("errorRates", "Determine optimal separation threshold"), br(),
                                               textOutput("errorRateNote", h5),
                                               numericInput("classificationSepThresh", "Separation threshold:", value = "0"), br(),
                                               checkboxInput("testLabels", "I have a list of ground truth labels for my test set",value=FALSE),
                                               fileInput("classificationTestSet", 
                                                         label = "Upload test set for classification:", accept = "csv"),
                                               conditionalPanel(fileInput("testSetLabels", 
                                                                          label = "Ground truth labels for test set", accept = "csv"), condition = "input.testLabels == 1"), 
                                               actionButton("classify", "Classify test set")),
                                  mainPanel(
                                    conditionalPanel(fluidRow(column(downloadButton("downloadclassresults1", "Download table of test set classifications"), helpText("A summary of classification results will be displayed below"), withSpinner(verbatimTextOutput("classificationresultsLabels"), type = 6), br(), helpText("Sensitivity, specificity and balanced accuracy are given as percentages"), tableOutput("classificationaccuracy"), width = 4), column(br(), plotOutput("ROC_curve"), width = 8)), plotOutput("postprobabilities"), tableOutput("classificationsLabels"), condition = "input.testLabels == 1"),
                                    conditionalPanel(downloadButton("downloadclassresults0", "Download table of test set classifications"), helpText("A confusion matrix of classification results will be displayed below"), withSpinner(tableOutput("classificationresultsNoLabels"), type = 6), condition = "input.testLabels == 0"))
                         ),
                         tabPanel("Cluster analysis",
                                  sidebarPanel(width = 3,
                                               p("Use this page to identify heterogenous subsets within your data set using cluster analysis"),
                                               fileInput("forclusters", label = "Upload file for clustering:", accept = "csv"),
                                               numericInput("numclust", label = "Number of clusters:", value = "2")),
                                  mainPanel(tabsetPanel(type = "tabs",
                                                        tabPanel("Determine optimal number of clusters", withSpinner(plotOutput("optimalclusts"), type = 6)),
                                                        tabPanel("Hierarchical clustering", withSpinner(plotOutput("hierclusts"), type = 6), downloadButton("downloadhierclusts", "Download current table of cluster assignments"), br(), br(), dataTableOutput("hierclustclusters")),
                                                        tabPanel("k-means clustering", withSpinner(plotOutput("kmeans"), type = 6), downloadButton("downloadkmeans", "Download current table of cluster assignments"), br(), br(), dataTableOutput("kmeanclusters")),
                                                        tabPanel("Explore clusters", uiOutput("variableNameforDensityPlots"),
                                                                 tabsetPanel(type = "tabs",
                                                                             tabPanel("Mean", fluidRow(column(withSpinner(plotOutput("Hierarch1"), type = 6), width = 6), column(withSpinner(plotOutput("Kmean1"), type = 6), width = 6))), 
                                                                             tabPanel("Standard deviation", fluidRow(column(withSpinner(plotOutput("Hierarch2"), type = 6), width = 6), column(withSpinner(plotOutput("Kmean2"), type = 6), width = 6))),
                                                                             tabPanel("Skewness",fluidRow(column(withSpinner(plotOutput("Hierarch3"), type = 6), width = 6), column(withSpinner(plotOutput("Kmean3"), type = 6), width = 6))),
                                                                             tabPanel("Ascent", fluidRow(column(withSpinner(plotOutput("Hierarch4"), type = 6), width = 6), column(withSpinner(plotOutput("Kmean4"), type = 6), width = 6))),
                                                                             tabPanel("Descent", fluidRow(column(withSpinner(plotOutput("Hierarch5"), type = 6), width = 6), column(withSpinner(plotOutput("Kmean5"), type = 6), width = 6))),
                                                                             tabPanel("Maximum", fluidRow(column(withSpinner(plotOutput("Hierarch6"), type = 6), width = 6), column(withSpinner(plotOutput("Kmean6"), type = 6), width = 6)))))))),
                         tabPanel("Segmentation error removal", 
                                  sidebarPanel(width = 3,
                                               p("Use this page to identify segmentation errors within your data set and remove them prior to analysis."), br(),
                                               p("Note: training data sets of ground truth segmentation errors and correct segmentation are prerequisites for 
                                    model training. Please upload these, together with a test set, to identify segmentation errors."), br(),
                                               fileInput("segerrorset", 
                                                         label = "Upload file of ground truth segmentation errors:", accept = "csv"),
                                               fileInput("correctsegset", 
                                                         label = "Upload file of ground truth correctly segmented cells:", accept = "csv"),
                                               fileInput("segtestset", 
                                                         label = "Upload data set for segmentation error identification and removal", accept = "csv")),
                                  mainPanel(helpText("Click on the button below to download a list of cells identified as a segmentation error:"),
                                            downloadButton("downloadSegErrors", "Download list of predicted segmentation errors"), br(),
                                            helpText("Click on the button below to remove all cells predicted as segmentation error from the data set"),
                                            downloadButton("downloadSegErrorsRemoved", "Remove segmentation errors from data set"),
                                            tableOutput("predictedSegErrors")),
                         ))
              
)


server<-function(input,output,session){
  
  session$onSessionEnded(function() {
    stopApp()
  })
  
  output$cellID<-renderUI({req(input$Timeseries1)
    timeseriesvars<-readRDS(input$Timeseries1$datapath)
    selectizeInput("cellID", label = "Which cell would you like to look at?", choices = unique(timeseriesvars[,2]), options= list(maxOptions = 2000))
  })
  
  
  cellID<-reactive({req(input$Timeseries1)
    timeseriesvars<-readRDS(input$Timeseries1$datapath)
    req(input$cellID)
    num = dim(unique(timeseriesvars[,2]))[1]
    cellID <- vector(mode = "list", length = num)
    
    for (j in 1:num){
      cellID[[j]] = data.frame(unique(timeseriesvars[,2]))[j,1]
    }
    
    for(i in 1:num)
    {
      if(cellID[[i]] == input$cellID)
      {
        break
      }
    }
    i
  })
  
  output$variableNames<-renderUI({req(input$Timeseries1)
    timeseriesvars<-readRDS(input$Timeseries1$datapath)
    
    num = dim(unique(timeseriesvars[,2]))[1]
    timeseries <- vector(mode = "list", length = num)
    
    for (j in 1:num){
      timeseries[[j]] = subset(timeseriesvars[,-c(1,2,3)], timeseriesvars[,2] == as.numeric(unique(timeseriesvars[,2])[j,]))
    }
    
    colnames<-colnames(timeseries[[1]])
    selectizeInput("variableNames", label = "Which variable would you like to plot?", choices = colnames, options= list(maxOptions = 2000))
  })
  
  output$plot <- renderPlotly({req(input$Timeseries1)
    
    timeseriesvars<-readRDS(input$Timeseries1$datapath)
    
    num = dim(unique(timeseriesvars[,2]))[1]
    timeseries <- vector(mode = "list", length = num)
    
    for (j in 1:num){
      timeseries[[j]] = data.frame(subset(timeseriesvars[,-c(1,2,3)], timeseriesvars[,2] == as.numeric(unique(timeseriesvars[,2])[j,])))
    }
    
    
    title = paste(input$variableNames, "time series for cell", input$cellID, "from", input$cellType1, sep = " ")
    # plot(timeseries[[i]][,input$variableNames], type = "l", ylab = input$variableNames, xlab = "Frame", lwd = 1, main = title)
    
    fig = plot_ly(data.frame(timeseries[[cellID()]]), y = ~timeseries[[cellID()]][,input$variableNames], type = 'scatter', mode = 'lines', text = paste(input$variableNames,"=",round(timeseries[[cellID()]][,input$variableNames],2), sep = ""), hoverinfo = 'text')%>%
      layout(title = title, xaxis = list(title = 'Frame'), 
             yaxis = list(title = input$variableNames))
    fig
  })
  
  extractedVariables<-shiny::reactive({req(input$Timeseries1)
    extract<-readRDS(input$Timeseries1$datapath)
    CellPhe::varsFromTimeSeries(extract)})
  
  output$downloadData<-downloadHandler(
    filename = function(){
      paste(input$cellType1, "_outputfile.csv", sep = "")
    },
    content = function(file)
    {
      write.csv(extractedVariables(), file)
    }
    
  )
  
  output$extractedTimeSeriesVars<-renderTable({extractedVariables()[1:10,1:10]}, rownames = TRUE)
  
  
  output$variableNameforSumStats<-renderUI({req(extractedVariables())
    colnames<-unique(sapply(strsplit(colnames(extractedVariables()), "[_]"),"[[",1))
    colnames = subset(colnames, colnames != "trajArea")
    colnames = subset(colnames, colnames != "CellID")
    selectizeInput("variableNameforSumStats", label = "Display summary statistics for this feature:", choices = colnames, options= list(maxOptions = 2000))
  })
  
  output$summaryStats1<-renderTable({req(extractedVariables())
    req(input$variableNameforSumStats)
    tibble::tibble(!!!summary(extractedVariables()[,paste(input$variableNameforSumStats,"_mean", sep = "")]))
    
  })
  
  output$variablePlot1<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$cellType1)
    
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats,"_mean", sep = "")], type = "box", boxpoints = "all", name = input$cellType1, text = as.character(row.names(extractedVariables())), hoverinfo = 'text', pointpos = 0, fillcolor = "white")%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats,"_mean", sep = "")))
    fig
    
  })
  
  output$variablePlot1_summary<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$cellType1)
    
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats,"_mean", sep = "")], type = "box", name = input$cellType1, text = as.character(row.names(extractedVariables())), pointpos = 0)%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats,"_mean", sep = "")))
    fig
    
  })
  
  output$summaryStats2<-renderTable({req(extractedVariables())
    req(input$variableNameforSumStats)
    tibble::tibble(!!!summary(extractedVariables()[,paste(input$variableNameforSumStats,"_std", sep = "")]))
    
  })
  
  output$variablePlot2<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$cellType1)
    
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats,"_std", sep = "")], type = "box", boxpoints = "all", name = input$cellType1, text = as.character(row.names(extractedVariables())), hoverinfo = 'text', pointpos = 0, fillcolor = "white")%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats,"_std", sep = "")))
    fig
    
  })
  
  output$variablePlot2_summary<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$cellType1)
    
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats,"_std", sep = "")], type = "box", name = input$cellType1, text = as.character(row.names(extractedVariables())), pointpos = 0)%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats,"_std", sep = "")))
    fig
    
  })
  
  output$summaryStats3<-renderTable({req(extractedVariables())
    req(input$variableNameforSumStats)
    tibble::tibble(!!!summary(extractedVariables()[,paste(input$variableNameforSumStats,"_skew", sep = "")]))
    
  })
  
  output$variablePlot3<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$cellType1)
    
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats,"_skew", sep = "")], type = "box", boxpoints = "all", name = input$cellType1, text = as.character(row.names(extractedVariables())), hoverinfo = 'text', pointpos = 0, fillcolor = "white")%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats,"_skew", sep = "")))
    fig
  })
  
  output$variablePlot3_summary<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$cellType1)
    
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats,"_skew", sep = "")], type = "box", name = input$cellType1, text = as.character(row.names(extractedVariables())), pointpos = 0)%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats,"_skew", sep = "")))
    fig
    
  })
  
  output$summaryStats4<-renderTable(caption = "Original", caption.placement = getOption("xtable.caption.placement", "top"),
                                    {req(extractedVariables())
                                      req(input$variableNameforSumStats)
                                      tibble::tibble(!!!summary(extractedVariables()[,paste(input$variableNameforSumStats,"_asc", sep = "")]))
                                      
                                    })
  
  output$variablePlot4<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$cellType1)
    
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats,"_asc", sep = "")], type = "box", boxpoints = "all", name = input$cellType1, text = as.character(row.names(extractedVariables())), hoverinfo = 'text', pointpos = 0, fillcolor = "white")%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats,"_asc", sep = "")))
    fig
    
  })
  
  output$variablePlot4_summary<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$cellType1)
    
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats,"_asc", sep = "")], type = "box", name = input$cellType1, text = as.character(row.names(extractedVariables())), pointpos = 0)%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats,"_asc", sep = "")))
    fig
    
  })
  
  output$summaryStats4_l1<-renderTable(caption = "Level 1", caption.placement = getOption("xtable.caption.placement", "top"),
                                       {req(extractedVariables())
                                         req(input$variableNameforSumStats)
                                         tibble::tibble(!!!summary(extractedVariables()[,paste(input$variableNameforSumStats,"_l1_asc", sep = "")]))
                                         
                                       })
  
  output$variablePlot4_l<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$wavelevel)
    req(input$cellType1)
    
    name<-paste("l", input$wavelevel, sep="")
    name<-paste(name, "asc", sep = "_")
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats, name, sep = "_")], type = "box", boxpoints = "all", name = input$cellType1, text = as.character(row.names(extractedVariables())), hoverinfo = 'text', pointpos = 0, fillcolor = "white")%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats, name, sep = "_"), sep = ""))
    fig
    
  })
  
  output$variablePlot4_l_summary<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$wavelevel)
    req(input$cellType1)
    
    name<-paste("l", input$wavelevel, sep="")
    name<-paste(name, "asc", sep = "_")
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats, name, sep = "_")], type = "box", name = input$cellType1, pointpos = 0)%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats, name, sep = "_"), sep = ""))
    fig
    
  })
  
  output$summaryStats4_l2<-renderTable(caption = "Level 2", caption.placement = getOption("xtable.caption.placement", "top"),
                                       {req(extractedVariables())
                                         req(input$variableNameforSumStats)
                                         tibble::tibble(!!!summary(extractedVariables()[,paste(input$variableNameforSumStats,"_l2_asc", sep = "")]))
                                         
                                       })
  
  output$variablePlot4_l2<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$cellType1)
    
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats,"_l2_asc", sep = "")], type = "box", boxpoints = "all", name = input$cellType1, text = as.character(row.names(extractedVariables())), hoverinfo = 'text', pointpos = 0, fillcolor = "white")%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats,"_l2_asc", sep = "")))
    fig
    
  })
  
  output$summaryStats4_l3<-renderTable(caption = "Level 3", caption.placement = getOption("xtable.caption.placement", "top"),
                                       {req(extractedVariables())
                                         req(input$variableNameforSumStats)
                                         tibble::tibble(!!!summary(extractedVariables()[,paste(input$variableNameforSumStats,"_l3_asc", sep = "")]))
                                         
                                       })
  
  output$variablePlot4_l3<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$cellType1)
    
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats,"_l3_asc", sep = "")], type = "box", boxpoints = "all", name = input$cellType1, text = as.character(row.names(extractedVariables())), hoverinfo = 'text', pointpos = 0, fillcolor = "white")%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats,"_l3_asc", sep = "")))
    fig
    
  })
  
  output$summaryStats5<-renderTable(caption = "Original", caption.placement = getOption("xtable.caption.placement", "top"),
                                    {req(extractedVariables())
                                      req(input$variableNameforSumStats)
                                      tibble::tibble(!!!summary(extractedVariables()[,paste(input$variableNameforSumStats,"_des", sep = "")]))
                                      
                                    })
  
  output$variablePlot5<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$cellType1)
    
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats,"_des", sep = "")], type = "box", boxpoints = "all", name = input$cellType1, text = as.character(row.names(extractedVariables())), hoverinfo = 'text', pointpos = 0, fillcolor = "white")%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats,"_des", sep = "")))
    fig
    
  })
  
  output$variablePlot5_summary<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$cellType1)
    
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats,"_des", sep = "")], type = "box", name = input$cellType1, text = as.character(row.names(extractedVariables())), pointpos = 0)%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats,"_des", sep = "")))
    fig
    
  })
  
  output$summaryStats5_l1<-renderTable(caption = "Level 1", caption.placement = getOption("xtable.caption.placement", "top"),
                                       {req(extractedVariables())
                                         req(input$variableNameforSumStats)
                                         tibble::tibble(!!!summary(extractedVariables()[,paste(input$variableNameforSumStats,"_l1_des", sep = "")]))
                                         
                                       })
  
  output$variablePlot5_l<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$wavelevel)
    req(input$cellType1)
    
    name<-paste("l", input$wavelevel, sep="")
    name<-paste(name, "des", sep = "_")
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats, name, sep = "_")], type = "box", boxpoints = "all", name = input$cellType1, text = as.character(row.names(extractedVariables())), hoverinfo = 'text', pointpos = 0, fillcolor = "white")%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats, name, sep = "_"), sep = ""))
    fig
    
  })
  
  output$variablePlot5_l_summary<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$wavelevel)
    req(input$cellType1)
    
    name<-paste("l", input$wavelevel, sep="")
    name<-paste(name, "des", sep = "_")
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats, name, sep = "_")], type = "box", name = input$cellType1, pointpos = 0)%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats, name, sep = "_"), sep = ""))
    fig
    
  })
  
  output$summaryStats5_l2<-renderTable(caption = "Level 2", caption.placement = getOption("xtable.caption.placement", "top"),
                                       {req(extractedVariables())
                                         req(input$variableNameforSumStats)
                                         tibble::tibble(!!!summary(extractedVariables()[,paste(input$variableNameforSumStats,"_l2_des", sep = "")]))
                                         
                                       })
  
  output$variablePlot5_l2<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$cellType1)
    
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats,"_l2_des", sep = "")], type = "box", boxpoints = "all", name = input$cellType1, text = as.character(row.names(extractedVariables())), hoverinfo = 'text', pointpos = 0, fillcolor = "white")%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats,"_l2_des", sep = "")))
    fig
    
  })
  
  output$summaryStats5_l3<-renderTable(caption = "Level 3", caption.placement = getOption("xtable.caption.placement", "top"),
                                       {req(extractedVariables())
                                         req(input$variableNameforSumStats)
                                         tibble::tibble(!!!summary(extractedVariables()[,paste(input$variableNameforSumStats,"_l3_des", sep = "")]))
                                         
                                       })
  
  output$variablePlot5_l3<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$cellType1)
    
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats,"_l3_des", sep = "")], type = "box", boxpoints = "all", name = input$cellType1, text = as.character(row.names(extractedVariables())), hoverinfo = 'text', pointpos = 0, fillcolor = "white")%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats,"_l3_des", sep = "")))
    fig
    
  })
  
  output$summaryStats6<-renderTable(caption = "Original", caption.placement = getOption("xtable.caption.placement", "top"),
                                    {req(extractedVariables())
                                      req(input$variableNameforSumStats)
                                      tibble::tibble(!!!summary(extractedVariables()[,paste(input$variableNameforSumStats,"_max", sep = "")]))
                                      
                                    })
  
  output$variablePlot6<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$cellType1)
    
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats,"_max", sep = "")], type = "box", boxpoints = "all", name = input$cellType1, text = as.character(row.names(extractedVariables())), hoverinfo = 'text', pointpos = 0, fillcolor = "white")%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats,"_max", sep = "")))
    fig
    
  })
  
  output$variablePlot6_summary<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$cellType1)
    
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats,"_max", sep = "")], type = "box", name = input$cellType1, text = as.character(row.names(extractedVariables())), pointpos = 0)%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats,"_max", sep = "")))
    fig
    
  })
  
  output$summaryStats6_l1<-renderTable(caption = "Level 1", caption.placement = getOption("xtable.caption.placement", "top"),
                                       {req(extractedVariables())
                                         req(input$variableNameforSumStats)
                                         tibble::tibble(!!!summary(extractedVariables()[,paste(input$variableNameforSumStats,"_l1_max", sep = "")]))
                                         
                                       })
  
  output$variablePlot6_l<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$wavelevel)
    req(input$cellType1)
    
    name<-paste("l", input$wavelevel, sep="")
    name<-paste(name, "max", sep = "_")
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats, name, sep = "_")], type = "box", boxpoints = "all", name = input$cellType1, text = as.character(row.names(extractedVariables())), hoverinfo = 'text', pointpos = 0, fillcolor = "white")%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats, name, sep = "_"), sep = ""))
    fig
    
  })
  
  output$variablePlot6_l_summary<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$wavelevel)
    req(input$cellType1)
    
    name<-paste("l", input$wavelevel, sep="")
    name<-paste(name, "max", sep = "_")
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats, name, sep = "_")], type = "box", name = input$cellType1, pointpos = 0)%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats, name, sep = "_"), sep = ""))
    fig
    
  })
  
  output$summaryStats6_l2<-renderTable(caption = "Level 2", caption.placement = getOption("xtable.caption.placement", "top"),
                                       {req(extractedVariables())
                                         req(input$variableNameforSumStats)
                                         tibble::tibble(!!!summary(extractedVariables()[,paste(input$variableNameforSumStats,"_l2_max", sep = "")]))
                                         
                                       })
  
  output$variablePlot6_l2<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$cellType1)
    
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats,"_l2_max", sep = "")], type = "box", boxpoints = "all", name = input$cellType1, text = as.character(row.names(extractedVariables())), hoverinfo = 'text', pointpos = 0, fillcolor = "white")%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats,"_l2_max", sep = "")))
    fig
    
  })
  
  output$summaryStats6_l3<-renderTable(caption = "Level 3", caption.placement = getOption("xtable.caption.placement", "top"),
                                       {req(extractedVariables())
                                         req(input$variableNameforSumStats)
                                         tibble::tibble(!!!summary(extractedVariables()[,paste(input$variableNameforSumStats,"_l3_max", sep = "")]))
                                         
                                       })
  
  output$variablePlot6_l3<-renderPlotly({req(extractedVariables())
    req(input$variableNameforSumStats)
    req(input$cellType1)
    
    fig<-plot_ly(y = ~extractedVariables()[,paste(input$variableNameforSumStats,"_l3_max", sep = "")], type = "box", boxpoints = "all", name = input$cellType1, text = as.character(row.names(extractedVariables())), hoverinfo = 'text', pointpos = 0, fillcolor = "white")%>%
      layout((title = input$cellType1), 
             yaxis = list(title = paste(input$variableNameforSumStats,"_l3_max", sep = "")))
    fig
    
  })
  
  output$whichCell<-renderPrint({
    req(input$plot_click)
    req(extractedVariables())
    req(input$variableNameforSumStats)
    nearPoints(extractedVariables(),input$plot_click)
  })
  
  segmentationerrors<-reactive({req(input$segerrorset)
    req(input$correctsegset)
    req(input$segtestset)
    
    segerrors<-read.csv(input$segerrorset$datapath)
    correctsegs<-read.csv(input$correctsegset$datapath)
    testset<-read.csv(input$segtestset$datapath)
    withProgress(message = "Identifying segmentation errors",
                 data.frame(predictSegErrors_Ensemble(segerrors,correctsegs,50,10,testset,testset[,1],0.7)))
    
  })
  
  removesegmentationerrors<-reactive({req(segmentationerrors())
    req(input$segtestset)
    testset<-read.csv(input$segtestset$datapath)
    removePredictedSegErrors(testset,1,segmentationerrors())
  })
  
  
  output$predictedSegErrors<-renderTable({req(segmentationerrors())
    segmentationerrors()}, colnames = FALSE)
  
  output$downloadSegErrors<-downloadHandler(
    filename = function(){
      (paste("listOfPredictedSegErrors.csv"))
    },
    content = function(file)
    {
      write.csv(segmentationerrors(), file, row.names = FALSE, col.names = FALSE)
    }
    
  )
  
  output$downloadSegErrorsRemoved<-downloadHandler(
    filename = function(){
      ("segsRemoved.csv")
    },
    content = function(file)
    {
      write.csv(removesegmentationerrors(), file, row.names = FALSE)
    }
    
  )
  
  allData<-shiny::reactive({
    req(input$extracted1)
    req(input$extracted2)
    req(input$cell1name)
    req(input$cell2name)
    
    extracted1<-read.csv(input$extracted1$datapath)
    extracted2<-read.csv(input$extracted2$datapath)
    
    names = list(c(rep(input$cell1name, dim(extracted1)[1]), c(rep(input$cell2name, dim(extracted2)[1]))))
    
    data = rbind(extracted1, extracted2)
    
    cbind(names, data)
  })
  
  separationscores<-reactive({req(input$extracted1)
    req(input$extracted2)
    input$septhreshold
    extracted1<-read.csv(input$extracted1$datapath)
    extracted2<-read.csv(input$extracted2$datapath)
    CellPhe::calculateSeparationScores(extracted1[,-1], extracted2[,-1], 0)
  })
  
  separationscoresthresh<-reactive({req(input$extracted1)
    req(input$extracted2)
    input$septhreshold
    extracted1<-read.csv(input$extracted1$datapath)
    extracted2<-read.csv(input$extracted2$datapath)
    CellPhe::calculateSeparationScores(extracted1[,-1], extracted2[,-1], input$septhreshold)
  })
  
  
  output$variableNameforPlots<-renderUI({req(allData())
    colnames<-unique(sapply(strsplit(colnames(allData()[,-c(1,2)]), "[_]"),"[[",1))
    colnames = subset(colnames, colnames != "trajArea")
    selectizeInput("variableNameforPlots", label = "Which variable would you like to display?", choices = colnames, options= list(maxOptions = 2000))
  })
  
  output$comparisonPlot1<-renderPlotly({
    req(separationscores())
    req(input$cell1name)
    req(input$cell2name)
    req(allData())
    req(input$variableNameforPlots)
    groups<-factor(allData()[,1], levels = c(input$cell1name, input$cell2name))
    sepscore<-round(separationscores()[(separationscores()[,2] == paste(input$variableNameforPlots,"_mean",sep = "")) == TRUE,3], 2)
    
    xform <- list(categoryorder = "array",
                  categoryarray = groups)
    
    fig<-plot_ly(y = allData()[,paste(input$variableNameforPlots,"_mean", sep = "")], type = "box", boxpoints = "all", name = allData()[,1], text = as.character((allData()[,2])), hoverinfo = 'text', pointpos = 0, fillcolor = "white", color = groups, colors = c("red", "blue"))%>%
      layout(showlegend = FALSE, title = paste("Separation score:",sepscore,sep=" "),
             xaxis = xform,
             yaxis = list(title = paste(input$variableNameforPlots,"_mean", sep = "")))
    fig
    
  })
  
  output$comparisonPlot1_summary<-renderPlotly({
    req(separationscores())
    req(allData())
    req(input$variableNameforPlots)
    groups<-factor(allData()[,1], levels = c(input$cell1name, input$cell2name))
    sepscore<-round(separationscores()[(separationscores()[,2] == paste(input$variableNameforPlots,"_mean",sep = "")) == TRUE,3],2)
    xform <- list(categoryorder = "array",
                  categoryarray = groups)
    
    fig<-plot_ly(y = allData()[,paste(input$variableNameforPlots,"_mean", sep = "")], type = "box", name = allData()[,1], text = as.character((allData()[,2])), pointpos = 0, color = allData()[,1], colors = c("red", "blue"))%>%
      layout(showlegend = FALSE, title = paste("Separation score:",sepscore,sep=" "),
             xaxis = xform,
             yaxis = list(title = paste(input$variableNameforPlots,"_mean", sep = "")))
    fig
    
  })
  
  output$comparisonPlot2<-renderPlotly({
    req(separationscores())
    req(allData())
    req(input$variableNameforPlots)
    groups<-factor(allData()[,1], levels = c(input$cell1name, input$cell2name))
    sepscore<-round(separationscores()[(separationscores()[,2] == paste(input$variableNameforPlots,"_std",sep = "")) == TRUE,3],2)
    xform <- list(categoryorder = "array",
                  categoryarray = groups)
    
    fig<-plot_ly(y = allData()[,paste(input$variableNameforPlots,"_std", sep = "")], type = "box", boxpoints = "all", name = allData()[,1], text = as.character((allData()[,2])), hoverinfo = 'text', pointpos = 0, fillcolor = "white", color = allData()[,1], colors = c("red", "blue"))%>%
      layout(showlegend = FALSE, title = paste("Separation score:",sepscore,sep=" "),
             xaxis = xform,
             yaxis = list(title = paste(input$variableNameforPlots,"_std", sep = "")))
    fig
    
  })
  
  output$comparisonPlot2_summary<-renderPlotly({
    req(separationscores())
    req(allData())
    req(input$variableNameforPlots)
    groups<-factor(allData()[,1], levels = c(input$cell1name, input$cell2name))
    sepscore<-round(separationscores()[(separationscores()[,2] == paste(input$variableNameforPlots,"_std",sep = "")) == TRUE,3],2)
    xform <- list(categoryorder = "array",
                  categoryarray = groups)
    
    fig<-plot_ly(y = allData()[,paste(input$variableNameforPlots,"_std", sep = "")], type = "box", name = allData()[,1], text = as.character((allData()[,2])), pointpos = 0, color = allData()[,1], colors = c("red", "blue"))%>%
      layout(showlegend = FALSE, title = paste("Separation score:",sepscore,sep=" "),
             xaxis = xform,
             yaxis = list(title = paste(input$variableNameforPlots,"_std", sep = "")))
    fig
    
  })
  
  output$comparisonPlot3<-renderPlotly({
    req(separationscores())
    req(allData())
    req(input$variableNameforPlots)
    groups<-factor(allData()[,1], levels = c(input$cell1name, input$cell2name))
    sepscore<-round(separationscores()[(separationscores()[,2] == paste(input$variableNameforPlots,"_skew",sep = "")) == TRUE,3],2)
    xform <- list(categoryorder = "array",
                  categoryarray = groups)
    
    fig<-plot_ly(y = allData()[,paste(input$variableNameforPlots,"_skew", sep = "")], type = "box", boxpoints = "all", name = allData()[,1], text = as.character((allData()[,2])), hoverinfo = 'text', pointpos = 0, fillcolor = "white", color = allData()[,1], colors = c("red", "blue"))%>%
      layout(showlegend = FALSE, title = paste("Separation score:",sepscore,sep=" "),
             xaxis = xform,
             yaxis = list(title = paste(input$variableNameforPlots,"_skew", sep = "")))
    fig
    
  })
  
  output$comparisonPlot3_summary<-renderPlotly({
    req(separationscores())
    req(allData())
    req(input$variableNameforPlots)
    groups<-factor(allData()[,1], levels = c(input$cell1name, input$cell2name))
    sepscore<-round(separationscores()[(separationscores()[,2] == paste(input$variableNameforPlots,"_skew",sep = "")) == TRUE,3],2)
    xform <- list(categoryorder = "array",
                  categoryarray = groups)
    
    fig<-plot_ly(y = allData()[,paste(input$variableNameforPlots,"_skew", sep = "")], type = "box", name = allData()[,1], text = as.character((allData()[,2])), pointpos = 0, color = allData()[,1], colors = c("red", "blue"))%>%
      layout(showlegend = FALSE, title = paste("Separation score:",sepscore,sep=" "),
             xaxis = xform,
             yaxis = list(title = paste(input$variableNameforPlots,"_skew", sep = "")))
    fig
    
  })
  
  output$comparisonPlot4<-renderPlotly({
    req(separationscores())
    req(separationscores())
    req(allData())
    req(input$variableNameforPlots)
    groups<-factor(allData()[,1], levels = c(input$cell1name, input$cell2name))
    sepscore<-round(separationscores()[(separationscores()[,2] == paste(input$variableNameforPlots,"_asc",sep = "")) == TRUE,3],2)
    xform <- list(categoryorder = "array",
                  categoryarray = groups)
    
    fig<-plot_ly(y = allData()[,paste(input$variableNameforPlots,"_asc", sep = "")], type = "box", boxpoints = "all", name = allData()[,1], text = as.character((allData()[,2])), hoverinfo = 'text', pointpos = 0, fillcolor = "white", color = allData()[,1], colors = c("red", "blue"))%>%
      layout(showlegend = FALSE, title = paste("Separation score:",sepscore,sep=" "),
             xaxis = xform,
             yaxis = list(title = paste(input$variableNameforPlots,"_asc", sep = "")))
    fig
    
  })
  
  output$comparisonPlot4_summary<-renderPlotly({
    req(separationscores())
    req(allData())
    req(input$variableNameforPlots)
    groups<-factor(allData()[,1], levels = c(input$cell1name, input$cell2name))
    sepscore<-round(separationscores()[(separationscores()[,2] == paste(input$variableNameforPlots,"_asc",sep = "")) == TRUE,3],2)
    xform <- list(categoryorder = "array",
                  categoryarray = groups)
    
    fig<-plot_ly(y = allData()[,paste(input$variableNameforPlots,"_asc", sep = "")], type = "box", name = allData()[,1], text = as.character((allData()[,2])), pointpos = 0, color = allData()[,1], colors = c("red", "blue"))%>%
      layout(showlegend = FALSE, title = paste("Separation score:",sepscore,sep=" "),
             xaxis = xform,
             yaxis = list(title = paste(input$variableNameforPlots,"_asc", sep = "")))
    fig
    
  })
  
  output$comparisonPlot4_l<-renderPlotly({
    req(input$wavelevel1)
    req(separationscores())
    req(separationscores())
    req(allData())
    req(input$variableNameforPlots)
    groups<-factor(allData()[,1], levels = c(input$cell1name, input$cell2name))
    sepscore<-round(separationscores()[(separationscores()[,2] == paste(input$variableNameforPlots,"_asc",sep = "")) == TRUE,3],2)
    xform <- list(categoryorder = "array",
                  categoryarray = groups)
    
    name<-paste("l", input$wavelevel1, sep="")
    name<-paste(name, "asc", sep = "_")
    
    fig<-plot_ly(y = allData()[,paste(input$variableNameforPlots,name, sep = "_")], type = "box", boxpoints = "all", name = allData()[,1], text = as.character((allData()[,2])), hoverinfo = 'text', pointpos = 0, fillcolor = "white", color = allData()[,1], colors = c("red", "blue"))%>%
      layout(showlegend = FALSE, title = paste("Separation score:",sepscore,sep=" "),
             xaxis = xform,
             yaxis = list(title = paste(input$variableNameforPlots, name, sep = "_"), sep = ""))
    fig
    
  })
  
  output$comparisonPlot4_l_summary<-renderPlotly({
    req(input$wavelevel1)
    req(separationscores())
    req(separationscores())
    req(allData())
    req(input$variableNameforPlots)
    groups<-factor(allData()[,1], levels = c(input$cell1name, input$cell2name))
    sepscore<-round(separationscores()[(separationscores()[,2] == paste(input$variableNameforPlots,"_asc",sep = "")) == TRUE,3],2)
    xform <- list(categoryorder = "array",
                  categoryarray = groups)
    
    name<-paste("l", input$wavelevel1, sep="")
    name<-paste(name, "asc", sep = "_")
    
    fig<-plot_ly(y = allData()[,paste(input$variableNameforPlots,name, sep = "_")], type = "box", name = allData()[,1], text = as.character((allData()[,2])), pointpos = 0, color = allData()[,1], colors = c("red", "blue"))%>%
      layout(showlegend = FALSE, title = paste("Separation score:",sepscore,sep=" "),
             xaxis = xform,
             yaxis = list(title = paste(input$variableNameforPlots, name, sep = "_"), sep = ""))
    fig
    
  })
  
  output$comparisonPlot5<-renderPlotly({
    req(separationscores())
    req(allData())
    req(input$variableNameforPlots)
    groups<-factor(allData()[,1], levels = c(input$cell1name, input$cell2name))
    sepscore<-round(separationscores()[(separationscores()[,2] == paste(input$variableNameforPlots,"_des",sep = "")) == TRUE,3],2)
    xform <- list(categoryorder = "array",
                  categoryarray = groups)
    
    fig<-plot_ly(y = allData()[,paste(input$variableNameforPlots,"_des", sep = "")], type = "box", boxpoints = "all", name = allData()[,1], text = as.character((allData()[,2])), hoverinfo = 'text', pointpos = 0, fillcolor = "white", color = allData()[,1], colors = c("red", "blue"))%>%
      layout(showlegend = FALSE, title = paste("Separation score:",sepscore,sep=" "),
             xaxis = xform,
             yaxis = list(title = paste(input$variableNameforPlots,"_des", sep = "")))
    fig
    
  })
  
  output$comparisonPlot5_summary<-renderPlotly({
    req(separationscores())
    req(allData())
    req(input$variableNameforPlots)
    groups<-factor(allData()[,1], levels = c(input$cell1name, input$cell2name))
    sepscore<-round(separationscores()[(separationscores()[,2] == paste(input$variableNameforPlots,"_des",sep = "")) == TRUE,3],2)
    xform <- list(categoryorder = "array",
                  categoryarray = groups)
    
    fig<-plot_ly(y = allData()[,paste(input$variableNameforPlots,"_des", sep = "")], type = "box", name = allData()[,1], text = as.character((allData()[,2])), pointpos = 0, color = allData()[,1], colors = c("red", "blue"))%>%
      layout(showlegend = FALSE, title = paste("Separation score:",sepscore,sep=" "),
             xaxis = xform,
             yaxis = list(title = paste(input$variableNameforPlots,"_des", sep = "")))
    fig
    
  })
  
  output$comparisonPlot5_l<-renderPlotly({
    req(input$wavelevel1)
    req(separationscores())
    req(separationscores())
    req(allData())
    req(input$variableNameforPlots)
    groups<-factor(allData()[,1], levels = c(input$cell1name, input$cell2name))
    sepscore<-round(separationscores()[(separationscores()[,2] == paste(input$variableNameforPlots,"_des",sep = "")) == TRUE,3],2)
    xform <- list(categoryorder = "array",
                  categoryarray = groups)
    
    name<-paste("l", input$wavelevel1, sep="")
    name<-paste(name, "des", sep = "_")
    
    fig<-plot_ly(y = allData()[,paste(input$variableNameforPlots,name, sep = "_")], type = "box", boxpoints = "all", name = allData()[,1], text = as.character((allData()[,2])), hoverinfo = 'text', pointpos = 0, fillcolor = "white", color = allData()[,1], colors = c("red", "blue"))%>%
      layout(showlegend = FALSE, title = paste("Separation score:",sepscore,sep=" "),
             xaxis = xform,
             yaxis = list(title = paste(input$variableNameforPlots, name, sep = "_"), sep = ""))
    fig
    
  })
  
  output$comparisonPlot5_l_summary<-renderPlotly({
    req(input$wavelevel1)
    req(separationscores())
    req(separationscores())
    req(allData())
    req(input$variableNameforPlots)
    groups<-factor(allData()[,1], levels = c(input$cell1name, input$cell2name))
    sepscore<-round(separationscores()[(separationscores()[,2] == paste(input$variableNameforPlots,"_des",sep = "")) == TRUE,3],2)
    xform <- list(categoryorder = "array",
                  categoryarray = groups)
    
    name<-paste("l", input$wavelevel1, sep="")
    name<-paste(name, "des", sep = "_")
    
    fig<-plot_ly(y = allData()[,paste(input$variableNameforPlots,name, sep = "_")], type = "box", name = allData()[,1], text = as.character((allData()[,2])), pointpos = 0, color = allData()[,1], colors = c("red", "blue"))%>%
      layout(showlegend = FALSE, title = paste("Separation score:",sepscore,sep=" "),
             xaxis = xform,
             yaxis = list(title = paste(input$variableNameforPlots, name, sep = "_"), sep = ""))
    fig
    
  })
  
  output$comparisonPlot6<-renderPlotly({
    req(separationscores())
    req(allData())
    req(input$variableNameforPlots)
    groups<-factor(allData()[,1], levels = c(input$cell1name, input$cell2name))
    sepscore<-round(separationscores()[(separationscores()[,2] == paste(input$variableNameforPlots,"_max",sep = "")) == TRUE,3],2)
    xform <- list(categoryorder = "array",
                  categoryarray = groups)
    
    fig<-plot_ly(y = allData()[,paste(input$variableNameforPlots,"_max", sep = "")], type = "box", boxpoints = "all", name = allData()[,1], text = as.character((allData()[,2])), hoverinfo = 'text', pointpos = 0, fillcolor = "white", color = allData()[,1], colors = c("red", "blue"))%>%
      layout(showlegend = FALSE, title = paste("Separation score:",sepscore,sep=" "),
             xaxis = xform,
             yaxis = list(title = paste(input$variableNameforPlots,"_max", sep = "")))
    fig
    
  })
  
  output$comparisonPlot6_summary<-renderPlotly({
    req(separationscores())
    req(allData())
    req(input$variableNameforPlots)
    groups<-factor(allData()[,1], levels = c(input$cell1name, input$cell2name))
    sepscore<-round(separationscores()[(separationscores()[,2] == paste(input$variableNameforPlots,"_max",sep = "")) == TRUE,3],2)
    xform <- list(categoryorder = "array",
                  categoryarray = groups)
    
    fig<-plot_ly(y = allData()[,paste(input$variableNameforPlots,"_max", sep = "")], type = "box", name = allData()[,1], text = as.character((allData()[,2])), pointpos = 0, color = allData()[,1], colors = c("red", "blue"))%>%
      layout(showlegend = FALSE, title = paste("Separation score:",sepscore,sep=" "),
             xaxis = xform,
             yaxis = list(title = paste(input$variableNameforPlots,"_max", sep = "")))
    fig
    
  })
  
  output$comparisonPlot6_l<-renderPlotly({
    req(input$wavelevel1)
    req(separationscores())
    req(separationscores())
    req(allData())
    req(input$variableNameforPlots)
    groups<-factor(allData()[,1], levels = c(input$cell1name, input$cell2name))
    sepscore<-round(separationscores()[(separationscores()[,2] == paste(input$variableNameforPlots,"_max",sep = "")) == TRUE,3],2)
    xform <- list(categoryorder = "array",
                  categoryarray = groups)
    
    name<-paste("l", input$wavelevel1, sep="")
    name<-paste(name, "max", sep = "_")
    
    fig<-plot_ly(y = allData()[,paste(input$variableNameforPlots,name, sep = "_")], type = "box", boxpoints = "all", name = allData()[,1], text = as.character((allData()[,2])), hoverinfo = 'text', pointpos = 0, fillcolor = "white", color = allData()[,1], colors = c("red", "blue"))%>%
      layout(showlegend = FALSE, title = paste("Separation score:",sepscore,sep=" "),
             xaxis = xform,
             yaxis = list(title = paste(input$variableNameforPlots, name, sep = "_"), sep = ""))
    fig
    
  })
  
  output$comparisonPlot6_l_summary<-renderPlotly({
    req(input$wavelevel1)
    req(separationscores())
    req(separationscores())
    req(allData())
    req(input$variableNameforPlots)
    groups<-factor(allData()[,1], levels = c(input$cell1name, input$cell2name))
    sepscore<-round(separationscores()[(separationscores()[,2] == paste(input$variableNameforPlots,"_max",sep = "")) == TRUE,3],2)
    xform <- list(categoryorder = "array",
                  categoryarray = groups)
    
    name<-paste("l", input$wavelevel1, sep="")
    name<-paste(name, "max", sep = "_")
    
    fig<-plot_ly(y = allData()[,paste(input$variableNameforPlots,name, sep = "_")], type = "box", name = allData()[,1], text = as.character((allData()[,2])), pointpos = 0, color = allData()[,1], colors = c("red", "blue"))%>%
      layout(showlegend = FALSE, title = paste("Separation score:",sepscore,sep=" "),
             xaxis = xform,
             yaxis = list(title = paste(input$variableNameforPlots, name, sep = "_"), sep = ""))
    fig
    
  })
  
  output$sepscores<-renderTable(
    separationscoresthresh()[order(separationscoresthresh()[,3], decreasing = TRUE),2:3], colnames = FALSE)
  
  output$sepscoreBarPlot<-renderPlotly({req(separationscoresthresh())
    req(input$septhreshold)
    sepscores<-separationscoresthresh()[order(separationscoresthresh()[,3], decreasing = TRUE),]
    names = categoriseFeatures(sepscores[,2])
    sepscores = cbind(sepscores,names)
    sepscores[,4] = factor(sepscores[,4], levels = c("Texture", "Shape", "Size", "Movement", "Density"))
    
    fig<-plot_ly(data.frame(sepscores), y = ~sepscores[,3], x = 1:length(sepscores[,1]), color = ~sepscores[,4], colors = c("black", "violetred1", "turquoise4", "orchid", "skyblue"), text = sepscores[,2], hoverinfo = text)%>%
      layout( 
        yaxis = list(title = 'Separation score'))
    
    fig
    
  })
  
  output$downloadSepScores<-downloadHandler(
    filename = function(){
      (paste("SepScores_",input$septhreshold,"threshold.csv",sep=""))
    },
    content = function(file)
    {
      write.csv(separationscores(), file)
    }
    
  )
  
  balancedTraining1<-reactive({
    req(input$extracted1)
    req(input$extracted2)
    req(input$cell1name)
    req(input$cell2name)
    extracted1<-read.csv(input$extracted1$datapath)
    extracted2<-read.csv(input$extracted2$datapath)
    group1names<-rep(input$cell1name, dim(extracted1)[1])
    group2names<-rep(input$cell2name, dim(extracted2)[1])
    group1<-cbind(group1names,extracted1)
    colnames(group1)[colnames(group1) == 'group1names'] <- 'Group'
    group2<-cbind(group2names, extracted2)
    colnames(group2)[colnames(group2) == 'group2names'] <- 'Group'
    size1<-dim(group1)[1]
    size2<-dim(group2)[1]
    
    if(size1 > size2)
    {
      sample<-sample(1:size1, size2, replace = FALSE)
      group1<-group1[sample,]
    }
    
    if(size2 > size1)
    {
      sample<-sample(1:size2, size1, replace = FALSE)
      group2<-group2[sample,]
    }
    
    rbind(group1,group2)
    
  })
  
  
  errorRate1<-eventReactive(input$errorRates1, {
    req(allData())
    req(balancedTraining1())
    req(input$extracted1)
    req(input$extracted2)
    req(input$cell1name)
    req(input$cell2name)
    
    extracted1<-read.csv(input$extracted1$datapath)
    extracted2<-read.csv(input$extracted2$datapath)
    
    thresholds = c(0,0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5)
    
    separationscores<-lapply(thresholds, CellPhe::calculateSeparationScores, group1data = extracted1[,-1], group2data = extracted2[,-1])
    
    ErrRate = matrix(nrow = length(thresholds), ncol = 1)
    
    withProgress(message = "Calculating optimal separation threshold",
                 for (i in (1:length(thresholds)))
                 {
                   incProgress(amount = 1/length(thresholds))
                   correct = 0
                   if(length(separationscores[[i]][[1]]) > 5)
                   {
                     subtrain<-subsetBySeparationThreshold(balancedTraining1(), separationscores, i)
                     classificationresults = cellPopulationClassification(subtrain, subtrain, as.factor(balancedTraining1()[,1]))
                     for(j in c(1:dim(balancedTraining1())[1]))
                     {
                       if(classificationresults[j,4] == balancedTraining1()[j,1])
                       {
                         correct = correct+1
                       }
                     }
                   }
                   ErrRate[i] = 1-(correct/(dim(balancedTraining1())[1]))
                 }
    )
    
    
    # choosing optimal separation threshold
    
    for (i in c(1:(length(ErrRate)-1)))
    {
      increase = (ErrRate[i+1] - ErrRate[i])*100
      
      if((increase > 1) == TRUE)
      {
        break
      }
    }
    
    thresholds[i-1]
    
  })
  
  output$errorRateNote1<-renderText({req(errorRate1())
    paste("The optimal threshold is:", errorRate1())})
  
  
  output$PCAscores<-renderPlotly({req(allData())
    req(separationscoresthresh())
    req(input$septhreshold)
    palette("default")
    data<-allData()[,-c(1,2)]
    separationscores<-separationscoresthresh()
    forpca<-data[,separationscores[,2]]
    pca<-prcomp(scale(forpca))
    fig<-plot_ly(data = data.frame(pca$x), x = ~pca$x[,1], y = ~pca$x[,2], text = allData()[,2], hoverinfo = 'text', color = factor(allData()[,1], levels = c(input$cell1name, input$cell2name)), colors = c("red", "blue"))%>%
      layout(title = 'PCA scores plot', xaxis = list(title = 'PC1'), 
             yaxis = list(title = 'PC2'))
    fig
    # plot(pca$x[,1], pca$x[,2], col = as.factor(allData()[,1]), pch = 20, xlab = "PC1", ylab = "PC2", cex = 2, main = "PCA scores plot")
    
  })
  output$PCAbiplot<-renderPlot({req(allData())
    req(separationscoresthresh())
    req(input$septhreshold)
    
    data<-allData()[,-c(1,2)]
    
    separationscores<-separationscoresthresh()
    forpca<-data[,separationscores[,2]]
    pca<-prcomp(scale(forpca))
    features = categoriseDataFeatures(forpca)
    variables = categoriseSummaryStat(forpca)
    concat = paste(features, variables, sep = "_")
    rownames(pca$rotation) = concat
    biplot(pca, main = "PCA biplot", col = c("white", "black"))
    
  })
  
  output$UMAP<-renderPlotly({req(allData())
    req(separationscoresthresh())
    req(input$septhreshold)
    palette("default")
    data<-allData()[,-c(1,2)]
    separationscores<-separationscoresthresh()
    forumap<-data[,separationscores[,2]]
    
    umap<-umap::umap(scale(forumap))
    fig<-plot_ly(data = data.frame(umap$layout), x = ~umap$layout[,1], y = ~umap$layout[,2], text = allData()[,2], hoverinfo = 'text', color = factor(allData()[,1], levels = c(input$cell1name, input$cell2name)), colors = c("red", "blue"))%>%
      layout(title = 'UMAP', xaxis = list(title = 'UMAP1'), 
             yaxis = list(title = 'UMAP2'))
    fig
  })
  
  output$tSNE<-renderPlotly({req(allData())
    req(separationscoresthresh())
    req(input$septhreshold)
    palette("default")
    data<-allData()[,-c(1,2)]
    data<-data[ ,-caret::nearZeroVar(data)] 
    separationscores<-separationscoresthresh()
    fortsne<-data[,separationscores[,2]]
    
    tsne<-Rtsne::Rtsne(scale(fortsne), perplexity = 30, check_duplicates = FALSE)
    fig<-plot_ly(data = data.frame(tsne$Y), x = ~tsne$Y[,1], y = ~tsne$Y[,2], text = allData()[,2], hoverinfo = 'text', color = factor(allData()[,1], levels = c(input$cell1name, input$cell2name)), colors = c("red", "blue"))%>%
      layout(title = 't-SNE', xaxis = list(title = 't-SNE1'), 
             yaxis = list(title = 't-SNE2'))
    fig
    
  })
  
  balancedTraining<-reactive({
    req(input$group1)
    req(input$group2)
    req(input$group1name)
    req(input$group2name)
    group1<-read.csv(input$group1$datapath)
    group2<-read.csv(input$group2$datapath)
    group1names<-rep(input$group1name, dim(group1)[1])
    group2names<-rep(input$group2name, dim(group2)[1])
    group1<-cbind(group1names,group1)
    colnames(group1)[colnames(group1) == 'group1names'] <- 'Group'
    group2<-cbind(group2names, group2)
    colnames(group2)[colnames(group2) == 'group2names'] <- 'Group'
    size1<-dim(group1)[1]
    size2<-dim(group2)[1]
    
    if(size1 > size2)
    {
      sample<-sample(1:size1, size2, replace = FALSE)
      group1<-group1[sample,]
    }
    
    if(size2 > size1)
    {
      sample<-sample(1:size2, size1, replace = FALSE)
      group2<-group2[sample,]
    }
    
    rbind(group1,group2)
    
  })
  
  separationscoresForClass<-reactive({
    req(input$group1)
    req(input$group2)
    req(input$classificationSepThresh)
    
    group1<-read.csv(input$group1$datapath)
    group2<-read.csv(input$group2$datapath)
    
    CellPhe::calculateSeparationScores(group1[,-1], group2[,-1], input$classificationSepThresh)
  })
  
  
  classificationresults<-eventReactive(input$classify, {
    req(balancedTraining())
    req(input$classificationTestSet)
    req(separationscoresForClass())
    
    test = read.csv(input$classificationTestSet$datapath)
    test = data.frame(test)
    train = data.frame(balancedTraining())
    
    
    subtest<-test[,separationscoresForClass()[,2]]
    subtrain<-train[,separationscoresForClass()[,2]]
    
    cellPopulationClassification(subtrain, subtest, as.factor(train[,1]))
    
  })
  
  output$classificationresultsLabels<-renderPrint({
    req(classificationresults())
    req(input$testSetLabels)
    labels<-read.csv(input$testSetLabels$datapath, header = FALSE)
    results<-table(True = labels[,1], Predicted = classificationresults()[,4])
    results
  })
  
  output$classificationaccuracy <- renderTable({
    req(classificationresults())
    req(input$testSetLabels)
    labels<-read.csv(input$testSetLabels$datapath, header = FALSE)
    confusionmatrix<-caret::confusionMatrix(as.factor(classificationresults()[,4]), as.factor(labels[,1]))
    confusionmatrix$byClass[c(1,2,11)]*100
  }, rownames = TRUE, colnames = FALSE)
  
  output$ROC_curve<-renderPlot({
    req(classificationresults())
    req(balancedTraining())
    req(input$testSetLabels)
    labels<-read.csv(input$testSetLabels$datapath, header = FALSE)
    votes<-classificationresults()
    
    
    votes[1:3][votes[1:3] == unique(balancedTraining()[,1])[1]] <- 0
    votes[1:3][votes[1:3] == unique(balancedTraining()[,1])[2]] <- 1
    
    votes[,1:3] <- lapply(votes[,1:3], function(x) as.numeric(as.character(x)))
    probs<-rowSums(votes[,1:3])/3
    pred <- ROCR::prediction(1-probs, labels)
    perf <- ROCR::performance(pred,"tpr","fpr")
    
    auc_ROCR <- ROCR::performance(pred, measure = "auc")
    auc_ROCR <- auc_ROCR@y.values[[1]]
    title = paste("ROC curve, AUC:", round(auc_ROCR, 2), sep = " ")
    
    plot(perf, col = "darkgray", lwd = 2, main = title)
  })
  
  output$postprobabilities<-renderPlot({
    req(classificationresults())
    req(input$classificationTestSet)
    req(balancedTraining())
    req(input$testSetLabels)
    labels<-read.csv(input$testSetLabels$datapath, header = FALSE)
    votes<-classificationresults()
    test = read.csv(input$classificationTestSet$datapath)
    
    
    votes[1:3][votes[1:3] == unique(balancedTraining()[,1])[1]] <- 0
    votes[1:3][votes[1:3] == unique(balancedTraining()[,1])[2]] <- 1
    
    votes[,1:3] <- lapply(votes[,1:3], function(x) as.numeric(as.character(x)))
    probs<-rowSums(votes[,1:3])/3
    results<-cbind(labels, test[,1], probs)
    group1<-subset(results, results[,1] == unique(results[,1])[1])
    group2<-subset(results, results[,1] == unique(results[,1])[2])
    
    max1 = max(density((1-group1[,3])*100)$y)
    max2 = max(density((group2[,3])*100)$y)
    max = max(max1,max2)
    plot(density((1-group1[,3])*100), col = "red", lwd = 2, ylim = c(0,max), xlab = "Classifier votes for correct class (%)", main = "Summary of ensemble classification results")
    lines(density((group2[,3]*100)), col = "blue", lwd = 2)
    legend("topleft", legend = unique(results[,1]), fill = c("red", "blue"))
  })
  
  output$classificationresultsNoLabels<-renderTable({
    req(classificationresults())
    results<-table(Predicted = classificationresults()[,4])
    results<-as.data.frame(results)
    results
  })
  
  errorRate<-eventReactive(input$errorRates, {
    req(balancedTraining())
    req(input$group1)
    req(input$group2)
    req(input$classificationSepThresh)
    
    group1<-read.csv(input$group1$datapath)
    group2<-read.csv(input$group2$datapath)
    
    thresholds = c(0,0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5)
    
    separationscores<-lapply(thresholds, CellPhe::calculateSeparationScores, group1data = group1[,-1], group2data = group2[,-1])
    
    ErrRate = matrix(nrow = length(thresholds), ncol = 1)
    
    withProgress(message = "Calculating optimal separation threshold",
                 for (i in (1:length(thresholds)))
                 {
                   incProgress(amount = 1/length(thresholds))
                   correct = 0
                   if(length(separationscores[[i]][[1]]) > 5)
                   {
                     subtrain<-subsetBySeparationThreshold(balancedTraining(), separationscores, i)
                     classificationresults = cellPopulationClassification(subtrain, subtrain, as.factor(balancedTraining()[,1]))
                     for(j in c(1:dim(balancedTraining())[1]))
                     {
                       if(classificationresults[j,4] == balancedTraining()[j,1])
                       {
                         correct = correct+1
                       }
                     }
                   }
                   ErrRate[i] = 1-(correct/(dim(balancedTraining())[1]))
                 }
    )
    
    
    # choosing optimal separation threshold
    
    for (i in c(1:(length(ErrRate)-1)))
    {
      increase = (ErrRate[i+1] - ErrRate[i])*100
      
      if((increase > 1) == TRUE)
      {
        break
      }
    }
    
    thresholds[i-1]
    
  })
  
  output$errorRateNote<-renderText({req(errorRate())
    paste("The optimal threshold is:", errorRate())})
  
  tosave1<-reactive({
    
    req(input$testSetLabels)
    req(input$classificationTestSet)
    req(classificationresults())
    
    test = read.csv(input$classificationTestSet$datapath)
    labels = read.csv(input$testSetLabels$datapath)
    table<-cbind(test[,1], labels[,1], as.character(classificationresults()[,4]))
    colnames(table)<-c("CellID", "TrueLabel", "PredictedLabel")
    table
    
  })
  
  output$downloadclassresults1<-downloadHandler(
    
    filename = function(){
      paste("TestSetClassifications.csv")
    },
    content = function(file)
    {
      write.csv(tosave1(), file, row.names = FALSE)
    }
  )
  
  output$downloadclassresults0<-downloadHandler(
    filename = function(){
      paste("TestSetClassifications.csv", sep = "")
    },
    content = function(file)
    {
      req(input$classificationTestSet)
      req(classificationresults())
      
      test = read.csv(input$classificationTestSet$datapath)
      
      table<-cbind(test[,1], as.character(classificationresults()[,4]))
      colnames(table)<-c("CellID", "PredictedLabel")
      
      write.csv(table, file, row.names = FALSE)
    }
  )
  
  output$optimalclusts<-renderPlot({req(input$forclusters)
    data = read.csv(input$forclusters$datapath)
    data<-data[ ,-1] 
    k.max <- 10
    d <- dist(scale(data))
    wss <- sapply(1:k.max,
                  function(k){kmeans(d, k, nstart=50,iter.max = 15)$tot.withinss})
    wss
    plot(1:k.max, wss,
         type="b", pch = 19, frame = FALSE,
         xlab="Number of clusters",
         ylab="Total within-clusters sum of squares", cex.lab = 1.2, cex.axis = 1.2)})
  
  hierclusts<-reactive({req(input$forclusters)
    req(input$numclust)
    data = read.csv(input$forclusters$datapath)
    d = dist(scale(data[,-1]))
    hierclust<-factoextra::hcut(d, hc_func = "agnes", hc_method = "ward.D", hc_metric = "euclidean", k = input$numclust)
  })
  
  kmean<-reactive({req(input$forclusters)
    req(input$numclust)
    data = read.csv(input$forclusters$datapath)
    d = dist(scale(data[,-1]))
    kmean = kmeans(d,input$numclust,iter.max = 500)
  })
  
  output$hierclusts<-renderPlot({req(input$forclusters)
    req(input$numclust)
    data = read.csv(input$forclusters$datapath)
    d = dist(scale(data[,-1]))
    hierclust<-factoextra::hcut(d, hc_func = "agnes", hc_method = "ward.D", hc_metric = "euclidean", k = input$numclust)
    factoextra::fviz_dend(hierclust, show_labels = FALSE)+ theme(text = element_text(size = 20))})
  
  output$kmeans<-renderPlot({req(input$forclusters)
    req(input$numclust)
    data = read.csv(input$forclusters$datapath)
    d = dist(scale(data[,-1]))
    kmean = kmeans(d,input$numclust,iter.max = 500)
    factoextra::fviz_cluster(kmean, data = d, labelsize = 0, pointsize = 2, legend.title = "Cluster", xlab = "PC1", ylab = "PC2")+theme_minimal(base_size = 20)})
  
  output$hierclustclusters<-renderDataTable({req(input$forclusters)
    req(hierclusts())
    data = read.csv(input$forclusters$datapath)
    table<-cbind(data[,1], hierclusts()$cluster)
    colnames(table) = c("Cell ID", "Assigned cluster")
    table
  }, filter = 'top',
  rownames = FALSE, options = list(
    "pageLength" = 50))
  
  output$kmeanclusters<-renderDataTable({req(input$forclusters)
    req(kmean())
    data = read.csv(input$forclusters$datapath)
    table<-cbind(data[,1], kmean()$cluster)
    colnames(table) = c("Cell ID", "Assigned cluster")
    table
  }, filter = 'top',
  rownames = FALSE, options = list(
    "pageLength" = 50))
  
  output$downloadkmeans<-downloadHandler(
    filename = function(){
      paste("kmeanclusters.csv", sep = "")
    },
    content = function(file)
    {
      req(input$forclusters)
      req(kmean())
      data = read.csv(input$forclusters$datapath)
      table<-cbind(data[,1], kmean()$cluster)
      colnames(table) = c("Cell ID", "Assigned cluster")
      
      write.csv(table, file, row.names = FALSE)
    }
  )
  
  output$downloadhierclusts<-downloadHandler(
    filename = function(){
      paste("hierarchicalclusters.csv", sep = "")
    },
    content = function(file)
    {
      req(input$forclusters)
      req(hierclusts())
      data = read.csv(input$forclusters$datapath)
      table<-cbind(data[,1], hierclusts()$cluster)
      colnames(table) = c("Cell ID", "Assigned cluster")
      
      write.csv(table, file, row.names = FALSE)
    }
  )
  
  output$variableNameforDensityPlots<-renderUI({req(input$forclusters)
    data = read.csv(input$forclusters$datapath)
    colnames<-unique(sapply(strsplit(colnames(data[,-1]), "[_]"),"[[",1))
    colnames = subset(colnames, colnames != "trajArea")
    selectizeInput("variableNameforDensityPlots", label = "Which variable would you like to display?", choices = colnames, options= list(maxOptions = 2000))})
  
  output$Hierarch1<-renderPlot({req(input$forclusters)
    req(hierclusts())
    req(input$variableNameforDensityPlots)
    data = read.csv(input$forclusters$datapath)
    p<-ggplot(data, aes(x = data[,paste(input$variableNameforDensityPlots,"_mean", sep = "")], colour = as.factor(hierclusts()$cluster))) + geom_density()+theme_minimal(base_size = 15)
    p + xlab(paste(input$variableNameforDensityPlots,"_mean", sep = "")) + ylab("Density")+ggtitle("Coloured by hierarchical clusters")+scale_colour_discrete(name = "Cluster")
  })
  
  output$Hierarch2<-renderPlot({req(input$forclusters)
    req(hierclusts())
    req(input$variableNameforDensityPlots)
    data = read.csv(input$forclusters$datapath)
    p<-ggplot(data, aes(x = data[,paste(input$variableNameforDensityPlots,"_std", sep = "")], colour = as.factor(hierclusts()$cluster))) + geom_density()+theme_minimal(base_size = 15)
    p + xlab(paste(input$variableNameforDensityPlots,"_std", sep = "")) + ylab("Density")+ggtitle("Coloured by hierarchical clusters")+scale_colour_discrete(name = "Cluster")
  })
  
  output$Hierarch3<-renderPlot({req(input$forclusters)
    req(hierclusts())
    req(input$variableNameforDensityPlots)
    data = read.csv(input$forclusters$datapath)
    p<-ggplot(data, aes(x = data[,paste(input$variableNameforDensityPlots,"_skew", sep = "")], colour = as.factor(hierclusts()$cluster))) + geom_density()+theme_minimal(base_size = 15)
    p + xlab(paste(input$variableNameforDensityPlots,"_skew", sep = "")) + ylab("Density")+ggtitle("Coloured by hierarchical clusters")+scale_colour_discrete(name = "Cluster")
  })
  
  output$Hierarch4<-renderPlot({req(input$forclusters)
    req(hierclusts())
    req(input$variableNameforDensityPlots)
    data = read.csv(input$forclusters$datapath)
    p<-ggplot(data, aes(x = data[,paste(input$variableNameforDensityPlots,"_asc", sep = "")], colour = as.factor(hierclusts()$cluster))) + geom_density()+theme_minimal(base_size = 15)
    p + xlab(paste(input$variableNameforDensityPlots,"_asc", sep = "")) + ylab("Density")+ggtitle("Coloured by hierarchical clusters")+scale_colour_discrete(name = "Cluster")
  })
  
  output$Hierarch5<-renderPlot({req(input$forclusters)
    req(hierclusts())
    req(input$variableNameforDensityPlots)
    data = read.csv(input$forclusters$datapath)
    p<-ggplot(data, aes(x = data[,paste(input$variableNameforDensityPlots,"_des", sep = "")], colour = as.factor(hierclusts()$cluster))) + geom_density()+theme_minimal(base_size = 15)
    p + xlab(paste(input$variableNameforDensityPlots,"_des", sep = "")) + ylab("Density")+ggtitle("Coloured by hierarchical clusters")+scale_colour_discrete(name = "Cluster")
  })
  
  output$Hierarch6<-renderPlot({req(input$forclusters)
    req(hierclusts())
    req(input$variableNameforDensityPlots)
    data = read.csv(input$forclusters$datapath)
    p<-ggplot(data, aes(x = data[,paste(input$variableNameforDensityPlots,"_max", sep = "")], colour = as.factor(hierclusts()$cluster))) + geom_density()+theme_minimal(base_size = 15)
    p + xlab(paste(input$variableNameforDensityPlots,"_max", sep = "")) + ylab("Density")+ggtitle("Coloured by hierarchical clusters")+scale_colour_discrete(name = "Cluster")
  })
  
  output$Kmean1<-renderPlot({req(input$forclusters)
    req(kmean())
    req(input$variableNameforDensityPlots)
    data = read.csv(input$forclusters$datapath)
    p<-ggplot(data, aes(x = data[,paste(input$variableNameforDensityPlots,"_mean", sep = "")], colour = as.factor(kmean()$cluster))) + geom_density()+theme_minimal(base_size = 15)
    p + xlab(paste(input$variableNameforDensityPlots,"_mean", sep = "")) + ylab("Density")+ggtitle("Coloured by k-means clusters")+scale_colour_discrete(name = "Cluster")
  })
  
  output$Kmean2<-renderPlot({req(input$forclusters)
    req(kmean())
    req(input$variableNameforDensityPlots)
    data = read.csv(input$forclusters$datapath)
    p<-ggplot(data, aes(x = data[,paste(input$variableNameforDensityPlots,"_std", sep = "")], colour = as.factor(kmean()$cluster))) + geom_density()+theme_minimal(base_size = 15)
    p + xlab(paste(input$variableNameforDensityPlots,"_std", sep = "")) + ylab("Density")+ggtitle("Coloured by k-means clusters")+scale_colour_discrete(name = "Cluster")
  })
  
  output$Kmean3<-renderPlot({req(input$forclusters)
    req(kmean())
    req(input$variableNameforDensityPlots)
    data = read.csv(input$forclusters$datapath)
    p<-ggplot(data, aes(x = data[,paste(input$variableNameforDensityPlots,"_skew", sep = "")], colour = as.factor(kmean()$cluster))) + geom_density()+theme_minimal(base_size = 15)
    p + xlab(paste(input$variableNameforDensityPlots,"_skew", sep = "")) + ylab("Density")+ggtitle("Coloured by k-means clusters")+scale_colour_discrete(name = "Cluster")
  })
  
  output$Kmean4<-renderPlot({req(input$forclusters)
    req(kmean())
    req(input$variableNameforDensityPlots)
    data = read.csv(input$forclusters$datapath)
    p<-ggplot(data, aes(x = data[,paste(input$variableNameforDensityPlots,"_asc", sep = "")], colour = as.factor(kmean()$cluster))) + geom_density()+theme_minimal(base_size = 15)
    p + xlab(paste(input$variableNameforDensityPlots,"_asc", sep = "")) + ylab("Density")+ggtitle("Coloured by k-means clusters")+scale_colour_discrete(name = "Cluster")
  })
  
  output$Kmean5<-renderPlot({req(input$forclusters)
    req(kmean())
    req(input$variableNameforDensityPlots)
    data = read.csv(input$forclusters$datapath)
    p<-ggplot(data, aes(x = data[,paste(input$variableNameforDensityPlots,"_des", sep = "")], colour = as.factor(kmean()$cluster))) + geom_density()+theme_minimal(base_size = 15)
    p + xlab(paste(input$variableNameforDensityPlots,"_des", sep = "")) + ylab("Density")+ggtitle("Coloured by k-means clusters")+scale_colour_discrete(name = "Cluster")
  })
  
  output$Kmean6<-renderPlot({req(input$forclusters)
    req(kmean())
    req(input$variableNameforDensityPlots)
    data = read.csv(input$forclusters$datapath)
    p<-ggplot(data, aes(x = data[,paste(input$variableNameforDensityPlots,"_max", sep = "")], colour = as.factor(kmean()$cluster))) + geom_density()+theme_minimal(base_size = 15)
    p + xlab(paste(input$variableNameforDensityPlots,"_max", sep = "")) + ylab("Density")+ggtitle("Coloured by k-means clusters")+scale_colour_discrete(name = "Cluster")
  })
}

shinyApp(ui, server)

