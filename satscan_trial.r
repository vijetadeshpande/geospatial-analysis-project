library("rsatscan")
invisible(ss.options(reset = TRUE))

##
ss.options(list(CaseFile = "NYCfever.cas", PrecisionCaseTimes = 3))
ss.options(c("StartDate=2010/11/1", "EndDate=2012/11/1"))
ss.options(list(CoordinatesFile = "NYCfever.geo", AnalysisType = 4, 
                ModelType = 2, TimeAggregationUnits = 3))
ss.options(list(UseDistanceFromCenterOption = "y",
                MaxSpatialSizeInDistanceFromCenter = 3,
                NonCompactnessPenalty = 0))
ss.options(list(MaxTemporalSizeInterpretation = 1,
                MaxTemporalSize = 7))
ss.options(list(ProspectiveStartDate = "2010/11/1",
                ReportGiniClusters = "n",
                LogRunToHistoryFile = "n"))

head(NYCfevercas)
head(NYCfevergeo)

td = tempdir()
write.ss.prm(td, "NYCfever")
write.cas(NYCfevercas, td, "NYCfever")
write.geo(NYCfevergeo, td, "NYCfever")

## run satscan
NYCfever = satscan(td, "NYCfever", sslocation = "C:/Program Files (x86)/SaTScan")

