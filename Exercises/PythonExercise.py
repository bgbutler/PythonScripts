# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:34:15 2016

@author: adam.brzozowski
"""

import calendar
#import PIFModelMain as m
#from PIFModelMain import MapaWaterfall as m
import mysql.connector
from mysql.connector import errorcode
import pandas as pd
import datetime
#import numpy as np
import time
#import AdamDB as a
import math
from mysql.connector.constants import SQLMode
import numpy as np

StartTimePullDataDown = time.time()

#extracts database credentials for use in the program
Database = 'policy_management'
#S = a.dbCreds('read', 'policy_management')

User = #S['user']
Password =  #S['password']
Host =  #S['host']

StartTime = time.time()

#export data from db to dataframe
def sqlToDF(sql,cnx):
   df = pd.DataFrame()
   for chunk in pd.read_sql(sql,cnx,chunksize=1000):
       df=df.append(chunk)
       #print len(df)
   return df

# Prepares the query and puts the data in a dataframe
def getSqlData(sql,cnx,maxObs):
   if maxObs=='max':
       fQuery=sql
   else:
       fQuery=sql+" limit %s" % (maxObs)
   sqlData=sqlToDF(fQuery,cnx)
   return sqlData

#Concatenates a whole bunch of SQL text input commands
sqltxt = 'Select 					\n    PolicyNumber,\n    Carrier,					\n    Type,					\n    Tier,					\n    Term,					\n    Concat(Carrier, Type, Term, ifnull(Tier, \'\')) As \'Product\',					\n    OriginalEffectiveDate,\n    TerminationDate,\n    Premium,					\n    RenewalCommissionPercentage,			\n    rf.Alpha,\n    rf.Beta,\n    ifnull(ContractType, \'NA\') as \'ContractType\',					\n	Case when Cohort is not Null then Case when Right(Cohort,1) = \'P\' Then Left(Cohort,7) Else Cohort End Else \'NA\' End As \'Cohort\',\n    Cohort as \'HistoricalCohort\',\n    Case					\n        when					\n            ContractType = \'Securitization\'					\n                and Cohort in (\'2013-09\' , \'2013-11\')					\n        Then					\n            \'MAPA-STD\'					\n        when					\n            ContractType = \'Securitization\'					\n                and Cohort not in (\'2013-09\' , \'2013-11\')\n                and CohortAcquisitionDate < \'2015-10-01\'\n        Then					\n            \'MAPA-ACC\'	\n		When Cohort not in (\'2013-09\' , \'2013-11\') and CohortAcquisitionDate > \'2015-10-01\' and Right(Cohort,2) != \'SL\' Then\n			\'MAPA-OCT\'\n            \n		else \'NA\'\n            \n    End As \'MAPAPaybackRateCategory\', \n	PriorToRepayment,\n    PostRepayment,\n    cast(CohortAcquisitionDate as datetime) as \'CohortAcquisitionDate\',\n	year(OriginalEffectiveDate) as \'OriginalEffectiveYear\',\n    month(OriginalEffectiveDate) as \'OriginalEffectiveMonth\',\n    day(OriginalEffectiveDate) as \'OriginalEffectiveDay\'\n    \n    \nFrom					\n    policy_management.PolicyToday pt left join analytics.RetentionFactors rf on \n    \nTRIM(CONCAT(IFNULL(pt.Carrier, \'\'),\n                IFNULL(pt.Type, \'\'),\n                IFNULL(pt.Term, \'\'),\n                IFNULL(pt.tier, \'\'))) = rf.ProductCode\n\n    left join analytics.MAPASplitRatesToCU mr \n    \n    on \n    \n    Case when					\n            Carrier = \'PROGSS\'					\n                and ContractType = \'Securitization\'	and CohortAcquisitionDate < \'2015-10-01\'				\n        Then					\n            \'MAPA - PRG\'					\n        when					\n            ContractType = \'Securitization\'					\n                and Cohort in (\'2013-09\' , \'2013-11\')					\n        Then					\n            \'MAPA - STD\'					\n        when					\n            ContractType = \'Securitization\'					\n                and Cohort not in (\'2013-09\' , \'2013-11\')\n                and CohortAcquisitionDate < \'2015-10-01\'\n        Then					\n            \'MAPA - ACC\'	\n		When Cohort not in (\'2013-09\' , \'2013-11\') and CohortAcquisitionDate > \'2015-10-01\' and Right(Cohort,2) != \'SL\' Then\n			\'MAPA - OCT\'\n            \n    End = trim(mr.SecuritizationType)\n \nWhere					\n    RenewalCommissionPercentage > 0					\n        and Premium Is Not Null					\nand Premium > 0 and (Status = \'In Force\' )'
sqltxtMapa = 'Select a.Cohort, sum(a.PurchasePrice) as \'MAPABalance\', sum(a.Stretch) as \'STL Stretch\', sum(a.PurchasePrice + a.Stretch) as \'MAPA, Stretch\'\nfrom\n\n(Select Cohort, sum(PurchasePriceWithModification) as \'PurchasePrice\', sum(FifteenPercentStretch) as \'Stretch\'\nfrom policy_management.ThayerPolicies\nwhere right(Cohort, 2) != \'SL\'\nGroup by 1\n\nUnion\n\nSelect Cohort, sum(PurchasePriceWithModification) as \'PurchasePrice\', sum(FifteenPercentStretch) as \'Stretch\'\nfrom policy_management.ThayerTBRHistory\nWhere right(Cohort, 2) != \'SL\'\nGroup by 1) as a\ngroup by 1'
sqltxtCurrentMapa = 'SELECT Left(a.Cohort,7) as \'Cohort\', sum(a.Balance) as \'MAPABalance\' \nFROM \n	(Select Cohort, Min(NoteEndingBalance) as \'Balance\' from analytics.WaterfallsHistoricals Where NoteEndingBalance >= 0 Group by 1) as a\n\nGroup by 1\n\nUnion\nSelect Left(Cohort,7), sum(FinalPurchasePrice) as \'Balance\'\nfrom policy_management.PolicyToday\nWhere Right(Cohort, 2) != \'SL\' and CohortAcquisitionDate >= \'2016-01-01\'\nGroup by 1'
sqlMAPAMap = 'Select T1.Cohort, Case when T1.MAPAPaybackRateCategory is not Null then T1.MAPAPaybackRateCategory else \'NA\' End as \'MAPAPaybackRateCategory\'\nfrom\n\n(Select 					\n    PolicyNumber,\n    Carrier,					\n    Type,					\n    Tier,					\n    Term,					\n    Concat(Carrier, Type, Term, ifnull(Tier, \'\')) As \'Product\',					\n    OriginalEffectiveDate,\n    TerminationDate,\n    Premium,					\n    RenewalCommissionPercentage,			\n    rf.Alpha,\n    rf.Beta,\n    ifnull(ContractType, \'NA\') as \'ContractType\',					\n	Case when Cohort is not Null then Cohort Else \'NA\' End As \'Cohort\',				\n    Case					\n        when					\n            Carrier = \'PROGSS\'					\n                and ContractType = \'Securitization\'	and CohortAcquisitionDate < \'2015-10-01\'				\n        Then					\n            \'MAPA-PRG\'					\n        when					\n            ContractType = \'Securitization\'					\n                and Cohort in (\'2013-09\' , \'2013-11\')					\n        Then					\n            \'MAPA-STD\'					\n        when					\n            ContractType = \'Securitization\'					\n                and Cohort not in (\'2013-09\' , \'2013-11\')\n                and CohortAcquisitionDate < \'2015-10-01\'\n        Then					\n            \'MAPA-ACC\'	\n		When Cohort not in (\'2013-09\' , \'2013-11\') and CohortAcquisitionDate > \'2015-10-01\' and Right(Cohort,2) != \'SL\' Then\n			\'MAPA-OCT\'\n            \n    End As \'MAPAPaybackRateCategory\', \n	PriorToRepayment,\n    PostRepayment,\n    cast(CohortAcquisitionDate as datetime) as \'CohortAcquisitionDate\'\n    \nFrom					\n    policy_management.PolicyToday pt left join analytics.RetentionFactors rf on \n    \nTRIM(CONCAT(IFNULL(pt.Carrier, \'\'),\n                IFNULL(pt.Type, \'\'),\n                IFNULL(pt.Term, \'\'),\n                IFNULL(pt.tier, \'\'))) = rf.ProductCode\n\n    left join analytics.MAPASplitRatesToCU mr \n    \n    on \n    \n    Case when					\n            Carrier = \'PROGSS\'					\n                and ContractType = \'Securitization\'	and CohortAcquisitionDate < \'2015-10-01\'				\n        Then					\n            \'MAPA - PRG\'					\n        when					\n            ContractType = \'Securitization\'					\n                and Cohort in (\'2013-09\' , \'2013-11\')					\n        Then					\n            \'MAPA - STD\'					\n        when					\n            ContractType = \'Securitization\'					\n                and Cohort not in (\'2013-09\' , \'2013-11\')\n                and CohortAcquisitionDate < \'2015-10-01\'\n        Then					\n            \'MAPA - ACC\'	\n		When Cohort not in (\'2013-09\' , \'2013-11\') and CohortAcquisitionDate > \'2015-10-01\' and Right(Cohort,2) != \'SL\' Then\n			\'MAPA - OCT\'\n            \n    End = trim(mr.SecuritizationType)\n \nWhere					\n    RenewalCommissionPercentage > 0					\n        and Premium Is Not Null					\nand Premium > 0) As T1\nGroup by 1,2'
sqlStl ='Select CohortAcquisitionDate, sum(FinalCollateralizedPrice + FifteenPercentStretch) as \'StlBalances\'\nfrom policy_management.ThayerPolicies\nWhere Cohort is not null\nGroup by 1'


try:
    cnx = mysql.connector.connect(    user = User, 
                                      password = Password, 
                                      host = Host , 
                                      database = Database)
    cursor = cnx.cursor()    

    book = getSqlData(sqltxt,cnx,'max')
    OriginalCohortBalances = getSqlData(sqltxtMapa,cnx,'max')
    CurrentMapaBalances = getSqlData(sqltxtCurrentMapa,cnx,'max')
    MAPACohortMap = getSqlData(sqlMAPAMap,cnx,'max')
    StlBalances = getSqlData(sqlStl,cnx,'max')    
    
    cursor.close()
    cnx.close()
    
except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("USERNAME AND / OR PASSWORD IS INCORRECT")
    elif err.errno ==errorcode.ER_BAD_DB_ERROR:
        print("Database does not exist")
    else:
        print(err)
else:
    cursor.close()
    cnx.close()

# tabulates the distinct cohorts in the book for the loops used in the program
#Cohort = book.Cohort.value_counts()

MapaCohorts = set(book['MAPAPaybackRateCategory'])
Cohorts = book['Cohort'].unique()

book.Term = book.Term.apply(str)

EndPullDataTransform = time.time()

#records how long it takes to pull the data down
BookPullTime = EndPullDataTransform - StartTimePullDataDown



#PolicyDataResults = pd.DataFrame( columns = ['StartDate','EndDate', 'PolicyNumber', 'Carrier', 'Type', 'Tier', 'Term', 'Premium', 'RenewalCommissionPercentage' , 'OriginalEffectiveDate', 'TerminationDate', 'Cohort' ,'ContractType', 'Alpha', 'Beta'])
TpmFeeFlipDate = datetime.datetime(2016,1,1)
#StartDate = datetime.datetime(StartYear,StartMonth,StartDay) # start date for the analysis

AnalysisWindowYears = 10 # the number of years in the analysis
#EndDate = StartDate + datetime.timedelta(days = AnalysisWindow * 365.25)   # the end date for the analysis based on the analysis window

OldMAPA = {"MAPA-ACC", "MAPA-PRG", "MAPA-STD"} #the flags for the Old mapa in the database
NewMAPA = {"MAPA-OCT"}
OldMapaPreferredReturn = .08

OldMAPAPriorPayOff = {"MAPA-ACC": .85,"MAPA-STD":.80, "MAPA-PRG": 0}
OldMAPAAfterPayOff = {"MAPA-ACC": .80,"MAPA-STD":.72, "MAPA-PRG": .75}

ThresholdZero = 0
ThresholdOne = 1.25
ThresholdTwo = 1.40
ThresholdThree = 1.41

Thresholds = [ThresholdZero, ThresholdOne, ThresholdTwo, ThresholdThree]

NewMapaThayerSplitRate = {ThresholdZero: 1, ThresholdOne: 1, ThresholdTwo: .49, ThresholdThree: .05 }
AnnualizedPremiumDict = {'12':1, '6':2}

#handles the change in TPM fees for prior to 2016 and after
OldMapaTpmDictPrior = {'12': 10,'6': 5 }
OldMapaTpmDictAfter = {'12': 8,'6': 4 }
AnnualizedPremiumDict = {'12':1, '6':2}

#handles Mapa CuServiceFeePrior
CuServiceFeeDict = {'12': 15.0, '6': 7.5}
PolicyTermDict = {'6': 180.0, '12':365.0} #defines the number of days for the threshold analysis used to determine whether or not to use end of first period stick count retention

MapaGrossRenewalsFirstThreshold = datetime.datetime(2015,12,31)
MapaGrossRenewalsSecondThreshold = datetime.datetime(2016,01,01)
MapaThreshold = {'MapaGrossRenewalsFirstThreshold': 10000, 'MapaGrossRenewalsSecondThreshold': 15000}

#handles variables for the STL
StlDaysPerYear = 360
StlInterestRate = .20


Test = datetime.datetime(2014,01,01)

PolicySeries = []
PolicySeriesDict = {'StartDate':0, 'EndDate':1, 'PolicyNumber':2, 'Product':3, 'Carrier':4, 'Type':5, 'Tier':6, 'Term':7, 'Premium':8, 'RenewalCommissionPercentage':9, 'OriginalEffectiveDate':10, 'TerminationDate':11, 'Cohort':12, 'HistoricalCohort': 13, 'ContractType':14, 'Alpha':15, 'Beta':16, 'PriorToRepayment':17, 'PostRepayment':18, 'CohortAcquisitionDate': 19, 'Renewals': 20, 'RenewalCancellations': 21, 'PolicyArea': 22, 'PolicyCancelArea': 23,'MAPAPaybackRateCategory': 24, 'OriginalEffectiveYear': 25, 'OriginalEffectiveMonth':26 , 'OriginalEffectiveDay': 27 }
FieldList = ['StartDate', 'EndDate', 'PolicyNumber', 'Product', 'Carrier', 'Type', 'Tier', 'Term', 'Premium', 'RenewalCommissionPercentage', 'OriginalEffectiveDate', 'TerminationDate', 'Cohort', 'HistoricalCohort', 'ContractType', 'Alpha', 'Beta', 'PriorToRepayment', 'PostRepayment', 'CohortAcquisitionDate', 'Renewals', 'RenewalCancellations', 'PolicyArea', 'PolicyCancelArea','MAPAPaybackRateCategory']

MonthlyCashFlows = []

StartTimePolicySeries = time.time()

MonthlyCashProgress = []

StartYear = 2016
StartMonth = 1
StartDay = 1
#WaterfallRenewalsAccount = 0
 
#StartDate = datetime.datetime(StartYear,StartMonth,StartDay)
#EndDate = datetime.datetime(2023,9,01)

month = StartMonth
year = StartYear
day = StartDay
CumulativeInvestorPayments = 0

#Sensitivity variable for improvement vs disimprovement (-)
RetentionSensitivity = 0


#Function computes the policy area based on the decay curve for the policy
def getPolicyArea(StartedDate, EndedDate, Alpha, Beta, Term, OriginalEffectiveYear, OriginalEffectiveMonth, OriginalEffectiveDay): 
    
    OriginalEffectiveDate = datetime.datetime(OriginalEffectiveYear, OriginalEffectiveMonth, OriginalEffectiveDay)    
    
    PolicyArea = 0
    PolicyCancelArea = 0
    CurrentPolicyAge = 0
    PlusOneTermPolicyAge = 0
        
    if OriginalEffectiveDate <= StartedDate:
    
        print "OriginalEffectiveDate <= StartedDate"
        DeltaDays = PolicyTermDict[Term]
        CurrentScheduledRenewal = OriginalEffectiveDate + datetime.timedelta(days = DeltaDays)    
        
        while CurrentScheduledRenewal < StartedDate:
            CurrentScheduledRenewal += datetime.timedelta(days = DeltaDays)
        
        CurrentPolicyAge = (CurrentScheduledRenewal - OriginalEffectiveDate).days
        PlusOneTermPolicyAge = (CurrentScheduledRenewal - OriginalEffectiveDate).days  + DeltaDays

        if StartedDate <= CurrentScheduledRenewal <= EndedDate: #and CurrentPolicyAge % DeltaDays == 0:
            print "RenewalBetweenStartEndDate"
            #returns the 
            if CurrentPolicyAge /  DeltaDays  <=  1:
                    Alpha = Alpha * (1 + RetentionSensitivity)    
            else:
                    Alpha = 1    
                        
            Beta = Beta * 12 / 365 * (1 - RetentionSensitivity)
            PolicyArea = Alpha * math.exp(Beta * CurrentPolicyAge)
            PolicyCancelArea = Alpha * math.exp(Beta * PlusOneTermPolicyAge) - PolicyArea
        
        else: 
            PolicyArea = 0
            PolicyCancelArea = 0
    else:
        print "OriginalEffectiveDate > StartedDate"
        PolicyArea = 0
        PolicyCancelArea = 0
        CurrentScheduledRenewal = "NotScheduledInInterval"
    return {'PolicyArea':PolicyArea, 'PolicyCancelArea':PolicyCancelArea, 'RenewalDate':CurrentScheduledRenewal, 'StartedDate': StartedDate, 'EndedDate': EndedDate, 'CurrentPolicyAge': CurrentPolicyAge}            



#Function returns Tpm fee per policy based on the date the policy is evaluated
def getTpmFee(WaterfallDate, PolicyArea, Term): ############VETTED###############
    
    if WaterfallDate < TpmFeeFlipDate:
        TpmFee = PolicyArea * OldMapaTpmDictPrior[Term]
        
    else:
        TpmFee = PolicyArea * OldMapaTpmDictAfter[Term]
        
    return TpmFee    


#Function returns CU service fee per policy
def getCuServiceFee(PolicyArea, Term, Premium, RenewalCommissionPercentage): ##########VETTED###########
    
    AnnualizedPremium = Premium * AnnualizedPremiumDict[Term]
    
    #Handles annualized premium proration
    if AnnualizedPremium >= 1100:
        AnnualizedPremium = 1100
    else:
        AnnualizedPremium

    #print AnnualizedPremium
    
    CuServiceFee = PolicyArea * CuServiceFeeDict[Term]* AnnualizedPremium / 1100 * RenewalCommissionPercentage / .15
    return CuServiceFee 


#function that handles the Mapa Dynamics
#class MapaWaterfall:

#Waterfall method for the old mapa
def getOldMapa(EvaluationDate, TotalRenewalCommissions,WaterfallRenewalsAccount,MapaBalance, NetServiceFee, TotalTPMFee, MapaContract, TotalRenewalCancellations, ClawbackReserve):
    
    SplitPaidToGoji = 0
    SplitPaidToThayer = 0
    WaterfallThreshold = 15000
    TPMFeePaid = 0
    ServiceFeePaid = 0
    #MonthlyCashProgress.append([EvaluationDate,"Started OldMAPA Cohort", MapaContract])
    MapaBalance = MapaBalance * (1 + OldMapaPreferredReturn * 30/360)
    #Determines whether or not the threshold for gross renewals is met for the waterfall
    
    if (EvaluationDate >= MapaGrossRenewalsFirstThreshold) and (MapaContract != "MAPA-PRG"):
            WaterfallThreshold = MapaThreshold['MapaGrossRenewalsSecondThreshold']
    
    else:
            WaterfallThreshold = MapaThreshold['MapaGrossRenewalsFirstThreshold']

    #MonthlyCashProgress.append([EvaluationDate,WaterfallThreshold, MapaContract])
    print "WaterfallRenewalsAccount is: ", WaterfallRenewalsAccount
    print "TotalRenewalCommissions is: ", TotalRenewalCommissions
    print "WaterfallThreshold is: ", WaterfallThreshold
   
    #if the total gross renewals 
    if (TotalRenewalCommissions + WaterfallRenewalsAccount) > WaterfallThreshold:
        
        WaterfallRenewalsAccount = 0
        ClawbackReserve = 0
        TPMFeePaid = TotalTPMFee
        ServiceFeePaid = NetServiceFee
        TotalTPMFee = 0 
        NetServiceFee = 0
        
        #Calculates the preferred return for the MAPA investors
        #MonthlyCashProgress.append([EvaluationDate,WaterfallThreshold, MapaContract])
        #computes the distribution balance based on the convention used during that time. Prior to 2016, TPM was out of Goji's payout at the bottom of the waterfall
        if EvaluationDate >= datetime.datetime(2016,01,01):
            DistributionBalance = TotalRenewalCommissions - NetServiceFee - TotalRenewalCancellations - TotalTPMFee
                        
        else:
            DistributionBalance = TotalRenewalCommissions - NetServiceFee - TotalRenewalCancellations #TPM fees not witheld at the top of the waterfall prior to 2016   

       # MonthlyCashProgress.append([EvaluationDate,DistributionBalance, MapaContract])
        #Handles old mapa waterfall dynamics based on the waterfall balance
        if MapaBalance <= 0:
            print "MapaBalance is zero"
            SplitPaidToThayer = DistributionBalance * OldMAPAAfterPayOff[MapaContract]
            SplitPaidToGoji = DistributionBalance * (1 - OldMAPAAfterPayOff[MapaContract]) - TotalTPMFee
            FirstWaterfallBalance = DistributionBalance
            SecondWaterfallBalance = 0
        #MonthlyCashProgress.append([EvaluationDate,OldMAPAAfterPayOff[MapaContract], MapaContract])

        #Splits up the waterfall into the renewals required to repay the balance and the renewals available post repayment
        elif (MapaBalance - DistributionBalance * OldMAPAPriorPayOff[MapaContract] ) < 0:        
            print "Mapa balance minus the first distribution is less than zero"
            FirstWaterfallBalance = MapaBalance / OldMAPAPriorPayOff[MapaContract]
            SplitPaidToThayer = FirstWaterfallBalance * OldMAPAPriorPayOff[MapaContract]
            SplitPaidToGoji = FirstWaterfallBalance * (1 - OldMAPAPriorPayOff[MapaContract])
            MapaBalance = 0
        
        #The renewals post repayment are the remainder of the renewals from the first waterfall
            SecondWaterfallBalance = DistributionBalance - FirstWaterfallBalance
            SplitPaidToThayer = SplitPaidToThayer + SecondWaterfallBalance * OldMAPAPriorPayOff[MapaContract]
            SplitPaidToGoji =  SplitPaidToGoji + SecondWaterfallBalance * (1 - OldMAPAPriorPayOff[MapaContract]) - TotalTPMFee
    
        else: #(MapaBalance - DistributionBalance * OldMAPAPriorPayOff[MapaContract]) > 0:
            print "Mapa balance minus the first distribution is greater than zero"
            SplitPaidToThayer = DistributionBalance * OldMAPAPriorPayOff[MapaContract]
            SplitPaidToGoji = DistributionBalance * (1 - OldMAPAPriorPayOff[MapaContract]) - TotalTPMFee
            MapaBalance += -SplitPaidToThayer
            FirstWaterfallBalance = DistributionBalance
            SecondWaterfallBalance = 0
    
    else:
        
           SplitPaidToGoji = 0
           SplitPaidToThayer = 0
           DistributionBalance = 0
           WaterfallRenewalsAccount += TotalRenewalCommissions
           ClawbackReserve += TotalRenewalCancellations    
           print "Split Paid To Goji was not Greater Than Zero", WaterfallRenewalsAccount, ClawbackReserve, 
        #MonthlyCashProgress.append([EvaluationDate,'WaterfallThresholdNotExceeded', MapaContract])
           FirstWaterfallBalance = 0
           SecondWaterfallBalance = 0
          # MonthlyCashProgress.append([EvaluationDate,'WaterfallThresholdNotPassed', MapaContract])
    
    '''
        #returns renewals savings and cancellation accounts based on the waterfall amounts
    if SplitPaidToGoji > 0:
        WaterfallRenewalsAccount = 0
        ClawbackReserve = 0
        print "Split Paid To Goji was Greater Than Zero", WaterfallRenewalsAccount, ClawbackReserve
    else:
        WaterfallRenewalsAccount += TotalRenewalCommissions
        ClawbackReserve += TotalRenewalCancellations    
        print "Split Paid To Goji was not Greater Than Zero", WaterfallRenewalsAccount, ClawbackReserve, 
        #MonthlyCashProgress.append([EvaluationDate,'WaterfallThresholdNotExceeded', MapaContract])
        FirstWaterfallBalance = 0
        SecondWaterfallBalance = 0
        '''
        
    return {'WaterfallRenewalsAccount':WaterfallRenewalsAccount, 'ClawbackReserve':ClawbackReserve, 'TotalTPMFee': TPMFeePaid, 'NetServiceFee':ServiceFeePaid, 'MAPABalance': MapaBalance,'DistributionBalance': DistributionBalance,'SplitPaidToGoji': SplitPaidToGoji, 'SplitPaidToThayer': SplitPaidToThayer, 'FirstWaterfallBalance': FirstWaterfallBalance,'SecondWaterfallBalance': SecondWaterfallBalance, 'WaterfallThreshold': WaterfallThreshold}

#waterfall method for the new mapa  
def getNewMapa(EvaluationDate, TotalRenewalCommissions, WaterfallRenewalsAccount, MapaBalance, NetServiceFee, TotalTPMFee, TotalRenewalCancellations, CumulativeInvestorPayments, ClawbackReserve):
    
    SplitPaidToGoji = 0
    SplitPaidToThayer = 0
    WaterfallThreshold = 15000
    TPMFeePaid = 0
    ServiceFeePaid = 0
    NotionalMapaBalance = MapaBalance - CumulativeInvestorPayments
    
    if NotionalMapaBalance <= 0:
        NotionalMapaBalance = 0
    else:
        NotionalMapaBalance -= SplitPaidToThayer
    
    
    #MonthlyCashProgress.append([EvaluationDate,"Started New MAPA Cohort", MapaContract])
    #Determines whether or not the threshold for gross renewals is met for the waterfall
    if (EvaluationDate >= MapaGrossRenewalsFirstThreshold) and (MapaContract != "MAPA-PRG"):
            WaterfallThreshold = MapaThreshold['MapaGrossRenewalsSecondThreshold']
    
    else:
            WaterfallThreshold = MapaThreshold['MapaGrossRenewalsFirstThreshold']
    
    print "WaterfallRenewalsAccount is: ", WaterfallRenewalsAccount
    print "TotalRenewalCommissions is: ", TotalRenewalCommissions
    print "WaterfallThreshold is: ", WaterfallThreshold
    #########
    #MonthlyCashProgress.append([EvaluationDate,WaterfallThreshold, MapaContract])    
    
    # tests to verify that the gross renewals exceed the thresholding requirements
    if (TotalRenewalCommissions + WaterfallRenewalsAccount) >= WaterfallThreshold:
        #Funds available for distribution to SPV investors
        DistributionBalance = TotalRenewalCommissions - NetServiceFee + TotalRenewalCancellations - TotalTPMFee + WaterfallRenewalsAccount
        WaterfallRenewalsAccount = 0
        ClawbackReserve = 0
        TPMFeePaid = TotalTPMFee
        ServiceFeePaid = NetServiceFee
        TotalTPMFee = 0 
        NetServiceFee = 0
                 
       #TESTS TIER 1
        #Waterfall tests the position of the current waterfall up through the current time to determine the curent split rate for the first waterfall
        LowerThreshold = Thresholds[0]
        UpperThreshold = Thresholds[1]
        
        if (CumulativeInvestorPayments / MapaBalance) < Thresholds[len(Thresholds) - 1]:        
            counter = 0
            while (LowerThreshold < CumulativeInvestorPayments  <= UpperThreshold) is False and counter < (len(Thresholds) - 1):
                   LowerThreshold = Thresholds[counter]
                   UpperThreshold = Thresholds[counter + 1]
                   counter += 1
                   
        else:
            LowerThreshold = Thresholds[len(Thresholds) - 1] 
            
        #Determines the waterfall process based on whether subtracting the next waterfall 
        if (DistributionBalance * NewMapaThayerSplitRate[LowerThreshold] + CumulativeInvestorPayments) / MapaBalance <= LowerThreshold:
            SplitPaidToThayer = DistributionBalance * NewMapaThayerSplitRate[LowerThreshold]
            CumulativeInvestorPayments += SplitPaidToThayer
            SplitPaidToGoji = DistributionBalance * ( 1- NewMapaThayerSplitRate[LowerThreshold])
            FirstWaterfallBalance =  DistributionBalance           
            SecondWaterfallBalance = 0
            
            if NotionalMapaBalance <= 0:
                NotionalMapaBalance = 0
            else:
                NotionalMapaBalance -= SplitPaidToThayer

            
        #Determines the splits based on a split waterfall during a month which crosses a repayment threshold
        else:
            FirstWaterfallBalance = DistributionBalance * NewMapaThayerSplitRate[LowerThreshold]
            SecondWaterfallBalance = DistributionBalance - FirstWaterfallBalance
            
            SplitPaidToThayer = FirstWaterfallBalance * NewMapaThayerSplitRate[LowerThreshold] + SecondWaterfallBalance * NewMapaThayerSplitRate[UpperThreshold]          
            SplitPaidToGoji = FirstWaterfallBalance *(1- NewMapaThayerSplitRate[LowerThreshold]) + SecondWaterfallBalance *(1- NewMapaThayerSplitRate[UpperThreshold])
        
            if NotionalMapaBalance <= 0:
                NotionalMapaBalance = 0
            else:
                NotionalMapaBalance -= SplitPaidToThayer
        
        #########
        #MonthlyCashProgress.append([EvaluationDate,DistributionBalance, MapaContract])
    else:            
        SplitPaidToGoji = 0
        SplitPaidToThayer = 0
        DistributionBalance = 0
        WaterfallRenewalsAccount += TotalRenewalCommissions
        ClawbackReserve += TotalRenewalCancellations
        FirstWaterfallBalance = 0
        SecondWaterfallBalance = 0     
        
           ########
           #MonthlyCashProgress.append([EvaluationDate,'WaterfallThresholdNotPassed', MapaContract])    
    '''
    #returns renewals savings and cancellation accounts based on the waterfall amounts
    if SplitPaidToGoji > 0:
        WaterfallRenewalsAccount = 0
        ClawbackReserve = 0
        print "Split Paid To Goji was Greater Than Zero", WaterfallRenewalsAccount, ClawbackReserve
    else:
        WaterfallRenewalsAccount += TotalRenewalCommissions
        ClawbackReserve += TotalRenewalCancellations
        print "Split Paid To Goji was not Greater Than Zero", WaterfallRenewalsAccount, ClawbackReserve, 
        #MonthlyCashProgress.append([EvaluationDate,'WaterfallThresholdNotExceeded', MapaContract])
        FirstWaterfallBalance = 0
        SecondWaterfallBalance = 0        
        '''
    #print 'MAPABalance':MapaBalance, 'TotalTPMFee': TotalTPMFee, 'NetServiceFee':NetServiceFee, 'DistributionBalance':DistributionBalance, 'SplitPaidToGoji':SplitPaidToGoji,'SplitPaidToThayer': SplitPaidToThayer,'CumulativeInvestorPayments': CumulativeInvestorPayments,'WaterfallRenewalsAccount': WaterfallRenewalsAccount, 'ClawbackReserve':ClawbackReserve, 'FirstWaterfallBalance': FirstWaterfallBalance, 'SecondWaterfallBalance':SecondWaterfallBalance
    return {'MAPABalance':MapaBalance, 'NotionalMapaBalance': NotionalMapaBalance,  'TotalTPMFee': TPMFeePaid, 'NetServiceFee':ServiceFeePaid, 'DistributionBalance':DistributionBalance, 'SplitPaidToGoji':SplitPaidToGoji,'SplitPaidToThayer': SplitPaidToThayer,'CumulativeInvestorPayments': CumulativeInvestorPayments,'WaterfallRenewalsAccount': WaterfallRenewalsAccount, 'ClawbackReserve':ClawbackReserve, 'FirstWaterfallBalance': FirstWaterfallBalance, 'SecondWaterfallBalance':SecondWaterfallBalance, 'WaterfallThreshold': WaterfallThreshold}


#PRODUCES THE POLICY LEVEL CALCULATIONS THAT ARE TO BE BUCKETED ON STATEMENTS AND WATERFALLED
#For every month in the analysis window
for i in range(AnalysisWindowYears * 12):

    MonthTuple = calendar.monthrange(year,month)
    EndedDate = datetime.datetime(year,month,MonthTuple[1]) 
    StartedDate = datetime.datetime(year, month, day)

    if month == 12:
        month = 1
        year += 1
        
    else:
        month += 1
    
    print StartedDate, EndedDate 
    
    #For every renewable policy in the backbook generate the expected tail commissions based on the policy's retention, premium, and contract commission rates
    for index, row in book.iterrows():

        #if row['Cohort'] == cohort:
        PolicyArea = getPolicyArea(StartedDate, EndedDate, row['Alpha'], row['Beta'], row['Term'], row['OriginalEffectiveYear'],row['OriginalEffectiveMonth'], row['OriginalEffectiveDay'])
        Renewals = PolicyArea['PolicyArea'] * row['Premium'] * row['RenewalCommissionPercentage']    
        RenewalCancellations = PolicyArea['PolicyCancelArea'] * row['Premium'] * row['RenewalCommissionPercentage']        
        #PolicySeries.append([StartDate,EndDate, row['PolicyNumber'], row['Product'], row['Carrier'], row['Type'], row['Tier'], row['Term'], row['Premium'], row['RenewalCommissionPercentage'] , row['OriginalEffectiveDate'], row['TerminationDate'], row['Cohort'] ,row['ContractType'], row['Alpha'], row['Beta'] ,row['PriorToRepayment'], row['PostRepayment'], row['CohortAcquisitionDate'], Renewals, RenewalCancellations, PolicyArea['PolicyArea'], PolicyArea['PolicyCancelArea'], row['MAPAPaybackRateCategory']])
        if Renewals > 0:        
            PolicySeries.append([PolicyArea['StartedDate'], PolicyArea['EndedDate'], row['PolicyNumber'], row['Product'], row['Carrier'], row['Type'], row['Tier'], row['Term'], row['Premium'], row['RenewalCommissionPercentage'] , row['OriginalEffectiveDate'], row['TerminationDate'], row['Cohort'], row['HistoricalCohort'] ,row['ContractType'], row['Alpha'], row['Beta'] ,row['PriorToRepayment'], row['PostRepayment'], row['CohortAcquisitionDate'], Renewals, RenewalCancellations, PolicyArea['PolicyArea'], PolicyArea['PolicyCancelArea'], row['MAPAPaybackRateCategory']])

PolicyDataFrame = pd.DataFrame(PolicySeries)
EndTimePolicySeries = time.time()

TotalPolicySeriesTimeHours = (EndTimePolicySeries - StartTimePolicySeries)/ 3600




print "Start Writing To The Database"

#Writes all of the policy level cals to the database
WritingStart = time.time()

DatabaseA = 'analytics'
#S = a.dbCreds('write', DatabaseA)

UserA = #S['user']
PasswordA =  #S['password']
HostA = 


cnx = mysql.connector.connect(    user = UserA, 
                                  password = PasswordA, 
                                  host = HostA , 
                                  database = DatabaseA)

cnx.sql_mode = [SQLMode.NO_ZERO_IN_DATE]
cursor = cnx.cursor()    

sqlClearPolicyData = 'DELETE from analytics.PIFModelPolicySeries'
cursor.execute(sqlClearPolicyData)

PolicyDataFrame = pd.DataFrame() 
        
piece = 50000
startFrame = 0
endFrame = piece
Frames = int(math.ceil(len(PolicySeries) / endFrame)) + 1

for i in range(Frames):
    DatatoAppend = PolicySeries[startFrame:endFrame]
    startFrame += piece
    endFrame += piece
    PolicyDataFrame = pd.DataFrame(DatatoAppend,columns = FieldList) 
    PolicyDataFrame.to_sql(con=cnx, name='PIFModelPolicySeries', schema='analytics', if_exists='append', flavor='mysql',index=False, chunksize = 10000)
    print ("Currently writing records to the database ", startFrame)
cursor.close()
cnx.close()

WritingEnd = time.time()
TotalWritingTimeHours = (WritingEnd - WritingStart) / 3600

print "Finished Writing RecordsTo Database"



'''
AreaSize = 0
BookValue = 0
for i in range(len(PolicySeries)):
    BookValue+= PolicySeries[i][PolicySeriesDict['Renewals']]
    AreaSize+= PolicySeries[i][PolicySeriesDict['PolicyArea']]
print BookValue, AreaSize




for i in range(len(PolicySeries)):
    if PolicySeries[i][PolicySeriesDict['Renewals']] > 0:
        print PolicySeries[i][PolicySeriesDict['Renewals']]




for i in range(len(StlBalances)):

    if month == 12:
        month = 1
        year += 1
        
    else:
        month += 1
    
    MonthTuple = calendar.monthrange(year,month)
    EndDate = datetime.datetime(year,month,MonthTuple[1]) 
    StartDate = datetime.datetime(year, month, day)
    
    #For every renewable policy in the backbook generate the expected tail commissions based on the policy's retention, premium, and contract commission rates
    for index, row in book.iterrows():

'''



MonthlyCashFlows = []
MapaWaterfallsList = []
MapaWaterfallFinish = []
TPMFeeList = []
CuServiceFeeList = []

#For every Cohort compute the waterfalls based on the policy series dataset through time
for cohort in Cohorts:

    CashFlowSeries = []
    CumulativeInvestorPayments = 0
    FirstWaterfallBalance = 0
    SecondWaterfallBalance = 0
    CumulativeMOIC = 0 
    WaterfallThreshold = 0
    #Total 
    TotalRenewalCommissions = 0
    TotalRenewalCancellations = 0
    NetServiceFee = 0
    TotalRenewalCancelEvents = 0
    
    #Policy area reset
    SixMonthArea = 0
    TwelveMonthArea = 0
    
    #resets the MAPA fees to zero for then next cohort
    TotalCuServiceFees = 0
    TotalTPMFee = 0
    TotalRenewalEvents = 0
    X = MAPACohortMap[MAPACohortMap['Cohort'] == cohort]['MAPAPaybackRateCategory']
    WaterfallPaybackCategory = X.tolist()[0]
    WaterfallRenewalsAccount = 0
    
    if 'MAPA' in WaterfallPaybackCategory:
        
        if str(WaterfallPaybackCategory) in OldMAPA: 
        
            T = CurrentMapaBalances[CurrentMapaBalances['Cohort'] == cohort]
            #T = OriginalCohortBalances[OriginalCohortBalances['Cohort'] == cohort]
            MapaBalance = T['MAPABalance']
            
        elif str(WaterfallPaybackCategory) in NewMAPA: 
        
            T = CurrentMapaBalances[CurrentMapaBalances['Cohort'] == cohort]
            MapaBalance = T['MAPABalance']
        
        MapaBalance = MapaBalance.tolist()
        MapaBalance = MapaBalance[0]
        StartingMapaBalance = MapaBalance
        CashFlowSeries.append(-StartingMapaBalance)
    
    else:
        MapaBalance = 0
    
    MapaWaterfall = []    
    RollingIRR = 0  
    SplitPaidToGoji = 0
    SplitPaidToThayer = 0
    WaterfallRenewalsAccount = 0
    ClawbackReserve = 0
    
    year = StartYear
    month = StartMonth
    day = StartDay
        
    CurrentMonth = datetime.datetime(StartYear,StartMonth,StartDay)
    
        
    #Constructs the cash flows based on the PolicySeries object
    for i in range(AnalysisWindowYears * 12):
    
        MonthTuple = calendar.monthrange(year,month)
        EndDate = datetime.datetime(year,month,MonthTuple[1]) 
        StartDate = datetime.datetime(year, month, day)
        
        #calculates the statement month
        if month == 12:
            StatementMonth = 1
            StatementYear = year + 1
            
        else:
            StatementMonth = month + 1
            StatementYear = year
        
        #calculates the month the waterfalls happen
        if 'MAPA' in WaterfallPaybackCategory:
            if StatementMonth == 12:
                MapaWaterfallMonth = 1
                MapaWaterfallYear = StatementYear + 1
            
            else:
                MapaWaterfallMonth = StatementMonth + 1
                MapaWaterfallYear = StatementYear  
        else:
            MapaWaterfallMonth = StatementMonth
            MapaWaterfallYear = StatementYear
            
                
        #Calculates statement month start and ends
        StatementMonthTuple = calendar.monthrange(StatementYear,StatementMonth)
        StatementStartDate = datetime.datetime(StatementYear, StatementMonth, 1)
        StatementCutoffDate = datetime.datetime(StatementYear, StatementMonth, StatementMonthTuple[1])
        
        
        #Calculates mapa months start and ends
        MapaMonthTuple = calendar.monthrange(MapaWaterfallYear,MapaWaterfallMonth)
        MapaStartDate = datetime.datetime(MapaWaterfallYear, MapaWaterfallMonth, 1)
        MapaCutoffDate = datetime.datetime(MapaWaterfallYear, MapaWaterfallMonth, MapaMonthTuple[1])

        #sets up the year and month for the next calculation
        if month == 12:
            month = 1
            year += 1
            
        else:
            month += 1

        #Loops through the newly constructed policy series objects to construct the waterfalls
        for n in range(len(PolicySeries)):
        
            if PolicySeries[n][PolicySeriesDict['Cohort']] == cohort:
                
                if StartDate == PolicySeries[n][PolicySeriesDict['StartDate']] and EndDate == PolicySeries[n][PolicySeriesDict['EndDate']]:
                    
                    TotalRenewalCommissions += PolicySeries[n][PolicySeriesDict['Renewals']]
                    TotalRenewalCancellations += PolicySeries[n][PolicySeriesDict['RenewalCancellations']]    
                    
                    TotalRenewalEvents += PolicySeries[n][PolicySeriesDict['PolicyArea']]
                    TotalRenewalCancelEvents += PolicySeries[n][PolicySeriesDict['PolicyCancelArea']]
                    
                    #print 'MAPA', PolicySeries[n][PolicySeriesDict['Cohort']], row['MAPAPaybackRateCategory'], PolicySeries[n][PolicySeriesDict['StartDate']], PolicySeries[n][PolicySeriesDict['CohortAcquisitionDate']], 'MAPA' in row['MAPAPaybackRateCategory'] and PolicySeries[n][PolicySeriesDict['StartDate']] >= PolicySeries[n][PolicySeriesDict['CohortAcquisitionDate']]
                    #time.sleep(.1)
                    #determines whether or not to accrue a cu service fee or TPM fee based on the cohort acquisition date
                    if 'MAPA' in PolicySeries[n][PolicySeriesDict['MAPAPaybackRateCategory']] and PolicySeries[n][PolicySeriesDict['StartDate']] >= PolicySeries[n][PolicySeriesDict['CohortAcquisitionDate']] :                      
                        CuServiceFee = getCuServiceFee(PolicySeries[n][PolicySeriesDict['PolicyArea']],PolicySeries[n][PolicySeriesDict['Term']], PolicySeries[n][PolicySeriesDict['Premium']], PolicySeries[n][PolicySeriesDict['RenewalCommissionPercentage']])
                        NetServiceFee += CuServiceFee
                        
                        TPMFee = getTpmFee(PolicySeries[n][PolicySeriesDict['EndDate']], PolicySeries[n][PolicySeriesDict['PolicyArea']],PolicySeries[n][PolicySeriesDict['Term']])
                        TotalTPMFee += TPMFee
                        MapaContract = PolicySeries[n][PolicySeriesDict['MAPAPaybackRateCategory']]
                        #print "Crunched a service fee and TPM fee of", TPMFee, CuServiceFee
                        #time.sleep(1)
                    #else:

                        #print "Not crunching a MapaPolicy"
                    
                    TPMFeeList.append(TotalTPMFee)
                    CuServiceFeeList.append(NetServiceFee)

        #Paths the MAPA waterfalls through the waterfall and splits process
        if 'MAPA' in WaterfallPaybackCategory:
            
            #print WaterfallPaybackCategory, OldMAPA, NewMAPA
            #time.sleep(3)
            MapaWaterfallsList.append([cohort, "Successfully Started Waterfall"])
            
            
            #print str(WaterfallPaybackCategory), OldMAPA, str(WaterfallPaybackCategory) in OldMAPA
           # time.sleep(3)
            #Computes the cash flow waterfalls to the SPV investors structure
            if str(WaterfallPaybackCategory) in OldMAPA:
                MapaWaterfallsList.append([cohort, "Successfully Started Old MAPA Waterfall"])
                MapaWaterfall = getOldMapa(EndDate, TotalRenewalCommissions, WaterfallRenewalsAccount, MapaBalance, NetServiceFee, TotalTPMFee, WaterfallPaybackCategory, TotalRenewalCancellations, ClawbackReserve)
                CumulativeInvestorPayments += MapaWaterfall['SplitPaidToThayer']
                CumulativeMOIC = round(CumulativeInvestorPayments / StartingMapaBalance,2)
                MapaBalance = MapaWaterfall['MAPABalance']
                FirstWaterfallBalance = MapaWaterfall['FirstWaterfallBalance']
                SecondWaterfallBalance = MapaWaterfall['SecondWaterfallBalance']
                SplitPaidToGoji = MapaWaterfall['SplitPaidToGoji']
                SplitPaidToThayer = MapaWaterfall['SplitPaidToThayer']
                WaterfallRenewalsAccount = MapaWaterfall['WaterfallRenewalsAccount']
                ClawbackReserve = MapaWaterfall['ClawbackReserve']
                MapaWaterfallsList.append([cohort, "Successfully Finished Waterfall"]) 
               # WaterfallThreshold = MapaWaterfall['WaterfallThreshold']
                TotalTPMFee = MapaWaterfall['TotalTPMFee']
                NetServiceFee =  MapaWaterfall['NetServiceFee']
                
                MonthlyCashFlows.append([cohort, StartDate, EndDate, StatementStartDate,StatementCutoffDate,MapaStartDate, MapaCutoffDate,TotalRenewalCommissions,TotalRenewalCancellations,MapaBalance,FirstWaterfallBalance, SecondWaterfallBalance,SplitPaidToGoji, SplitPaidToThayer, TotalTPMFee, NetServiceFee, WaterfallRenewalsAccount,ClawbackReserve,TotalRenewalEvents,TotalRenewalCancelEvents,0, 0])        
              
                print "Crunched an old Mapa cohort having", MapaBalance, FirstWaterfallBalance,SecondWaterfallBalance, SplitPaidToGoji, SplitPaidToThayer,WaterfallRenewalsAccount,ClawbackReserve,
               # time.sleep(3)
            elif str(WaterfallPaybackCategory) in NewMAPA:
                MapaWaterfallsList.append([cohort, "Successfully Started New MAPA Waterfall"])
                MapaWaterfall = getNewMapa(EndDate,TotalRenewalCommissions, WaterfallRenewalsAccount, MapaBalance, NetServiceFee, TotalTPMFee, TotalRenewalCancellations, CumulativeInvestorPayments, ClawbackReserve)
                CumulativeInvestorPayments = MapaWaterfall['CumulativeInvestorPayments']             
                CumulativeMOIC = round(CumulativeInvestorPayments / StartingMapaBalance ,2)
                MapaBalance = MapaWaterfall['MAPABalance']
                NotionalMapaBalance = MapaWaterfall['NotionalMapaBalance']
                FirstWaterfallBalance = MapaWaterfall['FirstWaterfallBalance']
                SecondWaterfallBalance = MapaWaterfall['SecondWaterfallBalance']
                SplitPaidToGoji = MapaWaterfall['SplitPaidToGoji']
                SplitPaidToThayer = MapaWaterfall['SplitPaidToThayer']
                WaterfallRenewalsAccount = MapaWaterfall['WaterfallRenewalsAccount']
                ClawbackReserve = MapaWaterfall['ClawbackReserve']
                WaterfallThreshold = MapaWaterfall['WaterfallThreshold']
                TotalTPMFee = MapaWaterfall['TotalTPMFee']
                NetServiceFee =  MapaWaterfall['NetServiceFee']
                #MapaWaterfallsList.append([cohort, "Successfully Finished Waterfall"]) 
                MonthlyCashFlows.append([cohort, StartDate, EndDate, StatementStartDate,StatementCutoffDate,MapaStartDate, MapaCutoffDate,TotalRenewalCommissions,TotalRenewalCancellations,NotionalMapaBalance,FirstWaterfallBalance, SecondWaterfallBalance,SplitPaidToGoji, SplitPaidToThayer, TotalTPMFee, NetServiceFee, WaterfallRenewalsAccount,ClawbackReserve,TotalRenewalEvents,TotalRenewalCancelEvents,0, 0])        
                print "Crunched a new MAPA cohort having", MapaBalance, FirstWaterfallBalance,SecondWaterfallBalance, SplitPaidToGoji, SplitPaidToThayer, WaterfallRenewalsAccount,ClawbackReserve,MapaWaterfallsList
               # time.sleep(3)
            #Computes the rolling IRR for that Cohort
            for i in range(len(MonthlyCashFlows)):
                if cohort == MonthlyCashFlows[0]:
                    CashFlowSeries.append(SplitPaidToThayer)
                    RollingIRR = round(np.irr(CashFlowSeries),5)        
            
        else:
            print cohort, "Didn't Do A Waterfall"
            MapaWaterfallsList.append([cohort, "Didn't Do A Waterfall"])
            TotalTPMFee = 0
            NetServiceFee = 0
            MapaBalance = 0
            FirstWaterfallBalance = 0
            SecondWaterfallBalance = 0
            SplitPaidToGoji = 0
            SplitPaidToThayer = 0
            WaterfallRenewalsAccount = 0
            ClawbackReserve = 0
            CumulativeMOIC = 0
        
        ###INSERT EQUIVALENT ANNUAL COUPON
        
            MonthlyCashFlows.append([cohort, StartDate, EndDate, StatementStartDate,StatementCutoffDate,MapaStartDate, MapaCutoffDate,TotalRenewalCommissions,TotalRenewalCancellations,MapaBalance,FirstWaterfallBalance, SecondWaterfallBalance,SplitPaidToGoji, SplitPaidToThayer, TotalTPMFee, NetServiceFee, WaterfallRenewalsAccount,ClawbackReserve,TotalRenewalEvents,TotalRenewalCancelEvents,0, 0])        
        #print ("Appended monthly calculation for", cohort,StartDate, EndDate, StatementStartDate, StatementCutoffDate,MapaStartDate,MapaCutoffDate, TotalRenewalCommissions, TotalRenewalCancellations, MapaBalance, FirstWaterfallBalance, SecondWaterfallBalance, SplitPaidToGoji, SplitPaidToThayer, TotalTPMFee, NetServiceFee,  WaterfallRenewalsAccount,ClawbackReserve,TotalRenewalEvents,TotalRenewalCancelEvents,CumulativeMOIC, RollingIRR)
        
        TotalRenewalCommissions = 0
        TotalRenewalCancellations = 0
        TotalRenewalCancelEvents = 0
        TotalRenewalEvents = 0
        FirstWaterfallBalance = 0
        SecondWaterfallBalance = 0
        TotalRenewalEvents = 0 
        TotalRenewalCancelEvents = 0
        RollingIRR = 0        
        SplitPaidToGoji = 0
        SplitPaidToThayer = 0
        FirstWaterfallBalance = 0
        SecondWaterfallBalance = 0
        
        if SplitPaidToGoji > 0:
           print cohort,  "Split Paid To Goji was More Than Zero", "SplitPaidToGoji was: ", SplitPaidToGoji
           #WaterfallRenewalsAccount = 0
           #ClawbackReserve = 0
           NetServiceFee = 0
           TotalTPMFee = 0
           
        else:
            print cohort,  "Split Paid To Goji was not More Than Zero", "SplitPaidToGoji was: ", SplitPaidToGoji
        
        print "Total Waterfall Renewals Account is ", WaterfallRenewalsAccount
        print "Total Renewal Commissions Are", WaterfallRenewalsAccount
        print "WaterfallThreshold is: ", WaterfallThreshold
        print "TotalNetServiceFee is ", NetServiceFee
        print "TotalTPMFee is", TotalTPMFee
        print "Clawback reserve is", ClawbackReserve
EndTime = time.time()

TotalRunTimeHours = (EndTime - StartTime) / 3600

MonthlyCashFlowsTimeHours = (EndTimePolicySeries - EndTime)/ 3600

MonthlyCashFlowWritingStart = time.time()

CashFieldList = ['cohort', 'StartDate', 'EndDate', 'StatementStartDate','StatementCutoffDate','MapaStartDate', 'MapaCutoffDate','TotalRenewalCommissions','TotalRenewalCancellations','MapaBalance','FirstWaterfallBalance', 'SecondWaterfallBalance','SplitPaidToGoji', 'SplitPaidToThayer', 'TotalTPMFee', 'NetServiceFee', 'WaterfallRenewalsAccount','ClawbackReserve','TotalRenewalEvents','TotalRenewalCancelEvents', 'CumulativeMOIC', 'RollingIRR']



DatabaseA = 'analytics'
#S = a.dbCreds('write', DatabaseA)

UserA = #S['user']
PasswordA =  #S['password']
HostA = 


cnx = mysql.connector.connect(    user = UserA, 
                                  password = PasswordA, 
                                  host = HostA , 
                                  database = DatabaseA)

cursor = cnx.cursor()    
WaterfallDataFrame = pd.DataFrame() 
WaterfallDataFrame = pd.DataFrame(MonthlyCashFlows,columns = CashFieldList) 
WaterfallDataFrame.to_sql(con=cnx, name='PIFModelWaterfalls', schema='analytics', if_exists='append', flavor='mysql',index=False, chunksize = 10000)
cursor.close()
cnx.close()


MonthlyCashFlowWritingEnd = time.time()

MonthlyCashFlowTotalWritingTimeHours = (MonthlyCashFlowWritingEnd - MonthlyCashFlowWritingStart) / 3600

TotalProgramTimeHours = MonthlyCashFlowTotalWritingTimeHours + TotalRunTimeHours


print "The Total Calculation Time Was: " , TotalRunTimeHours
print "The Total Policy Series Calculation Time Was: " , TotalPolicySeriesTimeHours
print "The Total Monthly Cash Flows Run Time Was: " , MonthlyCashFlowsTimeHours
print "The Total Time To Write the Policy Series Was: " , TotalWritingTimeHours
print "The Total Time To Write the Monthly Cash Flows Series Was: " , MonthlyCashFlowTotalWritingTimeHours
print "The Total Program Time Was: ", TotalProgramTimeHours

MonitoringList = ['TotalRunTimeHours','TotalPolicySeriesTimeHours','MonthlyCashFlowsTimeHours','TotalWritingTimeHours','MonthlyCashFlowTotalWritingTimeHours', 'TotalProgramTimeHours']
print "made it past list"
MonitoringData = []
MonitoringData.append([TotalRunTimeHours,TotalPolicySeriesTimeHours,MonthlyCashFlowsTimeHours,TotalWritingTimeHours,MonthlyCashFlowTotalWritingTimeHours, TotalProgramTimeHours ])

print "made it past Monitoring data"

print "made it past monitoring list"

DatabaseA = 'analytics'
#S = a.dbCreds('write', DatabaseA)

UserA =
PasswordA = 
HostA = 

cnx = mysql.connector.connect(    user = UserA, 
                                  password = PasswordA, 
                                  host = HostA , 
                                  database = DatabaseA)
cnx.sql_mode = [SQLMode.NO_ZERO_IN_DATE]
cursor = cnx.cursor()    
print "made it past opening the connection"
MonitoringDataFrame = pd.DataFrame()
MonitoringDataFrame = pd.DataFrame(MonitoringData,columns = MonitoringList) 
print "made the dataframe"
MonitoringDataFrame.to_sql(con=cnx, name='PIFModelMonitoring', schema='analytics', if_exists='append', flavor='mysql',index=False, chunksize = 10000)
print "wrote the dataframe"
cursor.close()
cnx.close()
print "closed the connection"
