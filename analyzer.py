import ROOT
import RA2b
import numpy
import math
import os


class analyzer(object):
    """ Main analysis code for my thesis work.

    Analyze data using machine learning discriminant output.
    Machine learning code is found in classifier class.

    Use analyzer to predict backgrounds: Z, top+W, and QCD.
    
    Make data cards for Higgs combine tool for SUSY models.

    """
    def __init__(self, doSim=False):
        """If running closure test on simulation, use doSim=True.
        """

        ## run on data or simulation
        self.doData = not doSim

        ## define cuts and binning parameters
        self.mhtCut = True
        self.Bins = 50
        self.nSigBins = 25
        self.distRange = [0.,1.]
        self.binning = [x*0.02+0.01 for x in range(0, 50)]
        self.varBinning = {
            'Sig': [0.,0.02,0.04,0.06,0.08,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0],
            'TopW': [0.,0.02,0.04,0.06,0.08,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0],
            'QCD': [0.,0.2,0.4,0.5,0.6,0.7,0.8,0.9,0.92,0.94,0.96,0.98,1.0],
        }

        ## BDT background shapes
        self.BDTshapes = ['QCD','TopW', 'Sig']
        self.colorDict = {'QCD': 8, 'Zinv': 2, 'TopW': 4}

        ## prefix for discriminator output
        self.varPrefix = 'BDTG_NTrees2000_MaxDepth4_MNS2p_'

        ## define cuts different than baseline RA2b
        self.dPhiCut = 'none'
        self.htCut = ROOT.TCut('HT>800')
        self.hdpCut = ROOT.TCut('DeltaPhi1>0.1&&DeltaPhi2>0.1&&'
                                'DeltaPhi3>0.1&&DeltaPhi4>0.1')

        self.ldpCut = ROOT.TCut('DeltaPhi1<0.1||DeltaPhi2<0.1||'
                                'DeltaPhi3<0.1||DeltaPhi3<0.1')
        self.hpfCaloCut = ROOT.TCut('MET/CaloMET>0.9') 
        self.lpfCaloCut = ROOT.TCut('MET/CaloMET<0.9')

        ## dictionary for prediction to smooth
        ## default no smoothing anywhere
        ## likely not needed if we use LDP control sample for QCD
        self.doSmoothDict = {'QCD': False, 'TopW': False}


        ##
        ## Begin variables to compute
        ##

        ## Normalization dictionary
        # format: Norm[Bkg] = (CV,upsyst,dnsyst)
        self.Norm = {}

        # Signal region data
        self.dataHist = ROOT.TH1D()
 
        # Signal monte carlo
        self.signalHist = ROOT.TH1D()

        ## Z->ll/gamma Double ratio 
        ## format:
        self.Zdr = None

        ## R(Z/gamma) single ratio dictionary
        ## format: rzg[BDTshape] = ROOT.TH1D()
        self.rzg = {}

        ## overall Z prediction dictionary
        ## format: zPredDict[BDTshape] = ROOT.TH1D()
        self.zPredDict = {}

        ## Z prediction systematic 
        ## uncertainty histograms
        ## format: zSystDict[SystName] = systHistList
        self.zSystDict = {}

        ## top/W and QCD prediction dictionary
        ## format:
        self.predDict = {}

        ## HDP/LDP correction histograms
        self.qcdCorrHist = {}
        
        ## top/W and QCD systematic 
        ## uncertainty histograms
        ## format: 
        self.systDict = {}

        ## Signal systematics histogram
        ## format: 
        self.sigSystDict = {}

        ## Storing plots in dictionary saving time 
        ## later without having to remake plots
        self.canvasDict = {}

    def getZdr(self):

        ## blinding cuts
        bCut = (self.distRange[1]-self.distRange[0]-
                (self.nSigBins)/float(self.Bins))
        blindCut = ROOT.TCut(self.varPrefix+'Sig<'+str(bCut))

        ## double ratio computation takes
        ## a list of tgraphs
        graphs = []
        for BDTshape in self.BDTshapes:

            ## keep Sig reigion blind
            if BDTshape != 'Sig':
                cuts = self.htCut+self.hdpCut+self.hpfCaloCut+blindCut
            else:
                cuts = self.htCut+self.hdpCut+self.hpfCaloCut
    
            ## get dr graphs
            graphs.append(
                RA2b.getDoubleRatioGraph(self.varPrefix+BDTshape,
                                         applyPuWeight=True,
                                         extraCuts=cuts,
                                         dphiCut=self.dPhiCut))
    
        ## make plots viewable after runtime
        for graph in graphs:
            ROOT.SetOwnership(graph,0)

        ## Z/gamma double ratio 
        self.Zdr = RA2b.getDoubleRatioPlot(graphs)

    def getRZG(self):

        ## blinding cuts
        bCut = (self.distRange[1]-self.distRange[0]-
                (self.nSigBins)/float(self.Bins))
        blindCut = ROOT.TCut(self.varPrefix+'Sig<'+str(bCut))

        for BDTshape in self.BDTshapes:
            
            ## if not signal, apply blinding criteria
            cuts = self.hdpCut+self.hpfCaloCut+self.htCut
            if(BDTshape != 'Sig'):
                cuts+=blindCut
                
            ## photon MC
            gjets = RA2b.getDist('gjetsIDP',self.varPrefix+BDTshape,
                                 distRange=self.distRange,
                                 nBins=self.Bins,
                                 extraCuts=str(cuts),
                                 applyEffs=True, applyPuWeight=True)

            ## zinv MC
            zinv = RA2b.getDist('zinvIDP',self.varPrefix+BDTshape,
                                distRange=self.distRange,
                                nBins=self.Bins,
                                extraCuts=str(cuts),applyPuWeight=True)

            ## ratio becomes R(Z/gamma)
            zinv.Divide(gjets)

            rzg[BDTshape] = zinv

    def getZpred(self):

        ## blinding cuts
        bCut = (self.distRange[1]-self.distRange[0]-
                (self.nSigBins)/float(self.Bins))
        blindCut = ROOT.TCut(self.varPrefix+'Sig<'+str(bCut))

        ## loop over discriminant dists
        for BDTshape in BDTshapes:
                
            ## if not signal, apply blinding criteria
            cuts = self.hdpCut+self.hpfCaloCut+self.htCut
            if(BDTshape != 'Sig'):
                cuts+=blindCut
                
            ## photon data
            photon = RA2b.getDist('photonIDP',self.varPrefix+BDTshape,
                                  distRange=self.distRange,
                                  nBins=self.Bins,
                                  extraCuts=str(cuts),applyEffs=True)
        
            ## compute R(Z/gamma) unless already done so
            if(BDTshape not in rzg.keys()):
                getRZG()
              
            ## apply R(Z/gamma)
            photon.Multiply(self.rzg[BDTshape])
    
            ## compute double ratio unless already done so
            if(Zdr == None):
                getZdr()

            ## apply overall DR scaling factor 
            photon.Scale(self.Zdr[0][0])

            zPredDict[BDTshape] = photon

    def getZpredSyst(self):

        ## blinding cuts
        bCut = (self.distRange[1]-self.distRange[0]-
                (self.nSigBins)/float(self.Bins))
        blindCut = ROOT.TCut(self.varPrefix+'Sig<'+str(bCut))

        ## Uncertainties taken as inputs from other studies
        ## these uncertainties will be hard coded for now
        effSyst = 0.05
        btagSyst = 0.005
        trigSyst = 0.0035
        gTrig = 0.01
        fDir = 0.005
        gSF = 0.005
        gPur = 0.01

        ## dictionaries of total upper and 
        ## lower systematic uncertainties
        upSyst = {}
        dnSyst = {}
        for BDTshape in self.BDTshapes:

            ## check if prediction is stored yet
            ## if not, get prediction
            if BDTshape not in zPredDict.keys():
                getZpred()

            ## get cuts beyond baseline
            cuts = self.htCut+self.hdpCut+self.hpfCaloCut
            if(BDTshape!='Sig'):
                cuts += blindCut

            ## Zll dist without purity applied
            ## will use this as denominator to get 
            ## average purity
            zllTot = RA2b.getDist('zeeIDP',
                                  self.varPrefix+BDTshape,
                                  distRange=self.distRange,
                                  nBins=1,extraCuts=str(cuts))
            ## add Zmm to Zee to get total Zll
            zllTot.Add(RA2b.getDist('zmmIDP',self.varPrefix+BDTshape,
                                    distRange=self.distRange,
                                    nBins=1,extraCuts=str(cuts)))

            ## Zll dist weighted by purity error
            purityHist = RA2b.getDist('zeeIDP',
                                      self.varPrefix+BDTshape,
                                      distRange=self.distRange,
                                      nBins=1,extraCuts=str(cuts),
                                      applyEffs=True, doEffError=True)
            ## add Zmm to Zee to get total Zll
            purityHist.Add(RA2b.getDist('zmmIDP',
                                      self.varPrefix+BDTshape,
                                      distRange=self.distRange,
                                      nBins=1,extraCuts=str(cuts),
                                      applyEffs=True, doEffError=True))

            ## divide out hist without applying purity
            ## this will give us the average uncertainty
            purityHist.Divide(zllTot)
            

            ## clone zPredDict to get consistent binning
            ## structure for uncertainty histograms
            drUpHist = self.zPredDict[BDTshape].Clone()
            drDnHist = self.zPredDict[BDTshape].Clone()
            cvUpHist = self.zPredDict[BDTshape].Clone()
            cvDnHist = self.zPredDict[BDTshape].Clone()
            rzgUpHist = self.zPredDict[BDTshape].Clone()
            rzgDnHist = self.zPredDict[BDTshape].Clone()
            gTrigHist = self.zPredDict[BDTshape].Clone()
            fDirHist = self.zPredDict[BDTshape].Clone()
            gSFHist = self.zPredDict[BDTshape].Clone()
            gPurHist = self.zPredDict[BDTshape].Clone()
            TFhist = self.zPredDict[BDTshape].Clone()

            ## NphoZinv key: number of photon events and 
            ## the corresponding total transfer factor
            gStatHist = RA2b.getDist('photonIDP',self.varPrefix+BDTshape,
                                     distRange=self.distRange,nBins=self.Bins,
                                     extraCuts=str(cuts))
            TFhist.Divide(gStatHist)

            for binIter in range(1,self.Bins+1):

                ## Overall normalization uncertainty components:
                ## DR central val + average purity 
                ## + lepton SF + btag SF + trig SF
                cvUpErr = math.sqrt(drCvHist.GetBinContent(binIter)**2
                                    +purityHist.GetBinContent(binIter)**2
                                    +effSyst**2+btagSyst**2+trigSyst**2)
                cvDnErr = math.sqrt(drCvHist.GetBinContent(binIter)**2
                                    +purityHist.GetBinContent(binIter)**2
                                    +effSyst**2+btagSyst**2+trigSyst**2)
                upCvHist.SetBinContent(binIter,cvUpErr)
                dnCvHist.SetBinContent(binIter,cvDnErr)

                ## DR shape uncertainty:
                ## subtract off DR CV error from DR
                ## to avoid double counting
                drUpErr = max(self.Zdr[1][BDTshape][binIter-1][0]
                              -drCvHist.GetBinContent(binIter),0)
                drDnErr = max(self.Zdr[1][BDTshape][binIter-1][1]
                              -drCvHist.GetBinContent(binIter),0)
                drUpHist.SetBinContent(binIter, drUpErr)
                drDnHist.SetBinContent(binIter, drDnErr)

                ## R(Z/gamma) stat error 
                rzgUpHist.SetBinContent(binIter,self.rzg.GetBinError(binIter))
                rzgDnHist.SetBinContent(binIter,self.rzg.GetBinError(binIter))

                ## photon related efficiency errors
                gTrigHist.SetBinContent(binIter,gTrig)
                fDirHist.SetBinContent(binIter,fDir)
                gSFHist.SetBinContent(binIter,gSF)
                gPurHist.SetBinContent(binIter,gPur)

            ## add individual systematic uncertainties to
            ## the dictionary containing the Z systematics
            self.zSystDict['ZinvNorm'] = [upCvHist, dnCvHist]
            self.zSystDict['ZinvDR'] = [drUpHist, drDnHist]
            self.zSystDict['ZinvRZG'] = [rzgUpHist, rzgDnHist]
            self.zSystDict['NphoZinv'] = [photon,TFhist]
            self.zSystDict['ZinvGtrig'] = [gTrigHist]
            self.zSystDict['ZinvFdir'] = [fDirHist]
            self.zSystDict['ZinvGsf'] = [gSFHist]
            self.zSystDict['ZinvGpur'] = [gPurHist]


    def getData(self, nUnblindBins=0):

        ## Blind Cut Default: all signal data blind
        bCut = (self.distRange[1]-self.distRange[0]-
                (self.nSigBins-nUnblindBins)/float(self.Bins))
        blindCut = ROOT.TCut(self.varPrefix+'Sig<'+str(bCut))

        cuts = self.htCut+self.hdpCut+self.hpfCaloCut+blindCut

        self.dataHist = RA2b.getDist('sigIDP',self.varPrefix+'Sig', 
                                     distRange=self.distRange,
                                     nBins=self.Bins,extraCuts=str(cuts))

    def getCorrHistQCD(self):
        
        ## loop over discriminator distributions
        for BDTshape in BDTshapes:
            for binIter in range(1,self.Bins+1):
                x = i*0.02-0.01
                Bin = sum([int(x>Bin) for Bin in qcdbinning])
                qcdTF.SetBinContent(binIter,qcdTF2qcd.GetBinContent(Bin))
                qcdTF.SetBinError(binIter,qcdTF2qcd.GetBinError(Bin))


    def getNormFromFit(self, doCorrQCD=True):

        ## Simulation sample names
        D1 = "topWIDP"
        D2 = "qcdIDP"
        T1 = "topWslmIDP"
        T2 = "topWsleIDP"
        Q = 'qcdIDP'

        ## Data sample names
        if self.doData:
            D1 = 'sig'
            D2 = 'sigLDP'
            T1 = 'slmIDP'
            T2 = 'sleIDP'
            Q = 'sigIDP'
        
        ## define empty lists for appending
        nqcdList = []
        ntopList = []

        ## loop over classifier discriminant shapes
        for BDTshape in self.BDTshapes:
            ## Skip signal for getting bkg normalizations
            if BDTshape == 'Sig':
                continue 
            
            ## Key for dictionaries
            Key = BDTshape+'NormFit'

            ## declare canvas for plotting
            self.canvasDict[Key] = ROOT.TCanvas(BDTshape,BDTshape,0,0,900,600)

            ## Data Hist, high Dphi and high PF/Calo
            cuts = self.htCut+self.hdpCut+self.hpfCaloCut
            hData = RA2b.getDist(D1, self.varPrefix+BDTshape,
                                 distRange=self.distRange, nBins=self.Bins,
                                 extraCuts=str(cuts), applyMHTCut=self.mhtCut)
            
            ## add ldp for data and top/W for Sim
            hData2 = RA2b.getDist(D2, self.varPrefix+BDTshape,
                                  distRange=self.distRange, nBins=self.Bins,
                                  extraCuts=str(cuts), applyMHTCut=self.mhtCut)
            hData.Add(hData2)
            hData.SetTitle('BDToutput')
            
            ## subtract off Zinv measurement
            if(doData and zHistDict!=None):
                hData.Add(zHistDict[BDTshape],-1)

            ## top/W CR
            cuts = self.htCut+self.hdpCut+self.hpfCaloCut
            ## single muon
            hTopW = RA2b.getDist(T1, self.varPrefix+BDTshape,
                                 distRange=self.distRange=[0,1], 
                                 nBins=self.Bins, extraCuts=str(cuts), 
                                 applyMHTCut=self.mhtCut)
            ## single electron
            hTopWsle = RA2b.getDist(T2, self.varPrefix+BDTshape,
                                    distRange=self.distRange=[0,1], 
                                    nBins=self.Bins, extraCuts=str(cuts),
                                    applyMHTCut=self.mhtCut)
            ## single lepton
            hTopW.Add(hTopWsle)


            ## QCD CR
            cuts = self.htCut+self.ldpCut+self.hpfCaloCut
            hQCD = RA2b.getDist(Q, self.varPrefix+BDTshape,
                                distRange=self.distRange=[0,1], nBins=self.Bins,
                                extraCuts=str(cuts), applyMHTCut=self.mhtCut)

            ## Correct the shape for HDL/LDP differences
            if doCorrQCD:
                if BDTshape not in self.qcdCorrHist.keys():
                    self.qcdCorrHist = getCorrHistQCD()
                hQCD.Multiply(self.qcdCorrHist[BDTshape])

            ## range of fit
            fitRange_min = self.distRange[0]
            fitRange_max = self.distRange[1]
            
            ## RooFit objects
            ## define ROO vars
            ndata = hData.Integral()
            varX = ROOT.RooRealVar("varX",
                                   hData.GetXaxis().GetTitle(),
                                   fitRange_min,fitRange_max)
            ntopW  = ROOT.RooRealVar("topW",
                                     "estimated number of events 
                                     in the top+W backgrounds",
                                     ndata*2./3., 0.000001, ndata)
            nqcd  = ROOT.RooRealVar("nqcd",
                                    "estimated number of events 
                                    in the QCD background",
                                    ndata*1./3., 0.000001, ndata)
            ## define ROO data histograms
            data     = ROOT.RooDataHist("data",
                                        "data in signal region" , 
                                        ROOT.RooArgList(varX), hData)
            topW     = ROOT.RooDataHist("topW_shape",
                                        "signal and other MC backgrounds
                                        in signal region", 
                                        ROOT.RooArgList(varX), hTopW)
            qcd      = ROOT.RooDataHist("qcd_shape",
                                        "QCD distribution from control region", 
                                        ROOT.RooArgList(varX), hQCD)
            ## ROO PDFs (template hists from control regions)
            topW_model = ROOT.RooHistPdf("topW_model",
                                         "RooFit template for signal
                                         and other MC backgrounds", 
                                         ROOT.RooArgSet(varX), topW)
            qcd_model = ROOT.RooHistPdf("qcd_model",
                                        "RooFit template for QCD backgrounds", 
                                        ROOT.RooArgSet(varX), qcd)

            # Prepare extended likelihood fits
            topW_shape = ROOT.RooExtendPdf("topW_shape", "topW shape pdf",
                                           topW_model, ntopW)
            qcd_shape = ROOT.RooExtendPdf("qcd_shape","QCD shape pdf",
                                          qcd_model, nqcd)

            # Combine the models
            combModel = ROOT.RooAddPdf("combModel",
                                       "Combined model for top+W and QCD", 
                                       ROOT.RooArgList(topW_shape,qcd_shape))

            # Fit the data from the ROO PDF template combo model
            fitResult = combModel.fitTo(data, 
                                        ROOT.RooFit.Extended(True), 
                                        ROOT.RooFit.Save(True), 
                                        ROOT.RooFit.Minos(True),
                                        ROOT.RooFit.PrintLevel(-1))

            ################
            ## Begin plot fit
            xframe = varX.frame()
            data.plotOn(xframe)
            combModel.plotOn(xframe, ROOT.RooFit.VisualizeError(fitResult))
            data.plotOn(xframe) # plot again so that it is on top of the errors
            combModel.plotOn(xframe, ROOT.RooFit.LineColor(ROOT.kBlack))
            combModel.plotOn(xframe, ROOT.RooFit.Components("topW_shape"), 
                             ROOT.RooFit.LineColor(ROOT.kBlue))
            combModel.plotOn(xframe, ROOT.RooFit.Components("qcd_shape"), 
                             ROOT.RooFit.LineColor(ROOT.kGreen))
            xframe.GetXaxis().SetTitle(BDTshape+'-like BDT Output')
            xframe.GetYaxis().SetTitleOffset(1.35)
            xframe.GetYaxis().SetTitle('Events')
            xframe.SetTitle('')
            xframe.SetMinimum(0)
            xframe.Draw()
            ## End plot fit
            ################

            ## append list of normalization values and errors
            nqcdList.append((nqcd.getVal(),nqcd.getError()))
            ntopWList.append((ntopW.getVal(),ntopW.getError()))

        ## compute average norm with up and down systs
        nqcdAve = sum(nqcdList)/len(nqcdList)
        ntopAve = sum(ntopList)/len(ntopList)

        nqcdUpSyst = (max(nqcdList)[0]-nqcdAve)/nqcdAve
        nqcdDnSyst = (nqcdAve-min(nqcdList)[0])/nqcdAve
        nqcdUpSyst = math.sqrt(nqcdUpSyst**2+(max(nqcdList)[1]/nqcdAve)**2)
        nqcdDnSyst = math.sqrt(nqcdDnSyst**2+(min(nqcdList)[1]/nqcdAve)**2)

        ntopWUpSyst = (max(ntopWList)[0]-ntopWAve)/ntopWAve
        ntopWDnSyst = (ntopWAve-min(ntopWList)[0])/ntopWAve
        ntopWUpSyst = math.sqrt(ntopWUpSyst**2+(max(ntopWList)[1]/ntopWAve)**2)
        ntopWDnSyst = math.sqrt(ntopWDnSyst**2+(min(ntopWList)[1]/ntopWAve)**2)
    
        ## set the norm dict class variable
        self.Norm['QCD'] = (nqcdAve,nqcdUpSyst,nqcdDnSyst)
        self.Norm['TopW'] = (ntopWAve,ntopWUpSyst,ntopWDnSyst)

            
    def aveBins(self, hist, binList):

        ## average over binList range
        ## save average val and error
        valList = []
        errList = []
        for Bin in binList:
            valList.append(hist.GetBinContent(Bin))
            errList.append(hist.GetBinError(Bin))

        val = sum(valList)/len(binList)
        err = sum(errList)/len(binList)

        return (val, err)

    def smooth(self, hist, returnEntries=False):
        """ Smoothing model starts from basic assumption of a 
        monatonically decreasing distribution. This works well
        for QCD, especially if control region lacks sufficient 
        statistics. 

        Note:
             Since changing control region from low MET/CaloMET to 
        low delta phi, smoothing is no longer needed since the low
        delta phi control region has ample statistics.
        """

        ## clone unsmoothed prediction for consistent binning
        histSmooth = hist.Clone()
        histEntries = hist.Clone()

        for binIter in range(1,self.Bins+1):
            ## store bin values
            binVal = hist.GetBinContent(binIter)
            binEntries = hist.GetEntries()*(hist.GetBinContent(binIter)
                                            /hist.Integral())
            
            ## different criterian for first and last bins
            ## in all casses, criterion is met if nearest bins
            ## are consistent with a falling distribution
            if binIter == 1: # first bin
                criterion = (binVal>hist.GetBinContent(binIter+1) 
                             and binVal>hist.GetBinContent(binIter+2))
                binList = [binIter, binIter+1, binIter+2]
            elif binIter == self.Bins: # last bin
                criterion = (binVal<hist.GetBinContent(binIter-1) 
                             and binVal<hist.GetBinContent(binIter-2))
                binList = [binIter-2, binIter-1, binIter]
            else: # all other bins
                criterion = (binVal<hist.GetBinContent(binIter-1) 
                             and binVal>hist.GetBinContent(binIter+1))
                binList = [binIter-1, binIter, binIter+1]

            ## if criterion fails, do smoothing
            ## bin entries becomes sum of entries for
            ## all three bins
            if not(criterion):
                binEntries = 0
                for Bin in binList:
                    binEntries+=hist.GetEntries()*(
                        hist.GetBinContent(Bin)/hist.Integral())

                binVal = aveBins(hist, binList)[0]
                binErr = aveBins(hist, binList)[1]

            histSmooth.SetBinContent(binIter, binVal)
            histEntries.SetBinContent(binIter, binEntries)

        ## return either smoothed hist or smoothed hist entries
        if(returnEntries==True):
            histRet = histEntries.Clone()
        else:
            histRet = histSmooth.Clone()

        return histRet

    def getPred(self, doCorrQCD=True, setWeight=-1):

        ## use setWeight>=0 to get perturbed
        ## distributions for finding shape syst
        systWeight = None
        if(setWeight>-0.5):
            systWeight='systWeight['+str(setWeight)+']'
            
        ## get top/W and QCD predictions
        for BDTshape in BDTshapes:
            
            ## skip 'Sig' since 'Sig' is not a background
            if BDTshape == 'Sig':
                continue
            ## check to see if Norm has been computed yet
            elif BDTshape not in self.Norm.keys():
                getNormFromFit()
            ## QCD prediction fom LDP control region
            elif BDTshape == 'QCD':
                cuts = htCut+ldpCut+hpfCaloCut
                hist = RA2b.getDist('sigIDP',
                                    self.varPrefix+BDTshape,
                                    distRange=self.distRange,
                                    nBins=self.Bins,extraCuts=str(cuts),
                                    extraWeight=systWeight)
                if doCorrQCD:
                    if 'Sig' not in self.qcdCorrHist.keys():
                        self.qcdCorrHist = getCorrHistQCD()
                    hist.Multiply(self.qcdCorrHist['Sig'])
                    
            ## Top/W prediction from SLe and SLm control region
            elif BDTshape == 'TopW':
                cuts = htCut+ldpCut+hpfCaloCut
                hist = RA2b.getDist('slmIDP',
                                    self.varPrefix+BDTshape,
                                    distRange=self.distRange,
                                    nBins=self.Bins,extraCuts=str(cuts),
                                    extraWeight=systWeight)
                hist.Add(RA2b.getDist('sleIDP',
                                      self.varPrefix+BDTshape,
                                      distRange=self.distRange,
                                      nBins=self.Bins,extraCuts=str(cuts),
                                      extraWeight=systWeight))
                
            ## Get unblind normalization and scale to 
            ## normalization gotten from Norm dict
            nCR = 0.
            for i in range(1,self.Bins-self.nSigBins+1):             
                nCR += hist.GetBinContent(i)
            hist.Scale(self.Norm[BDTshape][0]/nCR)
                         
            ## smooth if configured to do so
            if self.doSmoothDict[BDTshape]:
                hist = smooth(hist)
                         
            ## store in prediction dictionary
            self.predDict[BDTshape] = hist
                         

    def getStatHist(self, hist, histEffs=None):

        ## stat hist, clone 
        hStat = hist.Clone()
        
        ## if there is variable efficiency factors applied
        ## then we'll need the transfer histogram to account
        ## for the correct efficiencies
        if(histEffs==None):
            hTF = hist.Clone()
        else:
            hTF = histEffs.Clone()

        ## Set hStat as number of absolute events
        for binInter in range(1,self.Bins+1):
            binEntries = (hist.GetEntries()*
                          hist.GetBinContent(binIter)/
                          hist.Integral())
            hStat.SetBinContent(binIter, binEntries)

        hTF.Divide(hStat)

        return [hStat,hTF]

    def getSystHists(self):

        dataFlag = ''
        if(doData==True):
            dataFlag = 'Data'

        ## 1% QCD purity uncertainty, 3% topW purity uncertainty
        purDict = {'QCD': 0.01, 'TopW': 0.03}

        ## get shape syst file 
        ## from running shape systematic study
        fShapeSyst = ROOT.TFile.Open(
            'hists/shapeSyst'+dataFlag+'.root','read')
        
        ## loop over background predictions
        for bkg in self.predDict.keys():

            ## case change between key and 
            ## file name
            lowerbkg = bkg[:3].lower()+bkg[3:]
            ## get stored shape systs
            upShapeHist = fShapeSyst.Get(lowerbkg+'SystUp'+dataFlag)
            dnShapeHist = fShapeSyst.Get(lowerbkg+'SystDn'+dataFlag)

            ## upper and lower norm uncertainties (float fraction error)
            normUpsyst = self.Norm[bkg][1]
            normDnSyst = self.Norm[bkg][2]

            ## clone predDict to get consistent binning
            ## structure for uncertainty histograms
            upsystSmooth = self.predDict[bkg].Clone()
            dnsystSmooth = self.predDict[bkg].Clone()
            dnsystSmoothTemp = self.predDict[bkg].Clone()
            purSystHist = self.predDict[bkg].Clone()
            upNormHist = self.predDict[bkg].Clone()
            dnNormHist = self.predDict[bkg].Clone()
            TFhist = self.predDict[bkg].Clone()            

            ## combine takes hist with N(data events)
            ## for statistical uncertainty
            statHist = getStatHist(self.predDict[bkg])[0]
            ## transfer factor histogram
            TFhist.Divide(statHist)

            ## if smoothing is being applied then
            ## fill smoothing histograms
            if doSmoothDict[bkg]:
                smoothNorm = {bkg: Norm[bkg]}
                smoothDict = {bkg: False}
                noSmoothPred = getPred(smoothNorm,smoothDict)
                S0 = noSmoothPred[bkg]
                S1 = smooth(S0)
                S2 = smooth(S1)
                SStat = smooth(S0, True)
                statHist = SStat.Clone()
                for sIter in range(1,upsystSmooth.GetNbinsX()+1):
                    s0 = S0.GetBinContent(sIter)
                    s1 = S1.GetBinContent(sIter)
                    s2 = S2.GetBinContent(sIter)
                    upsystSmooth.SetBinContent(sIter,max(s0,s1,s2))
                    dnsystSmoothTemp.SetBinContent(sIter,min(s0,s1,s2))
            upsystSmooth.Add(predHists[bkg],-1)
            dnsystSmooth.Add(dnsystSmoothTemp,-1)
            upsystSmooth.Divide(predHists[bkg])
            dnsystSmooth.Divide(predHists[bkg])

            ## fill purity and Norm syst histograms
            for binIter in range(1,self.Bins+1):
                purSystHist.SetBinContent(binIter, purDict[bkg])
                upNormHist.SetBinContent(binIter,normUpSyst)
                dnNormHist.SetBinContent(binIter,normDnSyst)

            ## store in syst uncertainty dictionary
            self.systHists['NCR'+bkg] = [statHist,TFhist]
            self.systHists[bkg+'Norm'] = [upNormHist, dnNormHist]
            self.systHists[bkg+'Shape'] = [upShapeHist, dnShapeHist]
            self.systHists[bkg+'Pur'] = [purSystHist]
            if self.doSmoothDict[bkg]:
                self.systHists[bkg+'Smooth'] = [upsystSmooth, dnsystSmooth]

        ## opened external file so we need to change the
        ## working directory back to 0
        for Key in self.systHists:
            for hist in systHists[Key]:
                hist.SetDirectory(0)


    def getSignalHistogram(self, sample, doSF=0, doISR=True, 
                           extraWeight=None, applyEffs=True,
                           JEtype='', applyISRWeight=True,
                           extraCuts=None, applyTrigWeight='cv'):

        ## if doSF<0, then do not apply btag SFs
        sampleSuffix = ''
        ## doSF==0 means apply btag SF central values
        ## doSF>0 means apply btag SF variations
        if doSF>=0:
            sampleSuffix = 'SF['doSF']'

        hist = RA2b.getDist(sample+JEtype,
                            self.varPrefix+'Sig'+sampleSuffix,
                            extraCuts=str(cuts), distRange=self.distRange,
                            nBins=self.Bin, applyISRWeight=doISR,
                            extraWeight=extraWeight, applyEffs=applyEffs,
                            extraCuts=extraCuts, 
                            applyTrigWeight=applyTrigWeight)
        return hist

    def setSignalHistogram(self, sample):

        self.signalHist = getSignalHistogram(sample)

    def getSystFromVarHists(UP,DN,CV=None):

        ## If no central value histogram is given,
        ## use average of up and down syst hists
        if CV==None:
            CV=UP.Clone()
            CV.Add(DN)
            CV.Scale(0.5)

        ## Actual histograms of systematic uncertainties
        ## clone signal hist for consistent binning
        upSyst = self.signalHist.Clone()
        dnSyst = self.signalHist.Clone()

        ## loop over bins to find upper and lower systematics
        for binIter in range(1,self.Bins+1):
            cv = CV.GetBinContent(binIter)
            up = UP.GetBinContent(binIter)
            dn = DN.GetBinContent(binIter)
            ## if central val is nonzero then take max and min
            ## as upper and lower systematics. If CV is zero then
            ## uncertainties will default to zero (CV)
            if(cv>0):
                upSyst.SetBinContent(binIter,(max(cv,up,dn)-cv)/cv)
                dnSyst.SetBinContent(binIter,(cv-min(cv,up,dn))/cv)

        return [upSyst,dnSyst]

    def getBTagSystHists(self, sample):

        ## interpretations on FastSim only
        ## only other sample requiring btag Systs is Zinv
        if('fast' in sample):
            suffix = 'Signal'
        else:
            suffix = 'Zinv'

        cuts = self.hdpCut+self.htCut+self.hpfCaloCut

        ## histogram array index dictionary
        ## format: [upperSyst,lowerSyst]
        indexDict = {
            'BTagSF'+suffix: [1,2],
            'CTagSF'+suffix: [3,4],
            'MTagSF'+suffix: [5,6],
            'BTagCF'+suffix: [7,8],
            'CTagCF'+suffix: [9,10],
            'MTagCF'+suffix: [11,12],
        }

        ## first get b-tag SF central value for comparison
        if sample not in self.signalHist.GetName():
            setSignalHistogram(sample)
            
        ## loop over each systematic from the indexDict
        for syst in indexDict:
            ## need temp list to store variation histograms
            tmpHistList = []
            for variation in indexDict[syst]:
                tmpHistList.append(getSignalHistogram(sample, doSF=variation))
                    
            self.sigSystDict[syst] = getSystFromVarHists(tmpHistList[0],
                                                         tmpHistList[1],
                                                         self.signalHist)
                
    def getScaleSystHists(self, sample):


        ## first set central value for comparison
        if sample not in self.signalHist.GetName():
            setSignalHistogram(sample)


        ## Eight different renomralization and factorization scale variations
        for scaleIter in range(8):
            scaleHists.append(
                getSignalHistogram(sample,extraWeight='ScaleWeights'
                                   '['+str(scaleIter)+']'))
            ## normalize to remove unphysical normalization differences
            scaleHists[i].Scale(
                self.signalHist.Integral()/scaleHists[scaleIter].Integral())

        ## clone signalHist to ensure consistent binning
        varUpHist = self.signalHist.Clone()
        varDnHist = self.signalHist.Clone()

        ## get max and min scale varation per bin and fill up and down hists
        for binIter in range(1,self.Bins+1):
            binList = []
            for scaleIter in range(8):
                binList.append(scaleHists[scaleIter].GetBinContent(binIter))
            varUpHist.SetBinContent(binIter,max(binList))
            varDnHist.SetBinContent(binIter,min(binList))

        self.sigSystDict['ScaleSignal'] = getSystFromVarHists(varUpHist,
                                                              varDnHist,
                                                              self.signalHist)
    def getISRSystHists(self, sample):

        ## first set central value for comparison
        if sample not in self.signalHist.GetName():
            setSignalHistogram(sample)
        
        ## get histograms with up and down ISR weights applied
        varUpHist = getSignalHistogram(sample, applyISRweight=False, 
                                       extraWeight='ISRup')
        varDnHist = getSignalHistogram(sample, applyISRweight=False, 
                                       extraWeight='ISRdn')

        self.sigSystDict['ISRSignal'] = getSystFromVarHists(varUpHist,
                                                            varDnHist,
                                                            self.signalHist)

    def getTrigSystHists(self, sample):

        ## first set central value for comparison
        if sample not in self.signalHist.GetName():
            setSignalHistogram(sample)

        ## get up and down variations
        varUpHist = getSignalHistogram(sample, applyTrigWeight='up')
        varDnHist = getSignalHistogram(sample, applyTrigWeight='dn')

        self.sigSystDict['Trig'] =  = getSystFromVarHists(varUpHist,
                                                          varDnHist,
                                                          self.signalHist)

    def getJetEnergySystHists(self, sample, JEtype):
        """
        For both jet energy correction and jet energy resolution systs.
        Using separate sample files, so things will speed up if 
        we don't apply all data/mc corrections and just take deviation.
        """

        ## get up and down variations
        varUpHist = getSignalHistogram(sample, JEtype=JEtype+'up', doSF=-1,
                                       doISR=False, applyEffs=False,
                                       applyISRWeight=False, 
                                       applyTrigWeight=False)
        varDnHist = getSignalHistogram(sample, JEtype=JEtype+'down', doSF=-1,
                                       doISR=False, applyEffs=False,
                                       applyISRWeight=False, 
                                       applyTrigWeight=False)

        ## set JE syst using average as CV
        self.sigSystDict[JEtype] = getSystFromVarHists(varUpHist,
                                                       varDnHist)

    def getJetIDSystHist(self):

        ## clone signalHist to ensure consistent binning
        systHist = self.signalHist.Clone()

        ## flat 1% uncertainty
        idError = 0.01

        ## fill the histogram
        for i in range(1,self.Bins+1):
            systHist.SetBinContent(i,idError)

        self.sigSystDict['JetIDSignal'] = systHist

    def getLumiSystHist(self):

        ## clone signalHist to ensure consistent binning
        systHist = self.signalHist.Clone()

        ## flat 2.6% uncertainty from lumi group
        lumiError = 0.026

        ## fill the histogram
        for i in range(1,self.Bins+1):
            systHist.SetBinContent(i,lumiError)

        self.sigSystDict['LumiSignal'] = systHist

    def getSignalSystHists(self, sample):

    def subtractSignalContamination(self, sample):
        """
        Be sure to do this after computing all systematic uncertainties
        ***Still need to implement T1tttt SL contamination subtaction**
        """

        ## Warning message in case we do things out of order
        if self.sigSystDict == {}:
            print "***Warning***"
            print "Any systematic uncertainties computed after"
            print "signal contamination subtraction will be biased."

        ## top/W contamination
        topContam = self.predDict['TopW'].Clone()

        ## get number of signal events in the sideband
        sigSidebandNorm = 0
        for binIter in range(1,self.Bins-self.nSigBins+1):
            sigSidebandNorm += self.signalHist.GetBinContent(binIter)

        ## Scale top contamination to sig sideband
        topContam.Scale(sigSidebandNorm/topContam.Integral())

        ## QCD contamination
        qcdContam = getSignalHistogram(sample, extraCuts=str(self.ldpCut))
                        
        ## Scale qcd contamination by qcd CR scaling
        qcdContam.Multiply(qcdCorrHist['Sig'])
        qcdContam.Scale(Norm['QCD'][0])
                
        ## Subtract off contamination
        self.signalHist.Add(topContam,-1)
        self.signalHist.Add(qcdContam,-1)

    def getGraphFromHists(self, hist, systUp, systDn, doQuad=True):

    def getPiePlots(self, graphs, rangeList=range(45,50), doSymmetric=True, colorDict=None, doUncert=False, binning=None):

    def Round(self, val):
        """
        Find out the order of magnitude before rounding to the appropriate 
        significant digit
        """

        ## Find out what order of magnitude
        ## anything below zero is zero order of magnitude
        if(val<=0):
            order = 0
        else:
            order = int(math.floor(math.log10(val)))
            
        ## Find out number of digits to keep after decimal point
        if order<=0:
            nFigs=3-order
        elif order>0:
            nFigs = max(order+3,0)

        return round(val, nFigs)

    def getLine(self, sampleDict, Key, Bin, spc, doSymmetric):

    def makeDataCards(self, sample, doSymmetric=True):

    def makeAllCards(self, sampleClass):
        
        ## Get full list of files available
        listOfFiles = os.listdir('/media/hdd/work/data/lpcTrees'
                                 '/Skims/Run2ProductionV11/scan/tree_signal')

        ## convert file name to RA2b readable sample name
        ## example: tree_T1qqqq_600_1_fast.root -> T1qqqq_600_1_fastIDP
        listOfSamples = []
        for File in listOfFiles:
            ## ignore files not in sample class
            if sampleClass not in File:
                continue
            listOfSamples.append(
                File.replace('tree_','').replace('.root','IDP'))

        ## make cards for each signal model point
        for sample in listOfSamples:
        
            ## reset signal histograms
            self.signalHist = ROOT.TH1D()
            self.sigSystDict = {}

            makeDataCards(sample)
