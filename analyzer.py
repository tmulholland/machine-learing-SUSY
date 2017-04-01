import ROOT
import RA2b
import numpy
import math
import os
import pickle

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

        ## BDT background shapes
        self.BDTshapes = ['Bkg', 'Sig[0]']
        self.colorDict = {'QCD': 8, 'Zinv': 2, 'TopW': 4}

        ## prefix for discriminator output
        self.varPrefix = 'BDT_'

        ## define cuts different than baseline RA2b
        self.dPhiCut = 'none'
        self.htCut = ROOT.TCut('HT>700')
        self.hdpCut = ROOT.TCut('DeltaPhi1>0.1&&DeltaPhi2>0.1&&'
                                'DeltaPhi3>0.1&&DeltaPhi4>0.1')

        self.ldpCut = ROOT.TCut('DeltaPhi1<0.1||DeltaPhi2<0.1||'
                                'DeltaPhi3<0.1||DeltaPhi3<0.1')
        self.hpfCaloCut = ROOT.TCut('MET/CaloMET>0.9') 
        self.lpfCaloCut = ROOT.TCut('MET/CaloMET<0.9')
        self.bCut = (self.distRange[1]-self.distRange[0]-
                     self.nSigBins/float(self.Bins))
        self.blindCut = ROOT.TCut(self.varPrefix+'Sig[0]<'+str(self.bCut))

        ## dictionary for prediction to smooth
        ## default no smoothing anywhere
        ## likely not needed if we use LDP control sample for QCD
        self.doSmoothDict = {'QCD': False, 'TopW': False}

        ## N(bins) for QCD DR 
        self.BinsDR = 10

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
        
        ## QCD purity histograms
        self.qcdPurDict = {}

        ## top/W and QCD systematic 
        ## uncertainty histograms
        ## format: 
        self.systDictTopW = {}
        self.systDictQCD = {}

        ## Signal systematics histogram
        ## format: 
        self.sigSystDict = {}

        ## Storing plots in dictionary saving time 
        ## later without having to remake plots
        self.canvasDict = {}

    def dumpAnalyzer(self, fileName=None):
        """ Use this to save work, i.e. so you dont have to rerun all of
        the time consuming background predictions. This is especially useful
        in case of a crash. This function uses pickle.
        """
        
        ## decide output name if none given
        if fileName==None:
            if self.doData:
                fileName='Data.pkl'
            else:
                fileName='MC.pkl'

        afile = open(fileName, 'wb')
        
        pickle.dump(self, afile)
        afile.close()

    def loadAnalyzer(self, fileName=None):
        """ Use this to load a stored analyzer (see above).
        This is especially useful
        in case of a crash. This function uses pickle.
        """
        
        ## decide output name if none given
        if fileName==None:
            if self.doData:
                fileName='Data.pkl'
            else:
                fileName='MC.pkl'
                
        afile = open(fileName, 'rb')
        
        newAnalyzer = pickle.load(afile)

        afile.close()
        
        return newAnalyzer

    def setZdr(self):
        """ Compute the Z prediction double ratio with error bands
        """

        ## double ratio computation takes
        ## a list of tgraphs
        graphs = []

        for BDTshape in self.BDTshapes:

            ## keep Sig reigion blind
            if 'Sig' not in BDTshape:
                cuts = self.hdpCut+self.htCut+self.hpfCaloCut+self.blindCut
            else:
                cuts = self.hdpCut+self.htCut+self.hpfCaloCut


            ## get dr graphs
            graphs.append(
                RA2b.getDoubleRatioGraph(self.varPrefix+BDTshape,
                                         applyPuWeight=True,
                                         extraCuts=str(cuts),
                                         doCG=False))
    
        ## make plots viewable after runtime
        for graph in graphs:
            ROOT.SetOwnership(graph,0)

        ## Z/gamma double ratio 
        self.Zdr = RA2b.getDoubleRatioPlot(graphs)

    def setRZG(self):
        """Compute ratio of Z over Gamma from simulation and store it 
        as in dictionary (one key per BDT output shape)
        """

        ## must do LDP shapes as well for QCD contamination
        shapeList = self.BDTshapes + [shape+'LDP' for shape in self.BDTshapes]
        ## must do btag SFs
        shapeList += ['Sig[1]','Sig[2]']

        for BDTshape in shapeList:
            
            keySuffix = ''
            ## if not signal, apply blinding criteria
            cuts = self.hpfCaloCut+self.htCut
            if('Sig' not in BDTshape):
                cuts+=self.blindCut

            if 'LDP' in BDTshape:
                cuts+=self.ldpCut
                BDTshape = BDTshape.replace('LDP','')
                keySuffix+='LDP'
            else:
                cuts+=self.hdpCut

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
                                extraCuts=str(cuts),applyPuWeight=True,
                                extraWeight='TrigWeight')

            ## Trigger syst variation for signal region (not for contam)
            if ('Sig[0]' in BDTshape and 'LDP' not in keySuffix):
                zinvUp = RA2b.getDist('zinvIDP',self.varPrefix+BDTshape,
                                    distRange=self.distRange,
                                    nBins=self.Bins,
                                    extraCuts=str(cuts),applyPuWeight=True,
                                    extraWeight='TrigWeightUp')

                zinvDn = RA2b.getDist('zinvIDP',self.varPrefix+BDTshape,
                                    distRange=self.distRange,
                                    nBins=self.Bins,
                                    extraCuts=str(cuts),applyPuWeight=True,
                                    extraWeight='TrigWeightDn')
                zinvUp.Divide(gjets)
                zinvDn.Divide(gjets)
                self.rzg[BDTshape+'TrigWeightUp'] = zinvUp
                self.rzg[BDTshape+'TrigWeightDn'] = zinvDn

            ## ratio becomes R(Z/gamma)
            zinv.Divide(gjets)

            self.rzg[BDTshape+keySuffix] = zinv

    def setZpred(self):
        """ Make full Z prediction (less systematic uncertainties)
        """

        ## must do LDP shapes as well for QCD contamination
        shapeList = self.BDTshapes + [shape+'LDP' for shape in self.BDTshapes]

        for BDTshape in shapeList:

            keySuffix = ''
            ## if not signal, apply blinding criteria
            cuts = self.hpfCaloCut+self.htCut
            if('Sig' not in BDTshape):
                cuts+=self.blindCut

            if 'LDP' in BDTshape:
                cuts+=self.ldpCut
                BDTshape = BDTshape.replace('LDP','')
                keySuffix+='LDP'
            else:
                cuts+=self.hdpCut
                
            ## photon data
            photon = RA2b.getDist('photonIDP',self.varPrefix+BDTshape,
                                  distRange=self.distRange,
                                  nBins=self.Bins,
                                  extraCuts=str(cuts),applyEffs=True)
        
            ## compute R(Z/gamma) unless already done so
            if(BDTshape not in self.rzg.keys()):
                self.setRZG()
              
            ## apply R(Z/gamma)
            photon.Multiply(self.rzg[BDTshape+keySuffix])
    
            ## compute double ratio unless already done so
            if(self.Zdr == None):
                self.setZdr()

            ## apply overall DR scaling factor 
            photon.Scale(self.Zdr[0][0])

            self.zPredDict[BDTshape+keySuffix] = photon

    def setZpredSyst(self):
        """ Make full Z prediction and systematic uncertainties. Will run
        Z prediction if not already stored. 
        """
        ## Uncertainties taken as inputs from other studies
        ## these uncertainties will be hard coded for now
        effSyst = 0.05
        btagSyst = 0.005
        trigSyst = 0.0035
        gTrig = 0.01
        fDir = 0.015 # 30% on 1-Fdir
        gSF = 0.005
        gPur = 0.01

        ## dictionaries of total upper and 
        ## lower systematic uncertainties
        upSyst = {}
        dnSyst = {}

        ## Only need to compute full systematics for signal region
        BDTshape = 'Sig[0]'

        ## check if prediction is stored yet
        ## if not, set prediction
        if BDTshape not in self.zPredDict.keys():
            self.setZpred()

        ## get cuts beyond baseline
        cuts = self.htCut+self.hdpCut+self.hpfCaloCut
        if('Sig' not in BDTshape):
            cuts += self.blindCut

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
        rzgStatHist = self.zPredDict[BDTshape].Clone()
        rzgUpBtagHist = self.zPredDict[BDTshape].Clone()
        rzgDnBtagHist = self.zPredDict[BDTshape].Clone()
        rzgUpTrigHist = self.zPredDict[BDTshape].Clone()
        rzgDnTrigHist = self.zPredDict[BDTshape].Clone()
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
            cvUpErr = math.sqrt(self.Zdr[0][1]**2
                                +purityHist.GetBinContent(1)**2
                                +effSyst**2+btagSyst**2+trigSyst**2)
            cvDnErr = math.sqrt(self.Zdr[0][1]**2
                                +purityHist.GetBinContent(1)**2
                                +effSyst**2+btagSyst**2+trigSyst**2)
            cvUpHist.SetBinContent(binIter,1+cvUpErr)
            cvDnHist.SetBinContent(binIter,1-cvDnErr)

            ## DR shape uncertainty:
            ## subtract off DR CV error from DR
            ## to avoid double counting
            drUpErr = max(self.Zdr[1][BDTshape][binIter-1][0]
                          -self.Zdr[0][1],0)
            drDnErr = max(self.Zdr[1][BDTshape][binIter-1][1]
                          -self.Zdr[0][1],0)
            drUpHist.SetBinContent(binIter, 1.+drUpErr)
            drDnHist.SetBinContent(binIter, 1.-drDnErr)

            ## R(Z/gamma) stat error, symmetric for now
            rzgStatHist.SetBinContent(binIter,
                                    1+self.rzg[BDTshape].GetBinError(binIter))
            ## R(Z/gamma) btag SF error
            btagUp = (max(self.rzg['Sig[0]'].GetBinContent(binIter),
                          self.rzg['Sig[1]'].GetBinContent(binIter),
                          self.rzg['Sig[2]'].GetBinContent(binIter))
                      -self.rzg['Sig[0]'].GetBinContent(binIter))
            btagDn = (self.rzg['Sig[0]'].GetBinContent(binIter)
                      -min(self.rzg['Sig[0]'].GetBinContent(binIter),
                           self.rzg['Sig[1]'].GetBinContent(binIter),
                           self.rzg['Sig[2]'].GetBinContent(binIter)))
            rzgUpBtagHist.SetBinContent(binIter,
                            1.+btagUp/self.rzg['Sig[0]'].GetBinContent(binIter))
            rzgDnBtagHist.SetBinContent(binIter,
                            1.-btagDn/self.rzg['Sig[0]'].GetBinContent(binIter))


            ## R(Z/gamma) trig SF error
            trigUp = (max(self.rzg['Sig[0]'].GetBinContent(binIter),
                          self.rzg['Sig[0]TrigWeightUp'].GetBinContent(binIter),
                          self.rzg['Sig[0]TrigWeightDn'].GetBinContent(binIter))
                      -self.rzg['Sig[0]'].GetBinContent(binIter))
            trigDn = (self.rzg['Sig[0]'].GetBinContent(binIter)
                      -min(self.rzg['Sig[0]'].GetBinContent(binIter),
                        self.rzg['Sig[0]TrigWeightUp'].GetBinContent(binIter),
                        self.rzg['Sig[0]TrigWeightDn'].GetBinContent(binIter)))
            rzgUpTrigHist.SetBinContent(binIter,
                            1.+trigUp/self.rzg['Sig[0]'].GetBinContent(binIter))
            rzgDnTrigHist.SetBinContent(binIter,
                            1.-trigDn/self.rzg['Sig[0]'].GetBinContent(binIter))


            ## photon related efficiency errors
            gTrigHist.SetBinContent(binIter,1+gTrig)
            fDirHist.SetBinContent(binIter,1+fDir)
            gSFHist.SetBinContent(binIter,1+gSF)
            gPurHist.SetBinContent(binIter,1+gPur)

        ## add individual systematic uncertainties to
        ## the dictionary containing the Z systematics
        self.zSystDict['ZinvNorm'] = [cvUpHist, cvUpHist]
        self.zSystDict['ZinvDR'] = [drUpHist, drDnHist]
        self.zSystDict['ZinvRZGstat'] = [rzgStatHist]
        self.zSystDict['ZinvRZGbtag'] = [rzgUpBtagHist, rzgDnBtagHist]
        self.zSystDict['ZinvRZGtrig'] = [rzgUpTrigHist, rzgDnTrigHist]
        self.zSystDict['ZinvNCR'] = [gStatHist,TFhist]
        self.zSystDict['ZinvGtrig'] = [gTrigHist]
        self.zSystDict['ZinvFdir'] = [fDirHist]
        self.zSystDict['ZinvGsf'] = [gSFHist]
        self.zSystDict['ZinvGpur'] = [gPurHist]


    def setData(self, nUnblindBins=25):
        """ Get data in signal region. Already unblinded so default is fully
        unblind. However, you can reblind the data by using setData(N) instead,
        where N is the number of bins you wish to unblind (i.e. N=0 is fully 
        blind)
        """

        ## Blind Cut Default: all signal data blind
        bCut = (self.distRange[1]-self.distRange[0]-
                (self.nSigBins-nUnblindBins)/float(self.Bins))
        blindCut = ROOT.TCut(self.varPrefix+'Sig[0]<'+str(bCut))

        cuts = self.htCut+self.hdpCut+self.hpfCaloCut+blindCut

        self.dataHist = RA2b.getDist('sigIDP',self.varPrefix+'Sig[0]', 
                                     distRange=self.distRange,
                                     nBins=self.Bins,extraCuts=str(cuts))

    def setCorrHistQCD(self):
        """ HDP/LDP correction histogram from QCD MC
        """

        ## using MC to get correction
        Name = 'qcdIDP'

        ## loop over discriminator distributions
        for BDTshape in self.BDTshapes+['Sig[1]']+['Sig[2]']:
            #numerator
            cuts = self.htCut+self.hdpCut+self.hpfCaloCut
            if 'Sig' not in BDTshape:
                cuts+=self.blindCut
            qcdNum = RA2b.getDist(Name,self.varPrefix+BDTshape,
                                  nBins=self.Bins,distRange=self.distRange,
                                  extraCuts=str(cuts))
            #denominator
            cuts = self.htCut+self.ldpCut+self.hpfCaloCut
            if 'Sig' not in BDTshape:
                cuts+=self.blindCut
            qcdDenom = RA2b.getDist(Name,self.varPrefix+BDTshape,
                                    nBins=self.Bins,distRange=self.distRange,
                                    extraCuts=str(cuts))
            ## correction hist
            qcdTF2qcd = qcdNum.Clone()
            qcdTF2qcd.Divide(qcdDenom)

            ## smooth out last bin
            for binIter in range(self.Bins+1):
                qcdTF2qcd.SetBinContent(binIter,
                                        min(2,qcdTF2qcd.GetBinContent(binIter)))
                qcdTF2qcd.SetBinError(binIter,
                                      min(1,qcdTF2qcd.GetBinError(binIter)))

            ## set correction hists
            self.qcdCorrHist[BDTshape] = qcdTF2qcd


    def setQcdCsFrac(self):
        """ Extract the QCD purity using fits with the QCD MC as signal
        template and single lepton LDP control sample as background template
        """

        D1 = "topWIDP"
        D2 = "qcdIDP"
        Q1 = 'qcdIDP'
        X1 = "topWslmLDP"
        X2 = "topWsleLDP"

        if self.doData:
            D1 = 'sigLDP'
            D2 = 'sig'
            X1 = 'slmLDP'
            X2 = 'sleLDP'


        BDTshape = 'Bkg'

        ## Data Hist, low Dphi and high PF/Calo
        cuts = self.htCut+self.ldpCut+self.hpfCaloCut+self.blindCut
        hData = RA2b.getDist(D1, self.varPrefix+BDTshape,
                             distRange=self.distRange, nBins=self.Bins,
                             extraCuts=str(cuts), applyMHTCut=self.mhtCut)
        hData2 = RA2b.getDist(D2, self.varPrefix+BDTshape,
                             distRange=self.distRange, nBins=self.Bins,
                             extraCuts=str(cuts), applyMHTCut=self.mhtCut)

        hData.Add(hData2)

        qcdPurHist = hData.Clone()

        ## subtract off Zinv measurement
        if(self.doData and self.zPredDict!=None):
            hData.Add(self.zPredDict[BDTshape+'LDP'],-1)

        ## QCD Hist shape always from MC
        hQCD = RA2b.getDist(Q1, self.varPrefix+BDTshape,
                            distRange=self.distRange, nBins=self.Bins,
                            extraCuts=str(cuts), applyMHTCut=self.mhtCut)

        nQCD0 = hQCD.Integral()
        ## SL LDP contamination
        hSLLDP = RA2b.getDist(X1, self.varPrefix+BDTshape,
                              distRange=self.distRange, nBins=self.Bins,
                              extraCuts=str(cuts), applyMHTCut=self.mhtCut,
                              applyEffs=True)
        hSLLDP_ = RA2b.getDist(X2, self.varPrefix+BDTshape,
                               distRange=self.distRange, nBins=self.Bins,
                               extraCuts=str(cuts), 
                               applyMHTCut=self.mhtCut,applyEffs=True)
        hSLLDP.Add(hSLLDP_)
        nSLLDP0 = hSLLDP.Integral()

        ## Key for dictionaries
        Key = BDTshape+'NormFitLDP'
        
        ## declare canvas for plotting
        self.canvasDict[Key] = ROOT.TCanvas(Key,Key,0,0,900,600)



        ## range of fit
        fitRange_min = self.distRange[0]
        fitRange_max = self.distRange[1]
            
        ## RooFit objects
        ## define ROO vars
        ndata = hData.Integral()
        varX = ROOT.RooRealVar("varX",
                               hData.GetXaxis().GetTitle(),
                               fitRange_min,fitRange_max)
        nqcd  = ROOT.RooRealVar("nqcd",
                                "estimated number of QCD events",
                                ndata*8./9., 0.000001, ndata)
        ncontam = ROOT.RooRealVar("ncontam",
                                  "top+W contamination in QCD events",
                                  ndata*1./9., 0.000001, ndata)

        ## define ROO data histograms
        data     = ROOT.RooDataHist("data",
                                    "data in signal region" , 
                                    ROOT.RooArgList(varX), hData)
        qcd      = ROOT.RooDataHist("qcd_",
                                    "QCD distribution from control region", 
                                    ROOT.RooArgList(varX), hQCD)
        contam      = ROOT.RooDataHist("contam_",
                                       "top+W contam in QCD control region", 
                                       ROOT.RooArgList(varX), hSLLDP)
        
        ## ROO PDFs (template hists from control regions)
        qcd_model = ROOT.RooHistPdf("qcd_model",
                                    "RooFit template for QCD backgrounds", 
                                    ROOT.RooArgSet(varX), qcd)
        contam_model = ROOT.RooHistPdf("contam_model",
                                       "RooFit template for top+W contam",
                                       ROOT.RooArgSet(varX), contam)
            
        # Prepare extended likelihood fits
        qcd_shape = ROOT.RooExtendPdf("qcd_shape","QCD shape pdf",
                                      qcd_model, nqcd)
        contam_shape = ROOT.RooExtendPdf("contam_shape","contam shape pdf",
                                         contam_model, ncontam)


        # Combine the models
        combModel = ROOT.RooAddPdf("combModel",
                                   "Combined model for top+W and QCD", 
                                   ROOT.RooArgList(qcd_shape,contam_shape))

        # Fit the data from the ROO PDF template combo model
        fitResult = combModel.fitTo(data, 
                                    ROOT.RooFit.Extended(True), 
                                    ROOT.RooFit.Save(True), 
                                    ROOT.RooFit.Minos(True),
                                    ROOT.RooFit.PrintLevel(-1))
            
        ################
        ## Begin plot fit
        xframe = varX.frame()
        data.plotOn(xframe,)
        combModel.plotOn(xframe)
        data.plotOn(xframe) # plot again so that it is on top of the errors
        combModel.plotOn(xframe, ROOT.RooFit.LineColor(ROOT.kBlack))
        combModel.plotOn(xframe, ROOT.RooFit.Components("qcd_shape"), 
                         ROOT.RooFit.LineColor(ROOT.kGreen))
        combModel.plotOn(xframe, ROOT.RooFit.Components("contam_shape"), 
                         ROOT.RooFit.LineColor(ROOT.kRed))
        xframe.GetXaxis().SetTitle(BDTshape+'-like BDT Output')
        xframe.GetYaxis().SetTitleOffset(1.35)
        xframe.GetYaxis().SetTitle('Events')
        xframe.SetTitle('')
        xframe.SetMinimum(0)
        xframe.Draw()
        ## End plot fit
        ################


        hCont = hSLLDP.Clone()
        hCont.Scale(ncontam.getVal()/nSLLDP0)
        ## add Zinv measurement for total contamination
        if(self.doData and self.zPredDict!=None):
            hCont.Add(self.zPredDict[BDTshape+'LDP'])
        
        for binIter in range(1,self.Bins+1):
            if hCont.GetBinContent(binIter)>qcdPurHist.GetBinContent(binIter):
                qcdPurHist.SetBinContent(binIter,hCont.GetBinContent(binIter)
                                         +qcdPurHist.GetBinContent(binIter))
        
        totalHist = qcdPurHist.Clone()
        qcdPurHist.Add(hCont,-1)
        passedHist = qcdPurHist.Clone()
        qcdPurHist.Divide(totalHist)

        self.qcdPurDict['Bkg'] = (qcdPurHist, passedHist,totalHist)        

        cuts = self.htCut+self.ldpCut+self.hpfCaloCut
        hContSig = RA2b.getDist(X1, self.varPrefix+'Sig[0]',
                                distRange=self.distRange, nBins=self.Bins,
                                extraCuts=str(cuts), applyMHTCut=self.mhtCut,
                                applyEffs=True)
        hContSig_ = RA2b.getDist(X2, self.varPrefix+'Sig[0]',
                                 distRange=self.distRange, nBins=self.Bins,
                                 extraCuts=str(cuts), 
                                 applyMHTCut=self.mhtCut,applyEffs=True)
        hContSig.Add(hContSig_)
        
        nContSig0 = 0.
        for binIter in range(1,26):
            nContSig0 += hContSig.GetBinContent(binIter)
        
        hContSig.Scale(ncontam.getVal()/nContSig0)

        ## add Zinv measurement for total contamination
        if(self.doData and self.zPredDict!=None):
            hContSig.Add(self.zPredDict['Sig[0]LDP'])

        self.qcdPurDict['Sig[0]'] = [hContSig]


    def setNormFromFit(self, doCorrQCD=True):
        """ Fit the Z subtrackted sideband data in the top+W-like 
        boosted decision tree output. Templates are from control sample data.
        """

        ## Simulation sample names
        D1 = "topWIDP"
        D2 = "qcdIDP"
        T1 = "topWslmIDP"
        T2 = "topWsleIDP"
        X1 = "topWslmLDP"
        X2 = "topWsleLDP"
        Q1 = 'qcdIDP'
        Q2 = 'topWIDP'

        ## Data sample names
        if self.doData:
            D1 = 'sig'
            D2 = 'sigLDP'
            T1 = 'slmIDP'
            T2 = 'sleIDP'
            X1 = "slmLDP"
            X2 = "sleLDP"
            Q1 = 'sigLDP'
            Q2 = 'sig'
        

        ## turn off annoying warnings
        ROOT.RooMsgService.instance().setSilentMode(True)
        ROOT.RooMsgService.instance().setGlobalKillBelow(-10000)
        ## loop over classifier discriminant shapes
        for BDTshape in self.BDTshapes:
            ## Skip signal for getting bkg normalizations
            if 'Sig' in BDTshape:
                continue 
            
            ## Key for dictionaries
            Key = BDTshape+'NormFit'

            ## declare canvas for plotting
            self.canvasDict[Key] = ROOT.TCanvas(BDTshape,BDTshape,0,0,900,600)

            ## Data Hist, high Dphi and high PF/Calo
            cuts = self.htCut+self.hdpCut+self.hpfCaloCut+self.blindCut
            hData = RA2b.getDist(D1, self.varPrefix+BDTshape,
                                 distRange=self.distRange, nBins=self.Bins,
                                 extraCuts=str(cuts), applyMHTCut=self.mhtCut)

            ## add ldp for data and top/W for Sim
            hData2 = RA2b.getDist(D2, self.varPrefix+BDTshape,
                                  distRange=self.distRange, nBins=self.Bins,
                                  extraCuts=str(cuts), applyMHTCut=self.mhtCut)
            nTopWTrue = hData.Integral()
            nQCDTrue = hData2.Integral()
            hData.Add(hData2)
            hData.SetTitle('BDToutput')
            
            ## subtract off Zinv measurement
            if(self.doData and self.zPredDict!=None):
                hData.Add(self.zPredDict[BDTshape],-1)

            if BDTshape not in self.qcdPurDict.keys():
                self.setQcdCsFrac()

            ## top/W CR
            cuts = self.htCut+self.hdpCut+self.hpfCaloCut+self.blindCut
            ## single muon
            hTopW = RA2b.getDist(T1, self.varPrefix+BDTshape,
                                 distRange=self.distRange, 
                                 nBins=self.Bins, extraCuts=str(cuts), 
                                 applyMHTCut=self.mhtCut,applyEffs=True)
            ## single electron
            hTopWsle = RA2b.getDist(T2, self.varPrefix+BDTshape,
                                    distRange=self.distRange, 
                                    nBins=self.Bins, extraCuts=str(cuts),
                                    applyMHTCut=self.mhtCut,applyEffs=True)
            ## single lepton
            hTopW.Add(hTopWsle)

            # get number of CR events for contamination constraint
            nSLHDP = float(hTopW.Integral())

            ## QCD CR
            cuts = self.htCut+self.ldpCut+self.hpfCaloCut+self.blindCut
            hQCD = RA2b.getDist(Q1, self.varPrefix+BDTshape,
                                distRange=self.distRange, nBins=self.Bins,
                                extraCuts=str(cuts), applyMHTCut=self.mhtCut)

            hTop = RA2b.getDist(Q2, self.varPrefix+BDTshape,
                                distRange=self.distRange, nBins=self.Bins,
                                extraCuts=str(cuts), applyMHTCut=self.mhtCut)
            hQCD.Add(hTop)

            hQCD.Multiply(self.qcdPurDict[BDTshape][0])# bkg pur dict is eff

            ## SL LDP contamination
            hSLLDP = RA2b.getDist(X1, self.varPrefix+BDTshape,
                                  distRange=self.distRange, nBins=self.Bins,
                                  extraCuts=str(cuts), applyMHTCut=self.mhtCut,
                                  applyEffs=True)
            hSLLDP_ = RA2b.getDist(X2, self.varPrefix+BDTshape,
                                   distRange=self.distRange, nBins=self.Bins,
                                   extraCuts=str(cuts), 
                                   applyMHTCut=self.mhtCut,applyEffs=True)
            hSLLDP.Add(hSLLDP_)

            # get number of CR events for contamination constraint
            nSLLDP = float(hSLLDP.Integral())

            ## Correct the shape for HDL/LDP differences
            if doCorrQCD:
                if BDTshape not in self.qcdCorrHist.keys():
                    self.setCorrHistQCD()
                hQCD.Multiply(self.qcdCorrHist[BDTshape])

            # get number of CR events for contamination constraint
            nQCDCR = float(hQCD.Integral())


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
                                     "estimated number of top+W events",
                                     ndata*4./9., 0.000001, ndata)
            nqcd  = ROOT.RooRealVar("nqcd",
                                    "estimated number of QCD events",
                                    ndata*4./9., 0.000001, ndata)
            ncontam = ROOT.RooRealVar("ncontam",
                                    "top+W contamination in QCD events",
                                    ndata*1./9., 0.000001, ndata)

            SLLDP_factor = ROOT.RooFormulaVar("LDP_factor",
                                              "factor applied to SLLDP",
                                              "@0*@1*"+
                                              str(nSLLDP/(nSLHDP*nQCDCR)),
                                              ROOT.RooArgList(nqcd,ntopW))

            ## define ROO data histograms
            data     = ROOT.RooDataHist("data",
                                        "data in signal region" , 
                                        ROOT.RooArgList(varX), hData)
            topW     = ROOT.RooDataHist("topW_",
                                        "signal and other MC backgrounds", 
                                        ROOT.RooArgList(varX), hTopW)
            qcd      = ROOT.RooDataHist("qcd_",
                                        "QCD distribution from control region", 
                                        ROOT.RooArgList(varX), hQCD)
            SLLDP    = ROOT.RooDataHist("SLLDP_",
                                        "SL LDP distribution",
                                        ROOT.RooArgList(varX), hSLLDP)

            contam      = ROOT.RooDataHist("contam_",
                                        "top+W contam in QCD control region", 
                                           ROOT.RooArgList(varX), hSLLDP)
 
            ## ROO PDFs (template hists from control regions)
            topW_model = ROOT.RooHistPdf("topW_model",
                                         "RooFit template for signal", 
                                         ROOT.RooArgSet(varX), topW)
            qcd_model = ROOT.RooHistPdf("qcd_model",
                                        "RooFit template for QCD backgrounds", 
                                        ROOT.RooArgSet(varX), qcd)
            SLLDP_model = ROOT.RooHistPdf("SLLDP_model",
                                          "RooFit template for SLLDP contam",
                                          ROOT.RooArgSet(varX), SLLDP)

            contam_model = ROOT.RooHistPdf("contam_model",
                                           "RooFit template for top+W contam",
                                           ROOT.RooArgSet(varX), contam)
            

            # Prepare extended likelihood fits
            topW_shape = ROOT.RooExtendPdf("topW_shape", "topW shape pdf",
                                           topW_model, ntopW)
            qcd_shape = ROOT.RooExtendPdf("qcd_shape","QCD shape pdf",
                                          qcd_model, nqcd)
            contam_shape = ROOT.RooExtendPdf("contam_shape","contam shape pdf",
                                          contam_model, ncontam)
            SLLDP_shape = ROOT.RooExtendPdf("SLLDP_shape","SLLDP shape pdf",
                                            SLLDP_model, SLLDP_factor)
            qcd_sub =  ROOT.RooAddPdf("qcd_sub",
                                      "top+W subtracted qcd model",
                                      ROOT.RooArgList(qcd_shape,SLLDP_shape))


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
            data.plotOn(xframe,)
            combModel.plotOn(xframe)
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

            ## compute normalization factors
            varX.setRange("all",0,1)
            
            qcdIntegral = qcd_sub.createIntegral(
                ROOT.RooArgSet(ROOT.RooArgList(varX)),"all").getVal()
            topIntegral = topW_shape.createIntegral(
                ROOT.RooArgSet(ROOT.RooArgList(varX)),"all").getVal()
            normPresFactor = float(ndata)/(qcdIntegral+topIntegral)
            qcdIntegral*=normPresFactor
            topIntegral*=normPresFactor

            ## set the norm dict class variable
            self.Norm['QCD'] = (nqcd.getVal(),nqcd.getError(),nQCDTrue)
            self.Norm['TopW'] = (ntopW.getVal(),ntopW.getError(),nTopWTrue)

            
    def aveBins(self, hist, binList):
        """ Helper function for use with smoothing. Not smoothing at the 
        moment so this function may deteriorate.
        """
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

    def setPred(self, doCorrQCD=True, setWeight=-1):
        """ Compute full top+W and QCD predictions less systematic 
        uncertainties
        """

        Q1 = 'qcdIDP'
        Q2 = 'topWIDP'
        L1 = 'topWsleIDP'
        L2 = 'topWslmIDP'

        if self.doData:
            Q1 = 'sigLDP'
            Q2 = 'sig'
            L1 = 'sleIDP'
            L2 = 'slmIDP'
            
        BDTshape = 'Sig[0]'

        ## get top/W and QCD predictions
        if 'TopW' not in self.Norm.keys():
            self.setNormFromFit()
        ## QCD prediction fom LDP control region
        cuts = self.htCut+self.ldpCut+self.hpfCaloCut
        histQCD = RA2b.getDist(Q1,self.varPrefix+BDTshape,
                               distRange=self.distRange,
                               nBins=self.Bins,extraCuts=str(cuts))
        histQCD2 = RA2b.getDist(Q2,self.varPrefix+BDTshape,
                               distRange=self.distRange,
                               nBins=self.Bins,extraCuts=str(cuts))
        histQCD.Add(histQCD2)
        if 'Sig[0]' not in self.qcdCorrHist.keys():
            self.setCorrHistQCD()
        histQCD.Multiply(self.qcdCorrHist['Sig[0]'])
        histQCD.Add(self.qcdPurDict['Sig[0]'][0],-1) ## sig pur dict is not eff

        ## Top/W prediction from SLe and SLm control region
        cuts = self.htCut+self.hdpCut+self.hpfCaloCut
        histL = RA2b.getDist(L1, self.varPrefix+BDTshape,
                            distRange=self.distRange,applyEffs=True,
                            nBins=self.Bins,extraCuts=str(cuts))
        histL.Add(RA2b.getDist(L2, self.varPrefix+BDTshape,
                              distRange=self.distRange,applyEffs=True,
                              nBins=self.Bins,extraCuts=str(cuts)))
                 
        ## Get unblind normalization and scale to 
        ## normalization gotten from Norm dict
        normQCD=0
        normL=0
        for i in range(1,self.Bins-self.nSigBins+1):             
            normQCD += histQCD.GetBinContent(i)
            normL += histL.GetBinContent(i)
        histQCD.Scale(self.Norm['QCD'][0]/normQCD)
        histL.Scale(self.Norm['TopW'][0]/normL)

        ## Get non-closure multiplicative factor for top+W ~20%
        fTop = ROOT.TFile.Open('topSystHistsMC.root','read')
        nonClosureFactor = fTop.Get('nonClosureFactor')
        histL.Multiply(nonClosureFactor)

        ## Get non-closure multiplicative factor for QCD ~20%
        ## REMOVED!!
        fQCD = ROOT.TFile.Open('qcdSystHists.root','read')
        nonClosureFactorQCD = fQCD.Get('nonClosureFactorQCD')
        ##histQCD.Multiply(nonClosureFactorQCD)

        ## set negative QCD yields (2 bins) equal to 0
        for binIter in range(1,self.Bins+1):
            if histQCD.GetBinContent(binIter)<0:
                histQCD.SetBinContent(binIter,0)

        ## store in prediction dictionary
        self.predDict['QCD'] = histQCD
        self.predDict['TopW'] = histL
            
        ## opened external file so we need to change the
        ## working directory back to 0
        for Key in self.predDict.keys():
            self.predDict[Key].SetDirectory(0)
             

    def getStatHist(self, sampleName):
        """ Computes the control sample statistics
        """
        cuts = self.htCut+self.hpfCaloCut
        if 'QCD' in sampleName:
            cuts+=self.ldpCut
        else:
            cuts+=self.hdpCut
            
        ## dictionary to determine appropriate dataset from sampleName

        mcSampleDict = {'QCD':  ['qcdIDP'],
                        'TopW': ['topWsleIDP','topWslmIDP'],
                        'Zinv': ['gjetsIDP']}
        dataSampleDict = {'QCD':  ['sigIDP'],
                          'TopW': ['sleIDP','slmIDP'],
                          'Zinv': ['photonIDP']}
                        
        DS = mcSampleDict[sampleName]
        if self.doData:
            DS = dataSampleDict[sampleName]
        
        ## no efficiencies or weights applied
        hStat = RA2b.getDist(DS[0], self.varPrefix+'Sig[0]',
                             distRange=self.distRange, nBins=self.Bins,
                             extraCuts=str(cuts),setWeight=1.)
        ## add second if there
        if len(DS)>1:
            hStat2 = RA2b.getDist(DS[1], self.varPrefix+'Sig[0]',
                                  distRange=self.distRange, nBins=self.Bins,
                                  extraCuts=str(cuts),setWeight=1.)

            hStat.Add(hStat2)

        return hStat

    def getNormSystHist(self, bkg):
        """ Turns the Norm dictionary into syst uncert histograms
        """
        ## set normalization error
        normSystHist = self.predDict[bkg].Clone()
        for binIter in range(1,self.Bins+1):
            normSystHist.SetBinContent(binIter,1+
                                       self.Norm[bkg][1]/self.Norm[bkg][0])
            normSystHist.SetBinError(binIter,0)

        return normSystHist

    def getStatSystHists(self, bkg):
        """ Computes the control sample statistics and transfer factor for 
        inputs to the higgs combine tool. 
        """

        TFhist = self.predDict[bkg].Clone()            

        ## combine takes hist with N(data events)
        ## for statistical uncertainty
        statHist = self.getStatHist(bkg)
        ## transfer factor histogram
        TFhist.Divide(statHist)
        
        return [statHist, TFhist]

    def setSystHistsTopW(self):
        """ Computes the full top+W prediction with systematic uncertainties
        will run setPred() if prediction dictionary is empty
        """
        ## check if need to run prediction first
        if 'TopW' not in self.predDict.keys():
            self.setPred()

        ## get shape syst file 
        ## from running shape systematic study
        if self.doData:
            fTop = ROOT.TFile.Open('topSystHistsData.root','read')
        else:
            fTop = ROOT.TFile.Open('topSystHistsMC.root','read')

        ## take 50% uncertainty on non closure multiplicative factor
        nonClosureFactor = fTop.Get('nonClosureFactor')
        nonClosureFactorSyst = nonClosureFactor.Clone()
        for binIter in range(1,nonClosureFactorSyst.GetNbinsX()+1):
            nonClosureFactorSyst.SetBinContent(
                binIter,(nonClosureFactorSyst.GetBinContent(binIter)-1)*0.5+1)

        ## top systs
        self.systDictTopW['TopWIsoSystHist'] = [
            fTop.Get('UpIsoSystHist'), 
            fTop.Get('DnIsoSystHist')]
        self.systDictTopW['TopWMTWSystHist'] = [
            fTop.Get('UpMTWSystHist'),
            fTop.Get('DnMTWSystHist')]
        self.systDictTopW['TopWPuritySystHist'] = [
            fTop.Get('UpPuritySystHist'), 
            fTop.Get('DnPuritySystHist')]
        self.systDictTopW['TopWMuIsoSystHist'] = [
            fTop.Get('UpMuIsoSystHist'),
            fTop.Get('DnMuIsoSystHist')]
        self.systDictTopW['TopWMuRecoSystHist'] = [
            fTop.Get('UpMuRecoSystHist'),
            fTop.Get('DnMuRecoSystHist')]
        self.systDictTopW['TopWElecIsoSystHist'] = [
            fTop.Get('UpElecIsoSystHist'),
            fTop.Get('DnElecIsoSystHist')]
        self.systDictTopW['TopWElecRecoSystHist'] = [
            fTop.Get('UpElecRecoSystHist'),
            fTop.Get('DnElecRecoSystHist')]
        self.systDictTopW['TopWDiLepContributionSystHist'] = [
            fTop.Get('UpDiLepContributionSystHist'),
            fTop.Get('DnDiLepContributionSystHist')]
        self.systDictTopW['TopWLepAccSystHist'] = [
            fTop.Get('UpLepAccSystHist'),
            fTop.Get('DnLepAccSystHist')]
        self.systDictTopW['TopWHadTauNonClosure'] = [
            fTop.Get('hadTauClosure')]
        self.systDictTopW['TopWnonClosure'] = [nonClosureFactorSyst]


        ## Norm and NStat uncertainties 
        self.systDictTopW['TopWNCR'] = self.getStatSystHists('TopW')
        self.systDictTopW['TopWNorm'] = [self.getNormSystHist('TopW')]

        ## opened external file so we need to change the
        ## working directory back to 0
        for Key in self.systDictTopW:
            for hist in self.systDictTopW[Key]:
                hist.SetDirectory(0)

    def setSystHistsQCD(self):
        """ Computes the full QCD prediction with systematic uncertainties
        will run setPred() if prediction dictionary is empty
        """

        ## check if need to run prediction first
        if 'QCD' not in self.predDict.keys():
            self.setPred()

        dataSample = 'qcdIDP'
        if self.doData:
            dataSample = 'sigIDP'

        fQCD = ROOT.TFile.Open('qcdSystHists.root','read')
            
        ## take 50% uncertainty on non closure multiplicative factor
        ## REMOVED!!
        nonClosureFactor = fQCD.Get('nonClosureFactorQCD')
        nonClosureFactorSyst = nonClosureFactor.Clone()
        for binIter in range(1,nonClosureFactorSyst.GetNbinsX()+1):
            nonClosureFactorSyst.SetBinContent(
                binIter,(nonClosureFactorSyst.GetBinContent(binIter)-1)*0.5+1)

        ## QCD double ratio
        cuts = self.htCut+self.hdpCut+self.lpfCaloCut
        qcdDataNum = RA2b.getDist(dataSample,self.varPrefix+'Sig[0]',
                                  distRange=self.distRange,nBins=self.BinsDR,
                                  extraCuts=str(cuts))
        qcdSimNum = RA2b.getDist('qcdIDP',self.varPrefix+'Sig[0]',
                                  distRange=self.distRange,nBins=self.BinsDR,
                                  extraCuts=str(cuts))

        cuts = self.htCut+self.ldpCut+self.lpfCaloCut
        qcdDataDen = RA2b.getDist(dataSample,self.varPrefix+'Sig[0]',
                                  distRange=self.distRange,nBins=self.BinsDR,
                                  extraCuts=str(cuts))
        qcdSimDen = RA2b.getDist('qcdIDP',self.varPrefix+'Sig[0]',
                                  distRange=self.distRange,nBins=self.BinsDR,
                                  extraCuts=str(cuts))

        dataRatio = qcdDataNum.Clone()
        dataRatio.Divide(qcdDataDen)
        
        simRatio = qcdSimNum.Clone()
        simRatio.Divide(qcdSimDen)

        doubleRatio = dataRatio.Clone()
        doubleRatio.Divide(simRatio)

        drGraph = ROOT.TGraphErrors(doubleRatio)
        Qdr = RA2b.getDoubleRatioPlot([drGraph],isQCD=True)

        drUpSyst = self.predDict['QCD'].Clone()
        drDnSyst = self.predDict['QCD'].Clone()
        for binIter in range(1,self.Bins+1):
            drUpSyst.SetBinContent(binIter,1.+Qdr[1]['Sig[0]'][binIter-1][0])
            drDnSyst.SetBinContent(binIter,1.-Qdr[1]['Sig[0]'][binIter-1][1])
        self.systDictQCD['QCDdr'] = [drUpSyst,drDnSyst]
        
        ## MC stat from HDP/LDP correction hist
        corrStat = self.qcdCorrHist['Sig[0]'].Clone()
        for binIter in range(1,self.Bins+1):
            corrStat.SetBinContent(binIter,
                            1+self.qcdCorrHist['Sig[0]'].GetBinError(binIter))
            corrStat.SetBinError(binIter,0)
        self.systDictQCD['QCDmcStat'] = [corrStat]

        ## btag SFs
        btagUpSyst = self.predDict['QCD'].Clone()
        btagDnSyst = self.predDict['QCD'].Clone()
        for binIter in range(1,self.Bins+1):
            bMax = max(self.qcdCorrHist['Sig[0]'].GetBinContent(binIter),
                       self.qcdCorrHist['Sig[1]'].GetBinContent(binIter),
                       self.qcdCorrHist['Sig[2]'].GetBinContent(binIter))
            bMin = min(self.qcdCorrHist['Sig[0]'].GetBinContent(binIter),
                       self.qcdCorrHist['Sig[1]'].GetBinContent(binIter),
                       self.qcdCorrHist['Sig[2]'].GetBinContent(binIter))
            upSyst = bMax-self.qcdCorrHist['Sig[0]'].GetBinContent(binIter)
            upSyst*=1./self.qcdCorrHist['Sig[0]'].GetBinContent(binIter)
            upSyst+=1.
            dnSyst = bMin-self.qcdCorrHist['Sig[0]'].GetBinContent(binIter)
            dnSyst*=1./self.qcdCorrHist['Sig[0]'].GetBinContent(binIter)
            dnSyst+=1.

            btagUpSyst.SetBinContent(binIter,upSyst)
            btagUpSyst.SetBinError(binIter,0)
            btagDnSyst.SetBinContent(binIter,dnSyst)
            btagDnSyst.SetBinError(binIter,0)
        self.systDictQCD['QCDbtag'] = [btagUpSyst,btagDnSyst]

        ## Contamination symmetric uncert
        purSyst = self.predDict['QCD'].Clone()
        contam = self.qcdPurDict['Sig[0]'][0].Clone()
        ## add back half of contamination
        contam.Scale(0.5)
        purSyst.Add(contam)

        purSyst.Add(self.predDict['QCD'],-1)
        purSyst.Divide(self.predDict['QCD'])
        for binIter in range(1,self.Bins+1):
            purSyst.SetBinContent(binIter,1.+purSyst.GetBinContent(binIter))
            purSyst.SetBinError(binIter,0)
        self.systDictQCD['QCDpurity'] = [purSyst]

        ## Norm and NStat uncertainties 
        self.systDictQCD['QCDNCR'] = self.getStatSystHists('QCD')
        self.systDictQCD['QCDNorm'] = [self.getNormSystHist('QCD')]

        ## opened external file so we need to change the
        ## working directory back to 0
        for Key in self.systDictQCD:
            for hist in self.systDictQCD[Key]:
                hist.SetDirectory(0)
        
    def getSignalHistogram(self, sample, doSF=0, doISR=True, setWeight=None,
                           extraWeight='TrigWeight', applyEffs=True,
                           JE='', extraCuts=None, cuts=None):
        """ Returns signal histogram in signal-like boosted decision tree 
        output. Use this to compute SUSY yields and compute variation for 
        systematic uncertainties
        Default parameters are no systematic variations. 
        """

        ## if doSF<0, then do not apply btag SFs
        sampleSuffix = ''
        ## doSF==0 means apply btag SF central values
        ## doSF>0 means apply btag SF variations
        if doSF>=0:
            sampleSuffix = '['+str(doSF)+']'

        if cuts==None:
            cuts = str(self.htCut+self.hdpCut+self.hpfCaloCut)

        hist = RA2b.getDist(sample+JE,
                            self.varPrefix+'Sig'+sampleSuffix, setWeight=None,
                            extraCuts=str(cuts), distRange=self.distRange,
                            nBins=self.Bins, applyISRWeight=doISR,
                            extraWeight=extraWeight, applyEffs=applyEffs)
        return hist

    def setSignalHistogram(self, sample):
        """ Stores default SUSY histogram for a given sample
        """
        self.signalHist = self.getSignalHistogram(sample)

    def getSystFromVarHists(self,UP,DN,CV=None):
        """ Function to compute systematic from a list of histograms.

        If CV is none, CV will default to the average of the two histograms.
        If you want asymmetric uncertainties, you should give self.signalHist
        as the third input parameter. Otherwise, uncertainties will be 
        symmetric. 
        """


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
                upSyst.SetBinContent(binIter,1+(max(cv,up,dn)-cv)/cv)
                dnSyst.SetBinContent(binIter,1-(cv-min(cv,up,dn))/cv)
            else:
                upSyst.SetBinContent(binIter,1)
                dnSyst.SetBinContent(binIter,1)

        return [upSyst,dnSyst]

    def setBTagSystHists(self, sample):
        """ 6 (up and down) systematic uncertainties based on the 
        btag, ctag, and light flavor jet mistagging.
        There are two sets:(both fullsim and fastsim)
        """

        ## interpretations on FastSim only
        ## only other sample requiring btag Systs is Zinv
        if('fast' in sample):
            suffix = 'Signal'
        else:
            suffix = 'Zinv'

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
            self.setSignalHistogram(sample)
            
        ## loop over each systematic from the indexDict
        for syst in indexDict:
            ## need temp list to store variation histograms
            tmpHistList = []
            for variation in indexDict[syst]:
                tmpHistList.append(self.getSignalHistogram(sample, 
                                                           doSF=variation))
                    
            self.sigSystDict[str('Signal')+syst] = self.getSystFromVarHists(
                tmpHistList[0],
                tmpHistList[1],
                self.signalHist)
                
    def setScaleSystHists(self, sample):
        """ Renomalization and factorization scale uncertainties
        """
        
        ## first set central value for comparison
        if sample not in self.signalHist.GetName():
            self.setSignalHistogram(sample)


        ## Eight different renomralization and factorization scale variations
        scaleHists = []
        for scaleIter in range(8):
            scaleHists.append(
                self.getSignalHistogram(sample,
                                        extraWeight='(TrigWeight*ScaleWeights'
                                   '['+str(scaleIter)+'])'))
            ## normalize to remove unphysical normalization differences
            scaleHists[scaleIter].Scale(
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

        self.sigSystDict['SignalScale'] = self.getSystFromVarHists(varUpHist,
                                                            varDnHist,
                                                            self.signalHist)
    def setISRSystHists(self, sample):
        """ Initial state radiation systematics 
        """

        ## first set central value for comparison
        if sample not in self.signalHist.GetName():
            self.setSignalHistogram(sample)
        
        ## get histograms with up and down ISR weights applied
        varUpHist = self.getSignalHistogram(sample, doISR=False, 
                                            extraWeight='(TrigWeight*ISRup)')
        varDnHist = self.getSignalHistogram(sample, doISR=False, 
                                            extraWeight='(TrigWeight*ISRdn)')

        self.sigSystDict['SignalISR'] = self.getSystFromVarHists(varUpHist,
                                                            varDnHist,
                                                            self.signalHist)

    def setTrigSystHists(self, sample):
        """ Trigger efficiency uncertainties

        Bayesian Neural Net regression analysis 
        to determine the HT and MHT dependence 
        of the trigger efficiency
        """

        ## first set central value for comparison
        if sample not in self.signalHist.GetName():
            self.setSignalHistogram(sample)

        ## get up and down variations
        varUpHist = self.getSignalHistogram(sample, extraWeight='TrigWeightUp')
        varDnHist = self.getSignalHistogram(sample, extraWeight='TrigWeightDn')

        self.sigSystDict['SignalTrig'] =  self.getSystFromVarHists(varUpHist,
                                                             varDnHist,
                                                             self.signalHist)

    def setJetEnergySystHists(self, sample, JEtype):
        """
        For both jet energy correction and jet energy resolution systs.
        Using separate sample files, so things will speed up if 
        we don't apply all data/mc corrections and just take deviation.
        """

        ## get up and down variations
        varUpHist = self.getSignalHistogram(sample, JE=JEtype+'up')
        varDnHist = self.getSignalHistogram(sample, JE=JEtype+'down')

        ## set JE syst using average as CV (only have "signal" trees)
        self.sigSystDict[str('Signal')+JEtype] = self.getSystFromVarHists(
            varUpHist,
            varDnHist)

    def setJetIDSystHist(self):
        """ JetID doesn't work for fastsim so we use the recommented 0.99
        scale factor with 1% uncertainty
        """

        ## clone signalHist to ensure consistent binning
        systHist = self.signalHist.Clone()

        ## flat 1% uncertainty
        idError = 0.01

        ## fill the histogram
        for i in range(1,self.Bins+1):
            systHist.SetBinContent(i,1+idError)

        self.sigSystDict['SignalJetID'] = [systHist]

    def setLumiSystHist(self):
        """ Luminosity uncertainty from the lumi group
        """
        ## clone signalHist to ensure consistent binning
        systHist = self.signalHist.Clone()

        ## flat 2.5% uncertainty from lumi group
        lumiError = 0.025

        ## fill the histogram
        for i in range(1,self.Bins+1):
            systHist.SetBinContent(i,1+lumiError)

        self.sigSystDict['SignalLumi'] = [systHist]

    def setSignalSystHists(self, sample):
        """ Set all systematic uncertainties for signal. 
        List:
        btagSFs(6), ISR, Scales, Trig, JECs, JERs, JetID, Lumi
        still need PU reweighting
        """

        ## first get b-tag SF central value for comparison
        if sample not in self.signalHist.GetName():
            self.setSignalHistogram(sample)

        ## btag systematic uncertainties
        self.setBTagSystHists(sample)

        ## Initial state radiation syst
        self.setISRSystHists(sample)

        ## renorm and factorization scales
        self.setScaleSystHists(sample)

        ## Trigger systematic from bayesian neural network
        self.setTrigSystHists(sample)

        ## Jet energy correction syst
        self.setJetEnergySystHists(sample,'JEC')

        ## Jet energy resolution syst
        self.setJetEnergySystHists(sample,'JER')

        ## Fastsim Jet ID syst
        self.setJetIDSystHist()

        ## Luminosity syst
        self.setLumiSystHist()

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
        cuts = self.ldpCut+self.htCut+self.hpfCaloCut
        qcdContam = self.getSignalHistogram(sample, extraCuts=str(cuts))
                        
        ## Scale qcd contamination by qcd CR scaling
        qcdContam.Multiply(self.systDictQCD['NCR'][1])
                
        ## Subtract off contamination
        self.signalHist.Add(topContam,-1)
        self.signalHist.Add(qcdContam,-1)

    def getGraphFromHists(self, hist, systDict):
        """
        Get graph with asymmetrical errors. 
        Default is to add syst errors in quadrature.
        """

        ## get central values from hist
        graph = ROOT.TGraphAsymmErrors(hist)
        
        ## loop over bins to set errors from systDict
        for binIter in range(hist.GetNbinsX()+1):

            ## add up in quadruature the fractional uncertainties
            upSystTot = 0.
            dnSystTot = 0.
            for systKey in systDict:
                
                ## stat error is a pure number, 
                ## using no.4 from: 
                ## www-cdf.fnal.gov/physics/statistics/notes/pois_eb.txt
                if systKey.endswith('NCR'):
                    upSystTot = math.sqrt(upSystTot**2+((0.5+ROOT.TMath.Sqrt(
                        systDict[systKey][0].GetBinContent(binIter)+0.25))/max(
                            systDict[systKey][0].GetBinContent(binIter),1))**2)
                    dnSystTot = math.sqrt(dnSystTot**2+((-0.5+ROOT.TMath.Sqrt(
                        systDict[systKey][0].GetBinContent(binIter)+0.25))/max(
                            systDict[systKey][0].GetBinContent(binIter),1))**2)
                elif len(systDict[systKey])==1: ## symmetrical systematic
                    upSystTot = math.sqrt(
                        upSystTot**2
                        +(systDict[systKey][0].GetBinContent(binIter)-1)**2)
                    dnSystTot = math.sqrt(
                        dnSystTot**2
                        +(systDict[systKey][0].GetBinContent(binIter)-1)**2)
                        
                else: ## asymmetrical systematic
                    upSystTot = math.sqrt(
                        upSystTot**2
                        +(systDict[systKey][0].GetBinContent(binIter)-1)**2)
                    dnSystTot = math.sqrt(
                        dnSystTot**2
                        +(systDict[systKey][1].GetBinContent(binIter)-1)**2)
                        
            upVal = hist.GetBinContent(binIter)
            if upVal==0:
                upVal=1
            graph.SetPointEYhigh(binIter-1,upSystTot*upVal)
            graph.SetPointEYlow(binIter-1,dnSystTot*hist.GetBinContent(binIter))

        return graph

    def getLine(self, lineList):
        """ Helper function to format the lines of the datacard
        """

        fullKeyList = (self.sigSystDict.keys()+self.systDictQCD.keys()
                       +self.systDictTopW.keys()+self.zSystDict.keys())

        blankSpace = ' '*(max([len(Key) for Key in fullKeyList])+5)

        row = ''
        for cell in lineList:
            row+=str(cell)+blankSpace[len(str(cell)):]

        row+='\n'
        
        return row

    def makeDataCard(self, sample, doSymmetric=True):
        """ Makes one datacard for a given signal sample
        """

        ## compute signal yield and subtract off contamination
        self.setSignalSystHists(sample)
        self.subtractSignalContamination(sample)

        # collect Zinv and signal predictions in predDict
        self.predDict['Zinv'] = self.zPredDict['Sig[0]']
        self.predDict['Signal'] = self.signalHist

        ## get root e.g. T1tttt, T1bbbb, T2tt, ...
        nRoot = 8-2*int(sample[1])
        Root = sample[:nRoot]
        
        ## actual datacard (txt file)
        card = open(Root+"/"+sample+'.txt', "w") 

        ## label top of file, first with sample mass parameters
        header ="# SUSY model: "
        header += sample.replace('_',' mGl=',1).replace('_',' mLSP=',1)[:-8]
        header += '\n'

        imax = self.nSigBins ## num channels
        jmax = 3 ## num backgrounds QCD + Zinv +topW
        ## num nuisance parameters use '*' for automatic counting
        kmax = '*'

        ## labels for samples
        processes = ['Signal','Zinv','TopW','QCD']

        ## list of systematics which are fully correlated
        corrSysts = ['SignalLumi', 'SignalIsoTrk', 'SignalJetID', 
                     'SignalPileUp', 'SignalScale', 'SignalISR',
                     'TopWNorm', 'TopWnonClosure', 'QCDNorm', 'ZinvNorm']
        corrSuffix = '_Cor'

        ## empty dict to add nuisances 
        fullNuisanceDict = {}

        ## first Bin (where signal region begins)
        fB = self.Bins-self.nSigBins+1
        ## signal Bins
        sB = range(fB,self.Bins+1)

        ## bin number and observation in signal region
        nBins = ['bin','','']
        nObs = ['observation','','']
        nBins2 = ['bin','','']
        process1 = ['process','','']+processes*self.nSigBins
        process2 = ['process','','']+[0,1,2,3]*self.nSigBins
        rate = ['rate','','']
        for iB in sB:
            nBins.append(iB)
            nObs.append(int(self.dataHist.GetBinContent(iB)))
            for jm in range(jmax):
                nBins.append('')
                nObs.append('')
            for proc in processes:
                rate.append(round(self.predDict[proc].GetBinContent(iB),6))
            nBins2+=[iB]*4
            for sigSyst in self.sigSystDict.keys():
                if sigSyst in corrSysts:
                    if iB!=sB[0]:
                        continue
                    sigLabel = sigSyst+'_Cor'
                    fullNuisanceDict[sigLabel] = [sigLabel,'lnN']
                    if len(sigSystDict[sigLabel])==1:
                        fullNuisanceDict[sigLabel].append(
                            round(sigSystDict[sigLabel][0],6))
                    else:
                        fullNuisanceDict[sigLabel].append(
                            str(round(sigSystDict[sigLabel][1],6))+'/'+
                            str(round(sigSystDict[sigLabel][1],6)))
                    ## add dashes
                    fullNuisanceDict[sigLabel]+='-'*(self.nSigBins*(jmax+1)-1)
                if 'NCR' in sigSyst:
                    sigLabel = sigSyst+'_'+str(iB)
                    fullNuisanceDict[sigLabel] = [sigLabel,'gmN']
                    #fullNuisanceDict[sigLabel].append(
                    ## add dashes
                    #fullNuisanceDict[sigLabel] = 
        ## Number of signal events

        card.write(header)
        card.write("------------\n")
        card.write('imax '+str(imax)+' number of channels\n')
        card.write('jmax '+str(jmax)+' number of backgrounds\n')
        card.write('kmax '+str(kmax)+' number of nuisance parameters\n')
        card.write("------------\n")
        card.write(self.getLine(nBins))
        card.write(self.getLine(nObs))
        card.write("------------\n")
        card.write(self.getLine(nBins2))
        card.write(self.getLine(process1))
        card.write(self.getLine(process2))
        card.write(self.getLine(rate))
        ## nuisances are here
        for iB in sB:
            for proc in processes:
                for Syst in fullNuisanceDict.keys():
                    ## group similar processes together
                    if proc not in Syst:
                        continue
                    ## do correlated systs here
                    if (Syst.endswith(corrSuffix) and iB==sB[0]):
                        card.write(self.getLine(fullNuisanceDict[Syst]))
                        continue
                    ## add uncorrelated syst here
                    if Syst.endswith(str(iB)):
                        card.write(self.getLine(fullNuisanceDict[Syst]))
                                            
        card.write("------------\n")
        card.close()

    def makeAllCards(self, sampleClass):
        """ For a given model class (e.g. T1bbbb), compute the full
        set of datacards for the entire mass plane
        """

        ## Get full list of files available
        listOfFiles = os.listdir('/media/hdd/work/data/lpcTrees'+
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

            self.makeDataCard(sample)

