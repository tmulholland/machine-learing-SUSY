"""
**Main machine learning classifier code**
This class uses ROOT TMVA to classify
signal events from QCD and top/W

classifier's available:
BDTG - Boosted decision tree with grad boosting
MLP - Multi-layer perceptron (artificial neural net)
FDA_GA - function discriminant analysis with GA minimizer
PDEFoam - likelihood estimator using self-adapting phase-space binning
"""
import ROOT
import RA2b
import os.path
import time
import numpy

class classifier(object):

    def __init__(self):
        
        ## Set classifer options to use
        ## default is boosted decision tree only
        self.useMethod = {
            'BDTG'   : 1, # boosted decision tree with grad boosting
            'MLP'    : 0, # multilayer perceptron (ANN)
            'FDA_GA' : 0, # function discriminant analysis with GA minimizer
            'PDEFoam': 0, # likelihood estimator using self-adapting phase-space binning
        }

        ## Training and testing data file name
        self.outFile = None
        
        ## set of features for classifier to learn on
        self.variableSet = [
            'DeltaPhi1',
            'DeltaPhi2',
            'DeltaPhi3',
            'DeltaPhi4',
            'MHT',
            'NJets',
            'HT',
            'BTags',
        ]

        ## Signal monte-carlo samples 
        self.signalSample = 'FullSimIDP' ## All FullSim signal benchmark points


        ## Background monte-carlo samples 
        self.bkgSet = [
            # 'qcdIDP', # qcd multijet mismeasurement
            # 'topWIDP', # combine top and W+jets samples
            # 'zinvIDP', # zinv
            'allbkgIDP'
        ]

        ## parameters tuned to optimize efficeincy
        ## and to prevent overtraining
        self.nTrees = [3000] 
        self.maxDepth = [4] 
        self.minNodeSize = '2%'    

        ## define cuts different than baseline RA2b
        self.mhtCut = True
        self.extraCuts = ('HT>800&&PFCaloMETRatio>0.9&&'
                          'DeltaPhi1>0.1&&DeltaPhi2>0.1&&'
                          'DeltaPhi3>0.1&&DeltaPhi4>0.1')

                          
        self.dPhiCut = 'none'

    
    ## train and evaluate training/testing samples
    def trainAndTest(self):

        ## set output file name if undefined
        if self.outFile==None: 

            self.outFile = 'NT'+str(
                len(self.nTrees))+'_MD'+str(len(self.maxDepth))
            self.outFile+= '_MNS'+self.minNodeSize[:-1]+'_'+self.signalSample
            self.outFile+= '_'+time.strftime('%m-%d')+'_'

            ## prevent overwriting 
            fileIter = 0
            while os.path.isfile(self.outFile+str(fileIter)+'.root'):
                fileIter+=1
        
            ## transform to ROOT.TFile type
            self.outFile = ROOT.TFile(self.outFile+str(fileIter)+'.root',
                                      'RECREATE')


        ## Declare the TMVA Factory object
        factory = ROOT.TMVA.Factory( 'TMVAMulticlass_'+self.signalSample, 
                                     self.outFile,
                                     '!V:!Silent:Color:!DrawProgressBar:'
                                     'Transformations=I;D;P;G,D:'
                                     'AnalysisType=multiclass' )

        ## Tell the Factory which features to learn with
        for variable in self.variableSet:
            factory.AddVariable(variable)


        ## declare dictionaries to store sample related info
        fileList = {}; TFileList = {}; TTreeList = {}
        ## Load up the training and testing samples (1 sig + all bkgs)
        for sample in [self.signalSample]+self.bkgSet:

            ## get file list from RA2b module
            fileList[sample] = RA2b.getFileList(sample)
            ## define dictionaries of file lists per sample
            TFileList[sample] = []
            TTreeList[sample] = []

            ## loop over file lists, getting sample weights,
            ## and adding samples to the factory
            for i in range(len(fileList[sample])):
                w = RA2b.getTreeWeight(fileList[sample][i])
                TFileList[sample].append(ROOT.TFile(fileList[sample][i]))
                TTreeList[sample].append(TFileList[sample][i].Get('tree'))
                factory.AddTree(TTreeList[sample][i],sample,w)


        ## get the cuts to apply before training
        cuts = RA2b.getCuts('sig',dphiCut=self.dPhiCut,applyMHTCut=self.mhtCut,
                            extraCuts=self.extraCuts)

        ## Prepare the factory for training and testing
        ## by normalizing to equal number of events
        factory.PrepareTrainingAndTestTree( cuts, 
                                            'SplitMode=Random:'
                                            'NormMode=EqualNumEvents:!V')

        # Boosted Decision Trees with gradient boosting
        if self.useMethod['BDTG']: 
            for nT in self.nTrees:
                for maxD in self.maxDepth:
                    factory.BookMethod( ROOT.TMVA.Types.kBDT, 
                                        'BDTG_NTrees'+str(nT)
                                        +'_MaxDepth'+str(maxD), 
                                        '!H:!V:NTrees='+str(nT)
                                        +':BoostType=Grad:Shrinkage=0.10:'
                                        'MinNodeSize='+self.minNodeSize
                                        +':GradBaggingFraction=0.50:'
                                        'nCuts=20:MaxDepth='+str(maxD))

        # Multi-layer perceptron (ANN)
        if self.useMethod['MLP']: 
            factory.BookMethod('MLP', 'MLP','!H:!V:NeuronType=tanh:'
                               'NCycles=1000:HiddenLayers=N+5,5:'
                               'TestRate=5:EstimatorType=MSE')

        # functional discriminant with GA minimizer
        if self.useMethod['FDA_GA']: 
            factory.BookMethod( ROOT.TMVA.Types.kFDA, 'FDA_GA', 
                                'H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:'
                                'ParRanges=(-1,1);(-10,10);'
                                '(-10,10);(-10,10);(-10,10):'
                                'FitMethod=GA:PopSize=300:Cycles=3:'
                                'Steps=20:Trim=True:SaveBestGen=1' )

        # PDE-Foam approach
        if self.useMethod['PDEFoam']: 
            factory.BookMethod( ROOT.TMVA.Types.kPDEFoam, 'PDEFoam', 
                                '!H:!V:TailCut=0.001:VolFrac=0.0666:'
                                'nActiveCells=500:nSampl=2000:nBin=5:'
                                'Nmin=100:Kernel=None:Compress=T' )

        ## train, test, and evaluate
        factory.TrainAllMethods()
        factory.TestAllMethods()
        factory.EvaluateAllMethods()

        ## Delete factory to prevent memory leak
        factory.Delete()
    
        ## close file before finishing script
        self.outFile.Close()

    def ROC(self, var='FullSimIDP'):

        ## define dictionaries to store ROC graphs
        ## and integrals of ROC curves
        graph = {}
        rocArea = {}

        ## Convert outFile to readable inFile
        inFile = ROOT.TFile(self.outFile.GetName())

        ## Get training and testing data samples
        TestTree = inFile.Get('TestTree')
        TrainTree = inFile.Get('TrainTree')

        ## Define hist for cloning
        templateHist = ROOT.TH1F('templateHist','templateHist',100,0.0,1.0)

        ## make ROC curves from all backgrounds
        ## and parameters
        for bkg in self.bkgSet:
            for NTrees in self.nTrees:
                for MaxDepth in self.maxDepth:

                    ##
                    ## Begin training computation
                    ##

                    ## Define unique key for storing in dictionary
                    Key = bkg+'-T'+str(NTrees)+'-D'+str(MaxDepth)

                    ## Declare training histograms
                    sigTrain = templateHist.Clone()
                    sigTrain.SetName('sigTrain')
                    bkgTrain = templateHist.Clone()
                    bkgTrain.SetName('bkgTrain')

                    ## Fill training histograms
                    TrainTree.Project(sigTrain.GetName(),
                                      'BDTG_NTrees'+str(NTrees)
                                      +'_MaxDepth'+str(MaxDepth)+'.'+var,
                                      'classID==0')
                    TrainTree.Project(bkgTrain.GetName(),
                                      'BDTG_NTrees'+str(NTrees)+'_MaxDepth'
                                      +str(MaxDepth)+'.'+var,
                                      'classID=='+str(self.bkgSet.index(bkg)+1))

                    print sigTrain.Integral()

                    ## make list of signal efficiencies 
                    sigEff = [1.]
                    for i in reversed(range(1,sigTrain.GetNbinsX()+1)):
                        sigEff.append(
                            sigEff[sigTrain.GetNbinsX()-i]
                            -sigTrain.GetBinContent(i)/sigTrain.Integral())
                    X = numpy.array(list(sigEff))
                    for i in range(len(X)):
                        X[i] = 1-X[i]
            
                    ## make list of background efficiencies
                    bkgRej = [1.]
                    for i in reversed(range(1,bkgTrain.GetNbinsX()+1)):
                        bkgRej.append(
                            bkgRej[bkgTrain.GetNbinsX()-i]
                            -bkgTrain.GetBinContent(i)/bkgTrain.Integral())
                    Y = numpy.array(list(bkgRej))

                    ## define ROC graph and store it in graph dictionary
                    graph[Key+'-train'] = ROOT.TGraph(
                        sigTrain.GetNbinsX()+1,X,Y)

                    ## Set point (0, 0) so integral covers 
                    ## full geometric range of ROC curve
                    graph[Key+'-train'].SetPoint(
                        graph[Key+'-train'].GetN(), 0,0)

                    ## Find area of ROC curve
                    rocArea[Key+'-train'] = graph[Key+'-train'].Integral()

                    ##
                    ## Begin testing computation
                    ##

                    ## Declare testing histograms
                    sigTest = templateHist.Clone()
                    sigTest.SetName('sigTest')
                    bkgTest = templateHist.Clone()
                    bkgTest.SetName('bkgTest')
                    
                    ## Fill testing histograms
                    TestTree.Project(sigTest.GetName(),
                                     'BDTG_NTrees'+str(NTrees)
                                     +'_MaxDepth'+str(MaxDepth)+'.'+var,
                                     'classID==0')

                    TestTree.Project(bkgTest.GetName(),
                                     'BDTG_NTrees'+str(NTrees)+'_MaxDepth'
                                     +str(MaxDepth)+'.'+var,
                                     'classID=='+str(self.bkgSet.index(bkg)+1))

                    ## make list of signal efficiencies 
                    sigEff = [1.]
                    for i in reversed(range(1,sigTest.GetNbinsX()+1)):
                        sigEff.append(
                            sigEff[sigTest.GetNbinsX()-i]
                            -sigTest.GetBinContent(i)/sigTest.Integral())
                    X = numpy.array(list(sigEff))
                    for i in range(len(X)):
                        X[i] = 1-X[i]
                        
                    ## make list of background efficiencies
                    bkgRej = [1.]
                    for i in reversed(range(1,bkgTest.GetNbinsX()+1)):
                        bkgRej.append(
                            bkgRej[bkgTest.GetNbinsX()-i]
                            -bkgTest.GetBinContent(i)/bkgTest.Integral())
                    Y = numpy.array(list(bkgRej))
                    
                    ## define ROC graph and store it in graph dictionary
                    graph[Key+'-test']=ROOT.TGraph(sigTest.GetNbinsX()+1,X,Y)
            
                    ## Set point (0, 0) so integral covers 
                    ## full geometric range of ROC curve
                    graph[Key+'-test'].SetPoint(graph[Key+'-test'].GetN(), 0,0)
                    
                    ## Find area of ROC curve
                    rocArea[Key+'-test'] = graph[Key+'-test'].Integral()

        return [graph, rocArea]
