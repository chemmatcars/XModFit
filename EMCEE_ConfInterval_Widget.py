from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QTreeWidgetItem, QWidget
from PyQt5.QtGui import QTextCursor, QFont
from PyQt5 import uic
from PyQt5 import Qt
import sys
import tabulate
from xmodfit import XModFit
from collections import OrderedDict
import time
import numpy as np

class EMCEE_ConfInterval_Window(QMainWindow):
    def __init__(self, emcee_walker=10, emcee_steps=100, emcee_burn=1, emcee_thin=1, emcee_cores=1, backendFile=None, parent=None):
        QMainWindow.__init__(self, parent)
        uic.loadUi('./UI_Forms/MCMC_ConfInterval_Window.ui', self)
        self.setWindowTitle('MCMC Confidence Interval Calculator')
        self.menuBar().setNativeMenuBar(False)
        self.emcee_walker = emcee_walker
        self.emcee_steps = emcee_steps
        self.emcee_burn = emcee_burn
        self.emcee_thin = emcee_thin
        self.emcee_cores = emcee_cores
        self.backendFile = backendFile
        self.MCMCWalkerLineEdit.setText(str(self.emcee_walker))
        self.MCMCStepsLineEdit.setText(str(self.emcee_steps))
        self.MCMCBurnLineEdit.setText(str(self.emcee_burn))
        self.MCMCThinLineEdit.setText(str(self.emcee_thin))
        self.ParallelCoresLineEdit.setText(str(self.emcee_cores))
        self.ParallelCoresLineEdit.setEnabled(False)
        self.progressBar.setValue(0)
        self.xmodfit=XModFit()

        self.init_signals()


    def init_signals(self):
        self.actionOpen_MCMC_run_file.triggered.connect(self.openMCMCRunFile)
        self.actionSave_MCMC_run_file.triggered.connect(self.saveMCMCRunFile)
        self.actionQuit.triggered.connect(self.close)
        self.startSamplingPushButton.clicked.connect(self.start_emcee_sampling)
        self.MCMCWalkerLineEdit.returnPressed.connect(self.MCMCWalker_changed)
        self.saveConfIntervalPushButton.clicked.connect(self.saveParameterError)
        self.addUserDefinedParamPushButton.clicked.connect(lambda x:self.addMCMCUserDefinedParam(parname=None, expression=None))
        self.removeUserDefinedParamPushButton.clicked.connect(self.removeMCMCUserDefinedParam)
        self.saveUserDefinedParamPushButton.clicked.connect(self.saveMCMCUserDefinedParam)
        self.loadUserDefinedParamPushButton.clicked.connect(self.loadMCMCUserDefinedParam)
        self.userDefinedParamTreeWidget.itemDoubleClicked.connect(self.openMCMCUserDefinedParam)

    def openMCMCRunFile(self):

        self.xmodfit.loadParameters()

    def saveMCMCRunFile(self):
        pass


    def EnableUserDefinedParameterButtons(self,enable=False):
        self.addUserDefinedParamPushButton.setEnabled(enable)
        self.removeUserDefinedParamPushButton.setEnabled(enable)
        self.saveUserDefinedParamPushButton.setEnabled(enable)
        self.loadUserDefinedParamPushButton.setEnabled(enable)

    def openMCMCUserDefinedParam(self,item,column):
        txt=item.text(0)
        if ':chain:' not in txt:
            parname,expression=txt.split('=')
            self.addMCMCUserDefinedParam(parname=parname, expression=expression)

    def addMCMCUserDefinedParam(self, parname=None, expression=None):
        self.MCMCUserDefinedParamWidget = QWidget()
        self.MCMCUserDefinedParamWidget.setWindowModality(Qt.ApplicationModal)
        uic.loadUi('./UI_Forms/User_Defined_Param_Widget.ui', self.MCMCUserDefinedParamWidget)
        if parname is not None:
            self.MCMCUserDefinedParamWidget.parnameLineEdit.setText(parname)
            self.MCMCUserDefinedParamWidget.parnameLineEdit.setEnabled(False)
            new=False
        else:
            new=True
        if expression is not None:
            self.MCMCUserDefinedParamWidget.expressionTextEdit.setText(expression)
        availParams = self.fit.result.var_names
        self.MCMCUserDefinedParamWidget.availParamListWidget.addItems(availParams)

        self.MCMCUserDefinedParamWidget.addPushButton.clicked.connect(lambda x: self.acceptUserDefinedParams(new=new))
        self.MCMCUserDefinedParamWidget.cancelPushButton.clicked.connect(self.MCMCUserDefinedParamWidget.close)
        self.MCMCUserDefinedParamWidget.availParamListWidget.itemDoubleClicked.connect(self.appendUserDefinedExpression)
        self.MCMCUserDefinedParamWidget.show()

    def appendUserDefinedExpression(self,item):
        txt=item.text()
        self.MCMCUserDefinedParamWidget.expressionTextEdit.insertPlainText(txt)
        self.MCMCUserDefinedParamWidget.expressionTextEdit.moveCursor(QTextCursor.End)

    def acceptUserDefinedParams(self,new=True,parname=None,expression=None):
        if parname is None:
            parname=self.MCMCUserDefinedParamWidget.parnameLineEdit.text()
        if parname=='':
            QMessageBox.warning(self, 'Name Error', 'Please provide a parameter name.', QMessageBox.Ok)
            return
        elif parname in self.param_chain.keys() and new:
            QMessageBox.warning(self, 'Name Error', 'Please provide a parameter name which is not already used.', QMessageBox.Ok)
            return
        else:
            if expression is None:
                txt=self.MCMCUserDefinedParamWidget.expressionTextEdit.toPlainText()
            else:
                txt=expression
            if txt != 'None':
                try:
                    self.userDefinedParamTreeWidget.itemSelectionChanged.disconnect(
                        self.userDefinedParameterTreeSelectionChanged)
                except:
                    pass
                if new:
                    l1 = QTreeWidgetItem([parname+'='+txt])
                else:
                    self.userDefinedParamTreeWidget.currentItem().setText(0,parname+'='+txt)
                self.param_chain[parname] = {}
                for i in range(self.chain_shape[1]):
                    ttxt=txt[0:]
                    for name in self.fit.result.var_names:
                        ttxt=ttxt.replace(name,"self.param_chain['%s'][%d]"%(name,i))
                    try:
                        self.param_chain[parname][i]=eval(ttxt)
                        if new:
                            l1_child = QTreeWidgetItem(['%s:chain:%d' % (parname, i)])
                            l1.addChild(l1_child)
                            self.userDefinedParamTreeWidget.addTopLevelItem(l1)
                        if expression is None:
                            self.MCMCUserDefinedParamWidget.close()

                    except:
                        QMessageBox.warning(self, 'Expression Error',
                                            'Some problems in the expression\n' + traceback.format_exc(),
                                            QMessageBox.Ok)
                        return
                self.userDefinedParamTreeWidget.itemSelectionChanged.connect(
                    self.userDefinedParameterTreeSelectionChanged)

            else:
                self.MCMCUserDefinedParamWidget.close()

    def removeMCMCUserDefinedParam(self):
        try:
            self.userDefinedParamTreeWidget.itemSelectionChanged.disconnect()
        except:
            pass
        for item in self.userDefinedParamTreeWidget.selectedItems():
            parname=item.text(0).split('=')[0]
            del self.param_chain[parname]
            index=self.userDefinedParamTreeWidget.indexOfTopLevelItem(item)
            self.userDefinedParamTreeWidget.takeTopLevelItem(index)
        self.userDefinedParamTreeWidget.itemSelectionChanged.connect(self.userDefinedParameterTreeSelectionChanged)


    def saveMCMCUserDefinedParam(self):
        fname=QFileDialog.getSaveFileName(caption='Save User-Defined parameter expression as',filter='Expression files (*.expr)'
                                          ,directory=self.curDir)[0]
        if fname!='':
            if os.path.splitext(fname)[0]=='':
                fname=fname+'.expr'
        else:
            return
        fh=open(fname,'w')
        txt ='#Parameter Expressions saved on %s\n'%(time.asctime())
        txt += '#Category:%s\n' % self.curr_category
        txt += '#Function:%s\n' % self.funcListWidget.currentItem().text()
        root = self.userDefinedParamTreeWidget.invisibleRootItem()
        child_count=root.childCount()
        for i in range(child_count):
            txt += '%s\n'%root.child(i).text(0)
        fh.write(txt)
        fh.close()


    def loadMCMCUserDefinedParam(self):
        fname=QFileDialog.getOpenFileName(self,'Open Expression file',filter='Expression files (*.expr)'
                                          ,directory=self.curDir)[0]
        if fname!='':
            try:
                self.userDefinedParamTreeWidget.itemSelectionChanged.disconnect()
            except:
                pass
            #Removing the existing User-Defined parameters, if present
            root = self.userDefinedParamTreeWidget.invisibleRootItem()
            child_count = root.childCount()
            for i in range(child_count):
                parname, expression = root.child(i).text(0).split('=')
                del self.param_chain[parname]
            self.userDefinedParamTreeWidget.clear()

            fh=open(fname,'r')
            lines=fh.readlines()
            for line in lines:
                if line[0]=='#':
                    if 'Category' in line:
                        _,category=line[1:].strip().split(':')
                        if category!=self.curr_category:
                            QMessageBox.warning(self,'File Error','The expression file is not generated from the same category of functions as used here.'
                                                ,QMessageBox.Ok)
                            return
                    elif 'Function' in line:
                        _,func=line[1:].strip().split(':')
                        if func!=self.curr_module:
                            QMessageBox.warning(self, 'File Error',
                                                'The expression file is not generated from the same function as used here.'
                                                , QMessageBox.Ok)
                            return
                else:
                    parname,expression=line.strip().split('=')
                    self.acceptUserDefinedParams(new=True,parname=parname,expression=expression)
        self.userDefinedParamTreeWidget.itemSelectionChanged.connect(self.userDefinedParameterTreeSelectionChanged)




    def MCMCWalker_changed(self):
        self.reuseSamplerCheckBox.setCheckState(Qt.Unchecked)
        self.update_emcee_parameters()


    def update_emcee_parameters(self):
        self.emcee_walker=int(self.MCMCWalkerLineEdit.text())
        self.emcee_steps=int(self.MCMCStepsLineEdit.text())
        self.emcee_burn=int(self.MCMCBurnLineEdit.text())
        self.emcee_thin = int(self.MCMCThinLineEdit.text())
        if self.reuseSamplerCheckBox.isChecked():
            self.reuse_sampler=True
        else:
            self.reuse_sampler=False
        self.emcee_cores = int(self.ParallelCoresLineEdit.text())

    def start_emcee_sampling(self):
        try:
            self.parameterTreeWidget.itemSelectionChanged.disconnect()
        except:
            pass
        self.parameterTreeWidget.clear()
        self.chainMPLWidget.clear()
        self.correlationMPLWidget.clear()
        self.cornerPlotMPLWidget.clear()
        self.confIntervalTextEdit.clear()
        self.update_emcee_parameters()
        if not self.errorAvailable:
            self.emcee_frac=self.emcee_burn/self.emcee_steps
        self.doFit(fit_method='emcee', emcee_walker=self.emcee_walker, emcee_steps=self.emcee_steps,
                       emcee_cores=self.emcee_cores, reuse_sampler=self.reuse_sampler, emcee_burn=self.emcee_burn,
                   emcee_thin=self.emcee_thin,backendFile='mcmc_chains.h5')


    def conf_interv_status(self,params,iterations,residual,fit_scale):
        self.confIntervalStatus.setText(self.confIntervalStatus.text().split('\n')[0]+'\n\n {:^s} = {:10d}'.format('Iteration',iterations))
        QApplication.processEvents()

    def perform_post_sampling_tasks(self):
        self.progressBar.setValue(self.emcee_walker*self.emcee_steps)
        self.fitIterLabel.setText('Time left (hh:mm:ss): 00:00:00' )
        self.chain=self.fit.result.chain
        self.chain_shape=self.chain.shape
        self.param_chain=OrderedDict()
        for i,key in enumerate(self.fit.result.flatchain.keys()):
            l1=QTreeWidgetItem([key])
            self.param_chain[key]=OrderedDict()
            for j in range(self.chain_shape[1]):
                self.param_chain[key][j]=self.chain[:,j,i]
                l1_child=QTreeWidgetItem(['%s:chain:%d'%(key,j)])
                l1.addChild(l1_child)
            self.parameterTreeWidget.addTopLevelItem(l1)
        self.parameterTreeWidget.itemSelectionChanged.connect(self.parameterTreeSelectionChanged)

        #Calculating autocorrelation
        acor=OrderedDict()
        Nrows=len(self.param_chain.keys())
        self.correlationMPLWidget.clear()
        ax1 = self.correlationMPLWidget.fig.add_subplot(1, 1, 1)
        corr_time=[]
        acor_mcmc=self.fit.fitter.sampler.get_autocorr_time(quiet=True)
        for i,key in enumerate(self.param_chain.keys()):
            tcor=[]
            for ikey in self.param_chain[key].keys():
                tdata=self.param_chain[key][ikey]
                res=sm.tsa.acf(tdata,nlags=len(tdata),fft=True)
                tcor.append(res)
            tcor=np.array(tcor)
            acor[key]=np.mean(tcor,axis=0)
            ax1.plot(acor[key],'-',label='para=%s'%key)
            corr_time.append([key,acor_mcmc[i]])#np.sum(np.where(acor[key]>0,acor[key],0))])
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Autocorrelation')
        l=ax1.legend(loc='best')
        l.set_draggable(True)
        self.correlationMPLWidget.draw()
        self.corrTimeTextEdit.clear()
        self.corrTimeTextEdit.setFont(QFont("Courier", 10))
        corr_text = tabulate(corr_time, headers=['Parameter', 'Correlation-time (Steps)'], stralign='left',
                             numalign='left', tablefmt='simple')
        self.corrTimeTextEdit.append(corr_text)

        #Plotting Acceptance Ratio
        self.acceptFracMPLWidget.clear()
        ax2=self.acceptFracMPLWidget.fig.add_subplot(1,1,1)
        ax2.plot(self.fit.result.acceptance_fraction,'-')
        ax2.set_xlabel('Walkers')
        ax2.set_ylabel('Acceptance Ratio')
        self.acceptFracMPLWidget.draw()

        self.calcConfIntervPushButton.clicked.connect(self.cornerPlot)
        self.tabWidget.setCurrentIndex(2)
        self.reset_cornerplot=True

        #Calculating User-Defined parameters
        self.EnableUserDefinedParameterButtons(enable=True)
        root = self.userDefinedParamTreeWidget.invisibleRootItem()
        child_count = root.childCount()
        if child_count>0:
            for i in range(child_count):
                parname, expression = root.child(i).text(0).split('=')
                self.param_chain[parname]={}
                for j in range(self.chain_shape[1]):
                    ttxt = expression[0:]
                    for name in self.fit.result.var_names:
                        ttxt = ttxt.replace(name, "self.param_chain['%s'][%d]" % (name, j))
                    self.param_chain[parname][j] = eval(ttxt)

    def cornerPlot(self):
        percentile = self.percentileDoubleSpinBox.value()
        first = int(self.MCMCBurnLineEdit.text())
        if self.reset_cornerplot:
            self.cornerPlotMPLWidget.clear()
            names = self.fit.result.var_names#[name for name in self.fit.result.var_names if name != '__lnsigma']
            values = [self.fit.result.params[name].value for name in names]
            ndim = len(names)
            quantiles=[1.0-percentile/100,0.5,percentile/100]
            corner.corner(self.fit.result.flatchain[names][first:], labels=names, bins=50, levels=(percentile/100,),
                          truths=values, quantiles=quantiles, show_titles=True, title_fmt='.6f',
                          use_math_text=True, title_kwargs={'fontsize': 3 * 12 / ndim},
                          label_kwargs={'fontsize': 3 * 12 / ndim}, fig=self.cornerPlotMPLWidget.fig)
            for ax3 in self.cornerPlotMPLWidget.fig.get_axes():
                ax3.set_xlabel('')
                ax3.set_ylabel('')
                ax3.tick_params(axis='y', labelsize=3 * 12 / ndim, rotation=0)
                ax3.tick_params(axis='x', labelsize=3 * 12 / ndim)
            self.cornerPlotMPLWidget.draw()
            self.tabWidget.setCurrentIndex(4)
            self.reset_cornerplot=False
        self.calcMCMCerrorbars(burn=first,percentile=percentile)


    def calcMCMCerrorbars(self,burn=0,percentile=85):
        mesg = [['Parameters', 'Value(50%)', 'Left-error(%.3f%s)' % (100 - percentile,'%'), 'Right-error(%.3f%s)' % (percentile,'%')]]
        for key in self.param_chain.keys():
            for chain in self.param_chain[key].keys():
                try:
                    pardata=np.vstack((pardata,self.param_chain[key][chain][burn:]))
                except:
                    pardata=[self.param_chain[key][chain][burn:]]
            pardata=np.ndarray.flatten(pardata)
            errors=np.quantile(pardata,[(100-percentile)/100,50/100,percentile/100])
            mesg.append([key, errors[1], errors[1]-errors[0], errors[2]-errors[1]])
        self.confIntervalTextEdit.clear()
        self.confIntervalTextEdit.setFont(QFont("Courier", 10))
        txt = tabulate(mesg, headers='firstrow', stralign='left', numalign='left', tablefmt='simple')
        self.confIntervalTextEdit.append(txt)


    def parameterTreeSelectionChanged(self):
        self.chainMPLWidget.clear()
        chaindata={}
        for item in self.parameterTreeWidget.selectedItems():
            if ':chain:' in item.text(0):
                key,i=item.text(0).split(':chain:')
                try:
                    chaindata[key].append(int(i))
                except:
                    chaindata[key]=[int(i)]
        NRows = len(chaindata.keys())
        if NRows>0:
            ax={}
            firstkey=list(chaindata.keys())[0]
            for j,key in enumerate(chaindata.keys()):
                try:
                    ax[key]=self.chainMPLWidget.fig.add_subplot(NRows, 1, j+1, sharex=ax[firstkey])
                except:
                    ax[key] = self.chainMPLWidget.fig.add_subplot(NRows, 1, j+1)
                for i in chaindata[key]:
                    ax[key].plot(self.param_chain[key][i],'-')
                ax[key].set_xlabel('MC steps')
                ax[key].set_ylabel(key)
            self.chainMPLWidget.draw()
            self.tabWidget.setCurrentIndex(0)

    def userDefinedParameterTreeSelectionChanged(self):
        self.userDefinedChainMPLWidget.clear()
        chaindata={}
        for item in self.userDefinedParamTreeWidget.selectedItems():
            if ':chain:' in item.text(0):
                key,i=item.text(0).split(':chain:')
                try:
                    chaindata[key].append(int(i))
                except:
                    chaindata[key]=[int(i)]
        NRows = len(chaindata.keys())
        if NRows>0:
            ax={}
            firstkey=list(chaindata.keys())[0]
            for j,key in enumerate(chaindata.keys()):
                try:
                    ax[key]=self.userDefinedChainMPLWidget.fig.add_subplot(NRows, 1, j+1, sharex=ax[firstkey])
                except:
                    ax[key] = self.userDefinedChainMPLWidget.fig.add_subplot(NRows, 1, j+1)
                for i in chaindata[key]:
                    ax[key].plot(self.param_chain[key][i],'-')
                ax[key].set_xlabel('MC steps')
                ax[key].set_ylabel(key)
            self.userDefinedChainMPLWidget.draw()
            self.tabWidget.setCurrentIndex(1)

    def saveParameterError(self):
        fname=QFileDialog.getSaveFileName(caption='Save Parameter Errors as',filter='Parameter Error files (*.perr)',directory=self.curDir)[0]
        if os.path.splitext(fname)[1]=='':
            fname=fname+'.perr'
        text=self.confIntervalTextEdit.toPlainText()
        fh=open(fname,'w')
        fh.write('# File save on %s\n'%time.asctime())
        fh.write('# Error calculated using MCMC Method\n')
        fh.write(text)
        fh.close()



if __name__=='__main__':
    app=QApplication(sys.argv)
    try:
        fname=sys.argv[1]
    except:
        fname=None
    #data={'meta':{'a':1,'b':2},'data':pd.DataFrame({'x':arange(1000),'y':arange(1000),'y_err':arange(1000)})}
    w=EMCEE_ConfInterval_Window()
    w.show()
#    w.showFullScreen()
    sys.exit(app.exec_())
