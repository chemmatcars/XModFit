from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QTreeWidgetItem, QWidget
from PyQt5.QtGui import QTextCursor, QFont
from PyQt5 import uic
from PyQt5 import Qt
import sys
import tabulate
from importlib import import_module, reload
from collections import OrderedDict
import traceback
import time
import os
import numpy as np
import pandas as pd
import emcee
import ray

os.environ["OMP_NUM_THREADS"]="1"

from Fit_Routines import Fit

if ray.is_initialized():
    ray.shutdown()
ray.init(log_to_driver=False)


def combine_hdf(files, newfile, nwalkers, ndim, nsteps):
    if os.path.exists(newfile):
        os.remove(newfile)
    backend = emcee.backends.HDFBackend(newfile)
    backend.reset(len(files) * nwalkers, ndim)
    backend.grow(nsteps, None)
    with backend.open(mode='a') as f:
        g = f['mcmc']
        for i, file in enumerate(files):
            with h5py.File(file, 'r') as ft:
                gt = ft['mcmc']
                g['accepted'][i * nwalkers:(i + 1) * nwalkers] = gt['accepted'][:]
                g['chain'][:, i * nwalkers:(i + 1) * nwalkers, :] = gt['chain'][:, :, :]
                g['log_prob'][:, i * nwalkers:(i + 1) * nwalkers] = gt['log_prob'][:, :]
            os.remove(file)
        g.attrs["nwalkers"] = len(files) * nwalkers
        g.attrs["ndim"] = ndim
        g.attrs["has_blobs"] = False
        g.attrs["iteration"] = nsteps


def plot_chains(hdf_file):
    backend = emcee.backends.HDFBackend(hdf_file, read_only=False)
    flat_samples = backend.get_chain(discard=100, thin=1, flat=True)
    labels = list(params.keys())
    fig = corner.corner(flat_samples, labels=labels, truths=initial, quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, bins=100, title_fmt='.3f', plot_contours=True,
                        smooth=0.0, smooth1d=1.0)

# @ray.remote
def run_mcmc(filenum, file_prefix, nwalkers, nsteps, params, parminmax, x, y, yerr, save_steps):
    fit = Fit(Linear, params, x, y, yerr)
    initial = np.array([params[key].value for key in params.keys()])
    parmin = np.array([parminmax[key][0] for key in parminmax.keys()])
    parmax = np.array([parminmax[key][1] for key in parminmax.keys()])
    pos = np.random.uniform(parmin, parmax, (nwalkers, len(initial)))
    nwalkers, ndim = pos.shape
    state = emcee.state.State(pos)
    file = file_prefix + '_%d.h5' % filenum
    if os.path.exists(file):
        os.remove(file)
    backend = emcee.backends.HDFBackend(file)
    backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, fit.log_probability, args=[parminmax]
                                    , backend=None, parameter_names=list(params.keys()))
    for sample in sampler.sample(pos, iterations=nsteps, store=True):
        # rayvars.increment.remote(filenum)
        if (sampler.iteration) % save_steps == 0:
            backend.grow(save_steps, None)
            with backend.open(mode='a') as f:
                g = f['mcmc']
                g['accepted'][...] = sampler.acceptance_fraction * sampler.iteration
                g['chain'][-save_steps:, :, :] = sampler.get_chain()[-save_steps:, :, :]
                g['log_prob'][-save_steps:, :] = sampler.get_log_prob()[-save_steps:, :]
                g.attrs["iteration"] = sampler.iteration
    return file



class EMCEE_ConfInterval_Window(QMainWindow):
    def __init__(self, fitFile=None, emcee_walker=10, emcee_steps=10000, emcee_burn=1, emcee_thin=1, emcee_cores=1, backendFile=None, parent=None):
        QMainWindow.__init__(self, parent)
        uic.loadUi('./UI_Forms/MCMC_ConfInterval_Window.ui', self)
        self.setWindowTitle('MCMC Confidence Interval Calculator')
        self.menuBar().setNativeMenuBar(False)
        self.fitFile=fitFile
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
        self.parallelCoresNumSpinBox.setMinimum(1)
        self.parallelCoresNumSpinBox.setValue(self.emcee_cores)
        self.NCores=int(ray.cluster_resources()['CPU'])
        self.parallelCoresNumSpinBox.setMaximum(self.NCores-2)
        self.parallelCoresNumSpinBox.valueChanged.connect(self.parallelCoresChanged)
        self.progressBar.setValue(0)
        self.backend=None

        self.init_signals()

    def init_signals(self):
        self.actionLoad_Fit_File.triggered.connect(lambda x:self.loadFitFile(fname=None))
        self.actionLoad_MCMC_File.triggered.connect(self.loadMCMCFile)
        self.actionNew_MCMC_File.triggered.connect(self.newMCMCFile)
        self.actionQuit.triggered.connect(self.close)
        self.startSamplingPushButton.clicked.connect(self.start_emcee_sampling)
        self.MCMCWalkerLineEdit.returnPressed.connect(self.MCMCWalker_changed)
        self.saveConfIntervalPushButton.clicked.connect(self.saveParameterError)
        self.addUserDefinedParamPushButton.clicked.connect(lambda x:self.addMCMCUserDefinedParam(parname=None, expression=None))
        self.removeUserDefinedParamPushButton.clicked.connect(self.removeMCMCUserDefinedParam)
        self.saveUserDefinedParamPushButton.clicked.connect(self.saveMCMCUserDefinedParam)
        self.loadUserDefinedParamPushButton.clicked.connect(self.loadMCMCUserDefinedParam)
        self.userDefinedParamTreeWidget.itemDoubleClicked.connect(self.openMCMCUserDefinedParam)
        self.tabWidget.setCurrentIndex(0)

    def parallelCoresChanged(self):
        self.emcee_cores=self.parallelCoresNumSpinBox.value()


    def loadFitFile(self, fname=None):
        """
        loads parameters from a parameter file
        """
        # if self.funcListWidget.currentItem() is not None:
        if fname is None:
            fname = QFileDialog.getOpenFileName(self, caption='Open Fit File',# directory=self.curDir,
                                                filter='Fit Files (*.xfit)')[0]
        else:
            fname = fname
        if fname != '':
            self.curDir = os.path.dirname(fname)
            try:
                fh = open(fname, 'r')
                lines = fh.readlines()
                category = lines[1].split(': ')[1].strip()
                func = lines[2].split(': ')[1].strip()
                module = 'Functions.%s.%s' % (category, func)
                funcClass=import_module(module)
                self.fit=Fit(getattr(funcClass,func),[1.0])
                # self.fit.func.init_params()
                lnum = 3
                sfline = None
                mfline = None
                for line in lines[3:]:
                    if line == '#Fixed Parameters:\n':
                        fline = lnum + 2
                    elif line == '#Single fitting parameters:\n':
                        sfline = lnum + 2
                    elif line == '#Multiple fitting parameters:\n':
                        mfline = lnum + 2
                    elif 'col_names' in line:
                        self.col_names=eval(line.split('=')[1].strip())
                    elif 'Fit Scale' in line:
                        self.fit_scale = line.split('=')[1].strip()
                    lnum += 1
                if sfline is None:
                    sendnum = lnum
                else:
                    sendnum = sfline - 2
                if mfline is None:
                    mendnum = lnum
                else:
                    mendnum = mfline - 2
                for line in lines[fline:sendnum]:
                    key, val = line[1:].strip().split('\t')
                    try:
                        val = eval(val.strip())
                    except:
                        val = val.strip()
                    self.fit.params[key] = val
                if sfline is not None:
                    for line in lines[sfline:mendnum]:
                        parname, parval, parfit, parmin, parmax, parexpr, parbrute = line[1:].strip().split('\t')
                        self.fit.params[parname] = float(parval)
                        self.fit.fit_params[parname].set(value=float(parval), vary=int(parfit), min=float(parmin),
                                                         max=float(parmax))
                        try:
                            self.fit.fit_params[parname].set(expr=eval(parexpr))
                        except:
                            self.fit.fit_params[parname].set(expr=str(parexpr))
                        try:
                            self.fit.fit_params[parname].set(brute_step=eval(parbrute))
                        except:
                            self.fit.fit_params[parname].set(brute_step=str(parbrute))

                if mfline is not None:
                    val = {}
                    expr = {}
                    pmin = {}
                    pmax = {}
                    pbrute = {}
                    pfit = {}
                    for line in lines[mfline:]:
                        tlist = line[1:].strip().split('\t')
                        if 'Fit Statistics' not in line:
                            if len(tlist) > 2:
                                parname, parval, parfit, parmin, parmax, parexpr, parbrute = tlist
                                val[parname] = float(parval)
                                pmin[parname] = float(parmin)
                                pmax[parname] = float(parmax)
                                pfit[parname] = int(parfit)
                                try:
                                    expr[parname] = eval(parexpr)
                                except:
                                    expr[parname] = str(parexpr)
                                try:
                                    pbrute[parname] = eval(parbrute)
                                except:
                                    pbrute[parname] = str(parbrute)
                                try:  # Here the expr is set to None and will be taken care at the for loop just after this for loop
                                    self.fit.fit_params[parname].set(val[parname], vary=pfit[parname],
                                                                     min=pmin[parname],
                                                                     max=pmax[parname], expr=None,
                                                                     brute_step=pbrute[parname])
                                except:
                                    self.fit.fit_params.add(parname, value=val[parname], vary=pfit[parname],
                                                            min=pmin[parname],
                                                            max=pmax[parname], expr=None,
                                                            brute_step=pbrute[parname])
                                mkey, pkey, num = parname[2:].split('_')
                                num = int(num)
                                try:
                                    self.fit.params['__mpar__'][mkey][pkey][num] = float(parval)
                                except:
                                    self.fit.params['__mpar__'][mkey][pkey].insert(num, float(parval))
                            else:
                                parname,parval=tlist
                                mkey,pkey,num=parname[2:].split('_')
                                num=int(num)
                                try:
                                    self.fit.params['__mpar__'][mkey][pkey][num]=parval
                                except:
                                    self.fit.params['__mpar__'][mkey][pkey].insert(num,parval)
                        else:
                            break
                    for parname in val.keys():  # Here is the expr is put into the parameters
                        try:
                            self.fit.fit_params[parname].set(value=val[parname], vary=pfit[parname],
                                                             min=pmin[parname],
                                                             max=pmax[parname], expr=expr[parname],
                                                             brute_step=pbrute[parname])
                        except:
                            self.fit.fit_params.add(parname, value=val[parname], vary=pfit[parname],
                                                    min=pmin[parname],
                                                    max=pmax[parname], expr=expr[parname],
                                                    brute_step=pbrute[parname])
                self.readData(fname, self.col_names)
                self.fitPlot()
                self.fitFileLineEdit.setText(fname)
                if len(self.fitPlotNames)>2: #For multi-column data and error-bars
                    x={}
                    y={}
                    yerr={}
                    for name in self.fitPlotNames:
                        if 'fit' not in name and 'calc' not in name:
                            x[name]=self.data['x_'+name].values
                            y[name]=self.data['y_'+name].values
                            yerr[name]=self.data['yerr_'+name].values
                else:
                    for name in self.fitPlotNames: #For single-column data and error-bars
                        if 'fit' not in name and 'calc' not in name:
                            x = self.data['x_' + name].values
                            y = self.data['y_' + name].values
                            yerr = self.data['yerr_' + name].values
                self.fit.set_x(x,y=y,yerr=yerr)
                self.ycalc=self.fit.evaluate()
                self.calcPlot()
                self.tabWidget.setCurrentIndex(0)
                self.fitParamsList=[]
                self.fitKeyList=[]
                for key in self.fit.fit_params.keys():
                    if self.fit.fit_params[key].vary:
                        self.fitParamsList.append(self.fit.fit_params[key].value)
                        self.fitKeyList.append(key)
                if not self.fitCalcMatch:
                    QMessageBox.warning(self, "Fit Function Warning", "The fitted values doesnot match with the calculated values from the function", QMessageBox.Ok)
                else:
                    self.MCMCWalker_changed()
            except:
                QMessageBox.warning(self, 'File Import Error',
                                    'Some problems in the fit file\n' + traceback.format_exc(), QMessageBox.Ok)


    def MCMCWalker_changed(self):
        self.mcmc_iterations=0
        self.reuseSamplerCheckBox.setChecked(False)
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
        self.emcee_cores = self.parallelCoresNumSpinBox.value()

    def loadMCMCFile(self):
        pass


    def newMCMCFile(self):
        file = QFileDialog.getSaveFileName(self, 'Start MCMC file as', filter='MCMC Files (*.h5)')[0]
        # self.iterations=0
        if file != '':
            if os.path.exists(file):
                os.remove(file)
            self.MCMCBackendFile = file
            self.MCMCFileLineEdit.setText(self.MCMCBackendFile)
            self.backend = emcee.backends.HDFBackend(self.MCMCBackendFile)
            self.autoCorrTime = np.array([[0, 1.0]])
            self.MCMC_starting_step = self.autoCorrTime[-1, 0]
            self.autoCorrTimeMPLWidget.clear()
            self.autoCorrPlot_ax1 = self.autoCorrTimeMPLWidget.fig.add_subplot(1, 1, 1)
            self.autoCorrPlot_sp, = self.autoCorrPlot_ax1.plot(self.autoCorrTime[:, 0], self.autoCorrTime[:, 1], '.')
            self.autoCorrPlot_ax1.set_xlabel('Iterations')
            self.autoCorrPlot_ax1.set_ylabel('correlation_time')
            self.autoCorrTimeMPLWidget.fig.canvas.draw()
            self.autoCorrTimeMPLWidget.fig.canvas.flush_events()
            self.tabWidget.setCurrentIndex(4)
            self.reuseSamplerCheckBox.setChecked(False)
            self.autoCorrTimeTextEdit.clear()
            self.autoCorrTimeTextEdit.append('#MCMC_steps\tCorr_Time')
            self.autoCorrTimeTextEdit.append(
                '%d\t%.3f' % (self.autoCorrTime[-1, 0], self.autoCorrTime[-1, 1]))
        else:
            self.backend = None



    def readData(self, fname, col_names):
        self.data=pd.read_csv(fname,names=col_names,comment='#',delimiter=' ')

    def fitPlot(self):
        self.fitPlotNames=[]
        for i in range(int(len(self.col_names)/4)):
            x=self.data[self.col_names[i*4]].values
            y=self.data[self.col_names[i*4+1]].values
            yerr=self.data[self.col_names[i*4+2]].values
            yfit=self.data[self.col_names[i*4+3]].values
            name=self.col_names[i*4][2:]
            self.fitPlotWidget.add_data(x,y,yerr=yerr,name=name)
            self.fitPlotNames.append(name)
            self.fitPlotWidget.add_data(x,yfit,name=name+'_fit',fit=True)
            self.fitPlotNames.append(name+'_fit')
        self.fitPlotWidget.Plot(self.fitPlotNames)


    def calcPlot(self):
        self.calcPlotNames=[]
        self.fitCalcMatch=True
        if type(self.ycalc)==dict:
            for key in self.ycalc.keys():
                self.fitPlotWidget.add_data(self.fit.x[key],self.ycalc[key],name=key+'_calc',fit=True)
                self.calcPlotNames.append(key+'_calc')
                self.fitCalcMatch=self.fitCalcMatch and np.allclose(self.fit.yfit[key],self.ycalc[key])
        else:
            self.fitPlotWidget.add_data(self.fit.x,self.ycalc,name='data_calc',fit=True)
            self.calcPlotNames.append('data_calc')
            self.fitCalcMatch = self.fitCalcMatch and np.allclose(self.fit.yfit, self.ycalc)
        self.fitPlotWidget.Plot(self.fitPlotNames + self.calcPlotNames)


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
        #self.fit.functionCalled.connect(self.updateMCMCStatus)
        self.setMCMC()
        self.runMCMC()


    def setMCMC(self):
        self.update_emcee_parameters()
        self.sampler = emcee.EnsembleSampler(self.emcee_walker, len(self.fitParamsList), self.fit.log_probability,
                                             args=(self.fitKeyList, self.fit_scale), backend=self.backend)

    def runMCMC(self):
        start_pos = np.array(self.fitParamsList)+1e-4*np.random.randn(self.emcee_walker, len(self.fitParamsList))
        # We'll track how the average autocorrelation time estimate changes
        index = 0
        autocorr = []

        # This will be useful to testing convergence
        old_tau = np.inf
        for sample in self.sampler.sample(start_pos, iterations=self.emcee_steps, progress=True):
            if self.sampler.iteration % 100:
                continue
                # Compute the autocorrelation time so far
                # Using tol=0 means that we'll always get an estimate even
                # if it isn't trustworthy
            tau = self.sampler.get_autocorr_time(tol=0)
            autocorr.append(np.mean(tau))
            index += 1

            # Check convergence
            converged = np.all(tau * 100 < self.sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            self.updateMCMCStatus(autocorr[-1])
            if converged:
                break
            old_tau = tau
        print("Iterations stopped or Convergence achieved")

        # self.sampler.run_mcmc(start_pos, self.emcee_steps, progress=True)

    def updateMCMCStatus(self, autocorr):
        self.fitIterLabel.setText("Iterations=%d, autocorr=%.3f"%(self.sampler.iteration, autocorr))
        QApplication.processEvents()

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
