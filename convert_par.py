import time
import os
import sys

def convert(ifname, ofname):
    mtime = os.path.getctime(ifname)
    ttime = time.ctime(mtime)
    try:
        fh = open(ifname)
        lines = fh.readlines()
        fh.close()
    except:
        print('Error: File %s not found'%ifname)
    for i, line in enumerate(lines):
        if 'Num_of_Layer' in line:
            ilayer = i
        if 'Para_Name' in line:
            player = i + 1
    Nlayers = int(lines[ilayer].strip().split()[1])
    rho = {}
    mu = {}
    sig = {}
    d = {}

    for i in range(Nlayers + 2):
        if i == 0:
            d[i] = ['d%d' % i, '0.0', 'False', '0.0', '0.0', 'inf']
            rho[i] = lines[player].split()
            if rho[i][2] == 'None' or rho[i][2] == 'NA':
                rho[i][2] = 'False'
            if rho[i][4] == 'None' or rho[i][4] == 'NA':
                rho[i][4] = '0.0'
            if rho[i][5] == 'None' or rho[i][5] == 'NA':
                rho[i][5] = 'inf'
            player += 1
            mu[i] = lines[player].split()
            if mu[i][2] == 'None' or mu[i][2] == 'NA':
                mu[i][2] = 'False'
            if mu[i][4] == 'None' or mu[i][4] == 'NA':
                mu[i][4] = '0.0'
            if mu[i][5] == 'None' or mu[i][5] == 'NA':
                mu[i][5] = 'inf'
            player += 1
            sig[i] = lines[player].split()
            if sig[i][2] == 'None' or sig[i][2] == 'NA':
                sig[i][2] = 'False'
            if sig[i][4] == 'None' or sig[i][4] == 'NA':
                sig[i][4] = '0.0'
            if sig[i][5] == 'None' or sig[i][5] == 'NA':
                sig[i][5] = 'inf'
            player += 1
        elif i == Nlayers + 1:
            d[i] = ['d%d' % i, '0.0', 'False', '0.0', '0.0', 'inf']
            rho[i] = lines[player].split()
            if rho[i][2] == 'None' or rho[i][2] == 'NA':
                rho[i][2] = 'False'
            if rho[i][4] == 'None' or rho[i][4] == 'NA':
                rho[i][4] = '0.0'
            if rho[i][5] == 'None' or rho[i][5] == 'NA':
                rho[i][5] = 'inf'
            player += 1
            mu[i] = lines[player].split()
            if mu[i][2] == 'None' or mu[i][2] == 'NA':
                mu[i][2] = 'False'
            if mu[i][4] == 'None' or mu[i][4] == 'NA':
                mu[i][4] = '0.0'
            if mu[i][5] == 'None' or mu[i][5] == 'NA':
                mu[i][5] = 'inf'
            player += 1
            sig[i] = ['sig%d' % i, '0.0', 'False', '0.0', '0.0', 'inf']
        else:
            d[i] = lines[player].split()
            if d[i][2] == 'None' or d[i][2] == 'NA':
                d[i][2] = 'False'
            if d[i][4] == 'None' or d[i][4] == 'NA':
                d[i][4] = '0.0'
            if d[i][5] == 'None' or d[i][5] == 'NA':
                d[i][5] = 'inf'
            player += 1
            rho[i] = lines[player].split()
            if rho[i][2] == 'None' or rho[i][2] == 'NA':
                rho[i][2] = 'False'
            if rho[i][4] == 'None' or rho[i][4] == 'NA':
                rho[i][4] = '0.0'
            if rho[i][5] == 'None' or rho[i][5] == 'NA':
                rho[i][5] = 'inf'
            player += 1
            mu[i] = lines[player].split()
            if mu[i][2] == 'None' or mu[i][2] == 'NA':
                mu[i][2] = 'False'
            if mu[i][4] == 'None' or mu[i][4] == 'NA':
                mu[i][4] = '0.0'
            if mu[i][5] == 'None' or mu[i][5] == 'NA':
                mu[i][5] = 'inf'
            player += 1
            sig[i] = lines[player].split()
            if sig[i][2] == 'None' or sig[i][2] == 'NA':
                sig[i][2] = 'False'
            if sig[i][4] == 'None' or sig[i][4] == 'NA':
                sig[i][4] = '0.0'
            if sig[i][5] == 'None' or sig[i][5] == 'NA':
                sig[i][5] = 'inf'
            player += 1
    qoff = lines[player].split()
    if qoff[2] == 'None' or qoff[2]=='NA':
        qoff[2] = 'False'
    if qoff[4] == 'None' or qoff[4]=='NA':
        qoff[4] = '-inf'
    if qoff[5] == 'None' or qoff[5]=='NA':
        qoff[5] = 'inf'
    player += 1
    y_scale = lines[player].split()
    if y_scale[2] == 'None' or y_scale[2] == 'NA':
        y_scale[2] = '0.0'
    if y_scale[4] == 'None' or y_scale[4] == 'NA':
        y_scale[4] = '0.0'
    if y_scale[5] == 'None' or y_scale[5] == 'NA':
        y_scale[5] = 'inf'
    player += 1
    q_res = lines[player].split()
    header = ''
    header += '#File saved on %s\n' % ttime
    header += '#Category: XRR\n'
    header += '#Function: XLayers\n'
    header += '#Xrange=np.linspace(0.001,0.7,200)\n'
    header += '#Fit Range=0:1\n'
    header += '#Fit Method=Levenberg-Marquardt\n'
    header += '#Fit Scale=Linear\n'
    header += '#Fit Iterations=1000\n'
    header += '#Fixed Parameters:\n'
    header += '#param\tvalue\n'
    header += 'E\t10.0\n'
    header += 'dz\t0.5\n'
    header += 'fix_sig\tFalse\n'
    header += 'rrf\tTrue\n'
    header += '#Single fitting parameters:\n'
    header += '#param\tvalue\tfit\tmin\tmax\texpr\tbrute_step\n'
    header += 'qoff\t%s\t%d\t%s\t%s\t%s\t%s\n' % (qoff[1], eval(qoff[2]), qoff[4], qoff[5], 'None', '0.1')
    header += 'yscale\t%s\t%d\t%s\t%s\t%s\t%s\n' % (y_scale[1], eval(y_scale[2]), y_scale[4], y_scale[5], 'None', '0.1')
    header += 'bkg\t%s\t%d\t%s\t%s\t%s\t%s\n' % ('0.0', False, '0', '1', 'None', '0.1')
    header += '#Multiple fitting parameters:\n'
    header += '#param\tvalue\tfit\tmin\tmax\texpr\tbrute_step\n'

    header += '__Model_Layers_000\ttop\n'
    for i in range(Nlayers):
        header += '__Model_Layers_%03d\tlayer%d\n' % (i + 1, i + 1)
    header += '__Model_Layers_%03d\tbottom\n' % (Nlayers + 1)

    header += '__Model_d_000\t%s\t%d\t%s\t%s\t%s\t%s\n' % (d[0][1], eval(d[0][2]), d[0][4], d[0][5], 'None', '0.1')
    for i in range(Nlayers + 1):
        header += '__Model_d_%03d\t%s\t%d\t%s\t%s\t%s\t%s\n' % (
        i + 1, d[i + 1][1], eval(d[i + 1][2]), d[i + 1][4], d[i + 1][5], 'None', '0.1')

    header += '__Model_rho_000\t%s\t%d\t%s\t%s\t%s\t%s\n' % (
    rho[0][1], eval(rho[0][2]), rho[0][4], rho[0][5], 'None', '0.1')
    for i in range(Nlayers + 1):
        header += '__Model_rho_%03d\t%s\t%d\t%s\t%s\t%s\t%s\n' % (
        i + 1, rho[i + 1][1], eval(rho[i + 1][2]), rho[i + 1][4], rho[i + 1][5], 'None', '0.1')

    header += '__Model_mu_000\t%s\t%d\t%s\t%s\t%s\t%s\n' % (mu[0][1], eval(mu[0][2]), mu[0][4], mu[0][5], 'None', '0.1')
    for i in range(Nlayers + 1):
        header += '__Model_mu_%03d\t%s\t%d\t%s\t%s\t%s\t%s\n' % (
        i + 1, mu[i + 1][1], eval(mu[i + 1][2]), mu[i + 1][4], mu[i + 1][5], 'None', '0.1')

    header += '__Model_sig_000\t%s\t%d\t%s\t%s\t%s\t%s\n' % (
    sig[Nlayers + 1][1], eval(sig[Nlayers + 1][2]), sig[Nlayers + 1][4], sig[Nlayers + 1][5], 'None', '0.1')
    for i in range(Nlayers + 1):
        header += '__Model_sig_%03d\t%s\t%d\t%s\t%s\t%s\t%s\n' % (
        i + 1, sig[i][1], eval(sig[i][2]), sig[i][4], sig[i][5], 'None', '0.1')
    fh = open(ofname, 'w')
    fh.writelines(header)
    fh.close()
    print('Success: File %s converted to %s'%(ifname,ofname))


if len(sys.argv)>=2:
    ifname=sys.argv[1]
    try:
        ofname=sys.argv[2]
        if os.path.splitext(ofname)[1]!='.par':
            print('Error: Please provide the extension of the file as .par')
        else:
            convert(ifname, ofname)
    except:
        ofname=os.path.splitext(ifname)[0]+'.par'
        convert(ifname, ofname)
else:
    print('Error: Please provide an parameter file with path')
    print('Usage:')
    print('python convert_par.py parmeter_file_w_path [output_parmeter_file_w_path]')