import os
import csv
import numpy
import time
import matplotlib.pyplot as plt

import benchmarks
from cec2017.simple import *
from cec2017.hybrid import *
from cec2017.composition import *

import PROA as proa


numpy.random.seed(42)


def selector(algo, func_details, popSize, Iter):
    function_name = func_details[0]
    lb = func_details[1]
    ub = func_details[2]
    dim = func_details[3]

    if (algo == 0):
        x = proa.PROA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    return x


# Select optimizers
PROA = True

Algorithm = [PROA]
objectivefunc = [(i+1) for i in range(57)] # Run individually

# Select number of repetitions for each experiment.
# To obtain meaningful statistical results, usually 30 independent runs
# are executed for each algorithm.
Runs = 30

# Select general parameters for all optimizers (population size, number of iterations)
PopSize = 1
iterations = 5000

# Export results ?
Export = False

ExportFolder = os.path.join("./results", "ALL_RES" + time.strftime("%d-%m-%Y-%H-%M-%S"))
os.mkdir(ExportFolder)

CsvFile = os.path.join(ExportFolder, "AllResults.csv")
ConvPlotFile = os.path.join(ExportFolder, "F")

# Check if it works at least once
atLeastOneIteration = False

# CSV Header for for the convergence
CnvgHeader = []

for l in range(0, iterations):
    CnvgHeader.append("Iter" + str(l + 1))

for i in range(0, len(Algorithm)):
    for fun_idx in objectivefunc:
        fun_idx -= 1
        if ((Algorithm[i] == True)):  # start experiment if an Algorithm is selected

            params=None
            objfunc=None
            exectime=0.0

            for k in range(0, Runs):
                func_details = benchmarks.getFunctionDetails(fun_idx)
                x = selector(i, func_details, PopSize, iterations)

                params = x.params
                objfunc = x.objfname
                exectime += x.executionTime

                if (Export == True):
                    with open(CsvFile, 'a') as out:
                        writer = csv.writer(out, delimiter=',')
                        if (atLeastOneIteration == False):  # just one time to write the header of the CSV file
                            header = numpy.concatenate(
                                [["ExptSetting", "objfname", "ExecutionTime"], CnvgHeader])
                            writer.writerow(header)
                        a = numpy.concatenate(
                            [[str(params), objfunc, x.executionTime], x.convergence])
                        writer.writerow(a)
                    out.close()

                    # save convergence curve
                    plt.figure()
                    plt.yscale('log')
                    plt.plot(x.convergence.tolist())
                    plt.savefig(f"{ConvPlotFile}{fun_idx+1}.jpg", dpi=300)

                atLeastOneIteration = True  # at least one experiment

if (atLeastOneIteration == False):  # Failed to run at least one experiment
    print("No Optimizer or Cost function is selected. Check lists of available optimizers and cost functions")

