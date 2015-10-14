###############################################

# This program runs a simulation of phenotypic assortment in cilates.
# Ciliates are characterized by nuclear dimorphism and a stochastic amitotic division process.
# The stochastic component of their chromosomal division during amitosis
# is the fundamental motivation behind the simulation. We use a simple fitness function of
# a weighted average of the two alleles (a "good" copy and a "bad" copy with some fitness cost).
# The ciliate's overall fitness is assessed using a multiplicative fitness function across all
# chromosome copies. These multiplicative fitnesses are normalized across the population by
# mapping them to a normal distribution that represents the distribution of fitnesses across
# the ciliate population.

# Phenotypic assortment occurs when after successive generations of amitotic division, certain 
# alleles are eliminated from the macronucleus and are thus no longer expressed.
# The simulation assesses the number of generations that phenotypic assortment takes to occur.
# It investigates how a variety of parameters affect phenotypic assortment times.

# Ciliates and chromosomes are represented as arrays that consist of the following elements:
# ciliate = [[chromosomeset],age,overall fitness]
# chromsome = [number of copies of normal allele, number of copies of deleterious allele, fitness]

###############################################

from __future__ import division
import time
import random 
import numpy
from scipy import stats

###############################################

# Parameters:
generations = 10000       # number of generations (typically 10000)
rounds = 100              # number of times to repeat simulation
chrom = 1                 # number of unique chromosomes
N = 46                    # number of copies of each unique chromosome (copies of chrom)
popsize = 100             # the size of the population of ciliates

###############################################

# Duplication and fitness function
# This function duplicates the ciliate population and recalculates each ciliates new fitness.
# It then returns the new population. 

# Parameters:
# population   - The ciliate population
# generations  - The maximum number of generations to let the ciliates divide
# N            - The number of copies of the identical chromosomes (chrom)
# badf         - The fitness of the deleterious allele
# avgfit       - The average fitness in the ciliate population
# stdfit       - The standard deviaiton of the fitness in the ciliate population

def dup_fitness(population,N,badf,avgfit,stdfit):
    temppop = []
    for ciliate in population:
        if stdfit == 0:
            fitprob = .5
        else:
            fitprob = stats.norm.cdf(ciliate[2],loc = avgfit,scale = stdfit)
        if random.random() <= fitprob:
            newchromset1 = []
            newchromset2 = []
            newage = (ciliate[1] + 1)
            newovrfit1 = 1
            newovrfit2 = 1
            for chromosome in ciliate[0]:
                if chromosome[0] == 0:
                    newallele2cil1 = numpy.random.binomial(2*chromosome[1],.5)
                    newallele2cil2 = (2*chromosome[1]) - newallele2cil1

                    if newallele2cil1:
                        newchromset1.append([0,newallele2cil1,0.0])
                    if newallele2cil2:
                        newchromset2.append([0,newallele2cil2,0.0])

                elif chromosome[1] == 0:
                    newallele1cil1 = numpy.random.binomial(2*chromosome[0],.5)
                    newallele1cil2 = (2*chromosome[0]) - newallele1cil1

                    if newallele1cil1:
                        newchromset1.append([newallele1cil1,0,1.0])
                    if newallele1cil2:
                        newchromset2.append([newallele1cil2,0,1.0])

                else:
                    newallele1cil1 = numpy.random.binomial(2*chromosome[0],.5)
                    newallele1cil2 = (2*chromosome[0]) - newallele1cil1

                    newallele2cil1 = numpy.random.binomial(2*chromosome[1],.5)
                    newallele2cil2 = (2*chromosome[1]) - newallele2cil1

                    if (newallele1cil1 + newallele2cil1):
                        newfitcil1 = ((newallele1cil1*1.0)+(newallele2cil1*badf))/(newallele1cil1 + newallele2cil1)
                        newchromset1.append([newallele1cil1,newallele2cil1,newfitcil1])
                    if (newallele1cil2 + newallele2cil2):
                        newfitcil2 = ((newallele1cil2*1.0)+(newallele2cil2*badf))/(newallele1cil2 + newallele2cil2)
                        newchromset2.append([newallele1cil2,newallele2cil2,newfitcil2])

            for k in newchromset1:
                newovrfit1 *= k[2]
            for l in newchromset2:
                newovrfit1 *= l[2]
            if len(newchromset1) == chrom:
                temppop.append([newchromset1,newage,newovrfit1])
            if len(newchromset2) == chrom:
                temppop.append([newchromset2,newage,newovrfit2])
        else:
            temppop.append(ciliate)

    return temppop


###############################################

# Master function
# Creates a ciliate population, then uses the dup_fitness function to duplicate
# the ciliate population and reassess each ciliate's fitness. It then checks to see
# if any of the chromosomes have been fixed (phenotypic assortment). It continues
# this process until the desired number of chromosomes have been fixed in the population,
# or until the generational limit is reached.

# Parameters:
# popsize      - The size of the ciliate population
# generations  - The maximum number of generations to let the ciliates divide
# N            - The number of copies of the identical chromosomes (chrom)
# badf         - The fitness of the deleterious allele
# chrom        - The number of unique chromosomes

def trial(popsize,generations,N,badf,chrom):
    i = 1
    initialciliate = [[[N/2,N/2,(badf + 1)/2] for m in range(chrom)],0,((badf + 1)/2)**chrom]
    population = [initialciliate for j in range(popsize)]
    fixrecord = [[0,0] for x in range(chrom)]
    while i <= generations:
        fitness = []
        for ciliate in population:
            fitness.append(ciliate[2])
        fitarray = numpy.array(fitness)
        avgfit = numpy.average(fitarray)
        stdfit = numpy.std(fitarray)
        population = dup_fitness(population,N,badf,avgfit,stdfit)
        if len(population) >= popsize:
            population = random.sample(population, popsize)
        for z in range(chrom):
            if fixrecord[z][0] == 0:
                totalcopy1 = sum([ciliate[0][z][0] for ciliate in population])
                totalcopy2 = sum([ciliate[0][z][1] for ciliate in population])
                if totalcopy1 == 0:
                    fixrecord[z][0] = 1
                    fixrecord[z][1] = i
                elif totalcopy1 == (totalcopy1 + totalcopy2):
                    fixrecord[z][0] = 2
                    fixrecord[z][1] = i

        if sum(t[0] for t in fixrecord) >= (2*chrom):
            break
        else:
            i += 1    

    newpoparray = numpy.array(population)
    fixrecarray = numpy.array(fixrecord)
    fixalleles = fixrecarray[:,0]
    fixgens = fixrecarray[:,1]
    pop = len(population)
    avgfit = sum(newpoparray[:,2])/pop
    avgage = sum(newpoparray[:,1])/pop
    allele1num = []
    allele2num = []    
    for ciliate in population:
        for y in range(chrom):
            allele1num.append(ciliate[0][y][0])
            allele2num.append(ciliate[0][y][1])
    totalcopy1 = sum(allele1num)
    totalcopy2 = sum(allele2num)
        
    totalchrom = totalcopy1 + totalcopy2
    for t in fixrecord:
        if t[0] == 0:
            t[1] = generations
    sortfix = sorted(fixrecord, key = lambda x: int(x[1]))
    fixrecord = sortfix
    
    fixchrom1 = fixrecord[0][0]
    fixgen1 = fixrecord[0][1]
        
    return fixchrom1,fixgen1, totalchrom, pop, avgfit, avgage 

###############################################

# Runs simulation and writes the results to a .csv file

print time.strftime("%d:%H:%M:%S")

outs = 'Popsize,N,badf,fixchrom1,fixgen1,total copy number,average fitness,average age,\n'
for badf in [0.0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]:
    for r in range(rounds):
        fixchrom1,fixgen1, totalchrom, pop, avgfit, avgage = trial(popsize,generations,N,badf,chrom)
        out = str(pop) + ',' + str(N) + ',' + str(badf) + ',' + str(fixchrom1) + ',' + str(fixgen1) + ',' + str(totalchrom) + ',' + str(avgfit) + ',' + str(avgage) + ',\n'
        outs += out
output = 'FitCurve_PhenoAssort_2binomial_' + str(chrom) + 'Chrom)_popsize' + str(popsize) + '.csv'   # name of output file
with open(output,'a') as f:
    f.write(str(outs))

print time.strftime("%d:%H:%M:%S")

###############################################