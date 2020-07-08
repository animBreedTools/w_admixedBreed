using DataFrames
using Distributions
using DelimitedFiles
using LinearAlgebra
using CSV

function w_bayesPR_shaoLei(genoTrain, phenoTrain, breedProp, weights, userMapData, chrs, fixedRegSize, varGenotypic, varResidual, chainLength, burnIn, outputFreq, onScreen)
    SNPgroups = prepRegionData(userMapData, chrs, genoTrain, fixedRegSize)
    these2Keep = collect((burnIn+outputFreq):outputFreq:chainLength) #print these iterations
    nRegions    = length(SNPgroups)
    println("number of regions: ", nRegions)
    X           = convert(Array{Float64}, genoTrain)
    println("X is this size", size(X))
    y           = convert(Array{Float64}, phenoTrain)
    println("y is this size", size(y))
    nTraits, nRecords , nMarkers   = size(y,2), size(y,1), size(X,2)
    w           = convert(Array{Float64}, weights)
    iD          = full(Diagonal(w))  # Dii is 1/wii=1/(r2/(1-r2))==> Dii is (1-r2)/r2 ==> iDii is r2/(1-r2)
    fileControlSt(fixedRegSize)
    p           = mean(X,dims=1)./2.0
    sum2pq      = sum(2*(1 .- p).*p) 
  
    #priors
    dfEffectVar = 4.0
    dfRes       = 4.0

    if varGenotypic==0.0
        varBeta      = fill(0.0005, nRegions)
        scaleVar     = 0.0005
        else
        varBeta      = fill(varGenotypic/sum2pq, nRegions)
        scaleVar     = varBeta[1]*(dfEffectVar-2.0)/dfEffectVar
    end
    if varResidual==0.0
        varResidual  = 0.0005
        scaleRes     = 0.0005
        else
        scaleRes    = varResidual*(dfRes-2.0)/dfRes    
    end
    
    #precomputation of vsB for convenience
    νS_β            = scaleVar*dfEffectVar
    df_β            = dfEffectVar
    νS_e            = scaleRes*dfRes
    df_e            = dfRes
    #initial values as "0"
    tempBetaVec     = zeros(Float64,nMarkers)
    μ               = mean(y)
    X              .-= ones(Float64,nRecords)*2p
    xpiDx            = diag((X.*w)'*X)  #w[i] is already iD[i,i]
    XpiD             = iD*X        #this is to iterate over columns in the body "dot(view(XpiD,:,l),ycorr)"
    println("size of xpiDx $(size(xpiDx))")
    println("size of XpiD $(size(XpiD))")
    
    #Can use equal numbers as this is just starting value!
    breedProp = convert(Array{Float64},breedProp)
    F = copy(breedProp)
    F .-=  mean(breedProp,1)
    F = [ones(nRecords) F]
    
    #for single-site sampler
    fpiDf            = diag((F.*w)'*F)  #w[i] is already iD[i,i]
    FpiD             = iD*F        #this is to iterate over columns in the body "dot(view(XpiD,:,l),ycorr)" already transposed    
    f               = [μ; mean(y .- μ)*vec(mean(breedProp,1))]
    ycorr           = y - F*f
    GC.gc()
    #MCMC starts here
    for iter in 1:chainLength
        #sample residual variance
        varE = sampleVarE_w(νS_e,ycorr,w,df_e,nRecords)
        #sample fixed effects, single-site gibbs sampling
        for fix in 1:size(F,2)
            ycorr    .+= F[:,fix]*f[fix]
            rhs      = view(FpiD,:,fix)'*ycorr
            invLhs   = 1.0/fpiDf[fix]
            meanMu   = invLhs*rhs
            f[fix]   = rand(Normal(meanMu,sqrt(invLhs*varE)))
            ycorr    .-= F[:,fix]*f[fix]
        end
        
        for r in 1:nRegions
            theseLoci = SNPgroups[r]
            regionSize = length(theseLoci)
            λ_r = varE/varBeta[r]
            for l in theseLoci::UnitRange{Int64}
                BLAS.axpy!(tempBetaVec[l], view(X,:,l), ycorr)
#                rhs = view(XpiD,:,l)'*ycorr
                rhs = BLAS.dot(view(XpiD,:,l),ycorr)
                lhs = xpiDx[l] + λ_r
                meanBeta = lhs\rhs
                tempBetaVec[l] = sampleBeta(meanBeta, lhs, varE)
                BLAS.axpy!(-1*tempBetaVec[l], view(X,:,l), ycorr)
            end
            varBeta[r] = sampleVarBeta(νS_β,tempBetaVec[theseLoci],df_β,regionSize)
        end
        outputControlSt(onScreen,iter,these2Keep,X,tempBetaVec,f',varBeta,varE,fixedRegSize)
    end
end

function w_bayesPR_BlockedGS(genoTrain, phenoTrain, breedProp, weights, userMapData, chrs, fixedRegSize, varGenotypic, varResidual, chainLength, burnIn, outputFreq, onScreen)
    SNPgroups = prepRegionData(userMapData, chrs, genoTrain, fixedRegSize)
    these2Keep = collect((burnIn+outputFreq):outputFreq:chainLength) #print these iterations
    nRegions    = length(SNPgroups)
    println("number of regions: ", nRegions)
    X           = convert(Array{Float64}, genoTrain)
    println("X is this size", size(X))
    y           = convert(Array{Float64}, phenoTrain)
    ####
    #center y
    y -= mean(y) 
    ####
    println("y is this size", size(y))
    nTraits, nRecords , nMarkers   = size(y,2), size(y,1), size(X,2)
    w           = convert(Array{Float64}, weights)
    iD          = full(Diagonal(w))  # Dii is 1/wii=1/(r2/(1-r2))==> Dii is (1-r2)/r2 ==> iDii is r2/(1-r2)
    fileControlSt(fixedRegSize)
    p           = mean(X,dims=1)./2.0
    sum2pq      = sum(2*(1 .- p).*p) 
  
    #priors
    dfEffectVar = 4.0
    dfRes       = 4.0

    if varGenotypic==0.0
        varBeta      = fill(0.0005, nRegions)
        scaleVar     = 0.0005
        else
        varBeta      = fill(varGenotypic/sum2pq, nRegions)
        scaleVar     = varBeta[1]*(dfEffectVar-2.0)/dfEffectVar
    end
    if varResidual==0.0
        varResidual  = 0.0005
        scaleRes     = 0.0005
        else
        scaleRes    = varResidual*(dfRes-2.0)/dfRes    
    end
    
    #precomputation of vsB for convenience
    νS_β            = scaleVar*dfEffectVar
    df_β            = dfEffectVar
    νS_e            = scaleRes*dfRes
    df_e            = dfRes
    #initial values as "0"
    tempBetaVec     = zeros(Float64,nMarkers)
    μ               = mean(y)
    X              .-= ones(Float64,nRecords)*2p
    xpiDx            = diag((X.*w)'*X)  #w[i] is already iD[i,i]
    XpiD             = iD*X        #this is to iterate over columns in the body "dot(view(XpiD,:,l),ycorr)"
    println("size of xpiDx $(size(xpiDx))")
    println("size of XpiD $(size(XpiD))")
    
    #Can use equal numbers as this is just starting value!
    breedProp = convert(Array{Float64},breedProp)
    F = copy(breedProp)
    F .-=  mean(breedProp,1)
    F = [ones(nRecords) F]
    
    #blocked sampler
    invFpiDF        = inv((F.*w)'*F)  #w[i] is already iD[i,i]
    FpiD            = F'iD        #this is to iterate over columns in the body "dot(view(XpiD,:,l),ycorr)" already transposed    
    f               = [μ; mean(y .- μ)*vec(mean(breedProp,1))]
    ycorr           = y - F*f
    GC.gc()
    #MCMC starts here
    for iter in 1:chainLength
        #sample residual variance
        varE = sampleVarE_w(νS_e,ycorr,w,df_e,nRecords)
        #sample fixed effects, single-site gibbs sampling
        
        ycorr    .+= F*f
        rhs      = view(FpiD,:,:)*ycorr
        invLhs   = view(invFpiDF,:,:)
        meanMu   = invLhs*rhs
        f       .= rand(MvNormal(meanMu,convert(Array,Symmetric(invLhs*varE))))
        ycorr    .-= F*f
    
        
        for r in 1:nRegions
            theseLoci = SNPgroups[r]
            regionSize = length(theseLoci)
            λ_r = varE/varBeta[r]
            for l in theseLoci::UnitRange{Int64}
                BLAS.axpy!(tempBetaVec[l], view(X,:,l), ycorr)
#                rhs = view(XpiD,:,l)'*ycorr
                rhs = BLAS.dot(view(XpiD,:,l),ycorr)
                lhs = xpiDx[l] + λ_r
                meanBeta = lhs\rhs
                tempBetaVec[l] = sampleBeta(meanBeta, lhs, varE)
                BLAS.axpy!(-1*tempBetaVec[l], view(X,:,l), ycorr)
            end
            varBeta[r] = sampleVarBeta(νS_β,tempBetaVec[theseLoci],df_β,regionSize)
        end
        outputControlSt(onScreen,iter,these2Keep,X,tempBetaVec,f',varBeta,varE,fixedRegSize)
    end
end

function bayesPR_shaoLei(genoTrain, phenoTrain, breedProp, weights, userMapData, chrs, fixedRegSize, varGenotypic, varResidual, chainLength, burnIn, outputFreq, onScreen)
    SNPgroups = prepRegionData(userMapData, chrs, genoTrain, fixedRegSize)
    these2Keep = collect((burnIn+outputFreq):outputFreq:chainLength) #print these iterations
    nRegions    = length(SNPgroups)
    println("number of regions: ", nRegions)
    X           = convert(Array{Float64}, genoTrain)
    println("X is this size", size(X))
    y           = convert(Array{Float64}, phenoTrain)
    println("y is this size", size(y))
    nTraits, nRecords , nMarkers   = size(y,2), size(y,1), size(X,2)
    w           = convert(Array{Float64}, weights)
    iD          = full(Diagonal(w))  # Dii is 1/wii=1/(r2/(1-r2))==> Dii is (1-r2)/r2 ==> iDii is r2/(1-r2)
    fileControlSt(fixedRegSize)
    p           = mean(X,dims=1)./2.0
    sum2pq      = sum(2*(1 .- p).*p) 
  
    #priors
    dfEffectVar = 4.0
    dfRes       = 4.0

    if varGenotypic==0.0
        varBeta      = fill(0.0005, nRegions)
        scaleVar     = 0.0005
        else
        varBeta      = fill(varGenotypic/sum2pq, nRegions)
        scaleVar     = varBeta[1]*(dfEffectVar-2.0)/dfEffectVar
    end
    if varResidual==0.0
        varResidual  = 0.0005
        scaleRes     = 0.0005
        else
        scaleRes    = varResidual*(dfRes-2.0)/dfRes    
    end
    
    #precomputation of vsB for convenience
    νS_β            = scaleVar*dfEffectVar
    df_β            = dfEffectVar
    νS_e            = scaleRes*dfRes
    df_e            = dfRes
    #initial values as "0"
    tempBetaVec     = zeros(Float64,nMarkers)
    μ               = mean(y)
    X              .-= ones(Float64,nRecords)*2p
    xpiDx            = diag((X.*w)'*X)  #w[i] is already iD[i,i]
    XpiD             = iD*X        #this is to iterate over columns in the body "dot(view(XpiD,:,l),ycorr)"
    println("size of xpiDx $(size(xpiDx))")
    println("size of XpiD $(size(XpiD))")
    
    ycorr           = y .- μ
    GC.gc()
    #MCMC starts here
    for iter in 1:chainLength
        #sample residual variance
        varE = sampleVarE_w(νS_e,ycorr,w,df_e,nRecords)
        #sample intercept
        ycorr    .+= μ
#        rhs = ones(nRecords)'iD*ycorr
#        invLhs = inv(ones(nRecords)'*iD*ones(nRecords))
        rhs      = sum(ycorr)
        invLhs   = 1.0/nRecords
        meanMu   = rhs*invLhs
        μ        = rand(Normal(meanMu,sqrt(invLhs*varE)))
        ycorr    .-= μ
        
        for r in 1:nRegions
            theseLoci = SNPgroups[r]
            regionSize = length(theseLoci)
            λ_r = varE/varBeta[r]
            for l in theseLoci::UnitRange{Int64}
                BLAS.axpy!(tempBetaVec[l], view(X,:,l), ycorr)
                rhs = view(XpiD,:,l)'*ycorr
                lhs = xpiDx[l] + λ_r
                meanBeta = lhs\rhs
                tempBetaVec[l] = sampleBeta(meanBeta, lhs, varE)
                BLAS.axpy!(-1*tempBetaVec[l], view(X,:,l), ycorr)
            end
            varBeta[r] = sampleVarBeta(νS_β,tempBetaVec[theseLoci],df_β,regionSize)
        end
        outputControlSt(onScreen,iter,these2Keep,X,tempBetaVec,μ,varBeta,varE,fixedRegSize)
    end
end

function bayesPR_NEW(genoTrain, phenoTrain, userMapData, chrs, fixedRegSize, varGenotypic, varResidual, chainLength, burnIn, outputFreq, onScreen)
    SNPgroups = prepRegionData(userMapData,chrs,genoTrain,fixedRegSize)
    these2Keep = collect((burnIn+outputFreq):outputFreq:chainLength) #print these iterations
    nRegions    = length(SNPgroups)
    println("number of regions: ", nRegions)
    X           = convert(Array{Float64}, genoTrain)
    genoTrain = 0
    println("X is this size", size(X))
    y           = convert(Array{Float64}, phenoTrain)
    println("y is this size", size(y))
    nTraits, nRecords , nMarkers   = size(y,2), size(y,1), size(X,2)
    fileControlSt(fixedRegSize)
    p           = mean(X,dims=1)./2.0
    sum2pq      = sum(2*(1 .- p).*p)

    #priors
    dfEffectVar = 4.0
    dfRes       = 4.0

    if varGenotypic==0.0
        varBeta      = fill(0.0005, nRegions)
        scaleVar     = 0.0005
        else
        varBeta      = fill(varGenotypic/sum2pq, nRegions)
        scaleVar     = varBeta[1]*(dfEffectVar-2.0)/dfEffectVar
    end
    if varResidual==0.0
        varResidual  = 0.0005
        scaleRes     = 0.0005
        else
        scaleRes    = varResidual*(dfRes-2.0)/dfRes
    end

    #precomputation of vsB for convenience
    νS_β            = scaleVar*dfEffectVar
    df_β            = dfEffectVar
    νS_e            = scaleRes*dfRes
    df_e            = dfRes
    #initial values as "0"
    tempBetaVec     = zeros(Float64,nMarkers)
    μ               = mean(y)
    X              .-= ones(Float64,nRecords)*2p
    xpx=[]
    for i in 1:nMarkers
    push!(xpx,dot(X[:,i],X[:,i]))
    end
#    xpx             = diag(X'X)
    ycorr           = y .- μ
    #MCMC starts here
    for iter in 1:chainLength
        #sample residual variance
        varE = sampleVarE(νS_e,ycorr,df_e,nRecords)
        #sample intercept
        ycorr    .+= μ
        rhs      = sum(ycorr)
        invLhs   = 1.0/nRecords
        meanMu   = rhs*invLhs
        μ        = rand(Normal(meanMu,sqrt(invLhs*varE)))
        ycorr    .-= μ
        for r in 1:nRegions
            theseLoci = SNPgroups[r]
            regionSize = length(theseLoci)
            λ_r = varE/varBeta[r]
            for l in theseLoci::UnitRange{Int64}
                BLAS.axpy!(tempBetaVec[l], view(X,:,l), ycorr)
                rhs = view(X,:,l)'*ycorr
                lhs = xpx[l] + λ_r
                meanBeta = lhs\rhs
                tempBetaVec[l] = sampleBeta(meanBeta, lhs, varE)
                BLAS.axpy!(-1*tempBetaVec[l], view(X,:,l), ycorr)
            end
            varBeta[r] = sampleVarBeta(νS_β,tempBetaVec[theseLoci],df_β,regionSize)
        end
        outputControlSt(onScreen,iter,these2Keep,X,tempBetaVec,μ,varBeta,varE,fixedRegSize)
    end
end

function sampleBeta_shaoLei!(tempBetaMat,nTraits,X1,X2,XpiD,XpiD2,Ri,locus,XpiDX,Ycorr1,Ycorr2,invB)
    Ycorr1 .+= view(X1,:,locus).*view(tempBetaMat,1,locus)
    Ycorr2 .+= view(X2,:,locus).*view(tempBetaMat,2,locus)
    rhs     = [view(XpiD,:,locus)'*view(Ycorr1,:,1)*Ri[1];view(XpiD2,:,locus)'*view(Ycorr2,:,1)*Ri[4]]
    invLhs  = inv(XpiDX[locus].*Ri .+ invB)    
    meanBeta = invLhs*rhs
    tempBetaMat[:,locus] = rand(MvNormal(meanBeta,convert(Array,Symmetric(invLhs))))
    Ycorr1 .-= view(X1,:,locus).*view(tempBetaMat,1,locus)
    Ycorr2 .-= view(X2,:,locus).*view(tempBetaMat,2,locus)
end

function outputControl_shaoLei(sum2pq,onScreen,iter,these2Keep,tempBetaMat,μ,covBeta,varE,fixedRegSize,nRegions)
    if iter in these2Keep
        out0 = open(pwd()*"/muOutMT$fixedRegSize", "a")
        writecsv(out0, μ)
        close(out0)
        for t in 1:2
            out1 = open(pwd()*"/beta"*"$t"*"OutMT$fixedRegSize", "a")
            writecsv(out1, tempBetaMat[t,:]')
            close(out1)
        end
        outCov = open(pwd()*"/covBetaOutMT$fixedRegSize", "a")
        printThis = [vcat(covBeta[r]...) for r in 1:nRegions]'
        writecsv(outCov, printThis)
        close(outCov)
        out3 = open(pwd()*"/varEOutMT$fixedRegSize", "a")
        writecsv(out3, vec(varE)')
        close(out3)
        coVarBeta = cov(tempBetaMat')
        genCov    = sum2pq.*coVarBeta
        out4 = open(pwd()*"/varUOutMT$fixedRegSize", "a")
        writecsv(out4, vec(genCov)')
        close(out4)
        if onScreen==true
            corBeta   = cor(tempBetaMat') 
            println("iter $iter \n coVarBeta (Overall): $coVarBeta \n genCov: $genCov \n corBeta: $corBeta \n varE: $varE \n")
        elseif onScreen==false
             @printf("iter %s\n", iter)
        end
    end
end

function prepRegionData(userMapData,chrs,genoTrain,fixedRegSize)
    accRegion = 0
    accRegionVec = [0]
    SNPgroups = []
    ###commented out for Ana's map file
#    mapData = readtable("$snpInfo", header=false, separator=',')
#    if size(mapData,2)<5
#        mapData = hcat(collect(1:size(mapData,1)),mapData,makeunique=true)
#    end
    headMap = [:row, :snpID, :snpOrder ,:chrID, :pos]
    #for Ana's map
    mapData = userMapData
    #
    rename!(mapData , names(mapData), headMap)
    print(mapData[1:5,:])
    mapData[:snpID] = ["M$i" for i in 1:size(mapData,1)] #to convert original IDs like "HAPMAP43437-BTA-101873"
    print(mapData[1:10,:])
    ###
    mapData = mapData[mapData[:chrID] .<= chrs,:]
    # if first col in genoTrain is ID
    # I find cols that are in mapData (<chrs), and select those
#    usedLoci = intersect(names(genoTrain),Symbol.(mapData[:snpID]))
#    mapData = mapData[[find(usedLoci[i].==Symbol.(mapData[:snpID]))[] for i in 1:length(usedLoci)],:] #trim map data
#    genoX = genoTrain[vcat(Symbol("ID"),usedLoci)]    #trim genoData
#     genoX = genoTrain[[1; [find(i -> i == j, names(genoTrain))[] for j in [Symbol(mapData[:snpID][i]) for i in 1:size(mapData,1)]]]]
    #genoX = genoTrain[[find(i -> i == j, names(genoTrain))[] for j in [Symbol(mapData[:snpID][i]) for i in 1:size(mapData,1)]]]
    #genoX = genoTrain
    totLoci = size(genoTrain[2:end],2) # first col is ID
    snpInfoFinal = DataFrame(Any, 0, 3)
    if fixedRegSize==99
        println("fixedRedSize $fixedRegSize")
        snpInfoFinal = mapData[:,[:snpID,:snpOrder,:chrID]]
        accRegion    = length(unique(mapData[:chrID]))
        elseif fixedRegSize==9999
            snpInfoFinal = mapData[:,[:snpID,:snpOrder,:chrID]]
            snpInfoFinal[:,:chrID]  = 1 #was ".=1"
            accRegion    = 1
        else
        for c in 1:chrs
            thisChr = mapData[mapData[:chrID] .== c,:]
            totLociChr = size(thisChr,1)
            TotRegions = ceil(Int,totLociChr/fixedRegSize)
            accRegion += TotRegions
            push!(accRegionVec, accRegion)
            tempGroups = sort(repeat(collect(accRegionVec[c]+1:accRegionVec[c+1]),fixedRegSize))
            snpInfo = DataFrame(Any, length(tempGroups), 3)
            snpInfo[1:totLociChr,1] = collect(1:totLociChr)
            snpInfo[1:totLociChr,2] = thisChr[:snpID]
            snpInfo[:,3] = tempGroups
            dropmissing!(snpInfo)
            snpInfoFinal = vcat(snpInfoFinal,snpInfo)
            @printf("chr %.0f has %.0f groups \n", c, TotRegions)
            println(by(snpInfo, :x3, nrow)[:,2])
        end
        end  #ends if control flow
#    print(snpInfoFinal)
    writecsv("snpInfo",convert(Array,snpInfoFinal))
    for g in 1:accRegion
        push!(SNPgroups,searchsorted(snpInfoFinal[:,3], g))
    end
    return SNPgroups #, genoX
end

function outputControlSt(onScreen,iter,these2Keep,X,tempBetaVec,μ,varBeta,varE,fixedRegSize)
    if iter in these2Keep
        out0 = open(pwd()*"/muOut$fixedRegSize", "a")
        writecsv(out0, μ)
        close(out0) 
        out1 = open(pwd()*"/betaOut$fixedRegSize", "a")
        writecsv(out1, tempBetaVec')
        close(out1)
        out2 = open(pwd()*"/varBetaOut$fixedRegSize", "a")
        writecsv(out2, varBeta')
        close(out2)
        out3 = open(pwd()*"/varEOut$fixedRegSize", "a")
        writecsv(out3, varE)
        close(out3)
        varUhat = var(X*tempBetaVec)
        out4 = open(pwd()*"/varUhatOut$fixedRegSize", "a")
        writecsv(out4, varUhat)
        close(out4)
        if onScreen==true
#            varU = var(X*tempBetaVec)
            @printf("iter %s varUhat %.2f varE %.2f\n", iter, varUhat, varE)
        elseif onScreen==false
             @printf("iter %s\n", iter)
        end
    end
end

function fileControlSt(fixedRegSize)
    for f in ["muOut$fixedRegSize" "betaOut$fixedRegSize" "varBetaOut$fixedRegSize" "varEOut$fixedRegSize" "varUhatOut$fixedRegSize"]
        if isfile(f)==true
            rm(f)
            println("$f removed")
        end
    end
end

function fileControl(nTraits,fixedRegSize)
    files2Remove = ["muOutMT$fixedRegSize", "varEOutMT$fixedRegSize", "covBetaOutMT$fixedRegSize", "varUOutMT$fixedRegSize"]
    for t in 1:nTraits
        push!(files2Remove,"beta"*"$t"*"Out$fixedRegSize")
        push!(files2Remove,"varBeta"*"$t"*"Out$fixedRegSize")
    end
    for f in files2Remove
        if isfile(f)==true
            rm(f)
            println("$f removed")
        end
    end
end

function sampleBeta(meanBeta, lhs, varE)
    return rand(Normal(meanBeta,sqrt(lhs\varE)))
end

function sampleVarBeta(νS_β,whichLoci,df_β,regionSize)
    return((νS_β + dot(whichLoci,whichLoci))/rand(Chisq(df_β + regionSize)))
end
function sampleVarE(νS_e,yCorVec,df_e,nRecords)
    return((νS_e + dot(yCorVec,yCorVec))/rand(Chisq(df_e + nRecords)))
end
function sampleVarE_w(νS_e,yCorVec,wVec,df_e,nRecords)
    return((νS_e + dot((yCorVec.*wVec),yCorVec))/rand(Chisq(df_e + nRecords)))
end
function sampleCovBeta(dfβ, regionSize, Vb , tempBetaMat, theseLoci)
    Sb = tempBetaMat[:,theseLoci]*tempBetaMat[:,theseLoci]'
    return rand(InverseWishart(dfβ + regionSize, Vb + Sb))
end
