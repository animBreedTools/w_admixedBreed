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
    
    breedProp = convert(Array{Float64},breedProp)
    bp               = mean(y)*vec(mean(breedProp,1))
    println(bp)
    F = breedProp
    FpiD = F'*iD
    iFpiDF = inv(F'*iD*F)
    
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
        #sample fixed effects breed proportions
        ycorr    .+= F*bp
        rhs      = FpiD*ycorr
        invLhs   = iFpiDF
        meanMu   = vec(invLhs*rhs)
        bp       = rand(MvNormal(meanMu,convert(Array,Symmetric(invLhs*varE)))) 
        ycorr    .-= F*bp
        
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
        outputControlSt(onScreen,iter,these2Keep,X,tempBetaVec,[μ bp'],varBeta,varE,fixedRegSize)
    end
end

function w_mtBayesPR_shaoLei(genoTrain::DataFrame,genoTrain2::DataFrame, phenoTrain, phenoTrain2,weights,weights2,snpInfo::String, chrs::Int64, fixedRegSize::Int64, varGenotypic::Array{Float64}, varResidual1::Float64,varResidual2::Float64,chainLength::Int64, burnIn::Int64, outputFreq::Int64, onScreen::Bool)
    SNPgroups = prepRegionData(snpInfo, chrs, genoTrain, fixedRegSize)
    these2Keep = collect((burnIn+outputFreq):outputFreq:chainLength) #print these iterations
    nRegions    = length(SNPgroups)
    println("number of regions: ", nRegions)
    dfEffect    = 3.0
    dfRes       = 3.0
    X1           = convert(Array{Float64}, genoTrain[:,2:end])  #first colum is ID
    X2           = convert(Array{Float64}, genoTrain2[:,2:end])  #first colum is ID
    genoTrain  = 0 #release memory
    genoTrain2 = 0
    println("X is this size", size(X1),size(X2))
    Y1           = convert(Array{Float64}, phenoTrain)
    Y2           = convert(Array{Float64}, phenoTrain2)
    println("Y1 is this size", size(Y1))
    println("Y2 is this size", size(Y2))
    nTraits, nRecords1, nRecords2 , nMarkers   = 2, size(Y1,1), size(Y2,1), size(X1,2)
    w           = convert(Array{Float64}, weights)
    iD          = full(Diagonal(w))  # Dii is wii=r2/(1-r2)==>iDii is (1-r2)/r2
    w2           = convert(Array{Float64}, weights2)
    iD2          = full(Diagonal(w2))  # Dii is wii=r2/(1-r2)==>iDii is (1-r2)/r2
    fileControl(nTraits,fixedRegSize)
    p1           = mean(X1,dims=1)./2.0
    sum2pq1      = sum(2*(1 .- p1).*p1)
    
    p2           = mean(X2,dims=1)./2.0
    sum2pq2      = sum(2*(1 .- p2).*p2)

    sum2pq       = sqrt.([sum2pq1; sum2pq2]*[sum2pq1; sum2pq2]')
    println(sum2pq)
    
    #priors
const    dfβ         = dfEffect + nTraits
const    scaleRes1    = varResidual1*(dfRes-2.0)/dfRes    
const    scaleRes2    = varResidual2*(dfRes-2.0)/dfRes    


    if varGenotypic==0.0
        covBeta  = fill([0.003 0;0 0.003],nRegions)
        Vb       = covBeta[1]
        else
        covBeta  = fill(varGenotypic./sum2pq,nRegions)
        Vb       = covBeta[1].*(dfβ-nTraits-1)
    end

    νS_e1           = scaleRes1*dfRes
    df_e            = dfRes
    νS_e2           = scaleRes2*dfRes

    
    #initial Beta values as "0"
    tempBetaMat     = zeros(Float64,nTraits,nMarkers)
    μ               = [mean(Y1) mean(Y2)]    
    X1             .-= ones(Float64,nRecords1)*2*p1    
    X2             .-= ones(Float64,nRecords2)*2*p2
    
    XpiDX = []
    for j in 1:nMarkers
        this = Array{Float64}(nTraits,nTraits)
        this[1,1] = dot((X1[:,j].*w),X1[:,j])
        this[2,2] = dot((X2[:,j].*w2),X2[:,j])
        this[1,2] =this[2,1] =0.0
        XpiDX = push!(XpiDX,this)
    end
    XpiD             = iD*X1
    XpiD2            = iD2*X2
        
    Ycorr1 = Y1 .- μ[1]
    Ycorr2 = Y2 .- μ[2]
    
    for iter in 1:chainLength
        #sample residual var
        R1 = sampleVarE_w(νS_e1,Ycorr1,w,df_e,nRecords1)
        R2 = sampleVarE_w(νS_e2,Ycorr2,w2,df_e,nRecords2)
        Rmat = [R1 0;0 R2]
        Ri = inv(Rmat)

        Ycorr1 = Ycorr1 .+ μ[1]
        Ycorr2 = Ycorr2 .+ μ[2] 
        
        rhs = sum(view(Ycorr1,:,1))
        invLhs = 1.0/nRecords1
        mean = rhs*invLhs
        μ[1] = rand(Normal(mean,sqrt(invLhs*Rmat[1,1])))
        
        rhs = sum(view(Ycorr2,:,1))
        invLhs = 1.0/nRecords2
        mean = rhs*invLhs
        μ[2] = rand(Normal(mean,sqrt(invLhs*Rmat[2,2])))

        Ycorr1 = Ycorr1 .- μ[1]
        Ycorr2 = Ycorr2 .- μ[2]
        
        for r in 1:nRegions
            theseLoci = SNPgroups[r]
            regionSize = length(theseLoci)
            invB = inv(covBeta[r])
            for locus in theseLoci::UnitRange{Int64}
                sampleBeta_shaoLei!(tempBetaMat,nTraits,X1,X2,XpiD,XpiD2,Ri,locus,XpiDX,Ycorr1,Ycorr2,invB)
            end
            covBeta[r] = sampleCovBeta(dfβ,regionSize,Vb,tempBetaMat, theseLoci)
        end
        outputControl_shaoLei(sum2pq,onScreen,iter,these2Keep,tempBetaMat,μ,covBeta,Rmat,fixedRegSize,nRegions)
    end
    GC.gc()
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
