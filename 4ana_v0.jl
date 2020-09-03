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
    iD          = Matrix(Diagonal(w))  # Dii is 1/wii=1/(r2/(1-r2))==> Dii is (1-r2)/r2 ==> iDii is r2/(1-r2)
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
    println("y is this size", size(y))
    nTraits, nRecords , nMarkers   = size(y,2), size(y,1), size(X,2)
    w           = convert(Array{Float64}, weights)
    iD          = Matrix(Diagonal(w))  # Dii is 1/wii=1/(r2/(1-r2))==> Dii is (1-r2)/r2 ==> iDii is r2/(1-r2)
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

#one trait multiple components
function bayesPR2_b(randomEffects, centered, phenoTrain, weights, locusID, userMapData, chrs, fixedRegSize, varGenotypic, varB, varResidual, chainLength, burnIn, outputFreq, onScreen)
    SNPgroups  = prepRegionData(userMapData, chrs, locusID, fixedRegSize)
    these2Keep = collect((burnIn+outputFreq):outputFreq:chainLength) #print these iterations
    nRegions    = length(SNPgroups)
    println("number of regions: ", nRegions)
    nMarkers = length(vcat(SNPgroups...))
    nRecords = size(phenoTrain,1)
    println("number of markers: ", nMarkers)
    println("number of records: ", nRecords)
    
    w           = convert(Array{Float64}, weights)
    iD          = Matrix(Diagonal(w))  # Dii is 1/wii=1/(r2/(1-r2))==> Dii is (1-r2)/r2 ==> iDii is r2/(1-r2)

    nRandComp = length(split(randomEffects, " "))
    sum2pq = Array{Float64}(undef,nRandComp)
    
    for i in 1:nRandComp
        this = split(randomEffects, " ")[i]
        println(this)
        @eval $(Symbol("M$i")) = convert(Array{Float64},eval(Symbol("$(split(randomEffects, " ")[$i])")))
       
        if centered==0
            p           = mean(eval(Symbol("M$i")),dims=1)./2.0
            sum2pq[i]   = sum(2*(1 .- p).*p)
                 
            nowM   = eval(Symbol("M$i"))
            nowM .-= ones(Float64,nRecords)*2p      
            @eval $(Symbol("M$i")) = $nowM
            else sum2pq[i] = centered[i] 
        end
        println(@eval $(Symbol("M$i"))[1:3,1:3])
    end
    nowM = 0
  println(whos()) 
    m1piDm1=[]
    m2piDm2=[]
    m3piDm3=[]
    m4piDm4=[]
    for i in 1:nMarkers
        push!(m1piDm1,dot(M1[:,i].*w,M1[:,i]))
        push!(m2piDm2,dot(M2[:,i].*w,M2[:,i]))
        push!(m3piDm3,dot(M3[:,i].*w,M3[:,i]))
        push!(m4piDm4,dot(M4[:,i].*w,M4[:,i]))
    end
       
    fileControlSt2(fixedRegSize)

    #priors
    dfEffectVar = 4.0  #noCor
    dfRes       = 4.0
    
    dfβ    = dfEffectVar + nRandComp
    
#    mat2pq = sqrt.(sum2pq*sum2pq')
    mat2pq = centered 

    if varGenotypic==0.0
#        covBeta  = fill(Matrix(Diagonal(fill((dfβ-nRandComp-1).*0.001,nRandComp))),nRegions)
        covBeta  = fill(Matrix(Diagonal(varB)),nRegions) #I added the use of varB here
        Vb       = covBeta[1]
        println("prior varB: $Vb")
        else
        covBeta  = fill(Matrix(Diagonal(varGenotypic./mat2pq)),nRegions) ##Array of arrays. covBeta[1] is the array for first region. It is not variance for 1,1
        Vb       = covBeta[1].*(dfβ-nRandComp-1)
    end
    
    Vb      = covBeta[1].*(dfEffectVar-2.0)/dfEffectVar

    νS_β            = diag(Vb.*dfEffectVar)
    df_β            = dfEffectVar
            
    if varResidual==0.0
        varResidual  = 0.0005
        scaleRes     = 0.0005
        else
        scaleRes    = varResidual*(dfRes-2.0)/dfRes    
    end
    
    y           = convert(Array{Float64}, phenoTrain)        
   
    #precomputation of vsE for convenience
    νS_e            = scaleRes*dfRes
    df_e            = dfRes
    #initial values as "0"
    tempBetaMat     = zeros(Float64,nRandComp,nMarkers)
    μ               = mean(y)
    ##########
#    m1piDm1         = diag((M1.*w)'*M1)  #w[i] is already iD[i,i]
    M1piD           = iD*M1        #this is to iterate over columns in the body "dot(view(XpiD,:,l),ycorr)"
#    m2piDm2         = diag((M2.*w)'*M2) #I do it up with push!
    M2piD           = iD*M2
#    m3piDm3         = diag((M3.*w)'*M3)
    M3piD           = iD*M3
#    m4piDm4         = diag((M4.*w)'*M4)
    M4piD           = iD*M4
    ##########
    
    ycorr           = y .- μ
    
    #MCMC starts here
    for iter in 1:chainLength
        #sample residual variance
        varE = sampleVarE_w(νS_e,ycorr,w,df_e,nRecords)
        
        #sample intercept
        ycorr  .+= μ
        rhs      = sum(ycorr)
        invLhs   = 1.0/nRecords
        meanMu   = rhs*invLhs
        μ        = rand(Normal(meanMu,sqrt(invLhs*varE)))
        ycorr  .-= μ
        
        for r in 1:nRegions
            theseLoci = SNPgroups[r]
            regionSize = length(theseLoci)
            lambda = diag(varE./(covBeta[r]))
            for locus in theseLoci::UnitRange{Int64}
                
                BLAS.axpy!(view(tempBetaMat,1,locus)[],view(M1,:,locus),ycorr)
                rhs = BLAS.dot(view(M1piD,:,locus),ycorr)
                lhs   = m1piDm1[locus] + lambda[1]
                meanBeta = lhs\rhs
                tempBetaMat[1,locus] = sampleBeta(meanBeta, lhs, varE)
                BLAS.axpy!(-1*view(tempBetaMat,1,locus)[],view(M1,:,locus),ycorr)
                
                BLAS.axpy!(view(tempBetaMat,2,locus)[],view(M2,:,locus),ycorr)
                rhs = BLAS.dot(view(M2piD,:,locus),ycorr)
                lhs   = m2piDm2[locus] + lambda[2]
                meanBeta = lhs\rhs
                tempBetaMat[2,locus] = sampleBeta(meanBeta, lhs, varE)
                BLAS.axpy!(-1*view(tempBetaMat,2,locus)[],view(M2,:,locus),ycorr)
                
                BLAS.axpy!(view(tempBetaMat,3,locus)[],view(M3,:,locus),ycorr)
                rhs = BLAS.dot(view(M3piD,:,locus),ycorr)
                lhs   = m3piDm3[locus] + lambda[3]
                meanBeta = lhs\rhs
                tempBetaMat[3,locus] = sampleBeta(meanBeta, lhs, varE)
                BLAS.axpy!(-1*view(tempBetaMat,3,locus)[],view(M3,:,locus),ycorr)

                BLAS.axpy!(view(tempBetaMat,4,locus)[],view(M4,:,locus),ycorr)
                rhs = BLAS.dot(view(M4piD,:,locus),ycorr)
                lhs   = m4piDm4[locus] + lambda[4]
                meanBeta = lhs\rhs
                tempBetaMat[4,locus] = sampleBeta(meanBeta, lhs, varE)
                BLAS.axpy!(-1*view(tempBetaMat,4,locus)[],view(M4,:,locus),ycorr)

                
            end
#            covBeta[r][1,1] = sampleVarBeta(νS_β[1],tempBetaMat[1,theseLoci],df_β,regionSize)
#            covBeta[r][2,2] = sampleVarBeta(νS_β[2],tempBetaMat[2,theseLoci],df_β,regionSize)
#            covBeta[r][3,3] = sampleVarBeta(νS_β[3],tempBetaMat[3,theseLoci],df_β,regionSize)

            cov1 = sampleVarBeta(νS_β[1],tempBetaMat[1,theseLoci],df_β,regionSize)
            cov2 = sampleVarBeta(νS_β[2],tempBetaMat[2,theseLoci],df_β,regionSize)
            cov3 = sampleVarBeta(νS_β[3],tempBetaMat[3,theseLoci],df_β,regionSize)
            cov4 = sampleVarBeta(νS_β[4],tempBetaMat[4,theseLoci],df_β,regionSize)

            covBeta[r]      = [cov1 0 0 0; 0 cov2 0 0; 0 0 cov3 0; 0 0 0 cov4]
        end
        outputControl2(nRandComp,onScreen,iter,these2Keep,tempBetaMat,μ,covBeta,varE,fixedRegSize,nRegions)
    end
    GC.gc()
end

#one trait multiple components, correlated
function bayesPR2(randomEffects, centered, phenoTrain, weights, locusID, userMapData, chrs, fixedRegSize, varGenotypic, varB, varResidual, chainLength, burnIn, outputFreq, onScreen)
    SNPgroups  = prepRegionData(userMapData, chrs, locusID, fixedRegSize)
    these2Keep = collect((burnIn+outputFreq):outputFreq:chainLength) #print these iterations
    nRegions    = length(SNPgroups)
    println("number of regions: ", nRegions)
    nMarkers = length(vcat(SNPgroups...))
    nRecords = size(phenoTrain,1)
    println("number of markers: ", nMarkers)
    println("number of records: ", nRecords)
   
    w           = convert(Array{Float64}, weights)
    iD          = Matrix(Diagonal(w))  # Dii is 1/wii=1/(r2/(1-r2))==> Dii is (1-r2)/r2 ==> iDii is r2/(1-r2)

    nRandComp = length(split(randomEffects, " "))
    sum2pq = Array{Float64}(undef,nRandComp)

    for i in 1:nRandComp
        this = split(randomEffects, " ")[i]
        println(this)
        @eval $(Symbol("M$i")) = convert(Array{Float64},eval(Symbol("$(split(randomEffects, " ")[$i])")))

        if centered==0
            p           = mean(eval(Symbol("M$i")),dims=1)./2.0
            sum2pq[i]   = sum(2*(1 .- p).*p)

            nowM   = eval(Symbol("M$i"))
            nowM .-= ones(Float64,nRecords)*2p
            @eval $(Symbol("M$i")) = $nowM
            else sum2pq[i] = centered[i]
        end
        println(@eval $(Symbol("M$i"))[1:3,1:3])
    end
    nowM = 0
 
    sqrtW = sqrt.(w) ####for having M1piDM1 as well as M1piDM2....
    MpM = []
    for j in 1:nMarkers
        tempM = Array{Float64}(undef,nRecords,0)
        for k in 1:nRandComp
            nowM  = @eval $(Symbol("M$k"))
            tempM = convert(Array{Float64},hcat(tempM,nowM[:,j].*sqrtW)) ###sqrtW here
        end
    MpM = push!(MpM,tempM'tempM)
    end
    nowM  = 0
    tempM = 0
       
    fileControlSt2(fixedRegSize)

    #priors
    dfEffectVar = 3.0
    dfRes       = 4.0
    
    dfβ    = dfEffectVar + nRandComp
    
 #   mat2pq = sqrt.(sum2pq*sum2pq')
    mat2pq = centered   

    if varGenotypic==0.0
 #       covBeta  = fill(Matrix(Diagonal(fill((dfβ-nRandComp-1).*0.001,nRandComp))),nRegions)
        covBeta  = fill(varB,nRegions) #I added the use of varB here
        Vb       = covBeta[1]
        println("prior varB: $Vb")
        else
        covBeta  = fill(varGenotypic./mat2pq,nRegions)
        Vb       = covBeta[1].*(dfβ-nRandComp-1)
    end
   
    #to avoide singularity in inv(covBeta[r]) for the first iteration
    #this is just the initial (starting) value
    covBeta  = fill(Matrix(Diagonal(covBeta[1])),nRegions)

    if varResidual==0.0
        varResidual  = 0.0005
        scaleRes     = 0.0005
        else
        scaleRes    = varResidual*(dfRes-2.0)/dfRes    
    end
    
    y           = convert(Array{Float64}, phenoTrain)        
   
    #precomputation of vsE for convenience
    νS_e            = scaleRes*dfRes
    df_e            = dfRes
    #initial values as "0"
    tempBetaMat     = zeros(Float64,nRandComp,nMarkers)
    μ               = mean(y)
    ##########
#    m1piDm1         = diag((M1.*w)'*M1)  #w[i] is already iD[i,i]
    M1piD           = iD*M1        #this is to iterate over columns in the body "dot(view(XpiD,:,l),ycorr)"
#    m2piDm2         = diag((M2.*w)'*M2) #I do it up with push!
    M2piD           = iD*M2
#    m3piDm3         = diag((M3.*w)'*M3)
    M3piD           = iD*M3
#    m4piDm4         = diag((M4.*w)'*M4)
    M4piD           = iD*M4
    ##########
    
    ycorr           = y .- μ
    
    #MCMC starts here
    for iter in 1:chainLength
        #sample residual variance
        varE = sampleVarE_w(νS_e,ycorr,w,df_e,nRecords)
        iVarE = 1/varE
        
        #sample intercept
        ycorr  .+= μ
        rhs      = sum(ycorr)
        invLhs   = 1.0/nRecords
        meanMu   = rhs*invLhs
        μ        = rand(Normal(meanMu,sqrt(invLhs*varE)))
        ycorr  .-= μ
        
        for r in 1:nRegions
            theseLoci = SNPgroups[r]
            regionSize = length(theseLoci)
            invB = inv(covBeta[r]) ###################check this
            for locus in theseLoci::UnitRange{Int64}
#                sampleCorRandomBeta!(M,MpM,tempBetaMat,locus,ycorr,varE,invB)
                BLAS.axpy!(view(tempBetaMat,1,locus)[],view(M1,:,locus),ycorr)
                BLAS.axpy!(view(tempBetaMat,2,locus)[],view(M2,:,locus),ycorr)
                BLAS.axpy!(view(tempBetaMat,3,locus)[],view(M3,:,locus),ycorr)
                BLAS.axpy!(view(tempBetaMat,4,locus)[],view(M4,:,locus),ycorr)
                rhs = [BLAS.dot(view(M1piD,:,locus),ycorr) ; BLAS.dot(view(M2piD,:,locus),ycorr) ; BLAS.dot(view(M3piD,:,locus),ycorr) ; BLAS.dot(view(M4piD,:,locus),ycorr)]*iVarE
                invLhs   = inv(MpM[locus]*iVarE + invB)
                meanBeta = invLhs*rhs
                tempBetaMat[:,locus] = rand(MvNormal(meanBeta,convert(Array,Symmetric(invLhs))))
                BLAS.axpy!(-1*view(tempBetaMat,1,locus)[],view(M1,:,locus),ycorr)
                BLAS.axpy!(-1*view(tempBetaMat,2,locus)[],view(M2,:,locus),ycorr)
                BLAS.axpy!(-1*view(tempBetaMat,3,locus)[],view(M3,:,locus),ycorr)
                BLAS.axpy!(-1*view(tempBetaMat,4,locus)[],view(M4,:,locus),ycorr)
            end
#            Random.seed!(iter)
            covBeta[r] = sampleCovBeta(dfβ,regionSize,Vb,tempBetaMat,theseLoci)
#            println(covBeta[r])
#            Random.seed!(iter)
#            covBeta[r] = sampleCovBeta_iW(dfβ,regionSize,Vb,tempBetaMat,theseLoci)
#            println(covBeta[r])
        end
        outputControl2(nRandComp,onScreen,iter,these2Keep,tempBetaMat,μ,covBeta,varE,fixedRegSize,nRegions)
    end
    GC.gc()
end

###now genoTrain is excluded, and it takes locusIDs, and map to trim
function prepRegionData(userMapData,chrs,locusID,fixedRegSize)
    accRegion = 0
    accRegionVec = [0]
    SNPgroups = []
    headMap = [:row, :snpID, :snpOrder ,:chrID, :pos]
    #for Ana's map
    mapData = userMapData
    #
    rename!(mapData, headMap)     ############## names(mapData) removed ############
    print(mapData[1:5,:])
    print(mapData[1:10,:])
    ###
    mapData = mapData[mapData[!,:chrID] .<= chrs,:]
    # if first col in genoTrain is ID
    # I find cols that are in mapData (<chrs), and select those
    usedLoci = intersect(Symbol.(locusID),Symbol.(mapData[!,:snpID]))
    mapData = mapData[[findall(usedLoci[i].==Symbol.(mapData[!,:snpID]))[] for i in 1:length(usedLoci)],:] #trim map data  ######## find -> findall #########
    println([findall(usedLoci[i].==Symbol.(mapData[!,:snpID]))[] for i in 1:length(usedLoci)])    ######## just added to check #######  

    totLoci = length(usedLoci) # first col is ID
    println("totalLoci in MAP: $totLoci")
    snpInfoFinal = DataFrame(Any, 0, 3)
    if fixedRegSize==99
        println("fixedRedSize $fixedRegSize")
        snpInfoFinal = mapData[:,[:snpID,:snpOrder,:chrID]]
        accRegion    = length(unique(mapData[!,:chrID]))
        elseif fixedRegSize==9999
            snpInfoFinal = mapData[:,[:snpID,:snpOrder,:chrID]]
            snpInfoFinal[!,:chrID]  .= 1 #was "1"
            accRegion    = 1
        else
        for c in 1:chrs
            thisChr = mapData[mapData[!,:chrID] .== c,:]
            totLociChr = size(thisChr,1)
            TotRegions = ceil(Int,totLociChr/fixedRegSize)
            accRegion += TotRegions
            push!(accRegionVec, accRegion)
            tempGroups = sort(repeat(collect(accRegionVec[c]+1:accRegionVec[c+1]),fixedRegSize))
            snpInfo = DataFrame(Any, length(tempGroups), 3)
            snpInfo[1:totLociChr,1] = collect(1:totLociChr)
            snpInfo[1:totLociChr,2] = thisChr[!,:snpID]
            snpInfo[:,3] = tempGroups
            dropmissing!(snpInfo)
            snpInfoFinal = vcat(snpInfoFinal,snpInfo)
            @printf("chr %.0f has %.0f groups \n", c, TotRegions)
            println(by(snpInfo, :x3, nrow)[:,2])
        end
        end  #ends if control flow
#    print(snpInfoFinal)
    CSV.write("snpInfo",snpInfoFinal,header=false)
    for g in 1:accRegion
        push!(SNPgroups,searchsorted(snpInfoFinal[:,3], g))
    end
    return SNPgroups #, genoX
end

function outputControlSt(onScreen,iter,these2Keep,X,tempBetaVec,μ,varBeta,varE,fixedRegSize)
    if iter in these2Keep
        out0 = open(pwd()*"/muOut$fixedRegSize", "a")
        CSV.write(out0, μ,header=false)
        close(out0) 
        out1 = open(pwd()*"/betaOut$fixedRegSize", "a")
        CSV.write(out1, tempBetaVec',header=false)
        close(out1)
        out2 = open(pwd()*"/varBetaOut$fixedRegSize", "a")
        CSV.write(out2, varBeta',header=false)
        close(out2)
        out3 = open(pwd()*"/varEOut$fixedRegSize", "a")
        CSV.write(out3, varE,header=false)
        close(out3)
        varUhat = var(X*tempBetaVec)
        out4 = open(pwd()*"/varUhatOut$fixedRegSize", "a")
        CSV.write(out4, varUhat,header=false)
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

function fileControlSt2(fixedRegSize)
    for f in ["muOut$fixedRegSize" "beta1Out$fixedRegSize" "beta2Out$fixedRegSize" "beta3Out$fixedRegSize" "beta4Out$fixedRegSize" "covBetaOut$fixedRegSize" "varEOut$fixedRegSize"]
        if isfile(f)==true
            rm(f)
            println("$f removed")
        end
    end
end

function outputControl2(nRandComp,onScreen,iter,these2Keep,tempBetaMat,μ,covBeta,varE,fixedRegSize,nRegions)
    if iter in these2Keep
        out0 = open(pwd()*"/muOut$fixedRegSize", "a")
        CSV.write(out0, μ,header=false)
        close(out0)
        for t in 1:nRandComp
            out1 = open(pwd()*"/beta"*"$t"*"Out$fixedRegSize", "a")
            CSV.write(out1, tempBetaMat[t,:]',header=false)
            close(out1)
        end
        outCov = open(pwd()*"/covBetaOut$fixedRegSize", "a")
        printThis = [vcat(covBeta[r]...) for r in 1:nRegions]'
        CSV.write(outCov, printThis,header=false)
        close(outCov)
        out3 = open(pwd()*"/varEOut$fixedRegSize", "a")
        CSV.write(out3, varE,header=false)
        close(out3)  
        if onScreen==true
            coVarBeta = cov(tempBetaMat')
            corBeta   = cor(tempBetaMat') 
            println("iter $iter \n coVarBeta (Overall): $coVarBeta \n corBeta: $corBeta \n varE: $varE \n")
        elseif onScreen==false
             @printf("iter %s\n", iter)
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

###Experimental iW ____ ALMOST DOUBLED THE SPEED
function sampleCovBeta_iW(dfβ, regionSize, Vb , tempBetaMat, theseLoci)
    Sb = tempBetaMat[:,theseLoci]*tempBetaMat[:,theseLoci]'
    return inv(rand(Wishart(dfβ + regionSize, convert(Array,Symmetric(inv(Vb + Sb))))))
end
