
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

'''Functions for Fuzzy Transform'''
def _statisticsProperties(verticesProperties, verticesProperties2, edgeList, alpha, depthLevel, gama):
    verticesFluxValue = []
    verticesFluxValue2 = []

    for i,vp in enumerate(verticesProperties):
        if not vp:
            continue
        verticesFluxValue.extend(np.array(vp)[:,0].tolist())
        verticesFluxValue2.extend(np.array(verticesProperties2[i])[:,0].tolist())

    meanVerticesFluxValue = np.mean(verticesFluxValue)
    sigmaVerticesFluxValue = np.std(verticesFluxValue)
    meanVerticesFluxValue2 = np.mean(verticesFluxValue2)
    sigmaVerticesFluxValue2 = np.std(verticesFluxValue2)
    levelWeights = np.exp(-0.3 * np.array(range(depthLevel))) * gama
    meanValue1 = np.sum(meanVerticesFluxValue * levelWeights)
    sigmaValue1 = np.sqrt(np.sum(pow(sigmaVerticesFluxValue,2) * pow(levelWeights,2)))
    meanValue2 = np.sum(meanVerticesFluxValue2 * levelWeights)
    sigmaValue2 = np.sqrt(np.sum(pow(sigmaVerticesFluxValue2,2) * pow(levelWeights,2)))
    meanValue = alpha * meanValue1 + (1-alpha) * meanValue2
    sigmaValue = np.sqrt(pow(alpha * sigmaValue1,2) + pow((1-alpha) * sigmaValue2,2))
    # Computing Edge Statistics
    edgeLength = [len(edge) for edge in edgeList]
    meanEdgeLength = np.sum(np.mean(edgeLength) * levelWeights)
    sigmaEdgeLength = np.sqrt(np.sum(pow(np.std(edgeLength) * levelWeights,2)))
    return meanValue, sigmaValue, meanEdgeLength, sigmaEdgeLength

# Importance Value of each edge
def _computeImportanceValue(vertexInd, eligibleEdgesInd, verticesProperties, vertices, edgeList, adjacencyMatrix, depthLevel , gama):
    edgeCheckCounter_iv = np.zeros((1,len(edgeList)))
    importanceValue = np.zeros((1,len(eligibleEdgesInd)))
    effectiveLength = np.zeros((1,len(eligibleEdgesInd))) if len(eligibleEdgesInd) > 0 else 0
    if ((depthLevel > 0 ) &  (eligibleEdgesInd.any())):
        searchEdgesInd = np.where(np.in1d(np.array(verticesProperties[vertexInd])[:,1],eligibleEdgesInd))[0]
        searchPointValue = gama * np.sum(np.array(verticesProperties[vertexInd])[searchEdgesInd,0])
        searchEdgeLength = gama * max([len(edgeList[x]) for x in eligibleEdgesInd])

        for i,eei in enumerate(eligibleEdgesInd):
            edgeCheckCounter_iv[0,eei] = 1
            linkedEdgesnumber_iv = adjacencyMatrix[vertexInd,np.where(adjacencyMatrix[vertexInd,:]!=0)[0]]
            if ((linkedEdgesnumber_iv == eei + 1).any()):
                searchPointNew = edgeList[eei][-1]
            else:
                searchPointNew = edgeList[eei][0]
            vertexIndNew = np.argwhere(np.all(vertices == searchPointNew, axis=1 ))[0,0]
            linkedEdgesNumberNew = abs(adjacencyMatrix[vertexIndNew,np.where(adjacencyMatrix[vertexIndNew,:]!=0)[0]])
            eligibleEdgesIndNew = linkedEdgesNumberNew[edgeCheckCounter_iv[0,linkedEdgesNumberNew-1] == 0] - 1
            importanceValueSub, effectiveLengthSub = _computeImportanceValue(vertexIndNew, eligibleEdgesIndNew, verticesProperties, vertices, edgeList, adjacencyMatrix, depthLevel - 1 , np.exp(-0.3) * gama)
            importanceValue[0,i] = searchPointValue + np.sum(importanceValueSub)
            effectiveLength[0,i] = searchEdgeLength + np.max(effectiveLengthSub)
    return importanceValue,effectiveLength
# Fuzzy Transform Pruning
def fuzzyTransform(skeleton, vertices, edgeList, edgeProperties, verticesProperties, verticesProperties2, adjacencyMatrix, returnDB = False, BT = 0 , maxBD = 0.7, alpha = 0.2, depthLevel = 7, gama = 1.05 ):
    mainSkeletonMap = np.zeros(skeleton.shape)
    branchMap = np.zeros(skeleton.shape)
    skeletonNew = np.zeros(skeleton.shape)

    # Find Statistic charactristics of values and lengths
    meanValue, sigmaValue, meanEdgeLength, sigmaEdgeLength = _statisticsProperties(verticesProperties,
                                                                                  verticesProperties2, edgeList, alpha,
                                                                                  depthLevel, gama)

    '''Creating Fuzzy Inference System For Main Skeleton Belief'''
    sigmaValue = 0.05 if sigmaValue == 0 else sigmaValue
    sigmaEdgeLength = 0.05 if sigmaEdgeLength == 0 else sigmaEdgeLength

    # Adding First input membership function
    minValue = meanValue - 6 * sigmaValue
    maxValue = meanValue + 6 * sigmaValue

    importanceValueInput = ctrl.Antecedent(np.arange(minValue, maxValue, 0.001), 'Importance Value')
    importanceValueInput['Low'] = fuzz.gaussmf(importanceValueInput.universe, minValue, sigmaValue)
    importanceValueInput['Medium'] = fuzz.gaussmf(importanceValueInput.universe, meanValue, sigmaValue)
    importanceValueInput['High'] = fuzz.gaussmf(importanceValueInput.universe, maxValue, sigmaValue)

    # Adding Second input membership function
    minEdgeLength = max(meanEdgeLength - 5 * sigmaEdgeLength, 0)
    maxEdgeLength = meanEdgeLength + 5 * sigmaEdgeLength

    edgeLengthInput = ctrl.Antecedent(np.arange(minEdgeLength, maxEdgeLength, 0.001), 'Edge Length')
    edgeLengthInput['Small'] = fuzz.gaussmf(edgeLengthInput.universe, minEdgeLength, sigmaEdgeLength)
    edgeLengthInput['Medium'] = fuzz.gaussmf(edgeLengthInput.universe, meanEdgeLength, sigmaEdgeLength)
    edgeLengthInput['Long'] = fuzz.gaussmf(edgeLengthInput.universe, maxEdgeLength, sigmaEdgeLength)

    # Adding Output membership function
    outputSigma = 0.05
    mainSkeletonBelief = ctrl.Consequent(np.arange(0, 1, 0.01), 'Main Skeleton Degree of Belief')
    mainSkeletonBelief['Very Low'] = fuzz.gaussmf(mainSkeletonBelief.universe, 0, outputSigma)
    mainSkeletonBelief['Low'] = fuzz.gaussmf(mainSkeletonBelief.universe, 0.25, outputSigma)
    mainSkeletonBelief['Average'] = fuzz.gaussmf(mainSkeletonBelief.universe, 0.5, outputSigma)
    mainSkeletonBelief['High'] = fuzz.gaussmf(mainSkeletonBelief.universe, 0.75, outputSigma)
    mainSkeletonBelief['Very High'] = fuzz.gaussmf(mainSkeletonBelief.universe, 1, outputSigma)

    # Adding Rules
    rule11 = ctrl.Rule(importanceValueInput['Low'] & edgeLengthInput['Small'], mainSkeletonBelief['Very Low'])
    rule12 = ctrl.Rule(importanceValueInput['Low'] & edgeLengthInput['Medium'], mainSkeletonBelief['Low'])
    rule13 = ctrl.Rule(importanceValueInput['Medium'] & edgeLengthInput['Small'], mainSkeletonBelief['Low'])
    rule14 = ctrl.Rule(importanceValueInput['Medium'] & edgeLengthInput['Medium'], mainSkeletonBelief['Average'])
    rule15 = ctrl.Rule(importanceValueInput['Low'] & edgeLengthInput['Long'], mainSkeletonBelief['Average'])
    rule16 = ctrl.Rule(importanceValueInput['High'] & edgeLengthInput['Small'], mainSkeletonBelief['Average'])
    rule17 = ctrl.Rule(importanceValueInput['Medium'] & edgeLengthInput['Long'], mainSkeletonBelief['High'])
    rule18 = ctrl.Rule(importanceValueInput['High'] & edgeLengthInput['Medium'], mainSkeletonBelief['High'])
    rule19 = ctrl.Rule(importanceValueInput['High'] & edgeLengthInput['Long'], mainSkeletonBelief['Very High'])

    mainSkeletonFIS_ctrl = ctrl.ControlSystem([rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19])
    mainSkeletonFIS = ctrl.ControlSystemSimulation(mainSkeletonFIS_ctrl)

    '''Creating Fuzzy Inference System For Branch Belief'''
    # Adding First Input membership function
    MSDB = ctrl.Antecedent(np.arange(0, 1, 0.001), 'Main Skeleton Degree of Belief')
    MSDB['Low'] = fuzz.trapmf(MSDB.universe, [0, 0, 0.3, 0.55])
    MSDB['High'] = fuzz.trapmf(MSDB.universe, [0.4, 0.7, 1, 1])

    # Adding Second Input membership function
    minEdgeLength1 = meanEdgeLength - 4 * sigmaEdgeLength
    maxEdgeLength1 = meanEdgeLength + 4 * sigmaEdgeLength

    edgeLengthInput1 = ctrl.Antecedent(np.arange(minEdgeLength1, maxEdgeLength1, 0.001), 'Edge Length')
    edgeLengthInput1['Small'] = fuzz.trapmf(edgeLengthInput1.universe,
                                            [minEdgeLength1, minEdgeLength1, minEdgeLength1 + sigmaEdgeLength,
                                             meanEdgeLength + sigmaEdgeLength])
    edgeLengthInput1['Long'] = fuzz.trapmf(edgeLengthInput1.universe,
                                           [meanEdgeLength - sigmaEdgeLength, maxEdgeLength1 - 2 * sigmaEdgeLength,
                                            maxEdgeLength1, maxEdgeLength1])

    # Adding Third Input membership function
    curvatureInput = ctrl.Antecedent(np.arange(-1, 1, 0.01), 'Curvature')
    curvatureInput['Averted'] = fuzz.trapmf(curvatureInput.universe, [-1, -1, -0.7, 0.2])
    curvatureInput['Straight'] = fuzz.trapmf(curvatureInput.universe, [-0.1, 0.7, 1, 1])

    # Adding Output membership function
    branchBelief = ctrl.Consequent(np.arange(0, 1, 0.01), 'Branch Degree of Belief')
    branchBelief['Low'] = fuzz.trapmf(branchBelief.universe, [0, 0, 0.2, 0.4])
    branchBelief['Average'] = fuzz.trimf(branchBelief.universe, [0.4, 0.5, 0.6])
    branchBelief['High'] = fuzz.trapmf(branchBelief.universe, [0.6, 0.8, 1, 1])

    # Adding Rules
    rule21 = ctrl.Rule(MSDB['Low'] & edgeLengthInput1['Small'], branchBelief['High'])
    rule22 = ctrl.Rule(MSDB['High'] & edgeLengthInput1['Small'] & curvatureInput['Averted'], branchBelief['Average'])
    rule23 = ctrl.Rule(MSDB['High'] & edgeLengthInput1['Small'] & curvatureInput['Straight'], branchBelief['Average'])
    rule24 = ctrl.Rule(MSDB['Low'] & edgeLengthInput1['Long'] & curvatureInput['Averted'], branchBelief['Average'])
    rule25 = ctrl.Rule(MSDB['Low'] & edgeLengthInput1['Long'] & curvatureInput['Straight'], branchBelief['Average'])
    rule26 = ctrl.Rule(MSDB['High'] & edgeLengthInput1['Long'], branchBelief['Low'])

    branchFIS_ctrl = ctrl.ControlSystem([rule21, rule22, rule23, rule24, rule25, rule26])
    branchFIS = ctrl.ControlSystemSimulation(branchFIS_ctrl)

    # Compute the Main Skeleton Degree of belief
    rootEdgeIndex = np.argmax(pow(edgeProperties[2, :], 2) * (edgeProperties[0, :] + edgeProperties[1, :]))
    searchVerticesQue = [[edgeList[rootEdgeIndex][0][0], edgeList[rootEdgeIndex][0][1], rootEdgeIndex],
                         [edgeList[rootEdgeIndex][-1][0], edgeList[rootEdgeIndex][-1][1], rootEdgeIndex]]
    edgeCheckCounter = np.zeros((1, len(edgeList)))
    searchVerticeNumber = 0
    # edgeCheckCounter[0, rootEdgeIndex] = 1

    while (searchVerticeNumber <= len(searchVerticesQue) - 1):
        searchVertexPoint = searchVerticesQue[searchVerticeNumber][0:2]
        searchVertexEdgeIndex = searchVerticesQue[searchVerticeNumber][2]
        searchVertexIndex = np.argwhere(np.all(vertices == searchVertexPoint, axis=1))[0, 0]
        linkedEdgesNumber = adjacencyMatrix[searchVertexIndex, np.where(adjacencyMatrix[searchVertexIndex, :] != 0)[0]]
        eligibleEdgesNumber = abs(linkedEdgesNumber)
        # Find the index of Edges that have not been searched
        eligibleEdgesInd = eligibleEdgesNumber[edgeCheckCounter[0, eligibleEdgesNumber - 1] == 0] - 1

        if (eligibleEdgesInd.any()):
            edgesImportanceValue1, edgeEffectiveLength = _computeImportanceValue(searchVertexIndex, eligibleEdgesInd,
                                                                                verticesProperties, vertices, edgeList,
                                                                                adjacencyMatrix, depthLevel, gama)
            edgesImportanceValue2, _ = _computeImportanceValue(searchVertexIndex, eligibleEdgesInd, verticesProperties2,
                                                              vertices, edgeList, adjacencyMatrix, depthLevel, gama)
            edgeEffectiveLength[edgeEffectiveLength < minEdgeLength] = minEdgeLength
            edgeEffectiveLength[edgeEffectiveLength > maxEdgeLength] = maxEdgeLength
            edgesImportanceValue = alpha * edgesImportanceValue1 + (1 - alpha) * edgesImportanceValue2
            edgesImportanceValue[edgesImportanceValue < minValue] = minValue
            edgesImportanceValue[edgesImportanceValue > maxValue] = maxValue
            for i, eei in enumerate(eligibleEdgesInd):
                mainSkeletonFIS.input['Edge Length'] = edgeEffectiveLength[0, i]
                mainSkeletonFIS.input['Importance Value'] = edgesImportanceValue[0, i]
                mainSkeletonFIS.compute()
                edgeMainSkeletonBelief = mainSkeletonFIS.output['Main Skeleton Degree of Belief']
                eligibleEdgePoints = np.array(edgeList[eei])
                mainSkeletonMap[eligibleEdgePoints.T.tolist()] = edgeMainSkeletonBelief
                if ((linkedEdgesNumber == eei + 1).any()):
                    searchVertexIndNew = edgeList[eei][-1]
                else:
                    searchVertexIndNew = edgeList[eei][0]
                searchVerticesQue.append([searchVertexIndNew[0], searchVertexIndNew[1], eei])
        edgeCheckCounter[0, eligibleEdgesInd] = 1
        searchVerticeNumber += 1

    # Compute the Branch Degree of belief and pruning
    searchVerticesQue = [[edgeList[rootEdgeIndex][0][0], edgeList[rootEdgeIndex][0][1], rootEdgeIndex],
                         [edgeList[rootEdgeIndex][-1][0], edgeList[rootEdgeIndex][-1][1], rootEdgeIndex]]
    edgeCheckCounter = np.zeros((1, len(edgeList)))
    searchVerticeNumber = 0
    edgeCheckCounter[0, rootEdgeIndex] = 1
    skeletonNew[np.array(edgeList[rootEdgeIndex]).T.tolist()] = 1

    while (searchVerticeNumber <= len(searchVerticesQue) - 1):
        searchVertexPoint = np.array(searchVerticesQue[searchVerticeNumber][0:2])
        searchVertexEdgeIndex = searchVerticesQue[searchVerticeNumber][2]
        searchEdgeMeanPoint = np.mean(edgeList[searchVertexEdgeIndex], axis=0)
        searchEdgeVector = searchVertexPoint - searchEdgeMeanPoint
        searchVertexIndex = np.argwhere(np.all(vertices == searchVertexPoint, axis=1))[0, 0]
        linkedEdgesNumber = adjacencyMatrix[searchVertexIndex, np.where(adjacencyMatrix[searchVertexIndex, :] != 0)[0]]
        eligibleEdgesNumber = abs(linkedEdgesNumber)
        # Find the index of Edges that have not been searched
        eligibleEdgesInd = eligibleEdgesNumber[edgeCheckCounter[0, eligibleEdgesNumber - 1] == 0] - 1
        addEdgesInd = np.array([])
        if (eligibleEdgesInd.any()):
            _, edgeEffectiveLength = _computeImportanceValue(searchVertexIndex, eligibleEdgesInd, verticesProperties,
                                                             vertices, edgeList, adjacencyMatrix, depthLevel, gama)
            edgeEffectiveLength[edgeEffectiveLength < minEdgeLength1] = minEdgeLength1
            edgeEffectiveLength[edgeEffectiveLength > maxEdgeLength1] = maxEdgeLength1
            eligibleEdgesSecondPoint = np.array([edgeList[x][1] for x in eligibleEdgesInd])
            eligibleEdgesMainSkeletonBelief = mainSkeletonMap[eligibleEdgesSecondPoint.T.tolist()]
            eligibleEdgesLength = edgeProperties[2, eligibleEdgesInd]
            eligibleEdgesMeanPoint = np.array([np.mean(edgeList[x], axis=0) for x in eligibleEdgesInd])
            eligibleEdgesVector = eligibleEdgesMeanPoint - searchVertexPoint
            edgesCurvatureMeasure = np.dot(eligibleEdgesVector, searchEdgeVector.T) / (
                np.linalg.norm(eligibleEdgesVector, axis=1) * np.linalg.norm(searchEdgeVector))
            edgesCurvatureMeasure[eligibleEdgesLength == 2] = 1
            branchBeliefDegree = np.zeros((1, len(eligibleEdgesInd)))
            # Fuzzy library does not support vector inputs, hence we should use loop
            for i, eei in enumerate(eligibleEdgesInd):
                branchFIS.input['Main Skeleton Degree of Belief'] = eligibleEdgesMainSkeletonBelief[i]
                branchFIS.input['Edge Length'] = eligibleEdgesLength[i]
                branchFIS.input['Curvature'] = edgesCurvatureMeasure[i]
                branchFIS.compute()
                edgeBranchBelief = branchFIS.output['Branch Degree of Belief']
                branchBeliefDegree[0, i] = edgeBranchBelief
            minBeliefDegree = np.min(branchBeliefDegree)
            # Finding if the edges have to be added to the Main Skeleton or not
            if (minBeliefDegree <= maxBD):
                beliefWindowSize = pow(BT, abs((maxBD - minBeliefDegree))) * pow((maxBD - minBeliefDegree),
                                                                                 (1.45 + 0.2 * BT)) * np.exp(
                    1.5 - 2.5 * (maxBD - minBeliefDegree))
            else:
                beliefWindowSize = 0

            if ((BT == 0) & (np.sum(branchBeliefDegree == minBeliefDegree) == len(branchBeliefDegree))):
                measureVector = (edgesCurvatureMeasure * eligibleEdgesMainSkeletonBelief) * edgeEffectiveLength
                addEdgesInd = np.array(eligibleEdgesInd[[np.argmax(measureVector)]])
            else:
                addEdgesInd = np.array(
                    eligibleEdgesInd[(branchBeliefDegree <= (minBeliefDegree + beliefWindowSize)).tolist()])
            # Adding selected edges' other point to the Que for next rounds
            for aei in addEdgesInd:
                addEdgePoints = np.array(edgeList[aei])
                skeletonNew[addEdgePoints.T.tolist()] = 1
                if ((linkedEdgesNumber == aei + 1).any()):
                    searchVertexIndNew = edgeList[aei][-1]
                else:
                    searchVertexIndNew = edgeList[aei][0]
                searchVerticesQue.append([searchVertexIndNew[0], searchVertexIndNew[1], aei])
        if (addEdgesInd.any()):
            edgeCheckCounter[0, addEdgesInd.tolist()] = 1
        searchVerticeNumber += 1

    if returnDB:
        searchVerticesQue = [[edgeList[rootEdgeIndex][0][0], edgeList[rootEdgeIndex][0][1], rootEdgeIndex],
                             [edgeList[rootEdgeIndex][-1][0], edgeList[rootEdgeIndex][-1][1], rootEdgeIndex]]
        edgeCheckCounter = np.zeros((1, len(edgeList)))
        searchVerticeNumber = 0
        # edgeCheckCounter[0, rootEdgeIndex] = 1

        while (searchVerticeNumber <= len(searchVerticesQue) - 1):
            searchVertexPoint = np.array(searchVerticesQue[searchVerticeNumber][0:2])
            searchVertexEdgeIndex = searchVerticesQue[searchVerticeNumber][2]
            searchEdgeMeanPoint = np.mean(edgeList[searchVertexEdgeIndex], axis=0)
            searchEdgeVector = searchVertexPoint - searchEdgeMeanPoint
            searchVertexIndex = np.argwhere(np.all(vertices == searchVertexPoint, axis=1))[0, 0]
            linkedEdgesNumber = adjacencyMatrix[
                searchVertexIndex, np.where(adjacencyMatrix[searchVertexIndex, :] != 0)[0]]
            eligibleEdgesNumber = abs(linkedEdgesNumber)
            # Find the index of Edges that have not been searched
            eligibleEdgesInd = eligibleEdgesNumber[edgeCheckCounter[0, eligibleEdgesNumber - 1] == 0] - 1
            if (eligibleEdgesInd.any()):
                _, edgeEffectiveLength = _computeImportanceValue(searchVertexIndex, eligibleEdgesInd,
                                                                 verticesProperties,
                                                                 vertices, edgeList, adjacencyMatrix, depthLevel,
                                                                 gama)
                edgeEffectiveLength[edgeEffectiveLength < minEdgeLength1] = minEdgeLength1
                edgeEffectiveLength[edgeEffectiveLength > maxEdgeLength1] = maxEdgeLength1
                eligibleEdgesSecondPoint = np.array([edgeList[x][1] for x in eligibleEdgesInd])
                eligibleEdgesMainSkeletonBelief = mainSkeletonMap[eligibleEdgesSecondPoint.T.tolist()]
                eligibleEdgesLength = edgeProperties[2, eligibleEdgesInd]
                eligibleEdgesMeanPoint = np.array([np.mean(edgeList[x], axis=0) for x in eligibleEdgesInd])
                eligibleEdgesVector = eligibleEdgesMeanPoint - searchVertexPoint
                edgesCurvatureMeasure = np.dot(eligibleEdgesVector, searchEdgeVector.T) / (
                    np.linalg.norm(eligibleEdgesVector, axis=1) * np.linalg.norm(searchEdgeVector))
                edgesCurvatureMeasure[eligibleEdgesLength == 2] = 1
                # Fuzzy library does not support vector inputs, hence we should use loop
                for i, eei in enumerate(eligibleEdgesInd):
                    branchFIS.input['Main Skeleton Degree of Belief'] = eligibleEdgesMainSkeletonBelief[i]
                    branchFIS.input['Edge Length'] = eligibleEdgesLength[i]
                    branchFIS.input['Curvature'] = edgesCurvatureMeasure[i]
                    branchFIS.compute()
                    edgeBranchBelief = branchFIS.output['Branch Degree of Belief']
                    # Extracting branchMap
                    eligibleEdgePoints = np.array(edgeList[eei])
                    branchMap[eligibleEdgePoints.T.tolist()] = edgeBranchBelief
                # Adding selected edges' other point to the Que for next rounds
                    if ((linkedEdgesNumber == eei + 1).any()):
                        searchVertexIndNew = edgeList[eei][-1]
                    else:
                        searchVertexIndNew = edgeList[eei][0]
                    searchVerticesQue.append([searchVertexIndNew[0], searchVertexIndNew[1], eei])
            edgeCheckCounter[0, eligibleEdgesInd] = 1
            searchVerticeNumber += 1
        return skeletonNew, mainSkeletonMap,branchMap
    else:
        return skeletonNew