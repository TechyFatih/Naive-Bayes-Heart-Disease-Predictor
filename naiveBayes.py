def naive_bayes(priorProbs, evidenceProbs):
    outcomes = []
    for prior in priorProbs:
        evidenceProb = 1
        for evidence in evidenceProbs:
            evidenceProb *= evidence
        evidenceProb *= prior
        outcomes.append(evidenceProb)
    return outcomes.index(max(outcomes))
