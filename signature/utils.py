def constructSigStr(sig: dict) -> str :
    sigStr = ""
    for trait in sig.values():
        sigStr += f"{trait}-"
    
    return sigStr