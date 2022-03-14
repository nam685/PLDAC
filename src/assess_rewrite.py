def assess(filename,canard_type, trec_type):
    _, val_tmp = createCQRdata(canard_type,trec_type)
    val_tmp["Raw"] = val_tmp["Source"].apply(lambda context: context[7:].split("||||",2)[0])
    predictions = pd.read_csv(filename)
    df = val_tmp.copy()
    df["Prediction"] = predictions["Generated Text"]
    return df