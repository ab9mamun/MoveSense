from imblearn.over_sampling import ADASYN

def create_daily_level(basepath):
    # splitbyparticipants.doit(basepath)
    # daily_level_creation.doit21(basepath)
    pass


def balance_data(X, y):
    oversample = ADASYN(random_state=0)
    s1, s2, s3 = X.shape
    # print(X.shape, y.shape)
    # print(Counter(y))

    X = X.reshape(s1, s2 * s3)

    failed = False
    try:
        X, y = oversample.fit_resample(X, y)
        # print("ADASYN balanced the data successfully")
    except ValueError:
        failed = True
        # print("ADASYN denied service because the number of samples in different classes are already very close")

    n = X.shape[0]
    X = X.reshape(n, s2, s3)
    if not failed:
        # print(X.shape, y.shape)
        # print(Counter(y))
        pass

    return X, y
