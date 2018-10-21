def stack_pred(estimator, X, y, Xt, k, method):
    
    kf = sklearn.model_selection.KFold(n_splits=k, shuffle=True, random_state=0)
    
    X_t_new = []
    ans = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        estimator.fit(X_train, y_train)
        if (method == 'predict'):
            X_t_new.append(estimator.predict(X_test))
            ans.append(estimator.predict(Xt))
        else:
            X_t_new.append(estimator.predict_proba(X_test))
            ans.append(estimator.predict_proba(Xt))
    for i in range(1, len(X_t_new)):
        X_t_new[0] = np.concatenate((X_t_new[0], X_t_new[i]), axis = 0) 
    sX = X_t_new[0]
    for i in range(1, len(ans)):
        for j in range(len(ans[0])):
            ans[0][j] = ans[0][j] + ans[i][j] 
    for i in range(len(ans[0])):
        ans[0][i] = ans[0][i] / len(ans)
     
    return sX, ans[0]
