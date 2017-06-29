warnings.filterwarnings("ignore", category=DeprecationWarning)

scores = []

for n_components in range(self.min_n_components, self.max_n_components + 1):

    model = self.base_model(n_components)
    logL = model.score(self.X, self.lengths)
    n = n_components
    f = self.X.shape[1]
    p = n*(n-1)+2*f*n
    logN = np.log(self.X.shape[0])
    BIC_score = -2 * logL + p * logN
    scores.append((BIC_score,n_components))

best_num_components=max(scores)


return self.base_model(best_num_components)
