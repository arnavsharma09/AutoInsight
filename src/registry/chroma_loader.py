import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = ".chroma"
COLLECTION_NAME = "statistical_methodologies"

METHODOLOGY_DESCRIPTIONS = [
    "Independent samples t-test compares the means of two unrelated groups when data is normally distributed and measured on a continuous scale.",
    "Paired samples t-test compares means of two related groups or repeated measurements from the same subjects under normality assumption.",
    "Welch t-test compares two group means when variances are unequal, robust to heteroscedasticity between groups.",
    "Mann-Whitney U test is a non-parametric test comparing two independent groups without normality assumption using rank sums.",
    "Wilcoxon signed-rank test is a non-parametric alternative to paired t-test for comparing two related samples.",
    "One-way ANOVA tests equality of means across three or more independent groups assuming normality and homogeneity of variance.",
    "Welch ANOVA is a one-way ANOVA variant robust to unequal variances across groups.",
    "Kruskal-Wallis H test is a non-parametric alternative to one-way ANOVA comparing distributions of three or more groups.",
    "Two-way ANOVA examines the effect of two categorical independent variables on a continuous dependent variable.",
    "Repeated measures ANOVA tests differences across three or more time points or conditions for the same subjects.",
    "Friedman test is a non-parametric alternative to repeated measures ANOVA using rank-based comparisons.",
    "Tukey HSD post-hoc test performs all pairwise comparisons after ANOVA while controlling family-wise error rate.",
    "Bonferroni correction adjusts significance thresholds for multiple comparisons to control false discovery rate.",
    "Dunn test with Bonferroni correction is a post-hoc test for Kruskal-Wallis for pairwise non-parametric comparisons.",
    "Games-Howell post-hoc test is used after Welch ANOVA when group variances are unequal.",
    "Pearson correlation coefficient measures the linear relationship between two continuous normally distributed variables.",
    "Spearman rank correlation is a non-parametric measure of monotonic relationship between two variables robust to outliers.",
    "Kendall tau correlation is a non-parametric rank correlation suitable for small samples or many tied ranks.",
    "Point-biserial correlation measures the relationship between a continuous and a binary variable.",
    "Phi coefficient measures association between two binary categorical variables.",
    "Ordinary least squares regression models the linear relationship between a continuous outcome and one or more predictors.",
    "Multiple linear regression extends OLS to model the effect of several predictors on a continuous outcome simultaneously.",
    "Polynomial regression fits a curved relationship between predictor and outcome by adding higher-order terms.",
    "Logistic regression models the probability of a binary outcome as a function of one or more predictor variables.",
    "Ordinal logistic regression models an ordinal categorical outcome as a function of predictor variables.",
    "Poisson regression models count data outcomes assuming a Poisson distribution of the response variable.",
    "Ridge regression is a regularized linear regression that penalizes large coefficients to handle multicollinearity.",
    "Lasso regression uses L1 regularization for variable selection and shrinkage in high-dimensional regression problems.",
    "Quantile regression estimates conditional quantiles of the response variable robust to outliers and skewed targets.",
    "Log-transformed OLS regression applies log transformation to a skewed continuous target before fitting OLS.",
    "Shapiro-Wilk test assesses whether a sample comes from a normally distributed population for small to medium samples.",
    "Kolmogorov-Smirnov test compares a sample distribution against a reference distribution or two samples against each other.",
    "Anderson-Darling test is a goodness-of-fit test sensitive to deviations in the tails of the distribution.",
    "Descriptive statistics summarize a dataset using mean, median, standard deviation, IQR, skewness, and kurtosis.",
    "Distribution fitting uses maximum likelihood estimation to fit candidate distributions and KS test to select the best fit.",
    "Box plot analysis visualizes distribution shape, central tendency, spread, and outliers for one or more groups.",
    "Histogram with kernel density estimate visualizes the empirical distribution of a continuous variable.",
    "Chi-square test of independence determines whether two categorical variables are statistically independent.",
    "Fisher exact test is used instead of chi-square when expected cell frequencies are below five in a contingency table.",
    "McNemar test assesses changes in paired categorical data such as before-after binary outcomes.",
    "Cochran Q test is a non-parametric test for differences among three or more matched proportions.",
    "Cramér V measures the strength of association between two categorical variables as an effect size.",
    "Mann-Kendall trend test detects monotonic upward or downward trends in time series data without normality assumption.",
    "Sen slope estimator provides a robust non-parametric estimate of the magnitude of trend in time series.",
    "Augmented Dickey-Fuller test checks whether a time series is stationary by testing for a unit root.",
    "ARIMA modelling fits an autoregressive integrated moving average model to a univariate time series for forecasting.",
    "Seasonal decomposition separates a time series into trend, seasonal, and residual components.",
    "Autocorrelation function analysis examines correlation between a time series and its own lagged values.",
    "Levene test checks homogeneity of variance across groups as a prerequisite for ANOVA.",
    "Bartlett test is a parametric test for equality of variances across groups assuming normality.",
    "Breusch-Pagan test detects heteroscedasticity in regression residuals.",
    "Durbin-Watson test checks for autocorrelation in regression residuals.",
    "Variance inflation factor measures multicollinearity among predictors in a regression model.",
    "Cohen d measures the standardized effect size for the difference between two group means.",
    "Eta squared measures the proportion of variance in the outcome explained by the group factor in ANOVA.",
    "Statistical power analysis determines the minimum sample size required to detect a given effect size.",
    "Confidence interval estimation provides a range of plausible values for a population parameter.",
    "Grubbs test detects a single outlier in a normally distributed dataset.",
    "IQR-based outlier detection identifies extreme values as points beyond 1.5 times the interquartile range.",
    "Robust regression using Huber loss reduces the influence of outliers on regression coefficient estimates.",
    "Winsorization caps extreme values at a specified percentile to reduce the effect of outliers on analysis."
]


def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection(client=None):
    if client is None:
        client = get_chroma_client()
    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def load_methodologies():
    client = get_chroma_client()
    collection = get_collection(client)

    existing = collection.count()
    if existing >= len(METHODOLOGY_DESCRIPTIONS):
        print(f"[ChromaDB] Collection already loaded ({existing} documents). Skipping.")
        return collection

    print(f"[ChromaDB] Loading {len(METHODOLOGY_DESCRIPTIONS)} methodology descriptions...")
    ids = [f"method_{i:03d}" for i in range(len(METHODOLOGY_DESCRIPTIONS))]
    collection.upsert(documents=METHODOLOGY_DESCRIPTIONS, ids=ids)
    print(f"[ChromaDB] Done. Total documents: {collection.count()}")
    return collection


def query_methodologies(query: str, n_results: int = 3):
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0]


if __name__ == "__main__":
    load_methodologies()
    print("\nTest query: 'compare salaries between departments'")
    results = query_methodologies("compare salaries between departments", n_results=3)
    for i, doc in enumerate(results):
        print(f"  [{i+1}] {doc}")