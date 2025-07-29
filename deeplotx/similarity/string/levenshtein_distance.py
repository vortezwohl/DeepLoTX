from overrides import override

from deeplotx.similarity.string.base_string_similarity import BaseStringSimilarity


class LevenshteinDistance(BaseStringSimilarity):
    def __init__(self, ignore_case: bool = False):
        super().__init__(ignore_case=ignore_case)

    @override
    def __call__(self, s_1: str, s_2: str) -> int:
        if self._ignore_case:
            s_1, s_2 = s_1.lower(), s_2.lower()
        m, n = len(s_1), len(s_2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s_1[i - 1] == s_2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j - 1] + 1,
                        dp[i - 1][j] + 1,
                        dp[i][j - 1] + 1
                    )
        return dp[m][n]

    @override
    def rank(self, s: str, S: list[str]) -> list[str]:
        lcs_list = sorted([(self.__call__(s, x), x) for x in S], key=lambda x: x[0], reverse=False)
        return [x[1] for x in lcs_list]
