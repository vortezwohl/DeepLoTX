from overrides import override

from deeplotx.similarity.string.base_string_similarity import BaseStringSimilarity


class LongestCommonSubstring(BaseStringSimilarity):
    def __init__(self, ignore_case: bool = False):
        super().__init__(ignore_case=ignore_case)

    @override
    def __call__(self, s_1: str, s_2: str) -> str:
        if self._ignore_case:
            s_1, s_2 = s_1.lower(), s_2.lower()
        m, n = len(s_1), len(s_2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_length = end_index = 0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s_1[i - 1] == s_2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] > max_length:
                        max_length = dp[i][j]
                        end_index = i - 1
                else:
                    dp[i][j] = 0
        if max_length == 0:
            return ''
        return s_1[end_index - max_length + 1: end_index + 1]

    @override
    def rank(self, s: str, S: list[str]) -> list[str]:
        lcs_list = sorted([(len(self.__call__(s, x)), x) for x in S], key=lambda x: x[0], reverse=True)
        return [x[1] for x in lcs_list]
