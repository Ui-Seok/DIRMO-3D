# 설명용 개별 사례

아래 사례는 장점과 한계를 설명하기 위해 선택했다. 전체 성능이나 대표성을 이 두 사례로 주장하지 않는다. 전체 비교는 REPORT.md의 동일 cohort 집계를 사용한다.

| 복합체 | 오차 | 방법 | Pocket 통과 | 전체 protein 통과 | 자유 영역 RMSD | 종합 성공 |
| --- | --- | --- | --- | --- | ---: | --- |
| 6h7k | large | input | True | True | 0.966 Å | False |
| 6h7k | large | UFF | False | False | 5.308 Å | False |
| 6h7k | large | base | True | True | 0.387 Å | True |
| 6h7k | large | v2 | True | True | 0.975 Å | False |
| 9abp | small | input | True | True | 0.580 Å | False |
| 9abp | small | UFF | True | True | 0.067 Å | True |
| 9abp | small | base | True | False | 0.716 Å | False |
| 9abp | small | v2 | True | True | 0.178 Å | True |

6h7k에서는 input 0.966 Å에서 base 0.387 Å로 복구되고 물리 검사도 통과했다. v2는 0.975 Å로 원래 pose에 더 가까워지지 못했다. UFF는 5.308 Å로 악화됐다.

9abp에서는 base가 0.580 → 0.716 Å로 악화되고 전체 protein 검사에도 실패했다. 같은 입력에서 UFF는 0.067 Å, v2는 0.178 Å로 개선됐다. Base가 평균적으로 우세해도 개별 복합체에서 항상 최선은 아니다.
