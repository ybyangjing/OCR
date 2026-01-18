#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional
import json


class StatisticalSignificance:

    
    def __init__(self, alpha: float = 0.05):
        """

        
        Args:
            alpha: Significance level, default0.05
        """
        self.alpha = alpha
    
    def paired_t_test(
        self, 
        scores_a: List[float], 
        scores_b: List[float],
        alternative: str = 'two-sided'
    ) -> Dict:
        """


        """
        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)
        
        if len(scores_a) != len(scores_b):
            raise ValueError("The lengths of the two score lists must be identical.")
        

        differences = scores_a - scores_b
        

        t_stat, p_value = stats.ttest_rel(scores_a, scores_b, alternative=alternative)
        

        ci = stats.t.interval(
            1 - self.alpha,
            len(differences) - 1,
            loc=np.mean(differences),
            scale=stats.sem(differences)
        )
        

        cohens_d = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences, ddof=1) > 0 else 0
        
        result = {
            'test_name': 'Paired t-test',
            'n_samples': len(scores_a),
            'mean_a': float(np.mean(scores_a)),
            'mean_b': float(np.mean(scores_b)),
            'mean_difference': float(np.mean(differences)),
            'std_difference': float(np.std(differences, ddof=1)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'confidence_interval_95': (float(ci[0]), float(ci[1])),
            'cohens_d': float(cohens_d),
            'significant': bool(p_value < self.alpha),
            'interpretation': self._interpret_paired_t(p_value, np.mean(differences))
        }
        
        return result
    
    def mcnemar_test(
        self,
        correct_a: List[bool],
        correct_b: List[bool]
    ) -> Dict:
        """


        """
        correct_a = np.array(correct_a, dtype=bool)
        correct_b = np.array(correct_b, dtype=bool)
        
        if len(correct_a) != len(correct_b):
            raise ValueError("Both lists must have same length")
        

        b = np.sum(correct_a & correct_b)
        c = np.sum(correct_a & ~correct_b)
        d = np.sum(~correct_a & correct_b)
        a = np.sum(~correct_a & ~correct_b)
        

        if c + d == 0:

            result = {
                'test_name': 'McNemar test',
                'n_samples': len(correct_a),
                'accuracy_a': float(np.mean(correct_a)),
                'accuracy_b': float(np.mean(correct_b)),
                'contingency_table': {
                    'both_correct': int(b),
                    'only_a_correct': int(c),
                    'only_b_correct': int(d),
                    'both_wrong': int(a)
                },
                'chi_square': None,
                'p_value': None,
                'significant': False,
                'interpretation': 'No disagreement between models, cannot perform test'
            }
        else:

            chi_square = (abs(c - d) - 1) ** 2 / (c + d)
            p_value = 1 - stats.chi2.cdf(chi_square, df=1)
            
            result = {
                'test_name': 'McNemar test',
                'n_samples': len(correct_a),
                'accuracy_a': float(np.mean(correct_a)),
                'accuracy_b': float(np.mean(correct_b)),
                'contingency_table': {
                    'both_correct': int(b),
                    'only_a_correct': int(c),
                    'only_b_correct': int(d),
                    'both_wrong': int(a)
                },
                'chi_square': float(chi_square),
                'p_value': float(p_value),
                'significant': bool(p_value < self.alpha),
                'interpretation': self._interpret_mcnemar(p_value, c, d)
            }
        
        return result
    
    def wilcoxon_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        alternative: str = 'two-sided'
    ) -> Dict:
        """

        """
        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)
        
        if len(scores_a) != len(scores_b):
            raise ValueError("The lengths of the two score lists must be identical.")
        

        try:
            stat, p_value = stats.wilcoxon(
                scores_a, scores_b,
                alternative=alternative,
                zero_method='wilcox'
            )
        except ValueError as e:

            result = {
                'test_name': 'Wilcoxon signed-rank test',
                'n_samples': len(scores_a),
                'median_a': float(np.median(scores_a)),
                'median_b': float(np.median(scores_b)),
                'statistic': None,
                'p_value': None,
                'significant': False,
                'interpretation': f'Cannot perform test: {str(e)}'
            }
            return result
        
        result = {
            'test_name': 'Wilcoxon signed-rank test',
            'n_samples': len(scores_a),
            'median_a': float(np.median(scores_a)),
            'median_b': float(np.median(scores_b)),
            'median_difference': float(np.median(scores_a - scores_b)),
            'statistic': float(stat),
            'p_value': float(p_value),
            'significant': bool(p_value < self.alpha),
            'interpretation': self._interpret_wilcoxon(p_value, np.median(scores_a - scores_b))
        }
        
        return result
    
    def bootstrap_confidence_interval(
        self,
        scores: List[float],
        n_bootstrap: int = 10000,
        ci_level: float = 0.95
    ) -> Dict:
        """

        """
        scores = np.array(scores)
        

        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        

        alpha = 1 - ci_level
        ci_lower = np.percentile(bootstrap_means, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        
        result = {
            'test_name': 'Bootstrap confidence interval',
            'n_samples': len(scores),
            'n_bootstrap': n_bootstrap,
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores, ddof=1)),
            'confidence_level': ci_level,
            'confidence_interval': (float(ci_lower), float(ci_upper)),
            'ci_width': float(ci_upper - ci_lower)
        }
        
        return result
    
    def compare_multiple_runs(
        self,
        runs_data: List[Dict],
        metric_name: str = 'cer'
    ) -> Dict:
        """

        """
        if len(runs_data) < 2:
            raise ValueError("Data from at least two runs")
        
        # 提取每次运行的指标值
        all_values = []
        for run in runs_data:
            if 'results' in run:
                values = [r.get(metric_name, 0) for r in run['results']]
                all_values.append(np.mean(values))
            elif metric_name in run:
                all_values.append(run[metric_name])
        
        all_values = np.array(all_values)
        

        mean = np.mean(all_values)
        std = np.std(all_values, ddof=1) if len(all_values) > 1 else 0
        sem = stats.sem(all_values) if len(all_values) > 1 else 0
        

        if len(all_values) > 1:
            ci = stats.t.interval(
                0.95,
                len(all_values) - 1,
                loc=mean,
                scale=sem
            )
        else:
            ci = (mean, mean)
        
        result = {
            'metric': metric_name,
            'n_runs': len(all_values),
            'mean': float(mean),
            'std': float(std),
            'sem': float(sem),
            'min': float(np.min(all_values)),
            'max': float(np.max(all_values)),
            'confidence_interval_95': (float(ci[0]), float(ci[1])),
            'values': [float(v) for v in all_values]
        }
        
        return result
    
    def _interpret_paired_t(self, p_value: float, mean_diff: float) -> str:

        if p_value >= self.alpha:
            return f"No significant difference (p={p_value:.4f} >= {self.alpha})"
        else:
            direction = "Model A performs better" if mean_diff < 0 else "Model B performs better"
            return f"Significant difference (p={p_value:.4f} < {self.alpha}): {direction}"
    
    def _interpret_mcnemar(self, p_value: float, c: int, d: int) -> str:

        if p_value >= self.alpha:
            return f"No significant difference (p={p_value:.4f} >= {self.alpha})"
        else:
            direction = "Model A performs better" if c > d else "Model B performs better"
            return f"Significant difference (p={p_value:.4f} < {self.alpha}): {direction}"
    
    def _interpret_wilcoxon(self, p_value: float, median_diff: float) -> str:

        if p_value >= self.alpha:
            return f"No significant difference (p={p_value:.4f} >= {self.alpha})"
        else:
            direction = "Model A performs better" if median_diff < 0 else "Model B performs better"
            return f"Significant difference (p={p_value:.4f} < {self.alpha}): {direction}"


def compare_two_models(
    results_a: List[Dict],
    results_b: List[Dict],
    model_name_a: str = "Model A",
    model_name_b: str = "Model B"
) -> Dict:

    stat_sig = StatisticalSignificance()
    
    # 提取指标
    cer_raw_a = [r['cer_raw'] for r in results_a]
    cer_raw_b = [r['cer_raw'] for r in results_b]
    
    cer_digits_a = [r['cer_digits'] for r in results_a]
    cer_digits_b = [r['cer_digits'] for r in results_b]
    
    exact_raw_a = [r['exact_match_raw'] for r in results_a]
    exact_raw_b = [r['exact_match_raw'] for r in results_b]
    
    exact_digits_a = [r['exact_match_digits'] for r in results_a]
    exact_digits_b = [r['exact_match_digits'] for r in results_b]
    

    comparison = {
        'model_a': model_name_a,
        'model_b': model_name_b,
        'n_samples': len(results_a),
        'cer_raw': {
            'paired_t_test': stat_sig.paired_t_test(cer_raw_a, cer_raw_b),
            'wilcoxon_test': stat_sig.wilcoxon_test(cer_raw_a, cer_raw_b)
        },
        'cer_digits': {
            'paired_t_test': stat_sig.paired_t_test(cer_digits_a, cer_digits_b),
            'wilcoxon_test': stat_sig.wilcoxon_test(cer_digits_a, cer_digits_b)
        },
        'exact_match_raw': {
            'mcnemar_test': stat_sig.mcnemar_test(exact_raw_a, exact_raw_b)
        },
        'exact_match_digits': {
            'mcnemar_test': stat_sig.mcnemar_test(exact_digits_a, exact_digits_b)
        }
    }
    
    return comparison


def print_statistical_report(comparison: Dict, output_file: Optional[str] = None):

    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("STATISTICAL SIGNIFICANCE ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nComparison: {comparison['model_a']} vs {comparison['model_b']}")
    report_lines.append(f"Number of samples: {comparison['n_samples']}")
    

    report_lines.append("\n" + "-" * 80)
    report_lines.append("CER (Raw) Analysis:")
    report_lines.append("-" * 80)
    
    t_test = comparison['cer_raw']['paired_t_test']
    report_lines.append(f"Paired t-test:")
    report_lines.append(f"  Mean {comparison['model_a']}: {t_test['mean_a']:.4f}")
    report_lines.append(f"  Mean {comparison['model_b']}: {t_test['mean_b']:.4f}")
    report_lines.append(f"  Mean difference: {t_test['mean_difference']:.4f} ± {t_test['std_difference']:.4f}")
    report_lines.append(f"  t-statistic: {t_test['t_statistic']:.4f}")
    report_lines.append(f"  p-value: {t_test['p_value']:.6f}")
    report_lines.append(f"  95% CI: [{t_test['confidence_interval_95'][0]:.4f}, {t_test['confidence_interval_95'][1]:.4f}]")
    report_lines.append(f"  Cohen's d: {t_test['cohens_d']:.4f}")
    report_lines.append(f"  ✓ {t_test['interpretation']}")
    
    wilcox = comparison['cer_raw']['wilcoxon_test']
    report_lines.append(f"\nWilcoxon signed-rank test:")
    report_lines.append(f"  Median {comparison['model_a']}: {wilcox['median_a']:.4f}")
    report_lines.append(f"  Median {comparison['model_b']}: {wilcox['median_b']:.4f}")
    if wilcox['statistic'] is not None:
        report_lines.append(f"  Statistic: {wilcox['statistic']:.4f}")
        report_lines.append(f"  p-value: {wilcox['p_value']:.6f}")
    report_lines.append(f"  ✓ {wilcox['interpretation']}")
    

    report_lines.append("\n" + "-" * 80)
    report_lines.append("CER (Digits) Analysis:")
    report_lines.append("-" * 80)
    
    t_test = comparison['cer_digits']['paired_t_test']
    report_lines.append(f"Paired t-test:")
    report_lines.append(f"  Mean {comparison['model_a']}: {t_test['mean_a']:.4f}")
    report_lines.append(f"  Mean {comparison['model_b']}: {t_test['mean_b']:.4f}")
    report_lines.append(f"  Mean difference: {t_test['mean_difference']:.4f} ± {t_test['std_difference']:.4f}")
    report_lines.append(f"  t-statistic: {t_test['t_statistic']:.4f}")
    report_lines.append(f"  p-value: {t_test['p_value']:.6f}")
    report_lines.append(f"  95% CI: [{t_test['confidence_interval_95'][0]:.4f}, {t_test['confidence_interval_95'][1]:.4f}]")
    report_lines.append(f"  Cohen's d: {t_test['cohens_d']:.4f}")
    report_lines.append(f"  ✓ {t_test['interpretation']}")
    

    report_lines.append("\n" + "-" * 80)
    report_lines.append("ExactMatch (Raw) Analysis:")
    report_lines.append("-" * 80)
    
    mcnemar = comparison['exact_match_raw']['mcnemar_test']
    report_lines.append(f"McNemar's test:")
    report_lines.append(f"  Accuracy {comparison['model_a']}: {mcnemar['accuracy_a']:.4f}")
    report_lines.append(f"  Accuracy {comparison['model_b']}: {mcnemar['accuracy_b']:.4f}")
    report_lines.append(f"  Contingency table:")
    report_lines.append(f"    Both correct: {mcnemar['contingency_table']['both_correct']}")
    report_lines.append(f"    Only {comparison['model_a']} correct: {mcnemar['contingency_table']['only_a_correct']}")
    report_lines.append(f"    Only {comparison['model_b']} correct: {mcnemar['contingency_table']['only_b_correct']}")
    report_lines.append(f"    Both wrong: {mcnemar['contingency_table']['both_wrong']}")
    if mcnemar['chi_square'] is not None:
        report_lines.append(f"  χ² statistic: {mcnemar['chi_square']:.4f}")
        report_lines.append(f"  p-value: {mcnemar['p_value']:.6f}")
    report_lines.append(f"  ✓ {mcnemar['interpretation']}")
    

    report_lines.append("\n" + "-" * 80)
    report_lines.append("ExactMatch (Digits) Analysis:")
    report_lines.append("-" * 80)
    
    mcnemar = comparison['exact_match_digits']['mcnemar_test']
    report_lines.append(f"McNemar's test:")
    report_lines.append(f"  Accuracy {comparison['model_a']}: {mcnemar['accuracy_a']:.4f}")
    report_lines.append(f"  Accuracy {comparison['model_b']}: {mcnemar['accuracy_b']:.4f}")
    report_lines.append(f"  Contingency table:")
    report_lines.append(f"    Both correct: {mcnemar['contingency_table']['both_correct']}")
    report_lines.append(f"    Only {comparison['model_a']} correct: {mcnemar['contingency_table']['only_a_correct']}")
    report_lines.append(f"    Only {comparison['model_b']} correct: {mcnemar['contingency_table']['only_b_correct']}")
    report_lines.append(f"    Both wrong: {mcnemar['contingency_table']['both_wrong']}")
    if mcnemar['chi_square'] is not None:
        report_lines.append(f"  χ² statistic: {mcnemar['chi_square']:.4f}")
        report_lines.append(f"  p-value: {mcnemar['p_value']:.6f}")
    report_lines.append(f"  ✓ {mcnemar['interpretation']}")
    
    report_lines.append("\n" + "=" * 80)
    

    report_text = "\n".join(report_lines)
    print(report_text)
    

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\nReport saved to: {output_file}")


if __name__ == "__main__":

    print("Statistical Significance Analysis Module")
    print("Import this module to use statistical tests in your evaluation scripts")
