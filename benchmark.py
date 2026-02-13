"""
Performance Benchmarking and Optimization Tools

This module provides comprehensive benchmarking utilities to measure and
optimize the performance of the ticket classification system.
"""

import time
import numpy as np
import pandas as pd
from typing import List, Dict, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
from datetime import datetime
import json

from ticket_classifier import (
    TechnicalSupportSystem,
    TicketData,
    EmbeddingGenerator
)


@dataclass
class BenchmarkResult:
    """Results from a benchmark test"""
    name: str
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p50_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    throughput_per_sec: float
    total_runs: int


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking suite for the
    ticket classification system.
    """
    
    def __init__(self, system: TechnicalSupportSystem):
        self.system = system
        self.results: List[BenchmarkResult] = []
    
    def _run_benchmark(
        self,
        name: str,
        func: Callable,
        n_runs: int = 100,
        warmup_runs: int = 10
    ) -> BenchmarkResult:
        """
        Run a benchmark test multiple times and collect statistics
        
        Args:
            name: Name of the benchmark
            func: Function to benchmark
            n_runs: Number of test runs
            warmup_runs: Number of warmup runs to exclude
            
        Returns:
            BenchmarkResult with timing statistics
        """
        print(f"Running benchmark: {name}")
        print(f"  Warmup: {warmup_runs} runs")
        
        # Warmup runs
        for _ in range(warmup_runs):
            func()
        
        print(f"  Testing: {n_runs} runs")
        
        # Actual benchmark runs
        times = []
        for i in range(n_runs):
            start_time = time.perf_counter()
            func()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
            
            if (i + 1) % 10 == 0:
                print(f"    Progress: {i + 1}/{n_runs}")
        
        times = np.array(times)
        
        result = BenchmarkResult(
            name=name,
            mean_time_ms=float(np.mean(times)),
            std_time_ms=float(np.std(times)),
            min_time_ms=float(np.min(times)),
            max_time_ms=float(np.max(times)),
            p50_time_ms=float(np.percentile(times, 50)),
            p95_time_ms=float(np.percentile(times, 95)),
            p99_time_ms=float(np.percentile(times, 99)),
            throughput_per_sec=1000.0 / np.mean(times),
            total_runs=n_runs
        )
        
        self.results.append(result)
        
        print(f"  Mean: {result.mean_time_ms:.2f} ms")
        print(f"  P95: {result.p95_time_ms:.2f} ms")
        print(f"  Throughput: {result.throughput_per_sec:.2f} req/sec")
        print()
        
        return result
    
    def benchmark_classification(
        self,
        sample_ticket: TicketData,
        n_runs: int = 100
    ) -> BenchmarkResult:
        """Benchmark ticket classification performance"""
        
        def classify():
            self.system.classifier.predict(sample_ticket)
        
        return self._run_benchmark(
            "Classification",
            classify,
            n_runs=n_runs
        )
    
    def benchmark_recommendation(
        self,
        sample_ticket: TicketData,
        top_k: int = 5,
        n_runs: int = 100
    ) -> BenchmarkResult:
        """Benchmark knowledge base recommendation performance"""
        
        def recommend():
            self.system.recommender.recommend(sample_ticket, top_k=top_k)
        
        return self._run_benchmark(
            f"Recommendation (top-{top_k})",
            recommend,
            n_runs=n_runs
        )
    
    def benchmark_full_processing(
        self,
        sample_ticket: TicketData,
        top_k: int = 5,
        n_runs: int = 100
    ) -> BenchmarkResult:
        """Benchmark complete ticket processing"""
        
        def process():
            self.system.process_ticket(sample_ticket, recommend_articles=True, top_k=top_k)
        
        return self._run_benchmark(
            "Full Processing",
            process,
            n_runs=n_runs
        )
    
    def benchmark_embedding_generation(
        self,
        texts: List[str],
        n_runs: int = 50
    ) -> BenchmarkResult:
        """Benchmark embedding generation"""
        
        def generate():
            self.system.embedding_generator.generate_embeddings(texts, show_progress=False)
        
        return self._run_benchmark(
            f"Embedding Generation ({len(texts)} texts)",
            generate,
            n_runs=n_runs
        )
    
    def benchmark_batch_processing(
        self,
        tickets: List[TicketData],
        n_runs: int = 20
    ) -> BenchmarkResult:
        """Benchmark batch ticket processing"""
        
        def process_batch():
            for ticket in tickets:
                self.system.process_ticket(ticket, recommend_articles=True)
        
        return self._run_benchmark(
            f"Batch Processing ({len(tickets)} tickets)",
            process_batch,
            n_runs=n_runs
        )
    
    def benchmark_varying_text_lengths(
        self,
        base_ticket: TicketData,
        length_multipliers: List[int] = [1, 2, 5, 10],
        n_runs: int = 50
    ) -> List[BenchmarkResult]:
        """Benchmark performance with varying text lengths"""
        results = []
        
        for multiplier in length_multipliers:
            ticket = TicketData(
                ticket_id=base_ticket.ticket_id,
                title=base_ticket.title,
                description=base_ticket.description * multiplier,
                category=base_ticket.category,
                priority=base_ticket.priority,
                resolution_time=base_ticket.resolution_time
            )
            
            result = self._run_benchmark(
                f"Text Length {multiplier}x",
                lambda: self.system.process_ticket(ticket, recommend_articles=False),
                n_runs=n_runs
            )
            results.append(result)
        
        return results
    
    def benchmark_concurrent_requests(
        self,
        sample_ticket: TicketData,
        n_concurrent: int = 10,
        n_runs: int = 20
    ) -> BenchmarkResult:
        """Simulate concurrent request processing"""
        import concurrent.futures
        
        def process_concurrent():
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as executor:
                futures = [
                    executor.submit(self.system.process_ticket, sample_ticket, False)
                    for _ in range(n_concurrent)
                ]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
        
        return self._run_benchmark(
            f"Concurrent Requests ({n_concurrent} workers)",
            process_concurrent,
            n_runs=n_runs
        )
    
    def generate_report(self, output_file: str = 'benchmark_report.txt') -> str:
        """Generate a comprehensive benchmark report"""
        report = []
        report.append("=" * 80)
        report.append("Performance Benchmark Report")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        if not self.results:
            report.append("No benchmark results available.")
            return "\n".join(report)
        
        # Summary table
        report.append("Summary")
        report.append("-" * 80)
        report.append(f"{'Benchmark':<40} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Throughput':<15}")
        report.append("-" * 80)
        
        for result in self.results:
            report.append(
                f"{result.name:<40} "
                f"{result.mean_time_ms:<12.2f} "
                f"{result.p95_time_ms:<12.2f} "
                f"{result.throughput_per_sec:<15.2f}"
            )
        
        report.append("")
        
        # Detailed results
        report.append("Detailed Results")
        report.append("-" * 80)
        
        for result in self.results:
            report.append(f"\n{result.name}")
            report.append(f"  Mean:       {result.mean_time_ms:.2f} ms")
            report.append(f"  Std Dev:    {result.std_time_ms:.2f} ms")
            report.append(f"  Min:        {result.min_time_ms:.2f} ms")
            report.append(f"  Max:        {result.max_time_ms:.2f} ms")
            report.append(f"  P50:        {result.p50_time_ms:.2f} ms")
            report.append(f"  P95:        {result.p95_time_ms:.2f} ms")
            report.append(f"  P99:        {result.p99_time_ms:.2f} ms")
            report.append(f"  Throughput: {result.throughput_per_sec:.2f} requests/sec")
            report.append(f"  Runs:       {result.total_runs}")
        
        report_text = "\n".join(report)
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(f"\nReport saved to: {output_file}")
        
        return report_text
    
    def export_to_json(self, output_file: str = 'benchmark_results.json') -> None:
        """Export benchmark results to JSON"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'results': [
                {
                    'name': r.name,
                    'mean_time_ms': r.mean_time_ms,
                    'std_time_ms': r.std_time_ms,
                    'min_time_ms': r.min_time_ms,
                    'max_time_ms': r.max_time_ms,
                    'p50_time_ms': r.p50_time_ms,
                    'p95_time_ms': r.p95_time_ms,
                    'p99_time_ms': r.p99_time_ms,
                    'throughput_per_sec': r.throughput_per_sec,
                    'total_runs': r.total_runs
                }
                for r in self.results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results exported to: {output_file}")
    
    def plot_results(self, output_file: str = 'benchmark_plot.png') -> None:
        """Generate visualization of benchmark results"""
        if not self.results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Benchmark Results', fontsize=16)
        
        # 1. Mean processing time comparison
        ax1 = axes[0, 0]
        names = [r.name for r in self.results]
        means = [r.mean_time_ms for r in self.results]
        ax1.barh(names, means, color='skyblue')
        ax1.set_xlabel('Mean Time (ms)')
        ax1.set_title('Mean Processing Time')
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Percentile comparison
        ax2 = axes[0, 1]
        x = np.arange(len(names))
        width = 0.25
        ax2.bar(x - width, [r.p50_time_ms for r in self.results], width, label='P50')
        ax2.bar(x, [r.p95_time_ms for r in self.results], width, label='P95')
        ax2.bar(x + width, [r.p99_time_ms for r in self.results], width, label='P99')
        ax2.set_ylabel('Time (ms)')
        ax2.set_title('Percentile Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Throughput
        ax3 = axes[1, 0]
        throughput = [r.throughput_per_sec for r in self.results]
        ax3.barh(names, throughput, color='lightgreen')
        ax3.set_xlabel('Throughput (requests/sec)')
        ax3.set_title('System Throughput')
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Variability (std dev)
        ax4 = axes[1, 1]
        std_devs = [r.std_time_ms for r in self.results]
        ax4.barh(names, std_devs, color='salmon')
        ax4.set_xlabel('Standard Deviation (ms)')
        ax4.set_title('Performance Variability')
        ax4.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
        plt.close()


class MemoryProfiler:
    """Profile memory usage of the system"""
    
    def __init__(self):
        try:
            import psutil
            self.psutil = psutil
            self.available = True
        except ImportError:
            print("Warning: psutil not installed. Memory profiling unavailable.")
            self.available = False
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        if not self.available:
            return {}
        
        process = self.psutil.Process()
        mem_info = process.memory_info()
        
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,
            'vms_mb': mem_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
    
    def profile_function(self, func: Callable, name: str = "Function") -> Dict:
        """Profile memory usage of a function"""
        if not self.available:
            return {}
        
        import gc
        gc.collect()
        
        before = self.get_memory_usage()
        start_time = time.time()
        
        result = func()
        
        end_time = time.time()
        after = self.get_memory_usage()
        
        return {
            'name': name,
            'execution_time_s': end_time - start_time,
            'memory_before_mb': before['rss_mb'],
            'memory_after_mb': after['rss_mb'],
            'memory_delta_mb': after['rss_mb'] - before['rss_mb'],
            'peak_memory_mb': after['rss_mb']
        }


def run_comprehensive_benchmark(system: TechnicalSupportSystem) -> None:
    """Run a comprehensive benchmark suite"""
    
    print("=" * 80)
    print("Comprehensive Performance Benchmark")
    print("=" * 80)
    print()
    
    # Create sample ticket
    sample_ticket = TicketData(
        ticket_id="BENCH-001",
        title="VPN connection timeout issue",
        description="Cannot connect to VPN. Getting repeated timeout errors when trying to establish connection.",
        category="Network",
        priority="High",
        resolution_time=0.0
    )
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark(system)
    
    # Run benchmarks
    print("1. Classification Performance")
    benchmark.benchmark_classification(sample_ticket, n_runs=100)
    
    print("2. Recommendation Performance")
    benchmark.benchmark_recommendation(sample_ticket, top_k=5, n_runs=100)
    
    print("3. Full Processing Performance")
    benchmark.benchmark_full_processing(sample_ticket, top_k=5, n_runs=100)
    
    print("4. Embedding Generation")
    texts = [sample_ticket.full_text] * 10
    benchmark.benchmark_embedding_generation(texts, n_runs=50)
    
    print("5. Varying Text Lengths")
    benchmark.benchmark_varying_text_lengths(sample_ticket, [1, 2, 5], n_runs=30)
    
    # Generate outputs
    report = benchmark.generate_report('benchmark_report.txt')
    print("\n" + report)
    
    benchmark.export_to_json('benchmark_results.json')
    benchmark.plot_results('benchmark_plot.png')
    
    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)


def compare_model_configurations():
    """Compare performance of different model configurations"""
    
    configurations = [
        {
            'name': 'Fast (MiniLM + LR)',
            'embedding_model': 'all-MiniLM-L6-v2',
            'classifier_type': 'logistic_regression',
            'use_tfidf': False
        },
        {
            'name': 'Balanced (MiniLM + GB)',
            'embedding_model': 'all-MiniLM-L6-v2',
            'classifier_type': 'gradient_boosting',
            'use_tfidf': True
        },
        {
            'name': 'Accurate (MPNet + GB)',
            'embedding_model': 'all-mpnet-base-v2',
            'classifier_type': 'gradient_boosting',
            'use_tfidf': True
        }
    ]
    
    print("Model Configuration Comparison")
    print("=" * 80)
    
    # Note: This is a template - actual implementation would require
    # training each configuration which is time-consuming
    print("To run this comparison:")
    print("1. Train models with each configuration")
    print("2. Run benchmark on each")
    print("3. Compare accuracy vs. speed tradeoffs")


if __name__ == '__main__':
    print("Performance Benchmarking Tools")
    print()
    print("To run benchmark:")
    print("  1. Load a trained system")
    print("  2. Call run_comprehensive_benchmark(system)")
    print()
    print("Example:")
    print("  from ticket_classifier import TechnicalSupportSystem")
    print("  system = TechnicalSupportSystem.load('./trained_model')")
    print("  run_comprehensive_benchmark(system)")
