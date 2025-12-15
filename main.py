"""
Main execution script for RAG analysis experiment.
Orchestrates data loading, model inference, evaluation, and reporting.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import HotpotQALoader, chunk_text
from rag_pipeline import VectorStore, LLMGenerator, RAGPipeline
from evaluator import Evaluator
from utils import (
    setup_logging,
    set_random_seed,
    load_config,
    save_json,
    print_results_table,
    create_experiment_dir,
)


def main(config_path="config/config.yaml"):
    """Main execution function.

    Args:
        config_path: Path to configuration file (default: config/config.yaml)
                     Use config/config_fast.yaml for quick testing
    """
    # Load configuration
    config = load_config(config_path)

    # Setup
    set_random_seed(config["seed"])
    logger = setup_logging(config["paths"]["logs_dir"])

    logger.info("=" * 80)
    logger.info("RAG ANALYSIS EXPERIMENT - HOTPOT QA")
    logger.info("=" * 80)

    # Create experiment directory
    experiment_dir = create_experiment_dir("experiments", config["project"]["name"])

    # ========== STEP 1: Load and prepare data ==========
    logger.info("\n[STEP 1] Loading and preparing dataset...")

    data_loader = HotpotQALoader(
        subset_size=config["dataset"]["subset_size"], random_seed=config["seed"]
    )

    # Load dataset
    data_loader.load_dataset(split=config["dataset"]["split"])

    # Create subset
    subset = data_loader.create_subset(strategy=config["dataset"]["sampling_strategy"])

    # Save subset
    os.makedirs(config["paths"]["data_dir"], exist_ok=True)
    data_loader.save_subset(config["paths"]["dataset_subset"])

    # Prepare corpus for retrieval
    corpus = data_loader.prepare_corpus()
    save_json(corpus, config["paths"]["corpus"])

    logger.info(f"Dataset prepared: {len(subset)} examples, {len(corpus)} passages")

    # ========== STEP 2: Build retrieval index ==========
    logger.info("\n[STEP 2] Building retrieval index...")

    vector_store = VectorStore(
        encoder_model=config["retrieval"]["encoder_model"],
        reranker_config=config["retrieval"].get("reranker", {}),
        hybrid_config=config["retrieval"].get("hybrid", {}),
    )
    vector_store.build_index(corpus)

    # Save index
    vector_store.save_index(config["paths"]["index"])

    logger.info("Retrieval index built and saved")

    # ========== STEP 3: Load LLM ==========
    logger.info("\n[STEP 3] Loading language model...")

    generator = LLMGenerator(
        model_name=config["model"]["name"],
        device=config["model"]["device"],
        load_in_8bit=config["model"]["load_in_8bit"],
    )

    logger.info("Model loaded successfully")

    # ========== STEP 4: Create RAG pipeline ==========
    logger.info("\n[STEP 4] Creating RAG pipeline...")

    rag_pipeline = RAGPipeline(
        vector_store,
        generator,
        enforce_citations=config.get("validation", {}).get("enforce_citations", False),
        context_max_sentences=config["retrieval"].get("context_max_sentences", 6),
        sentences_per_passage=config["retrieval"].get("sentences_per_passage", 2),
    )

    logger.info("RAG pipeline ready")

    # ========== STEP 5: Run inference ==========
    logger.info("\n[STEP 5] Running inference on all conditions...")

    evaluator = Evaluator()
    predictions = {}

    generation_kwargs = {
        "temperature": config["model"].get("temperature", 0.7),
        "top_p": config["model"].get("top_p", 1.0),
        "repetition_penalty": config["model"].get("repetition_penalty", 1.0),
        "max_new_tokens": config["model"].get("max_new_tokens", 100),
        "seed": config["seed"],
    }

    # Process each example
    for i, example in enumerate(subset):
        processed_example = data_loader.preprocess_example(example)
        question = processed_example["question"]

        logger.info(f"\nProcessing example {i+1}/{len(subset)}: {question[:50]}...")

        # Condition 1: No RAG baseline
        logger.info("  - Running no-RAG baseline...")
        no_rag_result = rag_pipeline.answer_without_rag(question, **generation_kwargs)

        evaluator.evaluate_single(processed_example, no_rag_result, condition="no_rag")

        # Conditions 2-N: RAG with different k values
        for k in config["retrieval"]["k_values"]:
            logger.info(f"  - Running RAG with k={k}...")
            rag_result = rag_pipeline.answer_with_rag(
                question, k=k, **generation_kwargs
            )

            evaluator.evaluate_single(
                processed_example, rag_result, condition=f"rag_k{k}"
            )

    logger.info("\nInference completed on all conditions")

    # ========== STEP 6: Aggregate and analyze results ==========
    logger.info("\n[STEP 6] Aggregating and analyzing results...")

    # Save detailed results
    evaluator.save_results(config["paths"]["results_csv"])

    # Aggregate by condition
    aggregated = {}
    for condition in config["evaluation"]["conditions"]:
        metrics = evaluator.aggregate_results(condition=condition)
        aggregated[condition] = metrics

    # Print results table
    print_results_table(aggregated)

    # Save aggregated results
    save_json(aggregated, config["paths"]["aggregated_results"])

    # ========== STEP 7: Statistical comparisons ==========
    logger.info("\n[STEP 7] Running statistical comparisons...")

    comparisons = {}

    # Compare no-RAG vs each RAG condition
    for k in config["retrieval"]["k_values"]:
        comparison = evaluator.compare_conditions("no_rag", f"rag_k{k}")
        comparisons[f"no_rag_vs_rag_k{k}"] = comparison

        logger.info(f"\nComparison: no_rag vs rag_k{k}")
        logger.info(f"  EM difference: {comparison['em_diff']:.3f}")
        logger.info(f"  F1 difference: {comparison['f1_diff']:.3f}")
        logger.info(
            f"  Hallucination reduction: {comparison['hallucination_diff']:.3f}"
        )
        logger.info(f"  McNemar p-value: {comparison['mcnemar_p_value']:.4f}")

    # Save comparisons
    save_json(comparisons, os.path.join(experiment_dir, "statistical_comparisons.json"))

    # ========== STEP 8: Generate report ==========
    logger.info("\n[STEP 8] Generating final report...")

    report = {
        "experiment_config": config,
        "dataset_info": {"n_examples": len(subset), "n_passages": len(corpus)},
        "aggregated_metrics": aggregated,
        "statistical_comparisons": comparisons,
    }

    save_json(report, os.path.join(experiment_dir, "experiment_report.json"))

    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.info(f"Results saved to: {experiment_dir}")
    logger.info("=" * 80)

    return report


if __name__ == "__main__":
    try:
        # Check for command-line argument
        import sys

        if len(sys.argv) > 1:
            config_file = sys.argv[1]
        else:
            # Prefer fast config by default if available
            fast_cfg = Path("config/config_fast.yaml")
            if fast_cfg.exists():
                config_file = str(fast_cfg)
            else:
                config_file = "config/config.yaml"

        print(f"Using configuration: {config_file}")
        report = main(config_file)
        print("\n✅ Experiment completed successfully!")
        print(f"\nNext steps:")
        print("  1. Review results in results/ directory")
        print("  2. Run visualization notebooks in notebooks/")
        print("  3. Perform error analysis on failure cases")

    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
