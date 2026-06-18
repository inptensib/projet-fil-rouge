"""Dual-Expert enhancement for FlatHydra Domain Expert.

Modules:
    losses              — focal_cross_entropy, effective-number class weights
    hierarchy_scorer    — HierarchyAwareScorer
    candidate_generator — TopKCandidateGenerator
    topk_metrics        — top-K recall, rare-bucket metrics, reports
    label_path_augmentation — AugmentedProductDataset
    category_definitions    — CategoryDefinitionStore
    llm_reranker        — LLMGeneralExpertReranker, PromptBuilder
    difficulty_router   — DifficultyRouter
    audit_logger        — AuditLogger
    dual_expert_classifier  — DualExpertProductClassifier (full pipeline)
"""
