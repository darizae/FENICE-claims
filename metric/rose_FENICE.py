import json
from typing import List, Dict, Any
from tqdm import tqdm
from metric.FENICE import FENICE


class RoSEFENICE(FENICE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Disable automatic claim extraction
        self.claims_cache = {}  # Will be populated from dataset
        self.claim_extractor = None

    def process_dataset(self, dataset_path: str, output_path: str):
        """Main processing pipeline for RoSE dataset"""
        # Load dataset
        with open(dataset_path, 'r') as f:
            full_dataset = json.load(f)

        results = {}

        # Process each subset (cnndm_test, xsum, etc.)
        for subset_name, subset_data in full_dataset.items():
            print(f"Processing subset: {subset_name}")
            subset_results = []

            # Batch processing for document caching
            documents = [item['source'] for item in subset_data]
            self.cache_sentences(documents)
            if self.use_coref:
                self.cache_coref(documents)

            # Process each record individually
            for item in tqdm(subset_data, desc=f"Processing {subset_name}"):
                record_result = self.process_record(item)
                subset_results.append(record_result)

            results[subset_name] = subset_results

        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    def process_record(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process single RoSE record with multiple claim sets"""
        record_id = item['record_id']
        document = item['source']
        summary = item['reference']

        # Initialize result structure
        result = {
            'record_id': record_id,
            'scores': {
                'reference_acus': {},
                'system_claims': {}
            }
        }

        # Process reference ACUs
        for acu_type, claims in item['reference_acus'].items():
            score = self.score_with_claims(document, summary, claims)
            result['scores']['reference_acus'][acu_type] = score

        # Process system claims
        for system_name, claims in item['system_claims'].items():
            score = self.score_with_claims(document, summary, claims)
            result['scores']['system_claims'][system_name] = score

        return result

    def score_with_claims(self, document: str, summary: str, claims: List[str]) -> float:
        """Compute FENICE score with predefined claims"""
        # Create fake batch for parent class compatibility
        fake_batch = [{'document': document, 'summary': summary}]

        # Checking used claims
        print(f"Using {len(claims)} custom claims:")
        for i, claim in enumerate(claims):
            print(f"{i + 1}. {claim}")

        # Cache claims (bypass extraction)
        summary_id = self.get_id(0, summary)
        self.claims_cache[summary_id] = claims

        # Get score using parent class logic
        scores = super().score_batch(fake_batch)
        return scores[0]['score']

    def cache_claims(self, summaries):
        """Override to disable automatic claim extraction"""
        pass  # No-op since we provide claims externally
