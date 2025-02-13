import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import List, Dict

import numpy as np
from datasets import tqdm

from metric.rose_FENICE import RoSEFENICE


class RoSEFENICEBatched(RoSEFENICE):
    def process_dataset(self, dataset_path: str, output_path: str):
        """Optimized batch processing with document grouping"""
        with open(dataset_path, 'r') as f:
            full_dataset = json.load(f)

        results = {}

        for subset_name, subset_data in full_dataset.items():
            print(f"Processing {subset_name} with batched optimization")

            # Group records by document to process once per document
            document_map = defaultdict(list)
            for item in subset_data:
                document_map[item['source']].append(item)

            # Process documents in parallel batches
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for doc, items in document_map.items():
                    futures.append(executor.submit(
                        self.process_document_batch, doc, items
                    ))

                subset_results = []
                for future in tqdm(as_completed(futures), total=len(futures)):
                    subset_results.extend(future.result())

            results[subset_name] = subset_results

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    def process_document_batch(self, document: str, items: List[Dict]):
        """Process all records sharing the same document"""
        # Cache document processing once
        doc_id = self.get_id(0, document)
        if doc_id not in self.sentences_cache:
            self.cache_sentences([document])
            if self.use_coref:
                self.cache_coref([document])

        # Batch all claims across summaries
        all_claims = []
        claim_mapping = []  # Tracks (item_idx, claim_type, system_name)

        for item_idx, item in enumerate(items):
            # Reference ACUs
            for acu_type, claims in item['reference_acus'].items():
                all_claims.extend(claims)
                claim_mapping.append((item_idx, 'reference_acus', acu_type, len(claims)))

            # System claims
            for sys_name, claims in item['system_claims'].items():
                all_claims.extend(claims)
                claim_mapping.append((item_idx, 'system_claims', sys_name, len(claims)))

        # Score all claims in parallel
        fake_batch = [{'document': document, 'summary': ''}] * len(all_claims)
        scores = self.score_batch(fake_batch, claims_override=all_claims)

        # Rebuild results
        results = [self._init_result_item(item) for item in items]
        score_ptr = 0

        for mapping in claim_mapping:
            item_idx, claim_type, name, num_claims = mapping
            claim_scores = scores[score_ptr:score_ptr + num_claims]
            avg_score = np.mean([s['score'] for s in claim_scores])

            results[item_idx]['scores'][claim_type][name] = avg_score
            score_ptr += num_claims

        return results

    def _init_result_item(self, item):
        return {
            'record_id': item['record_id'],
            'scores': {
                'reference_acus': {},
                'system_claims': {}
            }
        }

    def score_batch(self, batch: List[Dict], claims_override=None):
        """
        Child-class override that accepts the extra 'claims_override' argument,
        then calls the parent's score_batch without that argument.
        """
        if claims_override is not None:
            # Fill self.claims_cache or otherwise insert your externally provided claims
            for i, claims in enumerate(claims_override):
                summary_id = self.get_id(i, '')  # Or some other stable ID
                self.claims_cache[summary_id] = claims

        # Now call the parent *without* passing 'claims_override'
        return super().score_batch(batch)




