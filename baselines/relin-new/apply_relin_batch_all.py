"""
Batch RELIN Summary Generator for All Entities

Process all entities from both dbpedia and lmdb datasets,
generating top-5 and top-10 summaries for each.
"""

import re
import os
import sys
from collections import defaultdict
from pathlib import Path
from relin import RELIN, Entity, Feature


class NTriplesHandler:
    """Handle RDF N-Triples parsing and generation."""
    
    @staticmethod
    def parse_triple(line):
        """Parse N-Triples line and extract components."""
        line = line.strip()
        if not line or line.startswith('#'):
            return None
        
        # Pattern: <subject> <predicate> <object> .
        match = re.match(r'(<[^>]+>)\s+(<[^>]+>)\s+(.+?)\s+\.$', line)
        if not match:
            return None
        
        subject = match.group(1)
        predicate = match.group(2)
        obj = match.group(3)
        
        return (subject, predicate, obj, line.rstrip(' .'))
    
    @staticmethod
    def format_triple(subject, predicate, obj):
        """Format as N-Triples line."""
        return f"{subject} {predicate} {obj} ."
    
    @staticmethod
    def extract_local_name(uri):
        """Extract local name from URI."""
        if uri.startswith('http://'):
            return uri.split('/')[-1]
        return uri


class EntitySummarizer:
    """Summarize entities and generate output files."""
    
    def __init__(self, desc_file):
        self.desc_file = desc_file
        self.triples = []
        self.entities_data = defaultdict(list)
        self.features_map = defaultdict(dict)
    
    def parse_desc_file(self):
        """Parse the description file."""
        try:
            with open(self.desc_file, 'r', encoding='utf-8') as f:
                for line in f:
                    result = NTriplesHandler.parse_triple(line)
                    if result:
                        subject, predicate, obj, full_line = result
                        self.triples.append((subject, predicate, obj))
                        self.entities_data[subject].append((subject, predicate, obj))
            
            # Find and keep only the primary entity (most triples)
            if self.entities_data:
                primary_entity = max(self.entities_data.items(), key=lambda x: len(x[1]))[0]
                # Filter to keep only primary entity
                self.entities_data = {primary_entity: self.entities_data[primary_entity]}
            
            return True
        except Exception as e:
            print(f"    ❌ Error parsing file: {e}")
            return False
    
    def get_entity_summaries(self, k_values=[5, 10], lambda_param=0.85):
        """Generate entity summaries using RELIN."""
        results = {}
        
        for entity_uri, triples in self.entities_data.items():
            # Create Entity object
            entity = Entity(entity_uri)
            
            # Extract simple property names for RELIN
            prop_freq = defaultdict(int)
            val_freq = defaultdict(int)
            co_occur = defaultdict(int)
            
            # Build feature set
            for subject, predicate, obj in triples:
                prop_short = NTriplesHandler.extract_local_name(predicate)
                obj_short = NTriplesHandler.extract_local_name(obj)
                
                entity.add_feature(prop_short, obj_short)
                prop_freq[prop_short] += 1
                val_freq[obj_short] += 1
            
            # Initialize RELIN
            relin = RELIN(lambda_param=lambda_param, iterations=15)
            
            corpus_stats = {**prop_freq, **val_freq}
            relin.train_relatedness(corpus_stats, dict(co_occur))
            relin.prepare_informativeness([entity])
            
            # Get summaries for different k values
            entity_results = {}
            features = entity.get_features()
            
            for k in k_values:
                if k <= len(features):
                    summary = relin.summarize(entity, k=k)
                    entity_results[k] = summary
                else:
                    entity_results[k] = []
            
            results[entity_uri] = entity_results
        
        return results
    
    def map_selected_features_to_triples(self, summaries):
        """Map selected features back to original triples."""
        mapping = {}
        
        for entity_uri, entity_summaries in summaries.items():
            mapping[entity_uri] = {}
            
            for k, summary in entity_summaries.items():
                selected_triples = []
                
                # Get selected features
                selected_features = set((f.prop, f.value) for f, _ in summary)
                
                # Match against original triples
                for subject, predicate, obj in self.entities_data[entity_uri]:
                    prop_short = NTriplesHandler.extract_local_name(predicate)
                    obj_short = NTriplesHandler.extract_local_name(obj)
                    
                    if (prop_short, obj_short) in selected_features:
                        selected_triples.append((subject, predicate, obj))
                
                mapping[entity_uri][k] = selected_triples
        
        return mapping
    
    def write_output_files(self, triple_mapping, output_base, dataset_name, entity_id):
        """Write selected triples to output files."""
        # Create hierarchical folder structure: output/{dataset}/{entity_id}/
        output_dir = os.path.join(output_base, dataset_name, str(entity_id))
        os.makedirs(output_dir, exist_ok=True)
        
        for entity_uri, k_data in triple_mapping.items():
            for k, triples in k_data.items():
                # Use entity_id for filename (e.g., "1_top5.nt", "101_top5.nt")
                filename = f"{entity_id}_top{k}.nt"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    for subject, predicate, obj in triples:
                        line = NTriplesHandler.format_triple(subject, predicate, obj)
                        f.write(line + '\n')


def process_dataset(base_path, dataset_name, output_base):
    """Process all entities in a dataset."""
    print(f"\n{'=' * 80}")
    print(f"Processing {dataset_name.upper()} Dataset")
    print(f"{'=' * 80}\n")
    
    entity_dirs = sorted([d for d in Path(base_path).iterdir() if d.is_dir()], 
                         key=lambda x: int(x.name) if x.name.isdigit() else 0)
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for i, entity_dir in enumerate(entity_dirs, 1):
        entity_id = entity_dir.name
        desc_file = entity_dir / f"{entity_id}_desc.nt"
        
        print(f"[{i:3d}/{len(entity_dirs)}] Entity {entity_id}...", end=" ")
        
        if not desc_file.exists():
            print(f"⊘ SKIP (no _desc.nt file)")
            skip_count += 1
            continue
        
        try:
            # Parse and process
            summarizer = EntitySummarizer(str(desc_file))
            if not summarizer.parse_desc_file():
                error_count += 1
                continue
            
            # Generate summaries
            summaries = summarizer.get_entity_summaries(k_values=[5, 10], lambda_param=0.85)
            
            # Map to triples
            triple_mapping = summarizer.map_selected_features_to_triples(summaries)
            
            # Write output files
            summarizer.write_output_files(triple_mapping, output_base, dataset_name, entity_id)
            
            # Count triples
            total_triples = sum(len(triples) for entity_data in triple_mapping.values() 
                              for triples in entity_data.values())
            
            print(f"✓ ({total_triples} triples)")
            success_count += 1
            
        except Exception as e:
            print(f"✗ ERROR: {str(e)[:50]}")
            error_count += 1
    
    print(f"\n{'-' * 80}")
    print(f"Results for {dataset_name.upper()}:")
    print(f"  ✓ Processed: {success_count}")
    print(f"  ⊘ Skipped:   {skip_count}")
    print(f"  ✗ Errors:    {error_count}")
    print(f"  Total:       {len(entity_dirs)}")
    
    return success_count, skip_count, error_count


def main():
    """Main batch processing."""
    print("\n" + "=" * 80)
    print("RELIN Batch Summary Generator - All Entities")
    print("=" * 80)
    
    base_dir = "/home/asepff/Documents/Github/dice/ToT4ES/relin/datasets/ESBM_benchmark_v1.2"
    faces_dir = "/home/asepff/Documents/Github/dice/ToT4ES/relin/datasets/FACES"
    output_base = "/home/asepff/Documents/Github/dice/ToT4ES/relin/output"
    
    # Process ESBM datasets
    dbpedia_success, dbpedia_skip, dbpedia_error = process_dataset(
        os.path.join(base_dir, "dbpedia_data"), 
        "dbpedia",
        output_base
    )
    
    lmdb_success, lmdb_skip, lmdb_error = process_dataset(
        os.path.join(base_dir, "lmdb_data"),
        "lmdb",
        output_base
    )
    
    # Process FACES dataset
    faces_success, faces_skip, faces_error = process_dataset(
        os.path.join(faces_dir, "faces_data"),
        "faces",
        output_base
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"\nDBpedia:")
    print(f"  ✓ Processed: {dbpedia_success}")
    print(f"  ⊘ Skipped:   {dbpedia_skip}")
    print(f"  ✗ Errors:    {dbpedia_error}")
    
    print(f"\nLMDB:")
    print(f"  ✓ Processed: {lmdb_success}")
    print(f"  ⊘ Skipped:   {lmdb_skip}")
    print(f"  ✗ Errors:    {lmdb_error}")
    
    print(f"\nFACES:")
    print(f"  ✓ Processed: {faces_success}")
    print(f"  ⊘ Skipped:   {faces_skip}")
    print(f"  ✗ Errors:    {faces_error}")
    
    total_success = dbpedia_success + lmdb_success + faces_success
    total_skip = dbpedia_skip + lmdb_skip + faces_skip
    total_error = dbpedia_error + lmdb_error + faces_error
    
    print(f"\nCombined:")
    print(f"  ✓ Total Processed: {total_success}")
    print(f"  ⊘ Total Skipped:   {total_skip}")
    print(f"  ✗ Total Errors:    {total_error}")
    
    print(f"\nOutput location: {output_base}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
