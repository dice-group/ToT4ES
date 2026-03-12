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
        self.all_triples = []  # All triples before filtering (for incoming edges)
        self.entities_data = defaultdict(list)
        self.features_map = defaultdict(dict)
    
    def parse_desc_file(self):
        """Parse the description file."""
        try:
            all_entities_data = defaultdict(list)
            with open(self.desc_file, 'r', encoding='utf-8') as f:
                for line in f:
                    result = NTriplesHandler.parse_triple(line)
                    if result:
                        subject, predicate, obj, full_line = result
                        self.all_triples.append((subject, predicate, obj))
                        all_entities_data[subject].append((subject, predicate, obj))
            
            # Find and keep only the primary entity (most triples)
            if all_entities_data:
                primary_entity = max(all_entities_data.items(), key=lambda x: len(x[1]))[0]
                self.entities_data = {primary_entity: all_entities_data[primary_entity]}
                self.triples = all_entities_data[primary_entity]
            
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
        """Map selected features back to original triples.
        
        Searches both outgoing triples (entity as subject) and incoming triples
        (entity as object) from the desc file, since RELIN considers both
        directions per paper Def. 2.
        """
        mapping = {}
        
        for entity_uri, entity_summaries in summaries.items():
            mapping[entity_uri] = {}
            entity_clean = entity_uri.strip('<>')
            
            for k, summary in entity_summaries.items():
                selected_triples = []
                
                # Get selected features
                selected_features = set((f.prop, f.value) for f, _ in summary)
                matched_features = set()
                
                # Match against outgoing triples (entity is subject)
                for subject, predicate, obj in self.entities_data[entity_uri]:
                    prop_short = NTriplesHandler.extract_local_name(predicate)
                    obj_short = NTriplesHandler.extract_local_name(obj)
                    
                    if (prop_short, obj_short) in selected_features:
                        selected_triples.append((subject, predicate, obj))
                        matched_features.add((prop_short, obj_short))
                
                # Match against incoming triples (entity is object)
                unmatched = selected_features - matched_features
                if unmatched:
                    for subject, predicate, obj in self.all_triples:
                        obj_clean = obj.strip('<>')
                        src_clean = subject.strip('<>')
                        if obj_clean == entity_clean and src_clean != entity_clean:
                            p_short = NTriplesHandler.extract_local_name(predicate)
                            s_short = NTriplesHandler.extract_local_name(subject)
                            if (p_short, s_short) in unmatched:
                                selected_triples.append((subject, predicate, obj))
                                unmatched.discard((p_short, s_short))
                
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
    """Process all entities in a dataset using corpus-wide statistics (paper-aligned).
    
    Two-pass approach:
      Phase 1: Parse all entities, build corpus-wide phrase counts and co-occurrences
               for PMI relatedness (Sect. 4.1) and conditional informativeness (Sect. 4.2).
      Phase 2: Initialize RELIN once with the global statistics.
      Phase 3: Generate summaries for each entity.
    """
    print(f"\n{'=' * 80}")
    print(f"Processing {dataset_name.upper()} Dataset")
    print(f"{'=' * 80}\n")
    
    entity_dirs = sorted([d for d in Path(base_path).iterdir() if d.is_dir()], 
                         key=lambda x: int(x.name) if x.name.isdigit() else 0)
    
    # ===== PHASE 1: Parse all entities and build corpus-wide statistics =====
    # Paper Sect. 4.1 (Eq. 6): PMI requires corpus-wide phrase occurrence/co-occurrence.
    # Paper Sect. 4.2 (Eq. 8): Informativeness requires counting across ALL entities E.
    print("  Phase 1: Parsing all entities and building corpus statistics...")
    
    parsed_entities = {}  # entity_id -> (Entity, entity_uri, EntitySummarizer)
    phrase_counts = defaultdict(int)        # phrase -> number of entities containing it
    co_occurrence_counts = defaultdict(int)  # (p1, p2) -> number of entities with both
    
    for entity_dir in entity_dirs:
        entity_id = entity_dir.name
        desc_file = entity_dir / f"{entity_id}_desc.nt"
        
        if not desc_file.exists():
            continue
        
        try:
            summarizer = EntitySummarizer(str(desc_file))
            if not summarizer.parse_desc_file():
                continue
            
            for entity_uri, triples in summarizer.entities_data.items():
                entity = Entity(entity_uri)
                phrases_in_entity = set()
                
                for subject, predicate, obj in triples:
                    prop_short = NTriplesHandler.extract_local_name(predicate)
                    obj_short = NTriplesHandler.extract_local_name(obj)
                    # Paper Def. 2: feature = (property, value) for outgoing edges
                    entity.add_feature(prop_short, obj_short)
                    phrases_in_entity.add(prop_short)
                    phrases_in_entity.add(obj_short)
                
                # Paper Def. 2: "We actually consider both incoming and outgoing
                # edges (i.e. where e appears as target and source node)."
                # Check all triples in the desc file where this entity is the object.
                entity_clean = entity_uri.strip('<>')
                for s, p, o in summarizer.all_triples:
                    obj_clean = o.strip('<>')
                    src_clean = s.strip('<>')
                    if obj_clean == entity_clean and src_clean != entity_clean:
                        p_short = NTriplesHandler.extract_local_name(p)
                        s_short = NTriplesHandler.extract_local_name(s)
                        # Incoming edge: property is same, value is the source
                        entity.add_feature(p_short, s_short)
                        phrases_in_entity.add(p_short)
                        phrases_in_entity.add(s_short)
                
                # Count how many entities mention each phrase (for PMI, Eq. 6)
                for phrase in phrases_in_entity:
                    phrase_counts[phrase] += 1
                
                # Count co-occurrences: how many entities mention both phrases
                phrases_list = sorted(phrases_in_entity)
                for idx_a in range(len(phrases_list)):
                    for idx_b in range(idx_a + 1, len(phrases_list)):
                        co_occurrence_counts[(phrases_list[idx_a], phrases_list[idx_b])] += 1
                
                parsed_entities[entity_id] = (entity, entity_uri, summarizer)
        except Exception:
            continue
    
    total_entities = len(parsed_entities)
    print(f"    Parsed {total_entities} entities, "
          f"{len(phrase_counts)} unique phrases, "
          f"{len(co_occurrence_counts)} co-occurrence pairs")
    
    if total_entities == 0:
        print("    No entities found. Skipping dataset.")
        return 0, 0, 0
    
    # ===== PHASE 2: Initialize RELIN with corpus-wide statistics =====
    print("  Phase 2: Initializing RELIN with corpus-wide statistics...")
    
    # Paper Sect. 6: lambda=0.85, iterations=10
    relin = RELIN(lambda_param=0.85, iterations=10)
    
    # Train relatedness: total_docs = number of entity descriptions ("documents")
    # add_co_occurrence stores both directions internally
    relin.train_relatedness(
        dict(phrase_counts), dict(co_occurrence_counts), total_docs=total_entities
    )
    
    # Prepare informativeness using ALL entities
    # Paper Eq. 8: P(fp|fq) = |{e in E | fp,fq in FS(e)}| / |{e in E | fq in FS(e)}|
    all_entity_objects = [data[0] for data in parsed_entities.values()]
    relin.prepare_informativeness(all_entity_objects)
    
    # ===== PHASE 3: Generate summaries for each entity =====
    print(f"  Phase 3: Generating summaries...\n")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for i, entity_dir in enumerate(entity_dirs, 1):
        entity_id = entity_dir.name
        
        print(f"[{i:3d}/{len(entity_dirs)}] Entity {entity_id}...", end=" ")
        
        if entity_id not in parsed_entities:
            print("⊘ SKIP")
            skip_count += 1
            continue
        
        try:
            entity, entity_uri, summarizer = parsed_entities[entity_id]
            features = entity.get_features()
            
            # Generate summaries using the globally-initialized RELIN
            entity_results = {}
            for k in [5, 10]:
                if k <= len(features):
                    summary = relin.summarize(entity, k=k)
                    entity_results[k] = summary
                else:
                    entity_results[k] = []
            
            summaries = {entity_uri: entity_results}
            
            # Map selected features back to original triples
            triple_mapping = summarizer.map_selected_features_to_triples(summaries)
            
            # Write output files
            summarizer.write_output_files(triple_mapping, output_base, dataset_name, entity_id)
            
            # Count triples per k value
            triple_counts = {}
            for entity_data in triple_mapping.values():
                for k, triples in entity_data.items():
                    triple_counts[k] = len(triples)
            
            counts_str = ", ".join(f"top{k}={n}" for k, n in sorted(triple_counts.items()))
            print(f"✓ ({counts_str})")
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
    
    base_dir = "../../datasets/ESBM_benchmark_v1.2"
    faces_dir = "../../datasets/FACES"
    output_base = "output"
    
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
