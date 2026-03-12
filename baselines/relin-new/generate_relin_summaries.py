"""
RELIN Summary Generator - Output as N-Triples

Generate RELIN-selected entity summaries and output them
as RDF N-Triples files for evaluation against golden truth.
"""

import re
import os
from collections import defaultdict
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
        self.triples = []  # Store all triples
        self.entities_data = defaultdict(list)  # Store by entity
        self.features_map = defaultdict(dict)  # Map features to triples
    
    def parse_desc_file(self):
        """Parse the description file."""
        print(f"Parsing {os.path.basename(self.desc_file)}...")
        
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
    
    def get_entity_summaries(self, k_values=[5, 10], lambda_param=0.85):
        """Generate entity summaries using RELIN."""
        print(f"\nApplying RELIN (λ={lambda_param})...")
        
        results = {}
        
        for entity_uri, triples in self.entities_data.items():
            print(f"\nProcessing entity: {NTriplesHandler.extract_local_name(entity_uri)}")
            
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
            print(f"  Total features: {len(features)}")
            
            for k in k_values:
                if k <= len(features):
                    summary = relin.summarize(entity, k=k)
                    entity_results[k] = summary
                    print(f"  Top {k} selected")
                else:
                    print(f"  k={k} exceeds feature count")
            
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
                
                print(f"\nWriting: {filename}")
                print(f"  Triples: {len(triples)}")
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    for subject, predicate, obj in triples:
                        line = NTriplesHandler.format_triple(subject, predicate, obj)
                        f.write(line + '\n')
                
                # Also verify against gold files if they exist
                self.compare_with_gold(str(entity_id), k, filepath, output_dir)
    
    def compare_with_gold(self, entity_name, k, output_file, output_dir):
        """Compare generated summary with gold file."""
        gold_file = os.path.join(output_dir, f"{entity_name}_gold_top{k}_{0}.nt")
        
        if os.path.exists(gold_file):
            with open(output_file, 'r') as f:
                generated = set(line.strip() for line in f if line.strip())
            
            with open(gold_file, 'r') as f:
                gold = set(line.strip() for line in f if line.strip())
            
            overlap = len(generated & gold)
            precision = overlap / len(generated) if generated else 0
            recall = overlap / len(gold) if gold else 0
            
            print(f"  Comparison with gold (top{k}_0):")
            print(f"    Generated: {len(generated)} triples")
            print(f"    Gold: {len(gold)} triples")
            print(f"    Overlap: {overlap} triples")
            print(f"    Precision: {precision:.2%}")
            print(f"    Recall: {recall:.2%}")


def main():
    """Main execution."""
    print("=" * 80)
    print("RELIN N-Triples Generator")
    print("=" * 80)
    
    # Define paths
    input_dir = "/home/asepff/Documents/Github/dice/ToT4ES/relin/datasets/ESBM_benchmark_v1.2/dbpedia_data/1"
    desc_file = os.path.join(input_dir, "1_desc.nt")
    output_base = "/home/asepff/Documents/Github/dice/ToT4ES/relin/output"
    
    # Extract dataset name and entity ID from input path
    # Path: .../ESBM_benchmark_v1.2/dbpedia_data/1 -> dataset: dbpedia, entity: 1
    path_parts = input_dir.split(os.sep)
    dataset_name = path_parts[-2].replace('_data', '')  # dbpedia_data -> dbpedia
    entity_id = path_parts[-1]  # 1
    
    # Check if file exists
    if not os.path.exists(desc_file):
        print(f"Error: File not found: {desc_file}")
        return
    
    # Process
    summarizer = EntitySummarizer(desc_file)
    summarizer.parse_desc_file()
    
    # Generate summaries
    summaries = summarizer.get_entity_summaries(k_values=[5, 10], lambda_param=0.85)
    
    # Map to triples
    print("\nMapping selected features to triples...")
    triple_mapping = summarizer.map_selected_features_to_triples(summaries)
    
    # Write output files
    print("\nGenerating output files...")
    summarizer.write_output_files(triple_mapping, output_base, dataset_name, entity_id)
    
    # Construct final output directory for display
    final_output_dir = os.path.join(output_base, dataset_name, entity_id)
    
    print("\n" + "=" * 80)
    print("✅ Summary generation complete!")
    print("=" * 80)
    print(f"\nOutput files generated:")
    print(f"  - {entity_id}_top5.nt")
    print(f"  - {entity_id}_top10.nt")
    print(f"\nLocation: {final_output_dir}")


if __name__ == "__main__":
    main()
