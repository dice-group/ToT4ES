from mlcroissant import Dataset

from wikies_datasets import get_wikies_dataset_config, resolve_jsonld_source


def print_first_item(record_name):
    for record in dataset.records(record_set=record_name):
        for key, val in record.items():
            if isinstance(val, bytes):
                val = str(val, "utf-8")
            print(f"{key}=[{val}]({type(val)})", end=", ")
        break
    print()


# Change this one value to switch between WikiES datasets.
DATASET_NAME = "wikiprofem-s"

dataset_config = get_wikies_dataset_config(DATASET_NAME)
jsonld_source = resolve_jsonld_source(DATASET_NAME, prefer_local=True)

print(f"Using dataset: {dataset_config['display_name']}")
print(f"JSON-LD source: {jsonld_source}")

dataset = Dataset(jsonld=jsonld_source)

print(dataset.metadata.record_sets)

print_first_item("entities")
print_first_item("root-entities")
print_first_item("predicates")
print_first_item("triples")
print_first_item("ground-truths")
""" The output of the above code:
wikes-dataset
[RecordSet(uuid="entities"), RecordSet(uuid="root-entities"), RecordSet(uuid="predicates"), RecordSet(uuid="triples"), RecordSet(uuid="ground-truths")]
id=[0](<class 'int'>), entity=[Q6387338](<class 'str'>), wikidata_label=[Ken Blackwell](<class 'str'>), wikidata_description=[American politician and activist](<class 'str'>), wikipedia_id=[769596](<class 'int'>), wikipedia_title=[Ken_Blackwell](<class 'str'>), 
entity=[9](<class 'int'>), category=[singer](<class 'str'>), 
id=[0](<class 'int'>), predicate=[P1344](<class 'str'>), predicate_label=[participant in](<class 'str'>), predicate_desc=[event in which a person or organization was/is a participant; inverse of P710 or P1923](<class 'str'>), 
subject=[1](<class 'int'>), predicate=[0](<class 'int'>), object=[778](<class 'int'>), 
root_entity=[9](<class 'int'>), subject=[9](<class 'int'>), predicate=[8](<class 'int'>), object=[31068](<class 'int'>), 
"""