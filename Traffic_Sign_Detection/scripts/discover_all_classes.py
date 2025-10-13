# discover_all_classes.py (fixed to preserve GTSRB class distinctions)
import os
import csv
import argparse
import re
from collections import defaultdict

def discover_gtsrb_classes(gtsrb_csv_path):
    """Discover all classes from GTSRB dataset"""
    print("Discovering GTSRB classes...")
    
    gtsrb_classes = set()
    if not os.path.exists(gtsrb_csv_path):
        print(f"GTSRB CSV not found at: {gtsrb_csv_path}")
        return gtsrb_classes
    
    try:
        with open(gtsrb_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'ClassId' in row:
                    gtsrb_classes.add(row['ClassId'])
        
        print(f"Found {len(gtsrb_classes)} GTSRB classes: {sorted(gtsrb_classes)}")
        return gtsrb_classes
    except Exception as e:
        print(f"Error reading GTSRB: {e}")
        return set()

def discover_pakistani_classes(pakistani_root):
    """Discover all classes from Pakistani dataset"""
    print("Discovering Pakistani classes...")
    
    pakistani_classes = set()
    if not os.path.exists(pakistani_root):
        print(f"Pakistani directory not found at: {pakistani_root}")
        return pakistani_classes
    
    try:
        for item in os.listdir(pakistani_root):
            item_path = os.path.join(pakistani_root, item)
            if os.path.isdir(item_path):
                pakistani_classes.add(item)
        
        print(f"Found {len(pakistani_classes)} Pakistani classes: {sorted(pakistani_classes)}")
        return pakistani_classes
    except Exception as e:
        print(f"Error reading Pakistani dataset: {e}")
        return set()

def get_concept_based_mapping(name, dataset_source=None):
    """
    Map names based on conceptual meaning rather than literal translation
    This ensures similar signs from different datasets go to the same unified class
    """
    original_name = name
    name_lower = name.lower()
    
    # CONCEPT-BASED MAPPING DICTIONARY - Only for Pakistani dataset
    # For GTSRB, we use the predefined mapping to preserve distinctions
    
    concept_map = {
        # ===== CURVE AND BEND WARNINGS =====
        r'left.bend': 'curve_left',
        r'curve.left': 'curve_left', 
        r'dangerous.curve.left': 'curve_left',
        r'hairpin.curve.left': 'curve_left',
        
        r'right.bend': 'curve_right',
        r'curve.right': 'curve_right',
        r'dangerous.curve.right': 'curve_right', 
        r'hairpin.curve.right': 'curve_right',
        
        r'zigzag.road': 'double_curve',
        r'double.curve': 'double_curve',
        r'winding.road': 'double_curve',
        
        # ===== SPEED BUMPS AND ROAD CONDITIONS =====
        r'speed.breaker': 'bumpy_road',
        r'bumpy.road': 'bumpy_road',
        r'road.bump': 'bumpy_road',
        
        r'slippery.road': 'slippery_road',
        r'slippery.road.surface': 'slippery_road',
        r'slippery.motorcycles': 'slippery_road',
        
        # ===== INTERSECTIONS AND CROSSINGS =====
        r'cross.roads': 'crossroads',
        r'crossroads': 'crossroads',
        r't.roads': 'crossroads',
        
        r'railway.crossing': 'railroad_crossing',
        r'railroad.crossing': 'railroad_crossing',
        r'rail.crossing': 'railroad_crossing',
        
        # ===== ROUNDABOUTS =====
        r'roundabout.ahead': 'roundabout',
        r'roundabout': 'roundabout',
        
        # ===== STOP AND YIELD =====
        r'give.way': 'yield',
        r'yield': 'yield',
        
        r'stop$': 'stop',
        r'stop[^a-z]': 'stop',
        r'stop.1': 'stop',
        r'stop.2': 'stop',
        
        # ===== NO PASSING/OVERTAKING - DISTINCT CLASSES =====
        r'no.passing$': 'no_passing',
        r'no.overtaking$': 'no_passing',
        
        r'no.passing.trucks': 'no_passing_trucks',
        r'no.passing.vehicles.over': 'no_passing_trucks',
        r'no.overtaking.trucks': 'no_passing_trucks',
        r'vehicles.over.3.5': 'no_passing_trucks',
        
        r'end.no.passing$': 'end_no_passing',
        r'end.of.no.passing$': 'end_no_passing',
        
        r'end.no.passing.trucks': 'end_no_passing_trucks',
        r'end.of.no.passing.vehicles': 'end_no_passing_trucks',
        r'end.no.overtaking.trucks': 'end_no_passing_trucks',
        
        # ===== PEDESTRIAN RELATED =====
        r'pedestrians$': 'pedestrians_crossing',
        r'pedestrians.crossing': 'pedestrians_crossing',
        
        # ===== VEHICLE RESTRICTIONS =====
        r'no.left.turn': 'no_left_turn',
        r'no.right.turn': 'no_right_turn', 
        r'no.u.turn': 'no_u_turn',
        r'u.turn': 'u_turn',
        
        # ===== PARKING =====
        r'no.parking': 'no_parking',
        r'parking$': 'parking',
        
        # ===== BRIDGE AND ROAD DIVIDES =====
        r'bridge.ahead': 'bridge_ahead',
        r'road.divides': 'road_divides',
        
        # ===== STEEP GRADES =====
        r'steep.descent': 'steep_descent',
        r'steep.ascent': 'steep_ascent',
        
        # ===== SPECIAL PAKISTANI SIGNS =====
        r'no.horns': 'no_horns',
        r'no.mobile.allowed': 'no_mobile_allowed',
        r'slow$': 'slow',
        r'sharp.right.turn': 'sharp_right_turn',
    }
    
    # Only apply concept mapping to Pakistani dataset
    if dataset_source == 'pakistani':
        for pattern, unified_concept in concept_map.items():
            if re.search(pattern, name_lower):
                return unified_concept
    
    return None

def normalize_class_name(name, dataset_source=None):
    """Normalize class names with concept-based mapping"""
    original_name = name
    name_lower = name.lower().replace(' ', '_').replace('-', '_')
    
    # First, try concept-based mapping (only for Pakistani dataset)
    concept_mapped = get_concept_based_mapping(name, dataset_source)
    if concept_mapped:
        return concept_mapped
    
    # Handle speed limits (most specific case)
    if ('speed_limit' in name_lower or 'speed' in name_lower) and any(char.isdigit() for char in name_lower):
        numbers = re.findall(r'\d+', name_lower)
        if numbers:
            return f"speed_limit_{numbers[0]}"
    
    # Dataset-specific handling for remaining cases
    if dataset_source == 'pakistani':
        # For Pakistani dataset, preserve original names for unmapped concepts
        name_clean = re.sub(r'\(.*?\)', '', name_lower)
        name_clean = re.sub(r'_+', '_', name_clean)
        name_clean = name_clean.strip('_')
        return name_clean if name_clean else original_name.lower()
    
    else:  # GTSRB or general case - return as is to preserve distinctions
        return name_lower

def create_comprehensive_mapping(gtsrb_classes, pakistani_classes):
    """Create a comprehensive mapping for all discovered classes"""
    
    unified_mapping = {}
    concept_validation = defaultdict(list)
    
    # Map GTSRB classes - PRESERVE ORIGINAL DISTINCTIONS
    gtsrb_standard_map = {
        '0': 'speed_limit_20', 
        '1': 'speed_limit_30', 
        '2': 'speed_limit_50',
        '3': 'speed_limit_60', 
        '4': 'speed_limit_70', 
        '5': 'speed_limit_80',
        '6': 'end_speed_limit_80', 
        '7': 'speed_limit_100', 
        '8': 'speed_limit_120',
        '9': 'no_passing',  # General no passing for all vehicles
        '10': 'no_passing_trucks',  # No passing for vehicles over 3.5 tons
        '11': 'right_of_way', 
        '12': 'priority_road', 
        '13': 'yield', 
        '14': 'stop', 
        '15': 'no_vehicles', 
        '16': 'no_trucks', 
        '17': 'no_entry', 
        '18': 'general_caution',
        '19': 'dangerous_curve_left', 
        '20': 'dangerous_curve_right',
        '21': 'double_curve', 
        '22': 'bumpy_road', 
        '23': 'slippery_road',
        '24': 'road_narrows_right', 
        '25': 'road_work', 
        '26': 'traffic_signals',
        '27': 'pedestrians_crossing', 
        '28': 'children_crossing', 
        '29': 'bicycles_crossing',
        '30': 'ice_snow', 
        '31': 'wild_animals', 
        '32': 'end_all_restrictions',
        '33': 'turn_right_ahead', 
        '34': 'turn_left_ahead', 
        '35': 'ahead_only',
        '36': 'go_straight_or_right', 
        '37': 'go_straight_or_left',
        '38': 'keep_right', 
        '39': 'keep_left', 
        '40': 'roundabout',
        '41': 'end_no_passing',  # End of general no passing restriction
        '42': 'end_no_passing_trucks'  # End of no passing for trucks restriction
    }
    
    # For GTSRB, use the predefined mapping directly without normalization
    for gtsrb_id in gtsrb_classes:
        unified_name = gtsrb_standard_map.get(gtsrb_id, f'gtsrb_unknown_{gtsrb_id}')
        unified_mapping[f'gtsrb_{gtsrb_id}'] = unified_name
        concept_validation[unified_name].append(f'gtsrb_{gtsrb_id}')
    
    # Map Pakistani classes with concept-based mapping
    for pakistani_class in pakistani_classes:
        unified_name = normalize_class_name(pakistani_class, 'pakistani')
        unified_mapping[f'pakistani_{pakistani_class}'] = unified_name
        concept_validation[unified_name].append(f'pakistani_{pakistani_class}')
    
    # Validate concept mapping
    print("\nCONCEPT MAPPING VALIDATION:")
    print("=" * 50)
    for unified_concept, sources in sorted(concept_validation.items()):
        print(f"{unified_concept}:")
        for source in sources:
            print(f"  - {source}")
    
    return unified_mapping

def save_discovery_report(gtsrb_classes, pakistani_classes, unified_mapping, output_dir):
    """Save a comprehensive discovery report"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw discovered classes
    with open(os.path.join(output_dir, 'discovered_classes_raw.txt'), 'w', encoding='utf-8') as f:
        f.write("RAW DISCOVERED CLASSES\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("GTSRB CLASSES:\n")
        for cls in sorted(gtsrb_classes):
            f.write(f"  {cls}\n")
        
        f.write("\nPAKISTANI CLASSES:\n")
        for cls in sorted(pakistani_classes):
            f.write(f"  {cls}\n")
    
    # Save unified mapping
    with open(os.path.join(output_dir, 'unified_class_mapping.txt'), 'w', encoding='utf-8') as f:
        f.write("UNIFIED CLASS MAPPING\n")
        f.write("=" * 50 + "\n\n")
        
        for original, unified in sorted(unified_mapping.items()):
            f.write(f"{original} → {unified}\n")
    
    # Create final classes.txt with unique unified names
    unified_classes = sorted(set(unified_mapping.values()))
    with open(os.path.join(output_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
        for cls in unified_classes:
            f.write(f"{cls}\n")
    
    print(f"\nDiscovery report saved to {output_dir}/")
    print(f"Found {len(gtsrb_classes)} GTSRB classes")
    print(f"Found {len(pakistani_classes)} Pakistani classes") 
    print(f"Created {len(unified_classes)} unified classes")
    
    return unified_classes

def main():
    parser = argparse.ArgumentParser(description='Discover all classes from datasets')
    parser.add_argument('--gtsrb_csv', required=True, help='Path to GTSRB Train.csv')
    parser.add_argument('--pakistani_root', required=True, help='Path to Pakistani dataset root')
    parser.add_argument('--output_dir', default='./output', help='Output directory for results')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CLASS DISCOVERY WITH CONCEPT MAPPING")
    print("=" * 60)
    
    # Discover classes from each dataset
    gtsrb_classes = discover_gtsrb_classes(args.gtsrb_csv)
    pakistani_classes = discover_pakistani_classes(args.pakistani_root)
    
    # Create comprehensive mapping
    unified_mapping = create_comprehensive_mapping(gtsrb_classes, pakistani_classes)
    
    # Save results
    unified_classes = save_discovery_report(gtsrb_classes, pakistani_classes, unified_mapping, args.output_dir)
    
    print(f"\nAll unified classes:")
    for i, cls in enumerate(unified_classes):
        print(f"  {i:3d}: {cls}")

if __name__ == "__main__":
    main()